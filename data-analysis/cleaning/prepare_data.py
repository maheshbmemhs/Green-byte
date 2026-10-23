#!/usr/bin/env python3
"""Builds the clean data set (data/clean/) from the raw Experiment Runner output in data/raw/."""
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "clean"

# RAPL package counter restarts after 2^18 J (seen in the data on both machines)
RAPL_WRAP_J = 262144.0

MODEL_LABELS = {"qwen3b": "Qwen2.5-VL-3B", "gemma4b": "Gemma 3 4B", "qwen7b": "Qwen2.5-VL-7B"}
MODEL_ORDER = ["qwen3b", "gemma4b", "qwen7b"]
SIZE_BY_TARGET = {100: "short", 500: "medium", 1000: "long"}
WORD_RANGE = {100: (50, 100), 500: (300, 500), 1000: (800, 1000)}
NETWORK_LABELS = {"normal": "standard", "degraded": "degraded", "bad": "bad"}


def energy_from_csv(path):
    out = dict(cpu_j=np.nan, gpu_j=np.nan, energy_j=np.nan, window_s=np.nan, start_epoch_s=np.nan,
               samples=0, rapl_wraps=0, status="missing_log")
    if not path.exists():
        return out
    df = pd.read_csv(path)
    out["samples"] = len(df)
    if len(df) < 2:
        out["status"] = "too_few_samples"
        return out
    t = df["Time"].to_numpy(float) / 1000.0  # epoch ms -> s
    if np.any(np.diff(t) < 0):
        out["status"] = "time_not_monotonic"
        return out
    out["window_s"] = t[-1] - t[0]
    out["start_epoch_s"] = t[0]  # run start (used to check order / warm-up effects)

    # counter overflow: a big drop means the counter restarted, add the range back
    diff = np.diff(df["PACKAGE_ENERGY (J)"].to_numpy(float))
    wrapped = diff < -RAPL_WRAP_J / 2
    diff[wrapped] += RAPL_WRAP_J
    if np.any(diff < -1e-6):
        out["status"] = "counter_decrease_unexplained"
        return out
    out["cpu_j"] = diff.sum()
    out["rapl_wraps"] = int(wrapped.sum())

    power_w = df["GPU0_POWER (mWatts)"].to_numpy(float) / 1000.0
    out["gpu_j"] = np.trapezoid(power_w, t)
    out["energy_j"] = out["cpu_j"] + out["gpu_j"]
    out["status"] = "ok_rapl_corrected" if wrapped.any() else "ok"
    return out


def request_info(path):
    req = json.loads(path.read_text())
    content = req["messages"][0]["content"]
    parts = [content] if isinstance(content, str) else content
    text, image_bytes = "", 0
    for p in parts:
        if isinstance(p, str):
            text += p
        elif p.get("type") == "text":
            text += p["text"]
        elif p.get("type") == "image_url":
            data = p["image_url"]["url"].split(",", 1)[1]
            image_bytes += len(base64.b64decode(data))
    return dict(input_bytes=len(text.encode("utf-8")) + image_bytes,
                request_max_tokens=req.get("max_tokens"),
                temperature=req.get("temperature"))


def response_info(path):
    resp = json.loads(path.read_text())
    choice = resp["choices"][0]
    text = choice["message"]["content"] or ""
    return dict(finish_reason=choice.get("finish_reason"),
                output_words=len(text.split()),
                output_bytes=len(text.encode("utf-8")),
                completion_tokens=resp["usage"]["completion_tokens"],
                prompt_tokens=resp["usage"]["prompt_tokens"],
                created=resp.get("created"))


def find_sessions():
    sessions = []
    for table in sorted(RAW.glob("**/main/run_table.csv")):
        if "attempts" in table.parts:
            continue
        main = table.parent
        model_dir = main.parent
        session_dir = model_dir.parent
        tbl = pd.read_csv(table, dtype=str, keep_default_na=False)
        approach = "remote" if "network_condition" in tbl.columns else "local"
        sessions.append(dict(main=main, model=model_dir.name, approach=approach,
                             session=str(session_dir.relative_to(RAW)), table=tbl,
                             has_warmup=(model_dir / "warmup").is_dir(),
                             session_json=session_dir / "session.json"))
    return sessions


def process_session(s):
    tbl = s["table"].set_index("__run_id")
    rows = []
    for run_dir in sorted(p for p in s["main"].iterdir() if p.is_dir() and p.name.startswith("run_")):
        meta = json.loads((run_dir / "run_metadata.json").read_text())
        t = tbl.loc[run_dir.name] if run_dir.name in tbl.index else {}
        r = dict(approach=s["approach"], model=meta["model"], session=s["session"],
                 run_id=run_dir.name, repetition=int(run_dir.name.rsplit("_", 1)[1]),
                 input_modality=meta["prompt_type"], target_words=int(meta["target_words"]),
                 max_tokens=int(meta["max_tokens"]),
                 network_condition=NETWORK_LABELS.get(meta.get("network_condition"), "none (local)"),
                 runner_status=t.get("__done", "missing_from_run_table") if len(t) else "missing_from_run_table",
                 had_warmup=s["has_warmup"], raw_run_dir=str(run_dir.relative_to(ROOT)))
        r.update(request_info(run_dir / "response.request.json"))
        r.update(response_info(run_dir / "response.json"))

        dur = pd.to_numeric(t.get("duration_s", ""), errors="coerce") if len(t) else np.nan
        r["duration_source"] = "run_table"

        if s["approach"] == "local":
            client = energy_from_csv(run_dir / "energibridge.csv")
            server = dict(cpu_j=0.0, gpu_j=0.0, energy_j=0.0, window_s=np.nan, start_epoch_s=np.nan, samples=0,
                          rapl_wraps=0, status="not_applicable")
            runner_client = pd.to_numeric(t.get("mean_energy_usage", ""), errors="coerce") if len(t) else np.nan
            runner_server = np.nan
        else:
            client = energy_from_csv(run_dir / "local_energy.csv")
            server = energy_from_csv(run_dir / "remote_energy.csv")
            runner_client = pd.to_numeric(t.get("local_mean_energy_usage", ""), errors="coerce")
            runner_server = pd.to_numeric(t.get("remote_mean_energy_usage", ""), errors="coerce")
        r["duration_s"] = dur

        for side, e in (("client", client), ("server", server)):
            r[f"{side}_cpu_j"] = e["cpu_j"]
            r[f"{side}_gpu_j"] = e["gpu_j"]
            r[f"{side}_energy_j"] = e["energy_j"]
            r[f"{side}_window_s"] = e["window_s"]
            r[f"{side}_start_epoch_s"] = e["start_epoch_s"]
            r[f"{side}_rapl_wraps"] = e["rapl_wraps"]
            r[f"{side}_energy_status"] = e["status"]
        r["runner_client_cpu_j"] = runner_client
        r["runner_server_cpu_j"] = runner_server
        rows.append(r)
    return rows


def session_time(rows):
    return max(r["created"] or 0 for r in rows)


def impute_local_duration(df):
    local = df.approach.eq("local")
    offset = (df.duration_s - df.client_window_s)[local].groupby(df.model).median()
    miss = local & df.duration_s.isna()
    df.loc[miss, "duration_s"] = df.loc[miss, "client_window_s"] + df.loc[miss, "model"].map(offset)
    df.loc[miss, "duration_source"] = "imputed_energibridge_window_plus_offset"
    return df


def derive(df):
    df = impute_local_duration(df)
    df["model_label"] = df["model"].map(MODEL_LABELS)
    df["content_size"] = df["target_words"].map(SIZE_BY_TARGET)
    df["e_total_j"] = df["client_energy_j"] + df["server_energy_j"]
    df["e_per_word_j"] = df["e_total_j"] / df["output_words"]
    df["throughput_bytes_per_s"] = df["output_bytes"] / df["duration_s"]
    lo = df["target_words"].map(lambda t: WORD_RANGE[t][0])
    hi = df["target_words"].map(lambda t: WORD_RANGE[t][1])
    df["within_word_range"] = df["output_words"].between(lo, hi)
    df["flag_truncated_at_max_tokens"] = df["finish_reason"].eq("length")
    df["flag_rapl_corrected"] = (df["client_rapl_wraps"] + df["server_rapl_wraps"]) > 0
    df["flag_runner_incomplete"] = df["runner_status"].ne("DONE")
    df["flag_energy_invalid"] = df["e_total_j"].isna()
    return df


def matched_relative(df):
    keys = ["model", "input_modality", "target_words", "repetition"]
    loc = df[df.approach == "local"][keys + ["e_total_j", "e_per_word_j", "output_words", "duration_s"]]
    rem = df[df.approach == "remote"][keys + ["network_condition", "e_total_j", "e_per_word_j",
                                              "output_words", "duration_s",
                                              "client_energy_j", "server_energy_j",
                                              "throughput_bytes_per_s"]]
    m = rem.merge(loc, on=keys, suffixes=("_remote", "_local"), how="inner")
    m["R_E"] = m["e_total_j_remote"] / m["e_total_j_local"]
    m["R_T"] = m["duration_s_remote"] / m["duration_s_local"]
    m["R_E_word"] = m["e_per_word_j_remote"] / m["e_per_word_j_local"]
    m["model_label"] = m["model"].map(MODEL_LABELS)
    m["content_size"] = m["target_words"].map(SIZE_BY_TARGET)
    return m


def to_r_csv(df, path):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].map({True: "TRUE", False: "FALSE"})
    df.to_csv(path, index=False, float_format="%.6g")


def write_report(df, matched, used, superseded):
    L = ["# Cleaning report", "",
         "Generated by `cleaning/prepare_data.py`. Raw data was only read.", "",
         "## Sessions used", ""]
    for s in used:
        L.append(f"- {s['approach']:6s} {s['model']:8s} `{s['session']}` "
                 f"({s['n']} runs, warm-up folder: {'yes' if s['has_warmup'] else 'NO'})")
    if superseded:
        L += ["", "Superseded (older) sessions, not used:"]
        L += [f"- {s['approach']} {s['model']} `{s['session']}`" for s in superseded]
    L += ["", "Excluded on purpose: `data/raw/remote_model/remote_model1/attempts/` (failed resume attempts; "
          "the runs were repeated and are present in the main session).", ""]

    L += ["## RAPL counter overflow (corrected)", "",
          f"The CPU package counter wraps at {RAPL_WRAP_J:.0f} J (2^18 J). When a reading drops by more "
          "than half of that, 2^18 J is added to all later readings (course announcement method). "
          "The runner had left these runs blank or failed on them.", ""]
    w = df[df.flag_rapl_corrected]
    for _, r in w.iterrows():
        side = "client" if r.client_rapl_wraps else "server"
        L.append(f"- `{r.raw_run_dir}` ({side})")
    L.append(f"\nTotal: {len(w)} runs.")

    inc = df[df.flag_runner_incomplete]
    L += ["", "## Runs the runner marked incomplete (recovered)", ""]
    for _, r in inc.iterrows():
        L.append(f"- `{r.raw_run_dir}`: status {r.runner_status}; energy from raw log, "
                 f"duration from {r.duration_source}")

    bad = df[df.flag_energy_invalid]
    L += ["", f"## Runs with unusable energy: {len(bad)}", ""]
    L += [f"- `{r.raw_run_dir}`: client={r.client_energy_status}, server={r.server_energy_status}"
          for _, r in bad.iterrows()]

    for side in ("client", "server"):
        ok = df[f"runner_{side}_cpu_j"].notna() & ~df.flag_rapl_corrected
        dev = (df.loc[ok, f"{side}_cpu_j"] - df.loc[ok, f"runner_{side}_cpu_j"]).abs().max()
        L.append(f"\nCheck: recomputed {side} CPU energy vs runner value, max |diff| = {dev:.4f} J "
                 f"over {ok.sum()} runs.")

    L += ["", "## Output length vs word target", "",
          "| approach | model | modality | target | max_tokens | median words | truncated | in range |",
          "|---|---|---|---|---|---|---|---|"]
    g = df.groupby(["approach", "model", "input_modality", "target_words"]).agg(
        mt=("max_tokens", "first"), w=("output_words", "median"),
        tr=("flag_truncated_at_max_tokens", "mean"), ok=("within_word_range", "mean"))
    for (a, m, p, t), r in g.iterrows():
        L.append(f"| {a} | {m} | {p} | {t} | {r.mt:.0f} | {r.w:.0f} | {r.tr:.0%} | {r.ok:.0%} |")

    L += ["", "## Measurement notes", "",
          f"- Median energy window minus request duration: client "
          f"{(df.client_window_s - df.duration_s)[df.approach == 'remote'].median():.2f} s (remote), "
          f"{(df.client_window_s - df.duration_s)[df.approach == 'local'].median():.2f} s (local); "
          f"server {(df.server_window_s - df.duration_s).median():.2f} s. The server window is longer "
          "because server profiling starts before and stops after the request.",
          "- Energy covers the full EnergiBridge window, as recorded by the runner.",
          "- DRAM energy is 0 J in every log (not reported by the hardware), so it is not included.",
          "- Outputs are deterministic (temperature 0): all repetitions of a treatment produce the same text.",
          "- Word count = whitespace split of the full response (title and field labels included)."]
    (OUT / "cleaning_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    groups = {}
    for s in find_sessions():
        s["rows"] = process_session(s)
        s["n"] = len(s["rows"])
        groups.setdefault((s["approach"], s["model"]), []).append(s)

    used, superseded, rows = [], [], []
    for key, ss in groups.items():
        ss.sort(key=lambda s: session_time(s["rows"]))
        used.append(ss[-1])
        superseded += ss[:-1]
        rows += ss[-1]["rows"]

    df = derive(pd.DataFrame(rows))
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    df = df.sort_values(["approach", "model", "input_modality", "target_words",
                         "network_condition", "repetition"]).reset_index(drop=True)
    df["model"] = df["model"].astype(str)
    matched = matched_relative(df)

    to_r_csv(df.drop(columns=["created"]), OUT / "all_runs_clean.csv")
    to_r_csv(matched, OUT / "matched_local_remote.csv")
    write_report(df, matched, used, superseded)

    print(f"runs: {len(df)}  (local {sum(df.approach == 'local')}, remote {sum(df.approach == 'remote')})")
    print(f"RAPL-corrected runs: {df.flag_rapl_corrected.sum()}, unusable energy: {df.flag_energy_invalid.sum()}")
    print(f"matched pairs: {len(matched)}")
    print(f"wrote {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
