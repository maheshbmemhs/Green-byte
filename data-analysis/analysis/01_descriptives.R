# Descriptive statistics per treatment (section 6.1)
source("analysis/R/setup.R")
message("01 descriptives")

runs <- load_runs()
metrics <- c("e_total_j", "client_energy_j", "server_energy_j", "e_per_word_j",
             "duration_s", "throughput_bytes_per_s", "output_words")
keys <- c("approach", "model", "input_modality", "content_size", "network_condition")

desc <- do.call(rbind, lapply(split(runs, runs[keys], drop = TRUE), function(g) {
  out <- g[1, keys]
  out$n_runs <- nrow(g)
  for (m in metrics) {
    v <- g[[m]]
    out[[paste0(m, "_mean")]] <- mean(v)
    out[[paste0(m, "_median")]] <- median(v)
    out[[paste0(m, "_sd")]] <- sd(v)
    out[[paste0(m, "_iqr")]] <- IQR(v)
    out[[paste0(m, "_var")]] <- var(v)
    out[[paste0(m, "_min")]] <- min(v)
    out[[paste0(m, "_max")]] <- max(v)
  }
  out
}))
desc <- desc[order(desc$approach, desc$model, desc$input_modality, desc$content_size,
                   desc$network_condition), ]
write.csv(desc, file.path(DIR_STATS, "descriptives.csv"), row.names = FALSE)
message("  stats:  ", file.path(DIR_STATS, "descriptives.csv"))

msd <- function(m, s, digits = 0, sd_digits = digits) {
  f <- function(x, d) formatC(x, format = "f", digits = d, big.mark = "{,}")
  sprintf("%s (%s)", f(m, digits), f(s, sd_digits))
}

rem <- desc[desc$approach == "remote", ]
rows <- list()
for (m in MODELS) for (mod in MODALITIES) for (s in SIZES) {
  g <- rem[rem$model == m & rem$input_modality == mod & rem$content_size == s, ]
  g <- g[match(NETWORKS, g$network_condition), ]
  rows[[length(rows) + 1]] <- data.frame(
    Model = MODEL_LABELS[[m]], Input = mod, Size = s,
    `E standard` = msd(g$e_total_j_mean[1], g$e_total_j_sd[1]),
    `E degraded` = msd(g$e_total_j_mean[2], g$e_total_j_sd[2]),
    `E bad` = msd(g$e_total_j_mean[3], g$e_total_j_sd[3]),
    `T standard` = msd(g$duration_s_mean[1], g$duration_s_sd[1], 1),
    `T degraded` = msd(g$duration_s_mean[2], g$duration_s_sd[2], 1),
    `T bad` = msd(g$duration_s_mean[3], g$duration_s_sd[3], 1), check.names = FALSE)
}
tab <- do.call(rbind, rows)
names(tab) <- c("Model", "Input", "Size", "$E$ std.", "$E$ degr.", "$E$ bad",
                "$T$ std.", "$T$ degr.", "$T$ bad")
write_latex_table(tab, "tab_descriptives_remote.tex",
                  paste("Remote inference: mean (SD) total energy $E_{total}$ (J, laptop + server)",
                        "and execution time $T$ (s) per treatment, 30 runs each."),
                  "tab:descriptives-remote", align = "lll rrr rrr", wide = TRUE, group = "Model")

loc <- desc[desc$approach == "local", ]
tab <- data.frame(Model = MODEL_LABELS[as.character(loc$model)], Input = loc$input_modality,
                  Size = loc$content_size,
                  `$E_{total}$ (J)` = msd(loc$e_total_j_mean, loc$e_total_j_sd),
                  `$E_{word}$ (J/word)` = msd(loc$e_per_word_j_mean, loc$e_per_word_j_sd, 2),
                  `$T$ (s)` = msd(loc$duration_s_mean, loc$duration_s_sd, 1),
                  `$\\Theta$ (B/s)` = msd(loc$throughput_bytes_per_s_mean, loc$throughput_bytes_per_s_sd),
                  check.names = FALSE)
cap <- "On-device inference: mean (SD) per treatment, 30 runs each."
write_latex_table(tab, "tab_descriptives_local.tex", cap, "tab:descriptives-local",
                  align = "lll rrrr", wide = TRUE, group = "Model")

cmp_rows <- list()
for (m in MODELS) for (mod in MODALITIES) for (s in SIZES) {
  g <- desc[desc$model == m & desc$input_modality == mod & desc$content_size == s, ]
  l <- g[g$approach == "local", ]
  r <- g[g$approach == "remote", ][match(NETWORKS, g$network_condition[g$approach == "remote"]), ]
  cmp_rows[[length(cmp_rows) + 1]] <- data.frame(
    Model = MODEL_LABELS[[m]], Input = mod, Size = s,
    `E on-dev.` = msd(l$e_total_j_mean, l$e_total_j_sd),
    `E std.` = msd(r$e_total_j_mean[1], r$e_total_j_sd[1]),
    `E degr.` = msd(r$e_total_j_mean[2], r$e_total_j_sd[2]),
    `E bad` = msd(r$e_total_j_mean[3], r$e_total_j_sd[3]),
    `T on-dev.` = msd(l$duration_s_mean, l$duration_s_sd, 1, 2),
    `T std.` = msd(r$duration_s_mean[1], r$duration_s_sd[1], 1, 2),
    `T degr.` = msd(r$duration_s_mean[2], r$duration_s_sd[2], 1, 2),
    `T bad` = msd(r$duration_s_mean[3], r$duration_s_sd[3], 1, 2), check.names = FALSE)
}
tab <- do.call(rbind, cmp_rows)
names(tab) <- c("Model", "Input", "Size", "On-device", "Std.", "Degr.", "Bad",
                "On-device", "Std.", "Degr.", "Bad")
write_latex_table(tab, "tab_descriptives_compact.tex",
                  paste("Mean (SD) of the total energy $E_{total}$ in J (left) and of the execution time $T$ in s",
                        "(right) per treatment, 30 runs each. Remote columns: standard, degraded and bad",
                        "network; remote energy is laptop + server. Median, variance, minimum and maximum",
                        "are in the replication package."),
                  "tab:descriptives", align = "lll rrrr rrrr", wide = TRUE, group = "Model",
                  header_above = "\\multicolumn{3}{c}{} & \\multicolumn{4}{c}{$E_{total}$ (J)} & \\multicolumn{4}{c}{$T$ (s)} \\\\ \\cmidrule(lr){4-7} \\cmidrule(lr){8-11}")

stats_table <- function(metric, unit, digits, file, label, what) {
  f <- function(x, d = digits) formatC(x, format = "f", digits = d, big.mark = "{,}")
  d <- desc
  d$setting <- ifelse(d$approach == "local", "On-device", paste("Remote,", d$network_condition))
  d$setting <- factor(d$setting, levels = c("On-device", paste("Remote,", NETWORKS)))
  d$model <- factor(d$model, levels = MODELS)
  d$input_modality <- factor(d$input_modality, levels = MODALITIES)
  d$content_size <- factor(d$content_size, levels = SIZES)
  d <- d[order(d$model, d$input_modality, d$content_size, d$setting), ]
  v <- function(s) d[[paste0(metric, "_", s)]]
  tab <- data.frame(Model = MODEL_LABELS[as.character(d$model)], Input = d$input_modality,
                    Size = d$content_size, Setting = d$setting,
                    Mean = f(v("mean")), Median = f(v("median")), SD = f(v("sd")),
                    Variance = f(v("var"), if (digits == 0) 0 else digits + 1),
                    Min = f(v("min")), Max = f(v("max")), check.names = FALSE)
  write_latex_table(tab, file,
                    sprintf("Summary statistics of %s per treatment (30 runs each).%s", what, if (metric == "e_total_j") " Remote energy is laptop + server." else ""),
                    label, align = "llll rrrrrr", wide = TRUE, group = "Model")
}
stats_table("e_total_j", "J", 0, "tab_stats_energy.tex", "tab:stats-energy", "the total energy $E_{total}$ (J)")
stats_table("duration_s", "s", 2, "tab_stats_time.tex", "tab:stats-time", "the execution time $T$ (s)")

# Pooled summary per model and setting (one-column table for the report)
runs$setting <- factor(ifelse(runs$approach == "local", "On-device", paste("Remote,", runs$network_condition)),
                       levels = c("On-device", paste("Remote,", NETWORKS)))
setting_short <- c("On-device" = "On-device", "Remote, standard" = "Remote std.",
                   "Remote, degraded" = "Remote degr.", "Remote, bad" = "Remote bad")
sci <- function(x) {
  e <- floor(log10(x))
  ifelse(e < 4, formatC(x, format = "f", digits = 1, big.mark = "{,}"), sprintf("$%.1f{\\cdot}10^{%d}$", x / 10^e, e))
}
pooled_rows <- function(metric, digits) {
  f <- function(x) formatC(x, format = "f", digits = digits, big.mark = "{,}")
  out <- character(0)
  for (m in MODELS) {
    g <- runs[runs$model == m, ]
    for (st in levels(runs$setting)) {
      v <- g[[metric]][g$setting == st]
      lab <- if (st == "On-device") MODEL_LABELS[[m]] else ""
      out <- c(out, paste0(paste(lab, setting_short[[st]], f(mean(v)), f(median(v)), f(sd(v)), sci(var(v)),
                                 f(min(v)), f(max(v)), sep = " & "), " \\\\"))
    }
    if (m != tail(MODELS, 1)) out <- c(out, "\\addlinespace[2pt]")
  }
  out
}
block <- function(title, rows) c(sprintf("\\multicolumn{8}{l}{\\textit{%s}} \\\\", title), "\\midrule", rows)
writeLines(c("\\begin{table}[!htb]", "\\centering", "\\small",
             paste0("\\caption{Summary statistics per model and setting. Each row pools the 180 runs of one ",
                    "model and setting (2 input modalities $\\times$ 3 content sizes $\\times$ 30 repetitions), ",
                    "so the spread mainly reflects the different workloads. Remote energy is laptop + server; ",
                    "Var.\\ = variance (J$^2$, s$^2$).}"),
             "\\label{tab:summary}",
             "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%",
             "\\begin{tabular}{ll rrrrrr}", "\\toprule",
             "Model & Setting & Mean & Median & SD & Var. & Min & Max \\\\", "\\midrule",
             block("Total energy $E_{total}$ (J)", pooled_rows("e_total_j", 0)), "\\midrule",
             block("Execution time $T$ (s)", pooled_rows("duration_s", 2)),
             "\\bottomrule", "\\end{tabular}}", "\\end{table}"),
           file.path(DIR_TABLES, "tab_summary_pooled.tex"))
message("  table:  ", file.path(DIR_TABLES, "tab_summary_pooled.tex"))

# Anomaly check: zero/negative values and outliers per treatment (1.5 and 3 x IQR)
k <- c("approach", "model", "input_modality", "content_size", "network_condition")
flag <- function(v, m) { q <- quantile(v, c(.25, .75)); i <- q[2] - q[1]; as.numeric(v < q[1] - m * i | v > q[2] + m * i) }
anom <- do.call(rbind, lapply(c("e_total_j", "duration_s", "output_words"), function(m) {
  o15 <- ave(runs[[m]], runs[k], FUN = function(v) flag(v, 1.5))
  o3 <- ave(runs[[m]], runs[k], FUN = function(v) flag(v, 3))
  do.call(rbind, lapply(c("local", "remote"), function(a) {
    s <- runs$approach == a
    data.frame(metric = m, approach = a, runs = sum(s), zero_or_negative = sum(runs[[m]][s] <= 0),
               missing = sum(is.na(runs[[m]][s])), outliers_1_5_iqr = sum(o15[s]), outliers_3_iqr = sum(o3[s]))
  }))
}))
write.csv(anom, file.path(DIR_STATS, "anomaly_check.csv"), row.names = FALSE)
message("  stats:  ", file.path(DIR_STATS, "anomaly_check.csv"))
