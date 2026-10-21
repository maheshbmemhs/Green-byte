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

msd <- function(m, s, digits = 0) {
  f <- function(x) formatC(x, format = "f", digits = digits, big.mark = "{,}")
  sprintf("%s (%s)", f(m), f(s))
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
  f1 <- function(x) formatC(x, format = "f", digits = 1)
  cmp_rows[[length(cmp_rows) + 1]] <- data.frame(
    Model = MODEL_LABELS[[m]], Input = mod, Size = s,
    `E on-dev.` = msd(l$e_total_j_mean, l$e_total_j_sd),
    `E std.` = msd(r$e_total_j_mean[1], r$e_total_j_sd[1]),
    `E degr.` = msd(r$e_total_j_mean[2], r$e_total_j_sd[2]),
    `E bad` = msd(r$e_total_j_mean[3], r$e_total_j_sd[3]),
    `T on-dev.` = f1(l$duration_s_mean), `T std.` = f1(r$duration_s_mean[1]),
    `T degr.` = f1(r$duration_s_mean[2]), `T bad` = f1(r$duration_s_mean[3]), check.names = FALSE)
}
tab <- do.call(rbind, cmp_rows)
names(tab) <- c("Model", "Input", "Size", "On-device", "Std.", "Degr.", "Bad",
                "On-device", "Std.", "Degr.", "Bad")
write_latex_table(tab, "tab_descriptives_compact.tex",
                  paste("Mean (SD) total energy $E_{total}$ in J (left) and mean execution time $T$ in s",
                        "(right) per treatment, 30 runs each. Remote columns: standard, degraded and bad",
                        "network; remote energy is laptop + server."),
                  "tab:descriptives", align = "lll rrrr rrrr", wide = TRUE, group = "Model",
                  header_above = "\\multicolumn{3}{c}{} & \\multicolumn{4}{c}{$E_{total}$ (J)} & \\multicolumn{4}{c}{$T$ (s)} \\\\ \\cmidrule(lr){4-7} \\cmidrule(lr){8-11}")
