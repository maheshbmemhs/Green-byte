# Summary table of all hypothesis tests + network effects table
source("analysis/R/setup.R")
message("08 hypothesis summary")

read_stats <- function(f) read.csv(file.path(DIR_STATS, f), stringsAsFactors = FALSE)
tests <- rbind(read_stats("rq1_anova.csv"), read_stats("rq1_3_anova.csv"), read_stats("rq2_anova.csv"))
tests <- tests[!duplicated(tests[c("model", "metric", "effect")]), ]

H <- data.frame(rbind(
  c("RQ1",      "Size ($R_E$)", "R_E", "size"),
  c("RQ1",      "Modality ($R_E$)", "R_E", "modality"),
  c("RQ1",      "Network ($R_E$)", "R_E", "network"),
  c("RQ1.1-A",  "Approach ($E$)", "E_total (standard network)", "approach"),
  c("RQ1.1-B",  "Size ($E$)",         "E_total (standard network)", "size"),
  c("RQ1.1-AB", "Approach $\\times$ size ($E$)",     "E_total (standard network)", "approach x size"),
  c("RQ1.2-A",  "Approach ($E$)", "E_total (standard network)", "approach"),
  c("RQ1.2-B",  "Modality ($E$)",       "E_total (standard network)", "modality"),
  c("RQ1.2-AB", "Approach $\\times$ modality ($E$)",   "E_total (standard network)", "approach x modality"),
  c("RQ1.3-N",  "Network ($E_{remote}$)",       "Total energy", "network"),
  c("RQ1.3-G",  "Network ($R_E$)",                 "R_E", "network"),
  c("RQ2",      "Network ($R_E$)",                 "R_E", "network"),
  c("RQ2",      "Network ($R_T$)",                 "R_T", "network"),
  c("RQ2",      "Network (throughput)",   "Throughput", "network")),
  stringsAsFactors = FALSE)
names(H) <- c("hypothesis", "h1", "metric", "effect")

out <- merge(H, tests[c("model", "metric", "effect", "primary_test", "F_art", "p_primary",
                        "eta_sq_primary")], by = c("metric", "effect"), sort = FALSE)
if (any(is.na(out$p_primary)) || nrow(out) != nrow(H) * length(MODELS)) {
  stop("hypothesis summary: some hypotheses have no matching test result")
}
out$rq_family <- sub("-.*", "", out$hypothesis)
out$p_bh <- ave(out$p_primary, out$rq_family, out$model, FUN = function(p) p.adjust(p, "BH"))
out$decision <- ifelse(out$p_bh < ALPHA, "reject H0", "fail to reject H0")
out$model <- factor(out$model, levels = MODELS)
out <- out[order(match(out$hypothesis, unique(H$hypothesis)), match(out$h1, H$h1), out$model), ]
write.csv(out, file.path(DIR_STATS, "hypothesis_summary.csv"), row.names = FALSE)
message(sprintf("  stats:  %s (%d tests, H0 rejected in %d after BH)", file.path(DIR_STATS, "hypothesis_summary.csv"),
                nrow(out), sum(out$decision == "reject H0")))

wide <- unique(out[c("hypothesis", "h1")])
for (m in MODELS) {
  g <- out[out$model == m, ]
  g <- g[match(paste(wide$hypothesis, wide$h1), paste(g$hypothesis, g$h1)), ]
  wide[[MODEL_LABELS[[m]]]] <- sprintf("%s%s (%.2f)", fmt_p(g$p_bh),
                                       ifelse(g$p_bh < ALPHA, "", "$^{\\dagger}$"), g$eta_sq_primary)
}
names(wide)[1:2] <- c("Hyp.", "Effect (metric)")
write_latex_table(wide, "tab_hypothesis_summary.tex",
                  paste("Hypothesis tests per model (ART ANOVA): effect tested and its metric ($E$ = total",
                        "energy on the standard network). Cells: $p$ after Benjamini--Hochberg correction",
                        "within each research question ($\\eta^2_p$). $H_0$ is rejected when $p < 0.05$;",
                        "$\\dagger$ = fail to reject $H_0$."),
                  "tab:hypotheses", align = "ll rrr", group = "Hyp.")

net <- rbind(read_stats("rq1_3_network_summary.csv"), read_stats("rq2_network_summary.csv"))
net <- net[net$metric %in% c("Total energy", "Client (laptop) energy", "Execution time", "Throughput"), ]
net$metric <- factor(net$metric, levels = c("Total energy", "Client (laptop) energy", "Execution time", "Throughput"))
net <- net[order(match(net$model, MODELS), net$metric), ]
inter <- function(effect) {
  t <- tests[tests$effect == effect, c("model", "metric", "p_primary")]
  t$p_primary[match(paste(net$model, net$metric), paste(t$model, t$metric))]
}
net$p_size_net <- inter("size x network"); net$p_mod_net <- inter("modality x network")
for (m in unique(net$model)) {
  i <- net$model == m
  adj <- p.adjust(c(net$p[i], net$p_size_net[i], net$p_mod_net[i]), "BH")
  k <- sum(i)
  net$p[i] <- adj[1:k]; net$p_size_net[i] <- adj[(k + 1):(2 * k)]; net$p_mod_net[i] <- adj[(2 * k + 1):(3 * k)]
}
write.csv(net, file.path(DIR_STATS, "network_effects_bh.csv"), row.names = FALSE)
short_metric <- c("Total energy" = "Energy", "Client (laptop) energy" = "Laptop energy",
                  "Execution time" = "Time", "Throughput" = "Throughput")
tab <- data.frame(Model = MODEL_LABELS[net$model], Metric = short_metric[as.character(net$metric)],
                  `$p$` = fmt_p(net$p), `$\\eta^2_p$` = sprintf("%.2f", net$eta_sq),
                  Ratio = sprintf("%.2f", net$ratio_bad_vs_standard),
                  `$\\delta$` = sprintf("%+.2f", net$cliffs_delta_bad_vs_standard),
                  Cells = sub(" of ", "/", net$cells_significant_bad_vs_standard),
                  `Size$\\times$net.` = fmt_p(net$p_size_net),
                  `Mod.$\\times$net.` = fmt_p(net$p_mod_net), check.names = FALSE)
write_latex_table(tab, "tab_network_effects.tex",
                  paste("Effect of network condition on remote inference (ART ANOVA, network main effect).",
                        "Ratio = median bad / standard; $\\delta$ = Cliff's delta, bad vs standard (median over",
                        "the six size $\\times$ modality cells); Cells = cells where bad differs from",
                        "standard after Benjamini--Hochberg. The last two columns give $p$ for the",
                        "interaction of network with content size and with input modality. All $p$-values",
                        "in this table are Benjamini--Hochberg corrected over the table, per model."),
                  "tab:network", align = "llrrrrrrr", group = "Model")
