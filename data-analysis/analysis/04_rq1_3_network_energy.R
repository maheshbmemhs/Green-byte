# RQ1.3: effect of the network condition on remote energy
source("analysis/R/setup.R")
message("04 RQ1.3 network and energy")

rem <- load_remote()
matched <- load_matched()
FACTORS <- c("content_size", "input_modality", "network_condition")
METRICS <- c(e_total_j = "Total energy", client_energy_j = "Client (laptop) energy",
             server_energy_j = "Server energy")

tests <- do.call(rbind, lapply(MODELS, function(m) {
  d <- rem[rem$model == m, ]
  rbind(do.call(rbind, lapply(names(METRICS), function(v)
          factorial_tests(d, v, FACTORS, "RQ1.3-N", m, METRICS[[v]]))),
        factorial_tests(matched[matched$model == m, ], "R_E", FACTORS, "RQ1.3-G", m, "R_E"))
}))
pairs <- do.call(rbind, lapply(MODELS, function(m) rbind(
  do.call(rbind, lapply(names(METRICS), function(v)
    cbind(network_pairs(rem[rem$model == m, ], v, paste("network pairs:", METRICS[[v]]), m),
          metric_label = METRICS[[v]]))),
  cbind(network_pairs(matched[matched$model == m, ], "R_E", "network pairs: R_E", m),
        metric_label = "R_E"))))
pairs <- add_bh(pairs)
summary_tab <- network_summary(tests, pairs)
write.csv(tests, file.path(DIR_STATS, "rq1_3_anova.csv"), row.names = FALSE)
write.csv(pairs, file.path(DIR_STATS, "rq1_3_posthoc.csv"), row.names = FALSE)
write.csv(summary_tab, file.path(DIR_STATS, "rq1_3_network_summary.csv"), row.names = FALSE)
message("  stats:  ", file.path(DIR_STATS, "rq1_3_*.csv"))

tab <- summary_tab
out <- data.frame(Model = MODEL_LABELS[tab$model], Metric = latex_metric(tab$metric), `$p$` = fmt_p(tab$p),
                  `$\\eta^2_p$` = sprintf("%.2f", tab$eta_sq),
                  Ratio = sprintf("%.2f", tab$ratio_bad_vs_standard),
                  `$\\delta$` = sprintf("%+.2f", tab$cliffs_delta_bad_vs_standard),
                  `Cells sig.` = tab$cells_significant_bad_vs_standard, check.names = FALSE)
write_latex_table(out, "tab_rq1_3_network_tests.tex",
                  paste("RQ1.3: effect of network condition on remote energy (ART ANOVA, factors",
                        "size $\\times$ modality $\\times$ network; network main effect).",
                        "Ratio = median bad / standard; $\\delta$ = Cliff's delta bad vs standard",
                        "(median over the six size $\\times$ modality cells); Cells sig. = cells where",
                        "bad differs from standard after Benjamini--Hochberg."),
                  "tab:rq1-3-network", align = "llrrrrr", group = "Model")

rem$series <- factor(paste0("Remote, ", rem$network_condition), levels = names(COL_SERIES))
net_box <- function(y, ylab, title) {
  ggplot(rem, aes(content_size, .data[[y]])) +
    geom_box(0.75) +
    scale_fill_manual(values = COL_SERIES) + scale_colour_manual(values = COL_SERIES) +
    scale_x_discrete(labels = SIZE_LABELS) +
    scale_y_continuous(labels = scales::label_comma(), limits = c(0, NA)) +
    facet_grid(model_label ~ input_modality, scales = "free_y",
               labeller = labeller(input_modality = c(text = "Text input", image = "Image input"))) +
    labs(title = title, x = "Requested content size (words)", y = ylab) +
    theme_greenlab()
}
save_fig(net_box("e_total_j", "Total energy E_total (J)",
                 "Remote inference: total energy per run (laptop + server)"),
         "fig_rq1_3_remote_energy", height = 6)
save_fig(net_box("client_energy_j", "Client (laptop) energy (J)",
                 "Remote inference: laptop energy per run (the part the network affects most)"),
         "fig_rq1_3_client_energy", height = 6)

parts <- rbind(data.frame(rem[c("model_label", "content_size", "series")], part = "Client (laptop)",
                          energy = rem$client_energy_j),
               data.frame(rem[c("model_label", "content_size", "series")], part = "Server",
                          energy = rem$server_energy_j))
p <- ggplot(parts, aes(content_size, energy)) +
  geom_box(0.75) +
  scale_fill_manual(values = COL_SERIES) + scale_colour_manual(values = COL_SERIES) +
  scale_x_discrete(labels = c(short = "Short", medium = "Medium", long = "Long")) +
  scale_y_continuous(labels = scales::label_comma(), limits = c(0, NA)) +
  facet_wrap(~ model_label + part, ncol = 2, scales = "free_y",
             labeller = function(l) list(paste(l$model_label, l$part, sep = " - "))) +
  labs(title = "Remote energy per run, laptop and server (text and image pooled)",
       x = "Requested content size", y = "Energy per run (J)") +
  theme_greenlab()
save_fig(p, "fig_rq1_3_energy_parts", height = 5, paper_height = 2.7)
