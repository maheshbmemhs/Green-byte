# RQ2: energy vs performance
source("analysis/R/setup.R")
message("05 RQ2 performance")

runs <- load_runs()
rem <- load_remote()
matched <- load_matched()
cap <- NULL
FACTORS <- c("content_size", "input_modality", "network_condition")
METRICS <- c(duration_s = "Execution time", throughput_bytes_per_s = "Throughput")

tests <- do.call(rbind, lapply(MODELS, function(m) {
  d <- rem[rem$model == m, ]
  rbind(do.call(rbind, lapply(names(METRICS), function(v)
          factorial_tests(d, v, FACTORS, "RQ2", m, METRICS[[v]]))),
        factorial_tests(matched[matched$model == m, ], "R_T", FACTORS, "RQ2", m, "R_T"))
}))
pairs <- do.call(rbind, lapply(MODELS, function(m) rbind(
  do.call(rbind, lapply(names(METRICS), function(v)
    cbind(network_pairs(rem[rem$model == m, ], v, paste("network pairs:", METRICS[[v]]), m),
          metric_label = METRICS[[v]]))),
  cbind(network_pairs(matched[matched$model == m, ], "R_T", "network pairs: R_T", m),
        metric_label = "R_T"))))
pairs <- add_bh(pairs)
summary_tab <- network_summary(tests, pairs)
write.csv(tests, file.path(DIR_STATS, "rq2_anova.csv"), row.names = FALSE)
write.csv(pairs, file.path(DIR_STATS, "rq2_posthoc.csv"), row.names = FALSE)
write.csv(summary_tab, file.path(DIR_STATS, "rq2_network_summary.csv"), row.names = FALSE)
message("  stats:  ", file.path(DIR_STATS, "rq2_*.csv"))

tab <- summary_tab
out <- data.frame(Model = MODEL_LABELS[tab$model], Metric = latex_metric(tab$metric), `$p$` = fmt_p(tab$p),
                  `$\\eta^2_p$` = sprintf("%.2f", tab$eta_sq),
                  Ratio = sprintf("%.2f", tab$ratio_bad_vs_standard),
                  `$\\delta$` = sprintf("%+.2f", tab$cliffs_delta_bad_vs_standard),
                  `Cells sig.` = tab$cells_significant_bad_vs_standard, check.names = FALSE)
write_latex_table(out, "tab_rq2_network_tests.tex",
                  paste("RQ2: effect of network condition on remote performance (ART ANOVA,",
                        "network main effect). Columns as in Table~\\ref{tab:rq1-3-network}."),
                  "tab:rq2-network", align = "llrrrrr", group = "Model")

runs$series <- factor(ifelse(runs$approach == "local", "On-device",
                             paste0("Remote, ", runs$network_condition)), levels = names(COL_SERIES))
perf_box <- function(y, ylab, title, d = runs, caption = cap) {
  ggplot(d, aes(content_size, .data[[y]])) +
    geom_box(0.8) +
    scale_fill_manual(values = COL_SERIES) + scale_colour_manual(values = COL_SERIES) +
    scale_x_discrete(labels = SIZE_LABELS) +
    scale_y_continuous(labels = scales::label_comma(), limits = c(0, NA)) +
    facet_grid(model_label ~ input_modality, scales = "free_y",
               labeller = labeller(input_modality = c(text = "Text input", image = "Image input"))) +
    labs(title = title, x = "Requested content size (words)", y = ylab, caption = caption) +
    theme_greenlab()
}
save_fig(perf_box("duration_s", "Execution time T (s)",
                  "Execution time per run: on-device vs remote per network condition"),
         "fig_rq2_execution_time", height = 6)
save_fig(perf_box("throughput_bytes_per_s", "Throughput (bytes/s)",
                  "Throughput: on-device vs remote per network condition"),
         "fig_rq2_throughput", height = 6)

remote_runs <- runs[runs$approach == "remote", ]
remote_runs$series <- droplevels(remote_runs$series)
save_fig(perf_box("duration_s", "Execution time T (s)", "Remote inference: execution time per run",
                  remote_runs, NULL), "fig_rq2_remote_execution_time", height = 6)
save_fig(perf_box("throughput_bytes_per_s", "Throughput (bytes/s)", "Remote inference: throughput",
                  remote_runs, NULL), "fig_rq2_remote_throughput", height = 6)

tr <- aggregate(cbind(R_E, R_T) ~ model + model_label + input_modality + content_size + network_condition,
                data = matched, FUN = median)
quad <- data.frame(x = c(0.032, 0.032, 3.8, 3.8), y = c(0.31, 11.5, 0.31, 11.5),
                   hj = c(0, 0, 1, 1), vj = c(0, 1, 0, 1),
                   label = c("remote cheaper & faster", "remote faster, costs more",
                             "remote cheaper, slower", "remote slower & costs more"))
p <- ggplot(tr, aes(R_T, R_E)) +
  geom_hline(yintercept = 1, colour = COL$axis) + geom_vline(xintercept = 1, colour = COL$axis) +
  geom_point(aes(colour = network_condition, shape = model_label, fill = network_condition),
             size = 2.2, stroke = 0.8) +
  geom_text(data = quad, aes(x = x, y = y, label = label, hjust = hj, vjust = vj),
            size = 2.2, colour = COL$muted) +
  scale_colour_manual(values = COL_NETWORK, labels = function(x) paste(tools::toTitleCase(x), "network")) +
  scale_fill_manual(values = COL_NETWORK, guide = "none") +
  scale_shape_manual(values = c(21, 24, 22)) +
  scale_x_log10(limits = c(0.03, 4)) + scale_y_log10(limits = c(0.3, 12)) +
  labs(title = "Energy-time trade-off",
       subtitle = "One point = one configuration (median of 30 matched pairs)",
       x = "R_T = T_remote / T_on-device (log)", y = "R_E = E_remote / E_on-device (log)",
       caption = if (is.null(cap)) NULL else paste(strwrap(cap, 60), collapse = "\n")) +
  theme_greenlab() + theme(panel.grid.major.x = element_line(colour = COL$grid, linewidth = 0.3),
                           legend.box = "vertical")
save_fig(p, "fig_rq2_tradeoff", width = 3.4, height = 3.9, paper_height = 2.9)

cls <- aggregate(cbind(R_E, R_E_word, R_T) ~ model + input_modality + content_size + network_condition,
                 data = matched, FUN = median)
cls$faster <- cls$R_T < 1
cls$category <- ifelse(cls$faster & cls$R_E < 1, "faster and cheaper",
                ifelse(cls$faster, "faster, costs more energy",
                ifelse(cls$R_E < 1, "slower, cheaper", "slower and costs more")))
cls$category_per_word <- ifelse(cls$faster & cls$R_E_word < 1, "faster and cheaper",
                         ifelse(cls$faster, "faster, costs more energy",
                         ifelse(cls$R_E_word < 1, "slower, cheaper", "slower and costs more")))
write.csv(cls, file.path(DIR_STATS, "rq2_tradeoff_classification.csv"), row.names = FALSE)

tt <- do.call(rbind, lapply(MODELS, function(m) {
  g <- cls[cls$model == m, ]
  data.frame(Model = MODEL_LABELS[[m]], Faster = sprintf("%d / %d", sum(g$faster), nrow(g)),
             `Faster \\& cheaper` = sum(g$category == "faster and cheaper"),
             `\\ldots{} per word` = sum(g$category_per_word == "faster and cheaper"),
             `Faster, costs more` = sum(g$category == "faster, costs more energy"),
             check.names = FALSE)
}))
write_latex_table(tt, "tab_rq2_tradeoff.tex",
                  paste("Energy--performance trade-off per model: number of the 18 configurations",
                        "(modality $\\times$ size $\\times$ network) where remote inference is faster",
                        "(median $R_T < 1$), faster and cheaper in total energy ($R_E < 1$) or per word",
                        "($R_{E,word} < 1$), or faster but more energy-intensive."),
                  "tab:tradeoff", align = "lrrrr")
