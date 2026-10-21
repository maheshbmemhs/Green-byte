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

split <- aggregate(cbind(client_energy_j, server_energy_j) ~ model_label + content_size + network_condition,
                   data = rem, FUN = mean)
long <- rbind(data.frame(split[1:3], part = "Server", energy = split$server_energy_j),
              data.frame(split[1:3], part = "Client (laptop)", energy = split$client_energy_j))
long$part <- factor(long$part, levels = c("Client (laptop)", "Server"))
share <- aggregate(cbind(client_energy_j, server_energy_j) ~ model_label, data = rem, FUN = sum)
share$label <- sprintf("laptop share: %.0f%%", 100 * share$client_energy_j /
                         (share$client_energy_j + share$server_energy_j))
share$content_size <- factor("short", levels = SIZES)
p <- ggplot(long, aes(network_condition, energy, fill = part)) +
  geom_col(width = 0.8, colour = "white", linewidth = 0.3) +
  geom_text(data = share, aes(x = -Inf, y = Inf, label = label), inherit.aes = FALSE,
            hjust = -0.1, vjust = 1.4, size = 2.3, colour = COL$ink2) +
  scale_fill_manual(values = c("Client (laptop)" = COL$client, "Server" = COL$server)) +
  scale_x_discrete(labels = c(standard = "S", degraded = "D", bad = "B")) +
  scale_y_continuous(labels = scales::label_comma()) +
  facet_grid(model_label ~ content_size, scales = "free_y",
             labeller = labeller(content_size = c(short = "Short", medium = "Medium", long = "Long"))) +
  labs(title = "Where remote energy is spent (mean per run, text and image pooled)",
       subtitle = "S / D / B = standard / degraded / bad network",
       x = "Network condition", y = "Mean energy per run (J)") +
  theme_greenlab()
save_fig(p, "fig_rq1_3_server_vs_client", height = 3.8)
