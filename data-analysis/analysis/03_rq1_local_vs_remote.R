# RQ1, RQ1.1, RQ1.2: on-device vs remote energy
source("analysis/R/setup.R")
message("03 RQ1 local vs remote")

runs <- load_runs()
matched <- load_matched()
cap <- NULL

std <- runs[runs$approach == "local" | runs$network_condition == "standard", ]
std$series <- factor(ifelse(std$approach == "local", "On-device", "Remote, standard"),
                     levels = names(COL_SERIES))

anova <- do.call(rbind, lapply(MODELS, function(m) rbind(
  factorial_tests(std[std$model == m, ], "e_total_j",
                  c("approach", "content_size", "input_modality"), "RQ1.1 + RQ1.2", m,
                  "E_total (standard network)"),
  factorial_tests(matched[matched$model == m, ], "R_E",
                  c("content_size", "input_modality", "network_condition"), "RQ1", m, "R_E"))))
write.csv(anova, file.path(DIR_STATS, "rq1_anova.csv"), row.names = FALSE)

post <- do.call(rbind, lapply(MODELS, function(m) do.call(rbind, lapply(MODALITIES, function(mod)
  do.call(rbind, lapply(SIZES, function(s) {
    c <- std[std$model == m & std$input_modality == mod & std$content_size == s, ]
    rmt <- c[c$approach == "remote", ]; loc <- c[c$approach == "local", ]
    rmt <- rmt[order(rmt$repetition), ]; loc <- loc[order(loc$repetition), ]
    cbind(family = "RQ1.1/1.2 remote vs on-device", model = m, cell = paste(mod, s, sep = ", "),
          pairwise_independent(rmt$e_total_j, loc$e_total_j))
  }))))))
post <- add_bh(post)
write.csv(post, file.path(DIR_STATS, "rq1_posthoc.csv"), row.names = FALSE)
message("  stats:  ", file.path(DIR_STATS, "rq1_anova.csv"), ", rq1_posthoc.csv")

size_post <- do.call(rbind, lapply(MODELS, function(m) do.call(rbind, lapply(MODALITIES, function(mod)
  do.call(rbind, lapply(c("local", "remote"), function(a) {
    d <- std[std$model == m & std$input_modality == mod & std$approach == a, ]
    do.call(rbind, lapply(list(c("medium", "short"), c("long", "short"), c("long", "medium")), function(pr)
      cbind(family = paste("size", m, mod, a), model = m, approach = a, input_modality = mod,
            comparison = paste(pr[1], "vs", pr[2]),
            pairwise_independent(d$e_total_j[d$content_size == pr[1]], d$e_total_j[d$content_size == pr[2]]))))
  }))))))
size_post <- add_bh(size_post)
write.csv(size_post, file.path(DIR_STATS, "rq1_1_size_posthoc.csv"), row.names = FALSE)

modality_post <- do.call(rbind, lapply(MODELS, function(m) do.call(rbind, lapply(SIZES, function(s)
  do.call(rbind, lapply(c("local", "remote"), function(a) {
    d <- std[std$model == m & std$content_size == s & std$approach == a, ]
    cbind(family = paste("modality", m, a), model = m, approach = a, content_size = s,
          comparison = "image vs text",
          pairwise_independent(d$e_total_j[d$input_modality == "image"], d$e_total_j[d$input_modality == "text"]))
  }))))))
modality_post <- add_bh(modality_post)
write.csv(modality_post, file.path(DIR_STATS, "rq1_2_modality_posthoc.csv"), row.names = FALSE)
message("  stats:  ", file.path(DIR_STATS, "rq1_1_size_posthoc.csv"), ", rq1_2_modality_posthoc.csv")

effects_table <- function(a, metric) {
  a <- a[a$metric == metric, ]
  out <- data.frame(Effect = unique(a$effect))
  for (m in MODELS) {
    g <- a[a$model == m, ][match(out$Effect, a$effect[a$model == m]), ]
    out[[MODEL_LABELS[[m]]]] <- sprintf("%s (%.2f)", fmt_p(g$p_primary), g$eta_sq_primary)
  }
  out
}
note <- if (is.null(cap)) "" else paste0(" \\emph{", latex_escape(cap), "}")
write_latex_table(effects_table(anova, "E_total (standard network)"), "tab_rq1_1_2_anova.tex",
                  paste0("RQ1.1 and RQ1.2: ART ANOVA on $E_{total}$ (standard network), per model. ",
                         "Cells: $p$ ($\\eta^2_p$).", note), "tab:rq1-anova")
write_latex_table(effects_table(anova, "R_E"), "tab_rq1_re_anova.tex",
                  paste0("RQ1: ART ANOVA on $R_E$, per model. Cells: $p$ ($\\eta^2_p$).", note),
                  "tab:rq1-re-anova")
pt <- data.frame(Model = MODEL_LABELS[post$model], Cell = post$cell,
                 Ratio = sprintf("%.2f", post$median_ratio), `$p_{BH}$` = fmt_p(post$p_bh),
                 `Cliff's $\\delta$` = sprintf("%+.2f (%s)", post$cliffs_delta, post$cliffs_magnitude),
                 check.names = FALSE)
write_latex_table(pt, "tab_rq1_posthoc.tex",
                  paste0("Remote vs on-device $E_{total}$ per cell (Wilcoxon rank-sum, ",
                         "Benjamini--Hochberg). Ratio = median remote / median on-device.", note),
                  "tab:rq1-posthoc", align = "llrrr", group = "Model")

box_fig <- function(d, y, ylab, title) {
  ggplot(d, aes(content_size, .data[[y]])) +
    geom_box(0.7) +
    scale_fill_manual(values = COL_SERIES, drop = TRUE) +
    scale_colour_manual(values = COL_SERIES, drop = TRUE) +
    scale_x_discrete(labels = SIZE_LABELS) +
    scale_y_continuous(labels = scales::label_comma(), limits = c(0, NA)) +
    facet_grid(model_label ~ input_modality, scales = "free_y",
               labeller = labeller(input_modality = c(text = "Text input", image = "Image input"))) +
    labs(title = title, x = "Requested content size (words)", y = ylab, caption = cap) +
    theme_greenlab()
}
save_fig(box_fig(std, "e_total_j", "Total energy E_total (J)",
                 "Total energy per run: on-device vs remote (standard network)"),
         "fig_rq1_energy_local_vs_remote", height = 6)
save_fig(box_fig(std, "e_per_word_j", "Energy per word E_word (J/word)",
                 "Energy per output word: on-device vs remote (standard network)"),
         "fig_rq1_energy_per_word", height = 6)

ratio_heatmap <- function(var, title, name) {
  hm <- aggregate(as.formula(paste(var, "~ model + model_label + input_modality + content_size + network_condition")),
                  data = matched, FUN = median)
  names(hm)[names(hm) == var] <- "ratio"
  hm$model_key <- as.character(hm$model)
  hm$row <- factor(paste(hm$model_label, hm$input_modality, sep = ", "),
                   levels = rev(as.vector(outer(MODALITIES, MODEL_LABELS, function(a, b) paste(b, a, sep = ", ")))))
  lim <- max(abs(log10(hm$ratio)))
  p <- ggplot(hm, aes(content_size, row, fill = log10(ratio))) +
    geom_tile(colour = "white", linewidth = 0.8) +
    geom_text(aes(label = sprintf("%.2f", ratio),
                  colour = ifelse(abs(log10(ratio)) > 0.55 * lim, "white", COL$ink)), size = 2.4) +
    scale_colour_identity() +
    scale_fill_gradient2(low = "#2a78d6", mid = "#f0efec", high = "#c43a3a", midpoint = 0,
                         limits = c(-lim, lim), breaks = log10(c(0.5, 1, 2, 4)),
                         labels = c("0.5", "1", "2", "4"), name = "ratio") +
    scale_x_discrete(labels = c(short = "Short", medium = "Medium", long = "Long")) +
    facet_wrap(~ network_condition, labeller = labeller(network_condition = function(x)
      paste(tools::toTitleCase(x), "network"))) +
    labs(title = title,
         subtitle = "Median of 30 matched runs. Blue: remote uses less energy; red: on-device uses less energy",
         x = NULL, y = NULL, caption = cap) +
    theme_greenlab() +
    theme(panel.grid = element_blank(), panel.grid.major.y = element_blank(), axis.line = element_blank(),
          axis.ticks = element_blank(), legend.position = "right",
          legend.title = element_text(size = 7, colour = COL$ink2))
  save_fig(p, name, height = 2.9)
}
ratio_heatmap("R_E", "Relative energy R_E = E_remote / E_on-device", "fig_rq1_relative_energy_heatmap")
ratio_heatmap("R_E_word", "Relative energy per output word = E_word,remote / E_word,on-device",
              "fig_rq1_relative_energy_per_word_heatmap")
