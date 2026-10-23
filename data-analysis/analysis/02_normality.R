# Normality checks: Shapiro-Wilk per treatment, Q-Q plots of the residuals
source("analysis/R/setup.R")
message("02 normality")

runs <- load_runs()
keys <- c("approach", "model", "input_modality", "content_size", "network_condition")
cells <- do.call(rbind, lapply(split(runs, runs[keys], drop = TRUE), function(g) {
  do.call(rbind, lapply(c("e_total_j", "duration_s", "throughput_bytes_per_s"), function(m) {
    sw <- shapiro.test(g[[m]])
    cbind(g[1, keys], metric = m, n = nrow(g), W = unname(sw$statistic), p = sw$p.value,
          normal = sw$p.value >= ALPHA)
  }))
}))
write.csv(cells, file.path(DIR_STATS, "normality_per_cell.csv"), row.names = FALSE)
message("  stats:  ", file.path(DIR_STATS, "normality_per_cell.csv"))
for (m in unique(cells$metric)) {
  g <- cells[cells$metric == m, ]
  message(sprintf("  %s: %d of %d cells reject normality", m, sum(!g$normal), nrow(g)))
}

rem <- runs[runs$approach == "remote", ]
std <- runs[runs$approach == "local" | runs$network_condition == "standard", ]
resid_of <- function(d, factors, label) {
  do.call(rbind, lapply(split(d, d$model), function(g) {
    r <- g$e_total_j - ave(g$e_total_j, g[factors], FUN = mean)
    sw <- shapiro.test(r)
    q <- qqnorm(r, plot.it = FALSE)
    data.frame(model_label = g$model_label[1], model_set = label, theoretical = q$x, sample = q$y,
               note = sprintf("W = %.3f\np = %.0e", sw$statistic, sw$p.value))
  }))
}
qq <- rbind(resid_of(std, c("approach", "content_size", "input_modality"),
                     "RQ1.1/1.2"),
            resid_of(rem, c("content_size", "input_modality", "network_condition"),
                     "RQ1.3"))
notes <- unique(qq[c("model_label", "model_set", "note")])

p <- ggplot(qq, aes(theoretical, sample)) +
  geom_qq_line(aes(sample = sample), colour = COL$axis, linewidth = 0.5) +
  geom_point(size = 0.5, colour = COL_NETWORK[["degraded"]]) +
  geom_text(data = notes, aes(x = -Inf, y = Inf, label = note), hjust = -0.08, vjust = 1.25,
            size = 1.9, lineheight = 0.9, colour = COL$ink2) +
  facet_grid(model_set ~ model_label, scales = "free_y") +
  labs(title = "Q-Q plots of model residuals (E_total)",
       subtitle = "Points should follow the line if residuals are normal; Shapiro-Wilk per panel",
       x = "Theoretical normal quantiles", y = "Residual of E_total (J)") +
  theme_greenlab() + theme(panel.grid.major.x = element_line(colour = COL$grid, linewidth = 0.3),
                           strip.text.x = element_text(size = 7))
save_fig(p, "fig_normality_qq", height = 4, paper_width = 3.4, paper_height = 2.6)
