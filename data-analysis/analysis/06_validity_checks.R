# Validity checks: output length and RAPL overflow example
source("analysis/R/setup.R")
message("06 validity checks")

runs <- load_runs()

w <- aggregate(cbind(output_words, flag_truncated_at_max_tokens) ~
                 approach + model_label + input_modality + content_size, data = runs, FUN = mean)
w$series <- paste(ifelse(w$approach == "local", "On-device", "Remote"), w$input_modality, sep = ", ")
w$cut <- w$flag_truncated_at_max_tokens > 0.5
offsets <- c("On-device, text" = -0.27, "On-device, image" = -0.09, "Remote, text" = 0.09, "Remote, image" = 0.27)
w$x <- as.numeric(w$content_size) + offsets[w$series]
ranges <- data.frame(x = 1:3, lo = c(50, 300, 800), hi = c(100, 500, 1000))
cols <- c("On-device, text" = COL$local, "On-device, image" = COL$local,
          "Remote, text" = COL_NETWORK[["degraded"]], "Remote, image" = COL_NETWORK[["degraded"]])
p <- ggplot(w, aes(x, output_words)) +
  geom_rect(data = ranges, aes(xmin = x - 0.4, xmax = x + 0.4, ymin = lo, ymax = hi),
            inherit.aes = FALSE, fill = "#f0efec") +
  geom_point(data = w[w$cut, ], shape = 1, size = 3.8, colour = COL$ink, stroke = 0.4) +
  geom_point(aes(colour = series, shape = series), size = 2, stroke = 0.9) +
  scale_colour_manual(values = cols, breaks = names(offsets)) +
  scale_shape_manual(values = c("On-device, text" = 16, "On-device, image" = 1,
                                "Remote, text" = 16, "Remote, image" = 1), breaks = names(offsets)) +
  scale_x_continuous(breaks = 1:3, labels = SIZE_LABELS) +
  facet_wrap(~ model_label) +
  labs(title = "Generated length vs requested word range (grey band)",
       subtitle = "Black ring = output cut off at max_tokens", x = "Requested content size (words)",
       y = "Output length (words)") +
  theme_greenlab()
save_fig(p, "fig_validity_word_target", height = 3, paper_width = 3.4, paper_height = 2.3,
         paper_extra = list(scale_x_continuous(breaks = 1:3, labels = c("Short", "Med.", "Long")),
                            guides(colour = guide_legend(nrow = 2), shape = guide_legend(nrow = 2)),
                            labs(x = "Requested content size"),
                            theme(axis.text = element_text(size = 6), strip.text = element_text(size = 7),
                                  legend.text = element_text(size = 6.5))))

r <- runs[runs$flag_rapl_corrected & runs$approach == "local", ][1, ]
eb <- read.csv(file.path(r$raw_run_dir, "energibridge.csv"), check.names = FALSE)
t <- (eb$Time - eb$Time[1]) / 1000
raw <- eb[["PACKAGE_ENERGY (J)"]]
corrected <- raw + cumsum(c(0, diff(raw) < -131072)) * 262144
d <- rbind(data.frame(t = t, value = raw / 1000, panel = "Raw counter (kJ)"),
           data.frame(t = t, value = corrected - corrected[1], panel = "Corrected energy since start (J)"))
d$panel <- factor(d$panel, levels = unique(d$panel))
p <- ggplot(d, aes(t, value)) +
  geom_line(aes(colour = panel), linewidth = 0.8, show.legend = FALSE) +
  scale_colour_manual(values = c(COL$muted, COL_NETWORK[["degraded"]])) +
  facet_wrap(~ panel, ncol = 1, scales = "free_y") +
  labs(title = "RAPL counter overflow in one run",
       subtitle = sprintf("%s, %s, %s, on-device\nNaive last - first: %s J; corrected: %s J",
                          r$model_label, r$input_modality, r$content_size,
                          format(round(raw[length(raw)] - raw[1]), big.mark = ","),
                          format(round(r$client_cpu_j), big.mark = ",")),
       x = "Time since start of run (s)", y = NULL) +
  theme_greenlab()
save_fig(p, "fig_validity_rapl_overflow", width = 3.6, height = 3.6)
