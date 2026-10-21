# Robustness checks for the test choices
source("analysis/R/setup.R")
message("07 robustness")

runs <- load_runs()
rem <- load_remote()
matched <- load_matched()
std <- runs[runs$approach == "local" | runs$network_condition == "standard", ]
NET_FACTORS <- c("content_size", "input_modality", "network_condition")

specs <- list(
  list(data = std, dv = "e_total_j", factors = c("approach", "content_size", "input_modality"),
       name = "E_total, on-device vs remote (standard)"),
  list(data = rem, dv = "e_total_j", factors = NET_FACTORS, name = "Remote E_total"),
  list(data = matched, dv = "R_E", factors = NET_FACTORS, name = "R_E"),
  list(data = matched, dv = "R_T", factors = NET_FACTORS, name = "R_T"))
diag <- do.call(rbind, lapply(specs, function(sp) do.call(rbind, lapply(MODELS, function(m)
  model_diagnostics(sp$data[sp$data$model == m, ], sp$dv, sp$factors, m, sp$name)))))
write.csv(diag, file.path(DIR_STATS, "model_diagnostics.csv"), row.names = FALSE)
message(sprintf("  diagnostics: residuals non-normal in %d of %d models; max/min cell SD up to %.0f",
                sum(diag$shapiro_p < ALPHA), nrow(diag), max(diag$max_to_min_cell_sd, na.rm = TRUE)))

ratio_metrics <- c(R_E = "R_E", R_E_word = "R_E_word", R_T = "R_T")
art <- do.call(rbind, lapply(MODELS, function(m) do.call(rbind, lapply(names(ratio_metrics), function(v)
  factorial_tests(matched[matched$model == m, ], v, NET_FACTORS, "robustness", m, v)))))
cr <- do.call(rbind, lapply(MODELS, function(m) do.call(rbind, lapply(names(ratio_metrics), function(v)
  cluster_robust_tests(matched[matched$model == m, ], v, NET_FACTORS, m, v)))))
cmp <- merge(cr, art[c("model", "metric", "effect", "p_art")], by = c("model", "metric", "effect"))
cmp$same_conclusion <- (cmp$p_cluster_robust < ALPHA) == (cmp$p_art < ALPHA)
write.csv(cmp, file.path(DIR_STATS, "ratio_cluster_robust_check.csv"), row.names = FALSE)
message(sprintf("  cluster-robust vs ART: same conclusion for %d of %d effects",
                sum(cmp$same_conclusion), nrow(cmp)))

# two local runs have an estimated duration: drop them and test again
imp <- unique(runs[runs$approach == "local" & runs$duration_source != "run_table",
                   c("model", "repetition")])
sens <- do.call(rbind, lapply(MODELS, function(m) {
  drop_reps <- imp$repetition[imp$model == m]
  d_all <- matched[matched$model == m, ]
  if (length(drop_reps) == 0) return(NULL)
  d_red <- d_all[!d_all$repetition %in% drop_reps, ]
  a <- factorial_tests(d_all, "R_T", NET_FACTORS, "sensitivity", m, "R_T")
  b <- factorial_tests(d_red, "R_T", NET_FACTORS, "sensitivity", m, "R_T")
  out <- merge(a[c("effect", "p_primary")], b[c("effect", "p_primary")], by = "effect",
               suffixes = c("_all_runs", "_without_imputed"))
  out$model <- m
  out$repetitions_dropped <- paste(drop_reps, collapse = ",")
  out$same_conclusion <- (out$p_primary_all_runs < ALPHA) == (out$p_primary_without_imputed < ALPHA)
  out
}))
write.csv(sens, file.path(DIR_STATS, "imputed_duration_sensitivity.csv"), row.names = FALSE)
message(sprintf("  imputed-duration sensitivity: same conclusion for %d of %d R_T effects",
                sum(sens$same_conclusion), nrow(sens)))

all_tests <- rbind(read.csv(file.path(DIR_STATS, "rq1_anova.csv")), read.csv(file.path(DIR_STATS, "rq1_3_anova.csv")),
                   read.csv(file.path(DIR_STATS, "rq2_anova.csv")))
all_tests <- all_tests[!duplicated(all_tests[c("model", "metric", "effect")]), ]
all_tests$same_conclusion <- (all_tests$p_param < ALPHA) == (all_tests$p_art < ALPHA)
write.csv(all_tests[c("model", "metric", "effect", "p_art", "p_param", "same_conclusion")],
          file.path(DIR_STATS, "art_vs_parametric_check.csv"), row.names = FALSE)
message(sprintf("  ART vs parametric factorial ANOVA: same conclusion for %d of %d effects",
                sum(all_tests$same_conclusion), nrow(all_tests)))

# remote qwen7b had no warm-up: compare the first 10 runs of each session with the rest
keys <- c("approach", "model", "input_modality", "content_size", "network_condition")
runs$deviation <- runs$e_total_j / ave(runs$e_total_j, runs[keys], FUN = median) - 1
ord <- do.call(rbind, lapply(split(runs, list(runs$approach, runs$model), drop = TRUE), function(g) {
  g <- g[order(g$client_start_epoch_s), ]
  first <- seq_len(nrow(g)) <= 10
  ct <- suppressWarnings(cor.test(seq_len(nrow(g)), g$deviation, method = "spearman", exact = FALSE))
  data.frame(approach = g$approach[1], model = as.character(g$model[1]), had_warmup = g$had_warmup[1],
             median_deviation_first10 = median(g$deviation[first]),
             median_deviation_rest = median(g$deviation[!first]),
             p_first10_vs_rest = suppressWarnings(wilcox.test(g$deviation[first], g$deviation[!first])$p.value),
             spearman_rho_with_order = unname(ct$estimate), p_spearman = ct$p.value)
}))
write.csv(ord, file.path(DIR_STATS, "order_warmup_check.csv"), row.names = FALSE)
q7 <- ord[ord$approach == "remote" & ord$model == "qwen7b", ]
message(sprintf("  warm-up check, remote qwen7b: first 10 runs %+.1f%% vs rest %+.1f%% (p = %.2f), rho = %+.2f",
                100 * q7$median_deviation_first10, 100 * q7$median_deviation_rest,
                q7$p_first10_vs_rest, q7$spearman_rho_with_order))
