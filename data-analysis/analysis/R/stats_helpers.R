# Statistical tests used by the analysis scripts
ALPHA <- 0.05
SHORT_NAMES <- c(approach = "approach", content_size = "size", input_modality = "modality",
                 network_condition = "network")

term_name <- function(term) paste(SHORT_NAMES[term], collapse = " x ")

all_terms <- function(factors) {
  unlist(lapply(seq_along(factors), function(k) combn(factors, k, simplify = FALSE)),
         recursive = FALSE)
}

effect_names <- function(rn) vapply(strsplit(trimws(rn), ":", fixed = TRUE), term_name, "")

param_anova <- function(d, dv, factors) {
  form <- as.formula(paste(dv, "~", paste(factors, collapse = " * ")))
  a <- summary(aov(form, data = d))[[1]]
  res <- a[trimws(rownames(a)) == "Residuals", ]
  a <- a[trimws(rownames(a)) != "Residuals", ]
  data.frame(effect = effect_names(rownames(a)),
             df = a$Df, df_error = res$Df, F_param = a$`F value`, p_param = a$`Pr(>F)`,
             partial_eta_sq_param = a$`Sum Sq` / (a$`Sum Sq` + res$`Sum Sq`))
}

art_anova <- function(d, dv, factors) {
  dd <- d[c(dv, factors)]
  for (f in factors) dd[[f]] <- factor(dd[[f]])
  names(dd)[1] <- "y"
  a <- as.data.frame(anova(art(as.formula(paste("y ~", paste(factors, collapse = " * "))), data = dd)))
  data.frame(effect = effect_names(as.character(a$Term)),
             F_art = a$`F value`, p_art = a$`Pr(>F)`,
             partial_eta_sq_art = a$`Sum Sq` / (a$`Sum Sq` + a$`Sum Sq.res`))
}

# ART ANOVA, unless the residuals are normal and the cell SDs are similar
factorial_tests <- function(d, dv, factors, rq, model, metric) {
  y <- d[[dv]]
  resid <- y - ave(y, d[factors], FUN = mean)
  sw <- shapiro.test(resid)
  res <- merge(param_anova(d, dv, factors), art_anova(d, dv, factors), by = "effect", sort = FALSE)
  if (any(is.na(res$effect) | res$effect == "NA")) stop("factorial_tests: unnamed effect")
  cell_sd <- tapply(resid, interaction(d[factors], drop = TRUE), sd)
  sd_ratio <- if (min(cell_sd) > 0) max(cell_sd) / min(cell_sd) else Inf
  normal <- sw$p.value >= ALPHA && sd_ratio < 2
  res$primary_test <- if (normal) "Factorial ANOVA" else "ART ANOVA"
  res$p_primary <- if (normal) res$p_param else res$p_art
  res$eta_sq_primary <- if (normal) res$partial_eta_sq_param else res$partial_eta_sq_art
  res$significant <- res$p_primary < ALPHA
  res$residual_shapiro_W <- unname(sw$statistic)
  res$residual_shapiro_p <- sw$p.value
  res$max_to_min_cell_sd <- sd_ratio
  cbind(rq = rq, model = model, metric = metric, res)
}

pairwise_paired <- function(a, b) {
  p <- if (all(a == b)) 1 else suppressWarnings(wilcox.test(a, b, paired = TRUE, exact = FALSE)$p.value)
  cd <- suppressWarnings(cliff.delta(a, b))
  cohen <- cohen.d(a, b)
  data.frame(n_pairs = length(a), median_a = median(a), median_b = median(b),
             median_ratio = median(a) / median(b), p_wilcoxon = p,
             cliffs_delta = unname(cd$estimate), cliffs_magnitude = as.character(cd$magnitude),
             cohens_d = unname(cohen$estimate), cohens_d_magnitude = as.character(cohen$magnitude))
}

RATIO_METRICS <- c("R_E", "R_T", "R_E_word")
PAIR_COLUMNS <- c("median_a", "median_b", "median_ratio", "p_wilcoxon", "cliffs_delta",
                  "cliffs_magnitude", "cohens_d", "cohens_d_magnitude")
NETWORK_PAIRS <- list(c("degraded", "standard"), c("bad", "standard"), c("bad", "degraded"))

network_pairs <- function(d, dv, family, model) {
  out <- list()
  for (mod in c("text", "image")) for (s in c("short", "medium", "long")) for (pr in NETWORK_PAIRS) {
    c <- d[d$input_modality == mod & d$content_size == s, ]
    a <- c[c$network_condition == pr[1], ]; b <- c[c$network_condition == pr[2], ]
    a <- a[order(a$repetition), dv]; b <- b[order(b$repetition), dv]
    # R_E / R_T of the three networks share one on-device run, so these are paired
    cmp <- if (dv %in% RATIO_METRICS) pairwise_paired(a, b) else pairwise_independent(a, b)
    cmp <- cmp[PAIR_COLUMNS]
    out[[length(out) + 1]] <- cbind(family = family, model = model, metric = dv,
                                    cell = paste(mod, s, sep = ", "),
                                    comparison = paste(pr[1], "vs", pr[2]), cmp)
  }
  do.call(rbind, out)
}

network_summary <- function(tests, pairs) {
  t <- tests[tests$effect == "network", ]
  do.call(rbind, lapply(seq_len(nrow(t)), function(i) {
    r <- t[i, ]
    pp <- pairs[pairs$model == r$model & pairs$metric_label == r$metric &
                  pairs$comparison == "bad vs standard", ]
    data.frame(model = r$model, metric = r$metric, p = r$p_primary, eta_sq = r$eta_sq_primary,
               test = r$primary_test, ratio_bad_vs_standard = median(pp$median_ratio),
               cliffs_delta_bad_vs_standard = median(pp$cliffs_delta),
               cells_significant_bad_vs_standard = sprintf("%d of %d", sum(pp$significant_bh), nrow(pp)))
  }))
}

add_bh <- function(post, family_col = "family") {
  post$p_bh <- ave(post$p_wilcoxon, post[[family_col]], FUN = function(p) p.adjust(p, "BH"))
  post$significant_bh <- post$p_bh < ALPHA
  post
}

pairwise_independent <- function(a, b) {
  p <- suppressWarnings(wilcox.test(a, b, paired = FALSE, exact = FALSE)$p.value)
  cd <- suppressWarnings(cliff.delta(a, b))
  cohen <- cohen.d(a, b)
  lt <- t.test(log(a), log(b), var.equal = FALSE)
  data.frame(n_a = length(a), n_b = length(b), median_a = median(a), median_b = median(b),
             median_ratio = median(a) / median(b), p_wilcoxon = p,
             geo_mean_ratio = exp(mean(log(a)) - mean(log(b))),
             ratio_ci_low = exp(lt$conf.int[1]), ratio_ci_high = exp(lt$conf.int[2]),
             cliffs_delta = unname(cd$estimate), cliffs_magnitude = as.character(cd$magnitude),
             cohens_d = unname(cohen$estimate), cohens_d_magnitude = as.character(cohen$magnitude))
}

# check for the ratios: log model with errors clustered by the shared on-device run
cluster_robust_tests <- function(d, dv, factors, model, metric) {
  for (f in factors) {
    d[[f]] <- droplevels(factor(d[[f]]))
    contrasts(d[[f]]) <- contr.sum(nlevels(d[[f]]))
  }
  d$log_y <- log(d[[dv]])
  fit <- lm(reformulate(paste(factors, collapse = " * "), response = "log_y"), data = d)
  X <- model.matrix(fit)
  assign <- attr(X, "assign")
  labels <- attr(terms(fit), "term.labels")
  cl <- interaction(d$input_modality, d$content_size, d$repetition, drop = TRUE)
  n_cl <- nlevels(cl)
  bread <- solve(crossprod(X))
  scores <- rowsum(X * residuals(fit), group = cl, reorder = FALSE)
  V <- (n_cl / (n_cl - 1)) * ((nrow(X) - 1) / (nrow(X) - ncol(X))) * bread %*% crossprod(scores) %*% bread
  do.call(rbind, lapply(seq_along(labels), function(j) {
    idx <- which(assign == j)
    b <- coef(fit)[idx]
    F <- as.numeric(crossprod(b, solve(V[idx, idx, drop = FALSE], b))) / length(idx)
    data.frame(model = model, metric = metric,
               effect = term_name(strsplit(labels[j], ":", fixed = TRUE)[[1]]),
               df = length(idx), df_error = n_cl - 1, F_cluster_robust = F,
               p_cluster_robust = pf(F, length(idx), n_cl - 1, lower.tail = FALSE))
  }))
}

model_diagnostics <- function(d, dv, factors, model, dataset) {
  do.call(rbind, lapply(c("raw", "log"), function(scale) {
    y <- if (scale == "log") log(d[[dv]]) else d[[dv]]
    r <- y - ave(y, d[factors], FUN = mean)
    sds <- tapply(r, interaction(d[factors], drop = TRUE), sd)
    sw <- shapiro.test(r)
    data.frame(dataset = dataset, model = model, scale = scale, n = length(y),
               shapiro_W = unname(sw$statistic), shapiro_p = sw$p.value,
               max_to_min_cell_sd = if (min(sds) > 0) max(sds) / min(sds) else NA_real_)
  }))
}
