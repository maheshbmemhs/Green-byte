# Shared setup: paths, data loading, plot style, LaTeX tables
if (!dir.exists("data/clean")) {
  stop("Run from the project folder (the one containing data/), e.g.  Rscript analysis/run_all.R")
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(ARTool)
  library(effsize)
})
source("analysis/R/stats_helpers.R")

DIR_CLEAN   <- "data/clean"
DIR_FIGURES <- "results/figures"
DIR_TABLES  <- "results/tables"
DIR_STATS   <- "results/stats"
for (d in c(DIR_FIGURES, DIR_TABLES, DIR_STATS)) dir.create(d, recursive = TRUE, showWarnings = FALSE)

MODELS   <- c("qwen3b", "gemma4b", "qwen7b")
MODEL_LABELS <- c(qwen3b = "Qwen2.5-VL-3B", gemma4b = "Gemma 3 4B", qwen7b = "Qwen2.5-VL-7B")
SIZES    <- c("short", "medium", "long")
SIZE_LABELS <- c(short = "Short\n50-100", medium = "Medium\n300-500", long = "Long\n800-1000")
MODALITIES <- c("text", "image")
NETWORKS <- c("standard", "degraded", "bad")

as_factors <- function(d) {
  d$model_label <- factor(MODEL_LABELS[d$model], levels = MODEL_LABELS)
  d$model <- factor(d$model, levels = MODELS)
  d$content_size <- factor(d$content_size, levels = SIZES)
  d$input_modality <- factor(d$input_modality, levels = MODALITIES)
  if ("network_condition" %in% names(d)) {
    d$network_condition <- factor(d$network_condition, levels = c(NETWORKS, "none (local)"))
  }
  d
}

load_runs <- function() {
  d <- read.csv(file.path(DIR_CLEAN, "all_runs_clean.csv"), stringsAsFactors = FALSE)
  as_factors(d)
}
load_remote <- function() {
  d <- read.csv(file.path(DIR_CLEAN, "server", "server_runs_clean.csv"), stringsAsFactors = FALSE)
  d <- as_factors(d)
  d$network_condition <- droplevels(d$network_condition)
  d
}
load_matched <- function() {
  d <- read.csv(file.path(DIR_CLEAN, "matched_local_remote.csv"), stringsAsFactors = FALSE)
  d <- as_factors(d)
  d$network_condition <- droplevels(d$network_condition)
  d
}

COL <- list(local = "#eb6834", client = "#1baf7a", server = "#6250d6",
            ink = "#0b0b0b", ink2 = "#52514e", muted = "#898781", grid = "#e1e0d9", axis = "#c3c2b7")
COL_NETWORK <- c(standard = "#86b6ef", degraded = "#2a78d6", bad = "#104281")
COL_SERIES <- c("On-device" = COL$local, "Remote, standard" = COL_NETWORK[["standard"]],
                "Remote, degraded" = COL_NETWORK[["degraded"]], "Remote, bad" = COL_NETWORK[["bad"]])

theme_greenlab <- function() {
  theme_minimal(base_size = 8, base_family = "Helvetica") +
    theme(plot.title = element_text(face = "bold", size = 9.5, colour = COL$ink),
          plot.subtitle = element_text(size = 7.5, colour = COL$ink2),
          plot.caption = element_text(size = 6.5, colour = COL$muted, face = "italic"),
          strip.text = element_text(face = "bold", size = 8, colour = COL$ink),
          axis.title = element_text(colour = COL$ink2), axis.text = element_text(colour = COL$ink2),
          panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(),
          panel.grid.major.y = element_line(colour = COL$grid, linewidth = 0.3),
          axis.line = element_line(colour = COL$axis, linewidth = 0.4),
          legend.position = "top", legend.title = element_blank(),
          legend.key.size = unit(8, "pt"), plot.title.position = "plot",
          plot.margin = margin(5, 10, 5, 5))
}

geom_box <- function(width = 0.75) {
  geom_boxplot(aes(fill = series, colour = series), width = width, linewidth = 0.3,
               median.colour = COL$ink, median.linewidth = 0.5, outlier.size = 0.6,
               key_glyph = draw_key_rect)
}

# saves pdf + png, and a version without title for the paper (results/figures/paper)
save_fig <- function(p, name, width = 7, height = 4.5, paper_width = width,
                     paper_height = min(height * 0.9, 4.3)) {
  ggsave(file.path(DIR_FIGURES, paste0(name, ".pdf")), p, width = width, height = height,
         device = "pdf", bg = "white")
  ggsave(file.path(DIR_FIGURES, paste0(name, ".png")), p, width = width, height = height,
         dpi = 300, bg = "white")
  dir.create(file.path(DIR_FIGURES, "paper"), showWarnings = FALSE)
  paper <- p + labs(title = NULL, subtitle = NULL, caption = NULL)
  ggsave(file.path(DIR_FIGURES, "paper", paste0(name, ".pdf")), paper, width = paper_width,
         height = paper_height, device = "pdf", bg = "white")
  message("  figure: ", file.path(DIR_FIGURES, name), ".pdf/.png (+ paper/)")
}

fmt_p <- function(p) ifelse(p < 0.001, "$<$0.001", sprintf("%.3f", p))
latex_escape <- function(x) gsub("_", "\\\\_", gsub("%", "\\\\%", x))
latex_metric <- function(x) ifelse(x %in% c("R_E", "R_T"), paste0("$", x, "$"),
                                   ifelse(x == "R_E_word", "$R_{E,word}$", latex_escape(x)))

write_latex_table <- function(df, file, caption, label, align = NULL, wide = FALSE, group = NULL,
                              header_above = NULL) {
  env <- if (wide) "table*" else "table"
  if (is.null(align)) align <- paste0(c("l", rep("r", ncol(df) - 1)), collapse = "")
  df[] <- lapply(df, as.character)
  rows <- character(0)
  for (i in seq_len(nrow(df))) {
    r <- df[i, ]
    if (!is.null(group) && i > 1) {
      if (df[[group]][i] != df[[group]][i - 1]) rows <- c(rows, "\\midrule") else r[[group]] <- ""
    }
    rows <- c(rows, paste0(paste(unlist(r), collapse = " & "), " \\\\"))
  }
  lines <- c(sprintf("\\begin{%s}[tp]", env), "\\centering", "\\small",
             sprintf("\\caption{%s}", caption), sprintf("\\label{%s}", label),
             "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%",
             sprintf("\\begin{tabular}{%s}", align), "\\toprule", header_above,
             paste0(paste(names(df), collapse = " & "), " \\\\"), "\\midrule",
             rows, "\\bottomrule", "\\end{tabular}}", sprintf("\\end{%s}", env))
  writeLines(lines, file.path(DIR_TABLES, file))
  message("  table:  ", file.path(DIR_TABLES, file))
}
