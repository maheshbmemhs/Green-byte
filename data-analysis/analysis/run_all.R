# Runs all analysis steps:  Rscript analysis/run_all.R
modules <- c("01_descriptives.R", "02_normality.R", "03_rq1_local_vs_remote.R",
             "04_rq1_3_network_energy.R", "05_rq2_performance.R", "06_validity_checks.R",
             "07_robustness.R", "08_hypothesis_summary.R")
for (m in modules) {
  source(file.path("analysis", m), local = new.env())
}
message("done: results/figures, results/tables, results/stats")
