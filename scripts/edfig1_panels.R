library(readr)
library(ggplot2)
# Extended Data Fig. 1: example curves per lane, drawn exactly as 8- grouping.R draws every network, without the title.
# a, Bordeaux and b, Konstanz after screening (per_lane\MFDs). c, Cagliari before screening: edfig1c_cagliari_raw.csv, written by
# make_manuscript_figures.py from the detector file, as the network file was built before its observations were removed.
args <- commandArgs(trailingOnly = TRUE)   # the repository, and where results go: Rscript edfig1_panels.R . results
ROOT <- if (length(args) >= 1) args[1] else "."
RESULTS <- if (length(args) >= 2) args[2] else file.path(ROOT, "results")
draw <- function(f, out) {
  df <- read_csv(f, show_col_types = FALSE)
  p <- ggplot(df, aes(x = density, y = flow)) +
    geom_point(color = "blue") +
    labs(x = "Density (veh/km/lane)", y = "Flow (veh/h/lane)") +
    theme_minimal() +
    theme(panel.background = element_rect(fill = "white"),
          plot.background  = element_rect(fill = "white"))
  ggsave(file.path(RESULTS, "figures", "panels", out), p, width = 8, height = 6, dpi = 300)
  cat(out, "points:", nrow(df), " max density:", max(df$density), " max flow:", max(df$flow), "\n")
}
draw(file.path(ROOT, "data", "per_lane", "MFDs", "data_bordeaux_grouped.csv"), "edfig1a_bordeaux.png")
draw(file.path(ROOT, "data", "per_lane", "MFDs", "data_constance_grouped.csv"), "edfig1b_konstanz.png")
draw(file.path(RESULTS, "figures", "panels", "edfig1c_cagliari_raw.csv"), "edfig1c_cagliari.png")
