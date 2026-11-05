# Locations start from ROAD_NETWORK_RAW (default: raw/, from the repository's root). Otherwise as the authors ran it. It needs the original UTD19 download, so it was not rerun for this package.
RAW_ROOT <- normalizePath(Sys.getenv("ROAD_NETWORK_RAW", "raw"), mustWork = FALSE)

library(dplyr)
library(readr)
library(ggplot2)

# ---------------------------------------------------------------
# 1. Folder that holds the grouped CSVs *and* receives the PNGs
# ---------------------------------------------------------------
mfd_path <- file.path(RAW_ROOT, "curves_as_reported")

# ---------------------------------------------------------------
# 2. All grouped files    data_<City>_grouped.csv
# ---------------------------------------------------------------
grouped_files <- list.files(
  mfd_path, pattern = "_grouped\\.csv$", full.names = TRUE
)

# Table for observation counts
nbObs <- data.frame(city = character(), numRows = integer())

# ---------------------------------------------------------------
# 3. Loop over files – skip plot if PNG already exists
# ---------------------------------------------------------------
for (file in grouped_files) {
  
  # pull out city name
  city_name <- basename(file) %>%              # "data_Boston_grouped.csv"
    sub("^data_", "",  .) %>%       # "Boston_grouped.csv"
    sub("_grouped\\.csv$", "", .)   # "Boston"
  
  # destination PNG path
  png_path <- file.path(mfd_path, paste0("MFD_", city_name, ".png"))
  
  # read grouped data (always – we need counts)
  df <- read_csv(file)
  
  # collect #rows
  nbObs <- rbind(nbObs,
                 data.frame(city = city_name,
                            numRows = nrow(df)))
  
  # skip if the plot is already there
  if (file.exists(png_path)) {
    message("✓ PNG already present for ", city_name,
            " – skipping MFD plot.")
    next
  }
  
  # create scatter plot
  mfd_plot <- ggplot(df, aes(x = density, y = flow)) +
    geom_point(color = "blue") +
    labs(title = paste("MFD for", city_name),
         x = "Density (veh/km/lane)",
         y = "Flow (veh/h/lane)") +
    theme_minimal() +
    theme(panel.background = element_rect(fill = "white"),
          plot.background  = element_rect(fill = "white"))
  
  # save
  ggsave(png_path, mfd_plot, width = 8, height = 6)
  message("• Wrote ", png_path)
}

# ---------------------------------------------------------------
# 4. Bar-chart of observation counts (always regenerated)
# ---------------------------------------------------------------
bar_plot <- ggplot(nbObs, aes(x = reorder(city, -numRows), y = numRows)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Number of Observations per City",
       x = "City", y = "Number of Observations") +
  theme_minimal() +
  theme(panel.background = element_rect(fill = "white"),
        plot.background  = element_rect(fill = "white"),
        axis.text.x      = element_text(angle = 45, hjust = 1))

ggsave(file.path(mfd_path, "NumObservations_per_City.png"),
       bar_plot, width = 10, height = 6)