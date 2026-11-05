# Locations start from ROAD_NETWORK_RAW (default: raw/, from the repository's root). Otherwise as the authors ran it. It needs the original UTD19 download, so it was not rerun for this package.
RAW_ROOT <- normalizePath(Sys.getenv("ROAD_NETWORK_RAW", "raw"), mustWork = FALSE)

setwd(file.path(RAW_ROOT, "utd19"))

library(readr)
library(ggplot2)

# Step 1: Create a list of file names
file_names <- paste0('data', 1:39, '.csv')

# Step 2: Initialize vectors to store results
cities <- character(length(file_names))
num_rows <- numeric(length(file_names))

# Step 3: Loop through the files
for (i in seq_along(file_names)) {
  # Read only the first few rows to extract city name and number of rows
  temp_data <- read_csv(file_names[i], col_types = cols_only(city = col_character()), n_max = 11)
  cities[i] <- temp_data$city[1]
  
  # Count the number of rows without reading the entire file
  num_rows[i] <- nrow(read_csv(file_names[i], col_types = cols()))
}

# Step 4: Modify column name to avoid spaces
data_summary <- data.frame(City = cities, NumberOfObservations = num_rows)

# Adjust the data for plotting (dividing the number of observations by 1e6 for millions)
data_summary$NumberOfObservations <- data_summary$NumberOfObservations / 1e6

# Plot with adjusted y-axis
ggplot(data = data_summary, mapping = aes(x = City, y = NumberOfObservations)) +
  geom_bar(stat = "identity") +
  labs(x = "City", y = "Number of Observations (in millions)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(breaks = seq(0, max(data_summary$NumberOfObservations), by = 2),  # Major ticks every 4 million
                     minor_breaks = seq(0, max(data_summary$NumberOfObservations), by = 1))  # Minor ticks every 2 million