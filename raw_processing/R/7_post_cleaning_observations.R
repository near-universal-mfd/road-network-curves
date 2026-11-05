# Locations start from ROAD_NETWORK_RAW (default: raw/, from the repository's root). Otherwise as the authors ran it. It needs the original UTD19 download, so it was not rerun for this package.
RAW_ROOT <- normalizePath(Sys.getenv("ROAD_NETWORK_RAW", "raw"), mustWork = FALSE)

setwd(file.path(RAW_ROOT, "utd19"))

library(readr)
library(ggplot2)
library(dplyr)
library(tidyr)

# Create a list of file names for both original and cleaned data
file_names_orig <- paste0('data', 1:39, '.csv')
file_names_clean <- paste0('data', 1:39, '_cor.csv')

# Initialize vectors to store results
cities <- character(length(file_names_orig))
num_rows_orig <- numeric(length(file_names_orig))
num_rows_clean <- numeric(length(file_names_orig))

# Loop through the files
for (i in seq_along(file_names_orig)) {
  # Read only the first few rows to extract city name
  temp_data <- read_csv(file_names_orig[i], col_types = cols_only(city = col_character()), n_max = 11)
  cities[i] <- temp_data$city[1]
  
  # Count the number of rows for original data
  num_rows_orig[i] <- nrow(read_csv(file_names_orig[i], col_types = cols()))
  
  # Count the number of rows for cleaned data, considering missing cleaned files
  if (!file.exists(file_names_clean[i]) || i %in% c(22, 36)) {
    num_rows_clean[i] <- 0  # Assign 0 for missing cleaned files
  } else {
    num_rows_clean[i] <- nrow(read_csv(file_names_clean[i], col_types = cols()))
  }
}

# Combine the data into a data frame
data_summary <- data.frame(City = cities, 
                           OriginalObservations = num_rows_orig, 
                           CleanedObservations = num_rows_clean)


# Filter cities with different numbers of observations in original and cleaned data
data_summary_filtered <- data_summary %>%
  filter(OriginalObservations != CleanedObservations)
# Reshape data for plotting
data_summary_long <- data_summary_filtered %>%
  pivot_longer(cols = c(OriginalObservations, CleanedObservations), 
               names_to = "DataType", 
               values_to = "NumberOfObservations")

# Adjust the data for plotting (dividing by 1e6 for millions)
data_summary_long$NumberOfObservations <- data_summary_long$NumberOfObservations / 1e6
# Plot with original and cleaned data side by side
ggplot(data = data_summary_long, aes(x = City, y = NumberOfObservations, fill = DataType)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "City", y = "Number of Observations (in millions)", fill = "Data Type") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 12, face = "bold")) +  # Adjusted size and added bold face
  scale_y_continuous(breaks = seq(0, max(data_summary_long$NumberOfObservations), by = 2), 
                     minor_breaks = seq(0, max(data_summary_long$NumberOfObservations), by = 1))