# Locations start from ROAD_NETWORK_RAW (default: raw/, from the repository's root). Otherwise as the authors ran it. It needs the original UTD19 download, so it was not rerun for this package.
RAW_ROOT <- normalizePath(Sys.getenv("ROAD_NETWORK_RAW", "raw"), mustWork = FALSE)

setwd(file.path(RAW_ROOT, "utd19"))
library(dplyr)
library(readr)

# Define the file numbers to loop over, excluding 22 and 36
file_numbers <- setdiff(1:39, c(22, 36))

# Loop over each file number
for (num in file_numbers) {
  # Construct the file name
  file_name <- paste0("data", num, "_cor.csv")
  
  # Read the CSV file
  data <- read.csv(file_name)
  
  # Count the number of rows before filtering
  rows_before <- nrow(data)
  
  # Filter out rows with negative values in specified columns, ignoring NA values
  cleaned_data <- data %>%
    filter((is.na(occ) | occ >= 0) & (is.na(flow) | flow >= 0) & (is.na(speed) | speed >= 0))
  
  # Count the number of rows after filtering
  rows_after <- nrow(cleaned_data)
  
  # Calculate the number of removed rows
  rows_removed <- rows_before - rows_after
  
  # Only proceed with rewriting the file if rows were removed
  if (rows_removed > 0) {
    cat("In", file_name, rows_removed, "rows were removed due to negative values.\n")
    
    # Write the cleaned data back to the CSV, overwriting the original file
    write.csv(cleaned_data, file_name, row.names = FALSE)
  }
}