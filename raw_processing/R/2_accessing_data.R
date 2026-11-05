# Locations start from ROAD_NETWORK_RAW (default: raw/, from the repository's root). Otherwise as the authors ran it. It needs the original UTD19 download, so it was not rerun for this package.
RAW_ROOT <- normalizePath(Sys.getenv("ROAD_NETWORK_RAW", "raw"), mustWork = FALSE)

#A<-read.csv('utd19_u.csv') # file is very large, make sure you have enough memory and processing power
setwd(file.path(RAW_ROOT, "utd19"))
library(readr)

# Read only the first row to get the column names
header <- read_csv('utd19_u.csv', n_max = 1)

# data1 <- read_csv('utd19_u.csv', n_max = 3443577)
# data2 <- read_csv('utd19_u.csv', skip = 3443578, n_max = 155232, col_names = colnames(header))
# data3 <- read_csv('utd19_u.csv', skip = 3598810, n_max = 1526032, col_names = colnames(header))
# data4 <- read_csv('utd19_u.csv', skip = 5124842, n_max = 112463, col_names = colnames(header))
# data5 <- read_csv('utd19_u.csv', skip = 5237305, n_max = 246842, col_names = colnames(header))
# data6 <- read_csv('utd19_u.csv', skip = 5484147, n_max = 844416, col_names = colnames(header))
# data7 <- read_csv('utd19_u.csv', skip = 6328563, n_max = 3578512, col_names = colnames(header))
# data8 <- read_csv('utd19_u.csv', skip = 9907075, n_max = 2820293, col_names = colnames(header))
# data9 <- read_csv('utd19_u.csv', skip = 12727368, n_max = 245952, col_names = colnames(header))
# data10 <- read_csv('utd19_u.csv', skip = 12973320, n_max = 4677926, col_names = colnames(header))
# data11 <- read_csv('utd19_u.csv', skip = 17651246, n_max = 291893, col_names = colnames(header))
# data12 <- read_csv('utd19_u.csv', skip = 17943139, n_max = 30638, col_names = colnames(header))
# data13 <- read_csv('utd19_u.csv', skip = 17973777, n_max = 847827, col_names = colnames(header))
# data14 <- read_csv('utd19_u.csv', skip = 18821604, n_max = 28378, col_names = colnames(header))
# data15 <- read_csv('utd19_u.csv', skip = 18849982, n_max = 20415435, col_names = colnames(header))
# data16 <- read_csv('utd19_u.csv', skip = 39265417, n_max = 138240, col_names = colnames(header))
# data17 <- read_csv('utd19_u.csv', skip = 39403657, n_max = 534684, col_names = colnames(header))
# data18 <- read_csv('utd19_u.csv', skip = 39938341, n_max = 34397322, col_names = colnames(header))
# data19 <- read_csv('utd19_u.csv', skip = 74335663, n_max = 4611254, col_names = colnames(header))
# data20 <- read_csv('utd19_u.csv', skip = 78946917, n_max = 26878297, col_names = colnames(header))
# dataP2 <- read_csv('utd19_u.csv', skip = 105825214, col_names = colnames(header))
# # Write dataframes to csv files
# write_csv(data1, 'data1.csv')
# write_csv(data2, 'data2.csv')
# write_csv(data3, 'data3.csv')
# write_csv(data4, 'data4.csv')
# write_csv(data5, 'data5.csv')
# write_csv(data6, 'data6.csv')
# write_csv(data7, 'data7.csv')
# write_csv(data8, 'data8.csv')
# write_csv(data9, 'data9.csv')
# write_csv(data10, 'data10.csv')
# write_csv(data11, 'data11.csv')
# write_csv(data12, 'data12.csv')
# write_csv(data13, 'data13.csv')
# write_csv(data14, 'data14.csv')
# write_csv(data15, 'data15.csv')
# write_csv(data16, 'data16.csv')
# write_csv(data17, 'data17.csv')
# write_csv(data18, 'data18.csv')
# write_csv(data19, 'data19.csv')
# write_csv(data20, 'data20.csv')
# write_csv(dataP2, 'dataP2.csv')
# data21 <- read_csv('dataP2.csv', skip = 1, n_max = 5070083, col_names = colnames(header))
# data22 <- read_csv('dataP2.csv', skip = 5070084, n_max = 1369849, col_names = colnames(header))
# data23 <- read_csv('dataP2.csv', skip = 6439933, n_max = 1142893, col_names = colnames(header))
# data24 <- read_csv('dataP2.csv', skip = 7582826, n_max = 2433600, col_names = colnames(header))
# data25 <- read_csv('dataP2.csv', skip = 10016426, n_max = 149270, col_names = colnames(header))
# data26 <- read_csv('dataP2.csv', skip = 10165696, n_max = 1548314, col_names = colnames(header))
# data27 <- read_csv('dataP2.csv', skip = 11714010, n_max = 351257, col_names = colnames(header))
# data28 <- read_csv('dataP2.csv', skip = 12065267, n_max = 490136, col_names = colnames(header))
# data29 <- read_csv('dataP2.csv', skip = 12555403, n_max = 1235392, col_names = colnames(header))
# data30 <- read_csv('dataP2.csv', skip = 13790795, n_max = 1327558, col_names = colnames(header))
# data31 <- read_csv('dataP2.csv', skip = 15118353, n_max = 600192, col_names = colnames(header))
# data32 <- read_csv('dataP2.csv', skip = 15718545, n_max = 2408368, col_names = colnames(header))
# data33 <- read_csv('dataP2.csv', skip = 18126913, n_max = 2080768, col_names = colnames(header))
# cities1 <- unique(data33$city) #Torino! Supposed to be Tokyo (all subsequent cities are shifted by 1)
# data34 <- read_csv('dataP2.csv', skip = 20207681, n_max = 1042330, col_names = colnames(header))
# data35 <- read_csv('dataP2.csv', skip = 21250011, n_max = 1602720, col_names = colnames(header))
# data36 <- read_csv('dataP2.csv', skip = 22852731, n_max = 1174140, col_names = colnames(header))
# data37 <- read_csv('dataP2.csv', skip = 24026871, n_max = 209614, col_names = colnames(header))
# data38 <- read_csv('dataP2.csv', skip = 24236485, n_max = 893672, col_names = colnames(header))
# data39 <- read_csv('dataP2.csv', skip = 25130157, col_names = colnames(header))
# Write the new dataframes to csv files
# write_csv(data21, 'data21.csv')
# write_csv(data22, 'data22.csv')
# write_csv(data23, 'data23.csv')
# write_csv(data24, 'data24.csv')
# write_csv(data25, 'data25.csv')
# write_csv(data26, 'data26.csv')
# write_csv(data27, 'data27.csv')
# write_csv(data28, 'data28.csv')
# write_csv(data29, 'data29.csv')
# write_csv(data30, 'data30.csv')
# write_csv(data31, 'data31.csv')
# write_csv(data32, 'data32.csv')
# write_csv(data33, 'data33.csv')
# write_csv(data34, 'data34.csv')
# write_csv(data35, 'data35.csv')
# write_csv(data36, 'data36.csv')
# write_csv(data37, 'data37.csv')
# write_csv(data38, 'data38.csv')
# write_csv(data39, 'data39.csv')

# Read the dataframes from csv files
data1 <- read_csv('data1.csv')
data2 <- read_csv('data2.csv')
data3 <- read_csv('data3.csv')
data4 <- read_csv('data4.csv')
data5 <- read_csv('data5.csv')
data6 <- read_csv('data6.csv')
data7 <- read_csv('data7.csv')
data8 <- read_csv('data8.csv')
data9 <- read_csv('data9.csv')
data10 <- read_csv('data10.csv')
data11 <- read_csv('data11.csv')
data12 <- read_csv('data12.csv')
data13 <- read_csv('data13.csv')
data14 <- read_csv('data14.csv')
data15 <- read_csv('data15.csv')
data16 <- read_csv('data16.csv')
data17 <- read_csv('data17.csv')
data18 <- read_csv('data18.csv')
data19 <- read_csv('data19.csv')
data20 <- read_csv('data20.csv')
data21 <- read_csv('data21.csv')
data22 <- read_csv('data22.csv')
data23 <- read_csv('data23.csv')
data24 <- read_csv('data24.csv')
data25 <- read_csv('data25.csv')
data26 <- read_csv('data26.csv')
data27 <- read_csv('data27.csv')
data28 <- read_csv('data28.csv')
data29 <- read_csv('data29.csv')
data30 <- read_csv('data30.csv')
data31 <- read_csv('data31.csv')
data32 <- read_csv('data32.csv')
data33 <- read_csv('data33.csv')
data34 <- read_csv('data34.csv')
data35 <- read_csv('data35.csv')
data36 <- read_csv('data36.csv')
data37 <- read_csv('data37.csv')
data38 <- read_csv('data38.csv')
data39 <- read_csv('data39.csv')

# Initialize variables
initial_skip <- 78946917  # Starting point, adjust based on your previous reads
skip <- initial_skip  # Use a separate variable for skip to preserve the initial value
chunk_size <- 200000  # Number of rows to read in each chunk

while (TRUE) {
  # Read a chunk of data
  data <- read_csv('utd19_u.csv', skip = skip, n_max = chunk_size, col_names = colnames(header))
  
  # Get the number of unique cities in the chunk
  unique_cities <- unique(data$city)
  
  if (length(unique_cities) == 1) {
    # If the chunk contains only one unique city, print the city and the boundary
    cat("Chunk starting at skip =", skip, "contains only city:", unique_cities, "\n")
    skip <- skip + nrow(data)  # Move to the next chunk after the current one
  } else {
    # More than one unique city found, search within this chunk for the boundary
    for (i in 2:nrow(data)) {
      if (data$city[i] != data$city[i-1]) {
        # Found the boundary within the chunk
        cat("Boundary found at row =", skip + i - 1, "\nCities:", data$city[i-1], "and", data$city[i], "\n")
        # Calculate n_max as the difference between the boundary row and the initial skip, plus one to include the boundary row
        n_max_corrected <- (skip + i - 1) - initial_skip
        cat("Use n_max =", n_max_corrected, "\n")
        break
      }
    }
    break  # Exit the while loop after finding the boundary
  }
  
  # Optional: Check to ensure the read operation returned data, to handle reaching the end of the file
  if (nrow(data) < chunk_size) {
    cat("Reached end of file or read less data than expected at skip =", skip, "\n")
    break  # Exit the loop if the end of the file is reached or less data is read
  }
}




####part of the code for 2nd paper (not in the MFD paper)
# List of detector IDs to keep (London)
detector_ids <- c(
  "CNTR_N01/383a1", "CNTR_N01/364b1", "CNTR_N01/383b1",
  "CNTR_N01/378a1", "CNTR_N01/378b1",
  "CNTR_N01/205x1", "CNTR_N01/175c2",
  "CNTR_N01/379a1", "CNTR_N01/206b1", "CNTR_N01/207a1", "CNTR_N01/379b1",
  "CNTR_N01/176x2", "CNTR_N01/206c1"
)

# Filter data18 to keep only the relevant detector IDs
LWRdf <- data18 %>% filter(detid %in% detector_ids)

# Load necessary libraries
library(dplyr)
library(tidyr)

# Step 1: Keep only the specified columns
LWRdf <- LWRdf %>% select(day, interval, detid, flow, occ)

# Step 2: Group by "day" and "interval", and add a unique identifier for each detector row within the group
LWRdf <- LWRdf %>%
  group_by(day, interval) %>%
  mutate(detector_num = row_number()) %>%
  ungroup()

# Step 3: Pivot the data to create columns for each detector's "detid", "flow", and "occ" values
LWRdf <- LWRdf %>%
  pivot_wider(
    names_from = detector_num,
    values_from = c(detid, flow, occ),
    names_sep = ""
  )

# Step 4: Reorder columns if necessary to maintain a structured sequence of "detid", "flow", "occ" for each detector
column_order <- c("day", "interval", 
                  paste0(rep(c("detid", "flow", "occ"), each = 13), 1:13))
LWRdf <- LWRdf %>% select(all_of(column_order))
# Create new columns "density1" to "density13" by multiplying "occ1" to "occ13" by 500
LWRdf <- LWRdf %>%
  mutate(across(starts_with("occ"), ~ . * 500, .names = "density{col}")) %>%
  select(-starts_with("occ"))  # Remove the original "occ" columns

# Remove 13 columns starting from the 3rd column
LWRdf <- LWRdf %>%
  select(-c(3:(3 + 13 - 1)))  # Adjusts to remove columns 3 to 18

# Step 2: Define the reorder pattern for each 16-column part
reorder_pattern <- c(7, 13, 10, 11, 6, 12, 5, 4, 1, 3, 2, 8, 9)

# Step 3: Split columns into parts
# Preserve the first 2 columns
preserved_cols <- LWRdf %>% select(1:2)

# Extract the two 13-column parts
flow_cols <- LWRdf %>% select(3:15)
density_cols <- LWRdf %>% select(16:28)

# Step 4: Reorder each part by moving each column to its new position
flow_cols <- flow_cols[, order(reorder_pattern)]
density_cols <- density_cols[, order(reorder_pattern)]

# Step 5: Rename columns as requested
colnames(flow_cols) <- paste0("flow", 1:13)
colnames(density_cols) <- paste0("density", 1:13)

# Step 6: Combine all parts into the final dataframe
LWRdf <- bind_cols(preserved_cols, flow_cols, density_cols)
write.csv(LWRdf,file=file.path(RAW_ROOT, "other/app.csv"))
