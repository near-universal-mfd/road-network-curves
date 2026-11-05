# Locations start from ROAD_NETWORK_RAW (default: raw/, from the repository's root). Otherwise as the authors ran it. It needs the original UTD19 download, so it was not rerun for this package.
RAW_ROOT <- normalizePath(Sys.getenv("ROAD_NETWORK_RAW", "raw"), mustWork = FALSE)

library(readr)
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
# Define the function to clean dataframes
clean_dataframe <- function(df) {
  # Count NAs in the specified columns
  na_count <- rowSums(is.na(df[c("occ", "flow", "speed")]))
  # Filter out rows with 2 or more NAs in the specified columns
  df_clean <- df[na_count < 2, ]
  return(df_clean)
}
# Loop over the dataframes and apply the function
for (i in 1:39) {
  # Construct the dataframe name and CSV file name
  df_name <- paste("data", i, sep = "")
  csv_name <- paste(df_name, "_cor.csv", sep = "")
  # Check if the dataframe exists in the environment
  if (exists(df_name)) {
    # Get the dataframe by name
    df <- get(df_name)
    # Clean the dataframe
    df_clean <- clean_dataframe(df)
    # Check if the cleaned dataframe is empty
    if (nrow(df_clean) == 0) {
      # Optional: Handle the empty dataframe case, e.g., print a message, skip saving, or save with a note
      message(paste("Cleaned dataframe", df_name, "is empty and was not saved."))
    } else {
      # Save the cleaned dataframe to a CSV file
      write.csv(df_clean, file = csv_name, row.names = FALSE)
    }
  }
}