# Locations start from ROAD_NETWORK_RAW (default: raw/, from the repository's root). Otherwise as the authors ran it. It needs the original UTD19 download, so it was not rerun for this package.
RAW_ROOT <- normalizePath(Sys.getenv("ROAD_NETWORK_RAW", "raw"), mustWork = FALSE)
DATAD <- normalizePath("raw_processing/R/dataD.xlsx", mustWork = FALSE)

setwd(file.path(RAW_ROOT, "utd19"))

library(readr)
library(ggplot2)
library(dplyr)
library(tidyr)
library(readxl)

DD <- read_excel(DATAD)

# Create a list of file names for both original and cleaned data
file_names_orig <- paste0('data', 1:39, '.csv')
file_names_clean <- paste0('data', 1:39, '_cor.csv') #remember 22 and 36 don't exist (no data)
df=read_csv(file_names_clean[8])
# Assuming df is your dataframe
df <- df %>%
  mutate(esk = if_else(speed == 0, 0, flow / speed),
         factor = if_else(esk == 0, NA_real_, 12*occ / esk))
summary(df)

# # Define the base path for saving the plots
# base_path <- file.path(RAW_ROOT, "other")
# 
# # Loop through each file name by index
# for (i in 1:length(file_names_clean)) {
#   # Skip the 22nd and 36th files
#   if (i == 22 | i == 36) {
#     next  # Skip the rest of the loop and proceed to the next iteration
#   }
#   
#   # Load the file into a dataframe
#   df <- read.csv(file_names_clean[i], header = TRUE, na.strings = c("", "NA"))
#   
#   # Plot for 'flow'
#   p_flow <- ggplot(df, aes(y = flow, fill = "Flow")) +
#     geom_boxplot() +
#     labs(title = paste("Boxplot of Flow - City", i), y = "Flow (veh/h/lane)") +
#     scale_fill_manual(values = "blue") +  # Customizing color
#     theme_light()  # Use a light theme for better readability
#   # Save the plot
#   ggsave(filename = paste0(base_path, "/flow_city", i, ".png"), plot = p_flow, width = 10, height = 8)
#   
#   # Plot for 'speed'
#   p_speed <- ggplot(df, aes(y = speed, fill = "Speed")) +
#     geom_boxplot() +
#     labs(title = paste("Boxplot of Speed - City", i), y = "Speed (km/h)") +
#     scale_fill_manual(values = "green") +  # Customizing color
#     theme_light()  # Use a light theme for better readability
#   # Save the plot
#   ggsave(filename = paste0(base_path, "/speed_city", i, ".png"), plot = p_speed, width = 10, height = 8)
#   
#   # Plot for 'occ'
#   p_occ <- ggplot(df, aes(y = occ, fill = "Occupancy")) +
#     geom_boxplot() +
#     labs(title = paste("Boxplot of Occupancy - City", i), y = "Occupancy") +
#     scale_fill_manual(values = "red") +  # Customizing color
#     theme_light()  # Use a light theme for better readability
#   # Save the plot
#   ggsave(filename = paste0(base_path, "/occ_city", i, ".png"), plot = p_occ, width = 10, height = 8)
#   
#   # Add any additional operations you need here
# }

# Define the base path for saving the exported files
base_path <- file.path(RAW_ROOT, "cleaned")

for (i in setdiff(1:39, c(22, 36))) {
  df <- read.csv(file_names_clean[i], header = TRUE, na.strings = c("", "NA"))
  
  # Get the city name from the data
  city_name <- as.character(df$city[1]) # take the first value (assuming it's consistent)
  city_name <- gsub(" ", "_", city_name) # replace spaces with underscores
  city_name <- gsub("[^A-Za-z0-9_]", "", city_name) # remove any weird characters
  
  df2 <- df %>% select(day, interval, detid, flow, occ, speed)
  
  if (i == 5) {
    df2$detid <- sub(".$", "", df2$detid)
  }
  
  df2 <- df2 %>% filter(!(flow == 0 & speed == 0))
  
  conditions <- DD[i, c("flow", "occ", "speed")]
  
  for (j in 1:3) {
    col_name <- names(conditions)[j]
    col_value <- conditions[j]
    
    if (col_value == "N") {
      next
    } else if (col_value != "Y") {
      df2 <- df2 %>% filter(get(col_name) <= as.numeric(col_value))
    }
  }
  
  if ("N" %in% conditions) {
    missing_col <- names(conditions)[which(conditions == "N")]
    
    if (missing_col == "speed") {
      df2 <- df2 %>% mutate(density = occ * 500)
      df2 <- df2 %>% mutate(speed = ifelse(density == 0, Inf, flow / density))
    } else if (missing_col == "occ") {
      df2 <- df2 %>% mutate(density = flow / speed)
      df2 <- df2 %>% mutate(occ = density / 500)
    }
  }
  
  df2$speed <- pmin(df2$speed, 105)
  
  # Remove rows where density calculation results in Inf
  df2 <- df2 %>% filter(!is.infinite(density))
  
  df2 <- select(df2, -occ)
  df2 <- na.omit(df2)
  
  # Now use city name in the file name!
  file_path <- paste0(base_path, "/data_", city_name, "_corre.csv")
  write.csv(df2, file_path, row.names = FALSE)
}