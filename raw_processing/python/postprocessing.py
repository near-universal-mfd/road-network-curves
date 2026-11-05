"""Screening of individual network curves after visual inspection

Removes points by flow thresholds and by the few custom rules below. The thresholds are in each source's own units. data/screening_decisions.csv lists them.

Run by the authors as cells of a notebook, in this order; each "cell" below is one of those cells. It needs the original downloads, so it was not rerun for this package. The only
changes are the locations, which now start from ROAD_NETWORK_RAW (default: raw/, from the repository's root), and the two rules first keyed to "essen" and "toronto" are keyed to "darmstadt" and "torino", whose data those two labels then held (see rename_grouped_files.R).
"""
import os

RAW_ROOT = os.environ.get("ROAD_NETWORK_RAW", "raw")


# %% cell 1 of 2
import pandas as pd
import os
import shutil

# Base folder path
base_path = os.path.join(RAW_ROOT, "curves_as_reported")
uncleaned_path = os.path.join(base_path, "Uncleaned files")

# Ensure the Uncleaned files directory exists
os.makedirs(uncleaned_path, exist_ok=True)

# Rules for each city with flow thresholds
flow_thresholds = {
    "Winchester": 2750,
    "Winwick": 4500,
    "Wakefield": 3200,
    "Trafford": 4000,
    "Three_Rivers": 5050,
    "Thorney": 5650,
    "Stockport": 3550,
    "St_Albans": 4400,
    "South_Oxfordshire": 1900,
    "South_Ribble": 5900,
    "South_Cambridgeshire": 4000,
    "Sevenoaks": 3400,
    "Sandwell": 3900,
    "Sandbach": 4900,
    "Rochdale": 4000,
    "Preston": 4000,
    "Prestwich": 5900,
    "North_East_Derbyshire": 3550,
    "Middleton": 3950,
    "Metropolitan_Borough_of_Solihull": 3500,
    "Manchester2": 3860,
    "Godstone": 3500,
    "Bury": 4110,
    "Borough_of_Runnymede": 4900,
}

def clean_test_valley(df):
    return df[(df['flow'] <= 3000) & (df['density'] <= 100)]

def clean_tameside(df):
    cleaned_rows = []
    for _, row in df.iterrows():
        q = row['flow']
        k = row['density']
        if 1000 < q < 3000:
            window = df[(df['flow'] >= q - 50) & (df['flow'] <= q + 50)]
            min_k = window['density'].min()
            if k <= 1.1 * min_k:
                cleaned_rows.append(row)
        else:
            cleaned_rows.append(row)
    return pd.DataFrame(cleaned_rows)

# Main cleaning function
def clean_city_data(city):
    csv_filename = f"data_{city}_grouped.csv"
    csv_filepath = os.path.join(base_path, csv_filename)

    if not os.path.exists(csv_filepath):
        print(f"CSV file not found for city: {city}")
    else:
        df = pd.read_csv(csv_filepath)

        if city == "Test_Valley":
            df_cleaned = clean_test_valley(df)
        elif city == "Tameside":
            df_cleaned = clean_tameside(df)
        elif city in flow_thresholds:
            threshold = flow_thresholds[city]
            df_cleaned = df[df['flow'] <= threshold]
        else:
            print(f"No cleaning rule specified for {city}")
            df_cleaned = df

        df_cleaned.to_csv(csv_filepath, index=False)
        print(f"Cleaned CSV saved for {city}.")

        # Move original CSV to Uncleaned files folder
        shutil.move(csv_filepath, os.path.join(uncleaned_path, csv_filename))
        print(f"Moved uncleaned CSV for {city} to Uncleaned files.")

    # Move corresponding image file if it exists
    png_filename = f"MFD_{city}.png"
    png_filepath = os.path.join(base_path, png_filename)

    if os.path.exists(png_filepath):
        shutil.move(png_filepath, os.path.join(uncleaned_path, png_filename))
        print(f"Moved PNG for {city} to Uncleaned files.")
    else:
        print(f"PNG file not found for city: {city}")

# List of cities to process
cities = [
    "Winchester", "Winwick", "Wakefield", "Trafford", "Three_Rivers", "Thorney",
    "Test_Valley", "Tameside", "Stockport", "St_Albans", "South_Oxfordshire",
    "South_Ribble", "South_Cambridgeshire", "Sevenoaks", "Sandwell", "Sandbach",
    "Rochdale", "Preston", "Prestwich", "North_East_Derbyshire", "Middleton",
    "Metropolitan_Borough_of_Solihull", "Manchester2", "Godstone", "Bury",
    "Borough_of_Runnymede"
]

for city in cities:
    clean_city_data(city)


# %% cell 2 of 2
import pandas as pd
import os
import shutil

# Base and destination paths
base_path = os.path.join(RAW_ROOT, "curves_as_reported")
uncleaned_path = os.path.join(base_path, "Uncleaned files")

# Ensure uncleaned directory exists
os.makedirs(uncleaned_path, exist_ok=True)

# Custom cleaning functions for new cities
def clean_birmingham(df):
    return df[~((df['flow'] < 375) & (df['density'] > 16.1))]

def clean_cagliari(df):
    # Line: from (8.5, 0) to (29.5, 200)
    # Line equation: flow = 200 / (29.5 - 8.5) * (density - 8.5)
    slope = 200 / (29.5 - 8.5)
    return df[~((df['flow'] < 200) & (df['flow'] < slope * (df['density'] - 8.5)))]

def clean_darmstadt(df):
    return df[~(((df['flow'] < 50) & (df['density'] > 26)) | ((df['flow'] < 75) & (df['density'] > 37.5)))]

def clean_torino(df):
    return df[df['flow'] <= 900]

# Mapping of city names to their cleaning functions
city_cleaning_map = {
    "birmingham": clean_birmingham,
    "cagliari": clean_cagliari,
    "darmstadt": clean_darmstadt,
    "torino": clean_torino,
}

def clean_city_data(city):
    csv_filename = f"data_{city}_grouped.csv"
    csv_filepath = os.path.join(base_path, csv_filename)

    if not os.path.exists(csv_filepath):
        print(f"CSV file not found for city: {city}")
    else:
        df = pd.read_csv(csv_filepath)
        cleaner = city_cleaning_map.get(city)

        if cleaner:
            df_cleaned = cleaner(df)
            df_cleaned.to_csv(csv_filepath, index=False)
            print(f"Cleaned CSV saved for {city}.")
        else:
            print(f"No cleaning function defined for city: {city}")
            return

        # Move original CSV to Uncleaned files
        shutil.move(csv_filepath, os.path.join(uncleaned_path, csv_filename))
        print(f"Moved uncleaned CSV for {city} to Uncleaned files.")

    # Move corresponding image if it exists
    png_filename = f"MFD_{city}.png"
    png_filepath = os.path.join(base_path, png_filename)

    if os.path.exists(png_filepath):
        shutil.move(png_filepath, os.path.join(uncleaned_path, png_filename))
        print(f"Moved PNG for {city} to Uncleaned files.")
    else:
        print(f"PNG file not found for city: {city}")

# List of new cities to process
new_cities = ["birmingham", "cagliari", "darmstadt", "torino"]

for city in new_cities:
    clean_city_data(city)

