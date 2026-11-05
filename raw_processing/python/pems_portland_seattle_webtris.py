"""Reading Seattle, Portland and WebTRIS into network series

Seattle: five k-means clusters of detectors, clusters 1 and 5 kept. Portland: 20-second records to 5-minute records, then a network curve. WebTRIS: 15-minute records for all sites from the API, sites assigned to towns by reverse geocoding, towns with fewer than 20 sites set aside. Its Portland cell averages detectors that report no speed as 0 km/h, and Seattle's speeds are mph: scripts/build_per_lane_data.py corrects both.

Run by the authors as cells of a notebook, in this order; each "cell" below is one of those cells. It needs the original downloads, so it was not rerun for this package. The only
changes are the locations, which now start from ROAD_NETWORK_RAW (default: raw/, from the repository's root).
"""
import os

RAW_ROOT = os.environ.get("ROAD_NETWORK_RAW", "raw")


# %% cell 1 of 15
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------
# 0. Paths & constants  (only things you may need to tweak)
# -------------------------------------------------------------------
city              = "seattle"
raw_path          = os.path.join(RAW_ROOT, f"traffic/{city}.csv")
cabinet_path      = os.path.join(RAW_ROOT, "detector_tables/Cabinet Location Information.csv")
cluster_template  = os.path.join(RAW_ROOT, "detector_tables/detectors_") + city + "{}.csv"    # 1-5
out_dir           = os.path.join(RAW_ROOT, f"curves_as_reported")
MPH2KPH           = 1.60934            # unused here but handy if you reuse script
SPACE_FACTOR      = 0.95               # time-mean → space-mean

# -------------------------------------------------------------------
# 1. Load raw Seattle traffic (no header row)
# -------------------------------------------------------------------
traffic = pd.read_csv(
    raw_path,
    header=None,
    names=["timestamp", "detid", "direction", "speed_kph", "volume_5min", "unused"],
    usecols=[0, 1, 2, 3, 4],             # skip 6th column
    parse_dates=["timestamp"],
    infer_datetime_format=True
)

# -------------------------------------------------------------------
# 2. Build mapping: detid  →  rounded coordinate key
# -------------------------------------------------------------------
cab = pd.read_csv(cabinet_path, usecols=["CabName", "Lat", "Lon"])
cab["coord_key"] = (
    cab["Lat"].round(6).astype(str) + "_" +
    cab["Lon"].round(6).astype(str)
)
detid_to_coord = dict(zip(cab["CabName"].astype(str).str.strip(), cab["coord_key"]))

# -------------------------------------------------------------------
# 3. Build mapping: coord_key  →  cluster_id   (1-5)
# -------------------------------------------------------------------
coord_to_cluster = {}
for k in range(1, 6):
    clust_file = cluster_template.format(k)
    dfc        = pd.read_csv(clust_file)

    # Robustly pick the latitude/longitude columns
    if {"Lat", "Lon"}.issubset(dfc.columns):
        latlon_df = dfc[["Lat", "Lon"]]
    else:                                   # assume first two cols are lat, lon
        latlon_df = dfc.iloc[:, :2].rename(columns={0: "Lat", 1: "Lon"})

    latlon_df["coord_key"] = (
        latlon_df["Latitude"].round(6).astype(str) + "_" +
        latlon_df["Longitude"].round(6).astype(str)
    )
    for ck in latlon_df["coord_key"]:
        coord_to_cluster[ck] = k

# -------------------------------------------------------------------
# 4. Tag every traffic row with its cluster (or NaN if no match)
# -------------------------------------------------------------------
traffic["coord_key"] = traffic["detid"].map(detid_to_coord)
traffic["cluster"]   = traffic["coord_key"].map(coord_to_cluster)

missing = traffic["cluster"].isna().sum()
print(f"{missing} rows could not be mapped to any cluster and will be skipped.")

traffic = traffic.dropna(subset=["cluster"])
traffic["cluster"] = traffic["cluster"].astype(int)

# -------------------------------------------------------------------
# 5. Per-record conversions
# -------------------------------------------------------------------
traffic["flow_vph"]    = traffic["volume_5min"] * 12          # veh/h
traffic["speed_space"] = traffic["speed_kph"]   * SPACE_FACTOR

# -------------------------------------------------------------------
# 6. Loop over the five clusters and produce outputs
# -------------------------------------------------------------------
os.makedirs(out_dir, exist_ok=True)

for k in range(1, 6):
    sub = traffic[traffic["cluster"] == k]
    if sub.empty:
        print(f"Cluster {k}: no data after mapping – skipped.")
        continue

    # Aggregate to one row per timestamp
    agg = (
        sub
        .groupby("timestamp", as_index=False)
        .agg(flow=("flow_vph", "mean"),
             speed=("speed_space", "mean"))
    )
    agg["density"] = agg["flow"] / agg["speed"]          # veh/km
    agg_out        = agg[["timestamp", "flow", "density"]]

    # ---------- 6a.  Save grouped CSV
    tag          = f"{city}_cluster{k}"
    csv_path     = os.path.join(out_dir, f"data_{tag}_grouped.csv")
    agg_out.to_csv(csv_path, index=False)
    print(f"Cluster {k}: grouped CSV → {csv_path}")

    # ---------- 6b.  Plot MFD
    png_path = os.path.join(out_dir, f"MFD_{tag}.png")
    plt.figure(figsize=(6, 5))
    plt.scatter(agg["density"], agg["flow"], s=8, alpha=0.8)
    plt.xlabel("Density (veh / km)")
    plt.ylabel("Flow (veh / h)")
    plt.title(f"MFD – Seattle · Cluster {k}")
    plt.grid(True, linestyle="--", linewidth=0.4)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Cluster {k}: MFD PNG   → {png_path}")


# %% cell 2 of 15
import os
import pandas as pd
import numpy as np

# ── 1.  Paths ────────────────────────────────────────────────────────────
raw_path = (
    os.path.join(RAW_ROOT, "traffic/Portland.csv")
)

out_path = (
    os.path.join(RAW_ROOT, "traffic/Portland_5min.csv")
)
os.makedirs(os.path.dirname(out_path), exist_ok=True)

# ── 2.  Read raw file (treat blanks as 0) ────────────────────────────────
traffic = pd.read_csv(
    raw_path,
    dtype={"detectorid": str},
    parse_dates=["starttime"],
    na_values=["", " "],
)

# replace NaNs in numeric columns with 0
numeric_cols = ["volume", "speed", "occupancy", "status", "dqflags"]
traffic[numeric_cols] = traffic[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

# ── 3.  Create a 5-minute time bucket and a “bad status” flag ────────────
traffic["time_bin"] = traffic["starttime"].dt.floor("5min")   # first row’s timestamp of each 5-min slot
traffic["bad_status"] = traffic["status"] > 3

# ── 4.  Aggregate 15×20-s rows → one 5-min row per detector/bin ──────────
def _agg_func(group: pd.DataFrame) -> pd.Series:
    # discard if any status > 3
    if group["bad_status"].any():
        return pd.Series(dtype=float)     # will be dropped later

    sum_vol = group["volume"].sum()
    flow    = sum_vol * 12.0              # veh/h (5-min → hourly)

    # weighted time-mean speed → space-mean → km/h
    if sum_vol > 0:
        speed_tm = (group["speed"] * group["volume"]).sum() / sum_vol
        speed_sm = speed_tm * 0.95 * 1.60934
    else:
        speed_sm = np.nan                 # blank if no vehicles

    # density (veh/km) – only if speed > 0
    density = flow / speed_sm if pd.notna(speed_sm) and speed_sm > 0 else 0.0

    return pd.Series({"flow": flow, "speed": speed_sm, "density": density})


agg = (
    traffic
    .groupby(["detectorid", "time_bin"], sort=False)
    .apply(_agg_func)
    .dropna(subset=["flow"])                # drop discarded groups
    .reset_index()
    .rename(columns={"time_bin": "starttime"})
)

# ── 5.  Save result ───────────────────────────────────────────────────────
agg[["detectorid", "starttime", "flow", "speed", "density"]].to_csv(out_path, index=False)
print(f"Aggregated 5-minute data written to:\n{out_path}")

# %% cell 3 of 15
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── 1.  Paths ────────────────────────────────────────────────────────────
five_min_path = os.path.join(RAW_ROOT, "traffic/Portland_5min.csv")
station_map    = os.path.join(RAW_ROOT, "freeway_detectors.csv")

mfd_dir   = os.path.join(RAW_ROOT, "curves_as_reported")
os.makedirs(mfd_dir, exist_ok=True)
grouped_path = os.path.join(mfd_dir, "data_portland_grouped.csv")
plot_path    = os.path.join(mfd_dir, "MFD_portland.png")

# ── 2.  Load 5-min data & station map ────────────────────────────────────
df = pd.read_csv(
    five_min_path,
    parse_dates=["starttime"],
    dtype={"detectorid": str},
    na_values=["", " "]
)

station_df = pd.read_csv(
    station_map,
    usecols=lambda c: c.strip().casefold().replace(" ", "_") in {"detectorid", "stationid"},
    dtype=str
).rename(columns=str.lower)

# ── 3.  Join to get stationid; drop rows without a match ─────────────────
df = df.merge(station_df, on="detectorid", how="left")
df = df.dropna(subset=["stationid"])

# ── 4.  Per-station (per-lane) averages at each timestamp ───────────────
station_agg = (
    df.groupby(["stationid", "starttime"], sort=False)
      .agg(flow=("flow", "mean"),   # mean = per lane
           speed=("speed", "mean"))
      .reset_index()
)
# density per station
station_agg["density"] = np.where(
    station_agg["speed"] > 0,
    station_agg["flow"] / station_agg["speed"],
    0.0
)

# ── 5.  Network-level average across stations at each timestamp ─────────
net = (
    station_agg.groupby("starttime", sort=False)
               .agg(flow=("flow", "mean"),
                    speed=("speed", "mean"))
               .reset_index()
)
net["density"] = np.where(net["speed"] > 0, net["flow"] / net["speed"], 0.0)

# ── 6.  Add day + interval (seconds from midnight) ──────────────────────
net["day"] = net["starttime"].dt.date.astype(str)
net["interval"] = (
    net["starttime"] - net["starttime"].dt.normalize()
).dt.total_seconds().astype(int)

grouped_out = net[["day", "interval", "flow", "speed", "density"]]

# ── 7.  Save grouped CSV --------------------------------------------------
grouped_out.to_csv(grouped_path, index=False)
print(f"Grouped dataset written to:\n{grouped_path}")

# ── 8.  Plot the MFD ------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.scatter(net["density"], net["flow"], s=8, alpha=0.8)
plt.xlabel("Density (veh / km)")
plt.ylabel("Flow (veh / h)")
plt.title("Macroscopic Fundamental Diagram – Portland")
plt.grid(True, linestyle="--", linewidth=0.4)
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"MFD plot saved to:\n{plot_path}")

# %% cell 4 of 15
# england_all_sites_csv_raw.py
"""
Pull all MIDAS detector data from WebTRIS for a 3-month window
and save *every* column exactly as delivered by the API.

Outputs ONE CSV at the user-supplied path.
"""

from __future__ import annotations
import datetime as dt
import itertools
import json
import requests
import pandas as pd
from typing import Iterator, Sequence

BASE = "https://webtris.highwaysengland.co.uk/api/v1.0"

OUT_CSV = (
    os.path.join(RAW_ROOT, "traffic/webtris_15min_all_sites.csv")
)

CHUNK     = 30        # max sites per call
PAGE_SIZE = 40000     # max rows per page


def _ddmmyyyy(d: dt.date) -> str:
    return d.strftime("%d%m%Y")


def _grouper(it: Sequence[int], n: int = CHUNK) -> Iterator[str]:
    itr = iter(it)
    while chunk := list(itertools.islice(itr, n)):
        yield ",".join(map(str, chunk))


def list_sites(site_type: int | None = 1) -> pd.DataFrame:
    url = f"{BASE}/sites" if site_type is None else f"{BASE}/sitetypes/{site_type}/sites"
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    return pd.json_normalize(r.json()["sites"])


def fetch_all_daily(
    site_ids: list[int],
    start: dt.date,
    end: dt.date,
    write_csv: str = OUT_CSV,
) -> None:
    """Stream raw 15-min pages to one CSV; keep all columns."""
    first_write = True

    for batch in _grouper(site_ids, CHUNK):
        page = 1
        while True:
            url = (
                f"{BASE}/reports/daily?"
                f"sites={batch}&start_date={_ddmmyyyy(start)}"
                f"&end_date={_ddmmyyyy(end)}&page={page}&page_size={PAGE_SIZE}"
            )
            r = requests.get(url, timeout=180)

            # 204 (no content) or 400 (HTML error) ⇒ stop paging this batch
            if r.status_code in (204, 400):
                break
            try:
                payload = r.json()
            except json.JSONDecodeError:  # caught HTML or gibberish
                break

            rows = payload.get("Rows", [])
            if not rows:        # empty page – done with this batch
                break

            df = pd.DataFrame(rows)  # ← keep every column as-is

            # append to CSV
            df.to_csv(write_csv, mode="w" if first_write else "a",
                      header=first_write, index=False)
            first_write = False

            # keep paging until “nextPage” disappears
            has_next = any(link.get("rel") == "nextPage"
                           for link in payload.get("Header", {}).get("links", []))
            if has_next:
                page += 1
            else:
                break

        print(f"Finished batch {batch[:50]}…", flush=True)


if __name__ == "__main__":
    START = dt.date(2024, 12, 1)
    END   = dt.date(2025, 2, 28)   # inclusive

    print("Querying detector list…")
    sites_df = list_sites(site_type=1)              # MIDAS only
    site_ids = sites_df["Id"].astype(int).tolist()
    print(f"Found {len(site_ids):,} MIDAS sites")

    print(f"Downloading {START} → {END} for *all* sites…")
    fetch_all_daily(site_ids, START, END)
    print(f"Complete ✓  Data saved to:\n{OUT_CSV}")

# %% cell 5 of 15
import pandas as pd
import os

# Path to your existing full CSV
IN_CSV = (
    os.path.join(RAW_ROOT, "traffic/webtris_15min_all_sites.csv")
)

# Output cleaned version in same folder
OUT_CSV = IN_CSV.replace(".csv", "_cleaned.csv")

# Columns to drop
DROP_COLS = [
    "0 - 520 cm", "521 - 660 cm", "661 - 1160 cm", "1160+ cm",
    "0 - 10 mph", "11 - 15 mph", "16 - 20 mph", "21 - 25 mph",
    "26 - 30 mph", "31 - 35 mph", "36 - 40 mph", "41 - 45 mph",
    "46 - 50 mph", "51 - 55 mph", "56 - 60 mph", "61 - 70 mph",
    "71 - 80 mph", "80+ mph"
]

# Read in chunks to avoid memory blowup
CHUNK_SIZE = 250_000
first_chunk = True

print("Cleaning CSV...")

for chunk in pd.read_csv(IN_CSV, chunksize=CHUNK_SIZE):
    available = [col for col in DROP_COLS if col in chunk.columns]
    cleaned = chunk.drop(columns=available)
    cleaned.to_csv(OUT_CSV, mode="w" if first_chunk else "a",
                   index=False, header=first_chunk)
    first_chunk = False

print(f"✅ Done. Cleaned file saved to:\n{OUT_CSV}")

# %% cell 6 of 15
import pandas as pd
import os

FILE = (
    os.path.join(RAW_ROOT, "traffic/webtris_15min_all_sites_cleaned.csv")
)

CHUNK_SIZE = 250_000
TEMP_FILE = FILE + ".tmp"
first_chunk = True

print("Processing cleaned CSV...")

for chunk in pd.read_csv(FILE, chunksize=CHUNK_SIZE):
    # Drop rows with missing volume or speed
    chunk = chunk.dropna(subset=["Avg mph", "Total Volume"])

    # Rename and transform
    chunk = chunk.rename(columns={"Avg mph": "speed", "Total Volume": "flow"})
    chunk["speed"] = chunk["speed"] * 1.60934     # mph → kph
    chunk["flow"]  = chunk["flow"] * 4             # 15 min → per hour

    # Write to temp file
    chunk.to_csv(TEMP_FILE, mode="w" if first_chunk else "a",
                 index=False, header=first_chunk)
    first_chunk = False

# Overwrite the original file
os.replace(TEMP_FILE, FILE)

print(f"✅ Done. File overwritten with cleaned & converted data:\n{FILE}")


# %% cell 7 of 15
import pandas as pd
import os

FILE = (
    os.path.join(RAW_ROOT, "traffic/webtris_15min_all_sites_cleaned.csv")
)

CHUNK_SIZE = 250_000
TEMP_FILE = FILE + ".tmp"
first_chunk = True

print("Adding 'density' column...")

for chunk in pd.read_csv(FILE, chunksize=CHUNK_SIZE):
    if "flow" not in chunk.columns or "speed" not in chunk.columns:
        raise ValueError("Missing 'flow' or 'speed' column in input file.")

    chunk["density"] = chunk["flow"] / chunk["speed"]

    chunk.to_csv(TEMP_FILE, mode="w" if first_chunk else "a",
                 index=False, header=first_chunk)
    first_chunk = False

# Replace the original file with the updated one
os.replace(TEMP_FILE, FILE)

print(f"✅ Done. 'density' column added and file overwritten:\n{FILE}")

# %% cell 8 of 15
def get_site_coordinates(site_type: int = 1) -> pd.DataFrame:
    """
    Fetch all sites and return only those with usable lat/lon + ID.
    """
    url = f"{BASE}/sitetypes/{site_type}/sites"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.json_normalize(r.json()["sites"])

    # Keep only useful location columns
    keep_cols = [
        "Id", "Name", "Latitude", "Longitude",
        "Road", "LocationText", "Region"
    ]
    df = df[[col for col in keep_cols if col in df.columns]]
    df = df.dropna(subset=["Latitude", "Longitude"])  # only geolocated ones

    return df
coords_df = get_site_coordinates()
coords_df.to_csv(os.path.join(RAW_ROOT, "detector_tables/detectors_england.csv"), index=False)
print(f"Saved coordinates for {len(coords_df)} sites.")

# %% cell 9 of 15
import pandas as pd
import re

# Load your file
file_path = os.path.join(RAW_ROOT, "detector_tables/detectors_england.csv")
df = pd.read_csv(file_path)

# Clean the "Name" column
def extract_site_name(text: str) -> str:
    match = re.search(r"at\s+([A-Z0-9/]+)", text)
    return match.group(1) if match else text  # fallback: original text if no match

df["Name"] = df["Name"].astype(str).apply(extract_site_name)

# Save back (overwrite or new file)
df.to_csv(file_path, index=False)
print(f"Cleaned 'Name' column and saved to:\n{file_path}")


# %% cell 10 of 15
import pandas as pd
import os
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm

# Path to the CSV file
file_path = os.path.join(RAW_ROOT, "detector_tables/detectors_england.csv")
output_dir = os.path.dirname(file_path)

# Load the dataset
df = pd.read_csv(file_path)

# Initialize geolocator and rate limiter to avoid hitting API limits
geolocator = Nominatim(user_agent="detector_city_splitter")
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1, return_value_on_exception=None)

# Create tqdm progress bar
tqdm.pandas()

# Function to get city from coordinates
def get_city(row):
    try:
        location = reverse((row['Latitude'], row['Longitude']), language='en')
        if location and 'address' in location.raw:
            address = location.raw['address']
            return address.get('city') or address.get('town') or address.get('village') or address.get('hamlet')
    except:
        return None
    return None

# Apply reverse geocoding
print("Reverse geocoding coordinates to get city names. This may take a while...")
df['City'] = df.progress_apply(get_city, axis=1)

# Drop rows with no city info
df = df.dropna(subset=['City'])

# Group by city and save each group
for city, group in df.groupby('City'):
    safe_city = "".join(c for c in city if c.isalnum() or c in (' ', '_', '-')).rstrip()
    output_file = os.path.join(output_dir, f"detectors_{safe_city}.csv")
    group.to_csv(output_file, index=False)
    print(f"Saved {len(group)} rows to {output_file}")

# %% cell 11 of 15
import pandas as pd
import os

# Define the path to the original file
file_path = os.path.join(RAW_ROOT, "detector_tables/england_detectors_with_cities.csv")

# Read the CSV file
df = pd.read_csv(file_path)

# Ensure the "City" column exists
if "City" not in df.columns:
    raise ValueError("The CSV file does not contain a 'City' column.")

# Get the folder and base name of the original file
folder = os.path.dirname(file_path)
base_name = os.path.splitext(os.path.basename(file_path))[0]

# Group by the 'City' column
for city, group in df.groupby("City"):
    # Drop the 'City' column
    group = group.drop(columns=["City"])

    # Sanitize city name for filename
    city_sanitized = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in city).replace(' ', '_')

    # Construct new file name
    new_file_name = f"{base_name}_{city_sanitized}.csv"
    new_file_path = os.path.join(folder, new_file_name)

    # Save the group to a new CSV
    group.to_csv(new_file_path, index=False)

print("Files created successfully.")

# %% cell 12 of 15
import os
import pandas as pd
from datetime import datetime, time

# --- Configuration ---
detectors_folder = os.path.join(RAW_ROOT, "detector_tables")
traffic_file_path = os.path.join(RAW_ROOT, "traffic/England_15min.csv")
prefix = "england_detectors"
target_day = datetime(2025, 6, 11)
cutoff_time = datetime.combine(target_day, time(16, 0))

# --- Load traffic data once ---
traffic_df = pd.read_csv(traffic_file_path)

# --- Process each relevant detector file ---
for filename in os.listdir(detectors_folder):
    if filename.startswith(prefix) and filename.endswith(".csv"):
        file_path = os.path.join(detectors_folder, filename)
        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))

        # Check modification date/time
        if mod_time.date() == target_day.date() and mod_time.time() >= cutoff_time.time():
            try:
                # Load detector file
                detector_df = pd.read_csv(file_path)

                # Ensure the 'Name' column exists
                if "Name" not in detector_df.columns:
                    print(f"Skipping {filename}: 'Name' column not found.")
                    continue

                # Extract list of detector names
                detector_names = detector_df["Name"].dropna().unique()

                # Filter traffic data by 'Site Name'
                filtered_traffic = traffic_df[traffic_df["Site Name"].isin(detector_names)]

                # Extract city name from filename
                city_part = filename.replace(prefix, "").replace(".csv", "").strip("_")

                # Construct new filename
                new_traffic_file = os.path.join(detectors_folder, f"England_15min_{city_part}.csv")

                # Save the filtered data
                filtered_traffic.to_csv(new_traffic_file, index=False)

                print(f"Saved traffic data for city '{city_part}' to {new_traffic_file}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")


# %% cell 13 of 15
import os
import pandas as pd
from datetime import datetime, time
import shutil

# --- Config ---
base_folder = os.path.join(RAW_ROOT, "detector_tables")
detectors_prefix = "england_detectors"
data_prefix = "England_15min"
city_marker = "cities_"

save_if_valid = os.path.join(RAW_ROOT, "detector_tables/Engl_Detectors")
data_if_valid = os.path.join(RAW_ROOT, "traffic/Engl_cities")
discard_path = os.path.join(RAW_ROOT, "traffic/Discarded Data")

cutoff_datetime = datetime(2025, 6, 11, 16, 0)

# --- Create output folders if they don't exist ---
for path in [save_if_valid, data_if_valid, discard_path]:
    os.makedirs(path, exist_ok=True)

# --- Find relevant files ---
detector_files = {}
data_files = {}

for filename in os.listdir(base_folder):
    if not filename.endswith(".csv"):
        continue

    full_path = os.path.join(base_folder, filename)
    mod_time = datetime.fromtimestamp(os.path.getmtime(full_path))
    if mod_time <= cutoff_datetime:
        continue

    if city_marker not in filename:
        continue

    try:
        city = filename.split(city_marker)[-1].replace(".csv", "").strip()
        if filename.startswith(detectors_prefix):
            detector_files[city] = full_path
        elif filename.startswith(data_prefix):
            data_files[city] = full_path
    except Exception as e:
        print(f"Failed to extract city from filename: {filename}, error: {e}")

# --- Process city files ---
total_before = 0
total_after = 0

for city in sorted(set(detector_files.keys()) & set(data_files.keys())):
    det_path = detector_files[city]
    dat_path = data_files[city]

    try:
        det_df = pd.read_csv(det_path)
        dat_df = pd.read_csv(dat_path)

        # Check column existence
        if "Name" not in det_df.columns or "Site Name" not in dat_df.columns:
            print(f"Skipping {city}: missing required columns.")
            continue

        before_count = len(det_df)
        total_before += before_count

        # Filter detectors
        valid_names = set(dat_df["Site Name"].dropna().unique())
        det_df = det_df[det_df["Name"].isin(valid_names)]

        after_count = len(det_df)
        total_after += after_count

        # Save to appropriate location
        if after_count >= 20:
            det_save_path = os.path.join(save_if_valid, f"detectors_england_cities_{city}.csv")
            dat_save_path = os.path.join(data_if_valid, f"England_15min_cities_{city}.csv")
        else:
            det_save_path = os.path.join(discard_path, f"detectors_england_cities_{city}.csv")
            dat_save_path = os.path.join(discard_path, f"England_15min_cities_{city}.csv")

        # Save the filtered detector and original data
        det_df.to_csv(det_save_path, index=False)
        shutil.copy2(dat_path, dat_save_path)

        print(f"Processed city '{city}': {before_count} → {after_count} detectors")
    except Exception as e:
        print(f"Error processing city '{city}': {e}")

# --- Final output ---
print(f"\nTotal detectors before filtering: {total_before}")
print(f"Total detectors after filtering: {total_after}")

# %% cell 14 of 15
import os

# Path to your target directory
directory = os.path.join(RAW_ROOT, "curves_as_reported")

# Prefix to look for and the part to remove
old_prefix = "data_England_15min_cities_"
new_prefix = "data_"

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv") and filename.startswith(old_prefix):
        new_filename = new_prefix + filename[len(old_prefix):]
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")

# %% cell 15 of 15
import os
import re

# Define the target directory
folder_path = os.path.join(RAW_ROOT, "detector_lists")

# List of cities to append '2' after
cities = ["London", "Manchester", "Birmingham"]

# Loop through all files in the directory
for filename in os.listdir(folder_path):
    if "england_cities_" in filename:
        old_path = os.path.join(folder_path, filename)

        # Remove "england_cities_"
        new_filename = filename.replace("england_cities_", "")

        # Append "2" after matching city names (only if "City2" not already present)
        for city in cities:
            if city in new_filename and f"{city}2" not in new_filename:
                new_filename = new_filename.replace(city, f"{city}2")

        new_path = os.path.join(folder_path, new_filename)

        # Avoid overwriting an existing file
        if os.path.exists(new_path):
            print(f"Skipped (target exists): {new_filename}")
            continue

        # Rename the file
        os.rename(old_path, new_path)
        print(f'Renamed: {filename} -> {new_filename}')

