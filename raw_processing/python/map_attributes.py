"""Map and covariate attributes per network

City-level attributes and WorldPop sums per detector hull; the WorldPop download; links, nodes and hulls from OpenStreetMap with OSMnx, with GeoQuery covariates; the k-means clusters of Seattle's detectors.

Run by the authors as cells of a notebook, in this order; each "cell" below is one of those cells. It needs the original downloads, so it was not rerun for this package. The only
changes are the locations, which now start from ROAD_NETWORK_RAW (default: raw/, from the repository's root).
"""
import os

RAW_ROOT = os.environ.get("ROAD_NETWORK_RAW", "raw")


# %% cell 1 of 4
# -*- coding: utf-8 -*-
"""
hull_area_sociecon_estimator.py
-------------------------------------------------
✓ Area (km²)
✓ GDP (Million US$)            – still area‑weighted from polygon attributes
✓ 4 city‑level indicators      – from NetworkProps.csv
✓ 36 gender/age raster sums    – F0…F85, M0…M85
(no more aggregate “Population” column)

Run with::

    python hull_area_sociecon_estimator.py
"""
from __future__ import annotations
import os, re, glob, pandas as pd, geopandas as gpd, rasterio
from rasterstats import zonal_stats
from shapely.geometry import box
from tqdm import tqdm
from pandas.api.types import is_numeric_dtype
import re

def normalize_name(name: str) -> str:
    """
    Canonical city key:
      • lowercase
      • trim leading/trailing whitespace
      • delete spaces, underscores and hyphens
      (digits and everything else stay as-is)
    """
    return re.sub(r"[\s_-]+", "", str(name).lower().strip())

# ───────────────────── PATHS (edit if needed) ────────────────────────────────
ROOT_DETECTORS = os.path.join(RAW_ROOT, "detector_lists")
GEO_DATA_DIR   = os.path.join(ROOT_DETECTORS, "Geographical Data")
OUTPUT_CSV     = os.path.join(ROOT_DETECTORS, "Network_Properties.csv")
CITY_PROPS_CSV = os.path.join(GEO_DATA_DIR, "NetworkProps.csv")
# ──────────────────────────────────────────────────────────────────────────────

COUNTRIES  = [
    "Austria","Canada","France","Germany","Italy","Lithuania",
    "Netherlands","Spain","Switzerland","Taiwan",
    "United Kingdom","United States"
]
TARGET_CRS = 6933                      # Equal‑area CRS for all vector work
GDP_COL    = "gdp_grid.none.mean"

CITY_KEY  = "City"
CITY_COLS = [
    "City Area (km^2)","Income (USD/yr)",
    "City Population","Employment Rate (%)"
]

# ─────────────────────────── CITY INDICATORS ────────────────────────────────

def load_city_props(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)
    for sep in (None, ",", ";"):       # robust delimiter sniff
        try:
            df = pd.read_csv(csv_path, sep=sep, engine="python", encoding="utf-8-sig")
            break
        except Exception:
            continue
    df.columns = df.columns.str.replace("\ufeff","",regex=False).str.strip()

    lookup = {c.lower(): c for c in df.columns}
    missing = [c for c in CITY_COLS + [CITY_KEY] if c.lower() not in lookup]
    if missing:
        raise KeyError("City CSV missing: " + ", ".join(missing))

    # Rename to canonical CITY_COLS / CITY_KEY
    df = df.rename(columns={lookup[k.lower()]: k for k in CITY_COLS + [CITY_KEY]})

    # Build a *normalized* index so different spellings still match
    df["__norm_key__"] = df[CITY_KEY].apply(normalize_name)
    return df.set_index("__norm_key__")

CITY_PROPS = load_city_props(CITY_PROPS_CSV)

# ───────────────────────── COUNTRY POLYGONS ──────────────────────────
GDP_CANDIDATES = [
    "gdp_grid.none.mean",   # ⇦ what the script expected
    "gdp_grid.ppp.mean",    # ⇦ e.g. Canada, EU layers
    "gdp_grid.usd.mean",    # ⇦ occasionally used
]

def load_country_layer(folder: str) -> gpd.GeoDataFrame:
    """
    Read <country>/<something>.geojson + matching CSV and return a layer
    with geometry, polygon area (m²) and a unified 'GDP' column.

    If the GDP field is missing we keep the polygons but set GDP = 0 and
    print a warning instead of raising an error, so a multi‑country hull
    never loses *all* its data because one country’s attribute table is
    incomplete.
    """
    gj_files  = glob.glob(os.path.join(folder, "*.geojson"))
    if not gj_files:
        raise FileNotFoundError(f"No GeoJSON in {folder}")
    gdf = gpd.read_file(gj_files[0]).set_crs("EPSG:4326")

    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV in {folder}")
    attrs = pd.read_csv(csv_files[0])

    # join on whichever ID column both files share
    key = next(k for k in ("shapeID", "asdf_id", "gqid") if k in gdf.columns and k in attrs.columns)
    gdf = gdf.merge(attrs, on=key, how="left")

    # normalise the GDP column name
    gdp_col = next((c for c in GDP_CANDIDATES if c in gdf.columns), None)
    if gdp_col:
        gdf = gdf.rename(columns={gdp_col: "GDP"})
    else:
        print(f"⚠ {os.path.basename(folder)}: GDP column missing – using 0")
        gdf["GDP"] = 0.0

    gdf = gdf.to_crs(TARGET_CRS)
    gdf["poly_area_m2"] = gdf.geometry.area
    return gdf[["geometry", "poly_area_m2", "GDP"]]

country_layers = {}
for c in COUNTRIES:
    try:
        country_layers[c] = load_country_layer(os.path.join(GEO_DATA_DIR, c))
    except Exception as e:
        print(f"⚠ {c}: {e} – skipped")

# ─────────────────────── RASTER CATALOG (36 tif / country) ───────────────────

def parse_rasters(folder: str) -> dict[str,str]:
    """Return {F0: path, M5: path, …} for every *.tif* in *folder*."""
    out = {}
    for tif in glob.glob(os.path.join(folder,"*.tif")):
        # ex:  twn_f_0_2020_constrained_UNadj.tif
        parts = os.path.basename(tif).split("_")
        if len(parts) < 3:
            continue
        g, age = parts[1].lower(), parts[2]
        if g in ("f","m") and age.isdigit():
            out[f"{g.upper()}{age}"] = tif
    return out

country_rasters = {c: parse_rasters(os.path.join(GEO_DATA_DIR, c)) for c in COUNTRIES}
demo_cols = sorted({k for d in country_rasters.values() for k in d},
                   key=lambda x: (x[0], int(x[1:])))

# ───────────────────────── PROCESS DETECTOR HULLS ────────────────────────────
hulls = sorted(
    f for f in glob.glob(os.path.join(ROOT_DETECTORS,"*.geojson"))
    if os.path.basename(f).lower().startswith("hull_")
)
if not hulls:
    raise SystemExit("❌ No hull_*.geojson found")

records = []
for hull_path in tqdm(hulls, unit="hull"):
    hull_name  = os.path.splitext(os.path.basename(hull_path))[0]
    basename   = re.sub(r"^hull[_\-]", "", hull_name, flags=re.I)  # keep case
    city_key   = normalize_name(basename)                              # ← canonical key

    # ---- lookup city‑level indicators ----------------------------------
    if city_key in CITY_PROPS.index:
        city_vals = CITY_PROPS.loc[city_key][CITY_COLS]
    else:
        city_vals = pd.Series([pd.NA]*len(CITY_COLS), index=CITY_COLS)

    # ---- geometry & area ------------------------------------------------
    hull_geom       = gpd.read_file(hull_path).to_crs(TARGET_CRS).unary_union
    hull_area_km2   = hull_geom.area / 1_000_000
    gdp_sum         = 0.0
    demo_sum        = {col: 0.0 for col in demo_cols}
    polygon_count   = 0
    hbbox           = box(*hull_geom.bounds)

    # ---- per‑country overlays & rasters ---------------------------------
    for c in country_layers:
        poly_gdf = country_layers[c]
        if not hbbox.intersects(box(*poly_gdf.total_bounds)):
            continue  # hull doesn't touch country

        # node‑level attribute means -------------------------------------
        nodes_file  = os.path.join(ROOT_DETECTORS, f"nodes_{basename}.csv")
        node_means  = {}
        if os.path.isfile(nodes_file):
            nodes_df  = pd.read_csv(nodes_file)
            skip_cols = {"NodeID", "X", "Y", "Type"}
            for col in nodes_df.columns:
                if col in skip_cols:
                    continue
                if is_numeric_dtype(nodes_df[col]):
                    node_means[col] = round(nodes_df[col].mean(skipna=True), 3)
        else:
            print(f"⚠ nodes file missing for {basename} – no node averages")

        # GDP from vector polygons ---------------------------------------
        inter = gpd.overlay(
            poly_gdf,
            gpd.GeoDataFrame(geometry=[hull_geom], crs=TARGET_CRS),
            how="intersection",
        )
        polygon_count += len(inter)
        if not inter.empty:
            weight  = inter.geometry.area / inter["poly_area_m2"]
            gdp_sum += (inter["GDP"] * weight).sum()

        # demographic rasters --------------------------------------------
        for col, tif in country_rasters[c].items():
            with rasterio.open(tif) as src:
                hull_in_raster_crs = (
                    gpd.GeoSeries([hull_geom], crs=TARGET_CRS)
                    .to_crs(src.crs)
                    .iloc[0]
                )
                zs = zonal_stats([hull_in_raster_crs], tif, stats="sum", nodata=src.nodata)
                val = zs[0]["sum"]
                if val is not None:
                    demo_sum[col] += val

    city_clean = basename.title()

    records.append({
        "City":              city_clean,
        "nb_of_polygons":    polygon_count,
        "Area (km^2)":       round(hull_area_km2, 3),
        "GDP (Million US$)": round(gdp_sum, 3),
        **city_vals.to_dict(),
        **{c: round(demo_sum[c], 3) for c in demo_cols},
        **node_means,
    })

# ──────────────────────────── WRITE CSV ──────────────────────────────────────
print(f"Writing → {OUTPUT_CSV}")
pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)
print("Done ✔")

# %% cell 2 of 4
# code to download all tif files for the country whose url is provided
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Target webpage
url = "https://hub.worldpop.org/geodata/summary?id=50341"

# Local download directory
download_dir = os.path.join(RAW_ROOT, "detector_lists/Geographical Data")

# Make sure the directory exists
os.makedirs(download_dir, exist_ok=True)

# Get the page content
response = requests.get(url)
if response.status_code != 200:
    raise Exception(f"Failed to access {url}: status code {response.status_code}")

# Parse the page with BeautifulSoup
soup = BeautifulSoup(response.content, "html.parser")

# Find all links
links = soup.find_all("a", href=True)

# Download each .tif file
for link in links:
    href = link["href"]
    if href.endswith(".tif") or ".tif?" in href:
        file_url = urljoin(url, href)
        filename = os.path.basename(file_url.split("?")[0])
        file_path = os.path.join(download_dir, filename)
        
        print(f"Downloading {file_url} to {file_path}...")
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

print("Download complete.")

# %% cell 3 of 4
# code to create files for links, nodes, hull, etc. of networks given their detectors files
import os
import glob
import re
import numbers
import numpy as np
import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import MultiPoint
from shapely.ops import nearest_points, unary_union
from functools import lru_cache
from shapely.strtree import STRtree
from shapely.geometry.base import BaseGeometry

# ----------------------------------------------------------------------
# 0. Small helpers -----------------------------------------------------
# ----------------------------------------------------------------------
@lru_cache(maxsize=None)
def _load_country_geodata(country: str, base_geo_dir: str):
    """
    Read and tidy the single *.geojson and *.csv for *country* exactly once.
    Subsequent calls with the same (country, base_geo_dir) tuple come straight
    from the LRU cache, so there is **no further disk I/O**.
    """
    c_dir = os.path.join(base_geo_dir, country)

    poly_path = glob.glob(os.path.join(c_dir, "*.geojson"))[0]
    csv_path  = glob.glob(os.path.join(c_dir, "*.csv"))[0]

    poly = gpd.read_file(poly_path).set_crs("EPSG:4326")
    if not poly.geometry.is_valid.all():
        poly = poly.copy()
        poly["geometry"] = poly.buffer(0)

    csv  = _tidy_geo_columns(pd.read_csv(csv_path))   # tidy once, here
    return poly, csv
def parse_lanes(val):
    """Try to coerce OSM 'lanes' tag to an int, default 1."""
    try:
        if pd.isnull(val):
            return 1
        if isinstance(val, list) and val:
            return int(val[0])
        return int(val)
    except Exception:
        return 1

def parse_maxspeed(ms):
    """Convert OSM `maxspeed` tag formats → km/h (float) or None."""
    if ms is None:
        return None
    if isinstance(ms, numbers.Number) and np.isnan(ms):
        return None

    if isinstance(ms, (list, tuple, np.ndarray)):
        for item in ms:
            val = parse_maxspeed(item)
            if val is not None:
                return val
        return None

    s = str(ms)
    m = re.search(r"\d+(?:\.\d+)?", s)
    if not m:
        return None

    value = float(m.group())
    if "mph" in s.lower():
        value *= 1.60934
    return round(value, 2)

# helper: unwrap single-value lists that sometimes appear in OSM tags
def _first(val):
    return val[0] if isinstance(val, (list, tuple, np.ndarray)) and val else val


# ----------------------------------------------------------------------
# 1. Build nodes / links  (robust tag handling) ------------------------
# ----------------------------------------------------------------------
def build_topology(G_proj):
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(
        G_proj, nodes=True, edges=True, fill_edge_geometry=True
    )

    # ---------- nodes ------------------------------------------------
    nodes_df = (
        nodes_gdf.reset_index()
                 .rename(columns={"osmid": "NodeID", "x": "X", "y": "Y"})
                 [["NodeID", "X", "Y"]]
    )

    inflow_counts  = edges_gdf.groupby("v").size()
    outflow_counts = edges_gdf.groupby("u").size()

    def classify_node(n):
        i = inflow_counts.get(n, 0)
        o = outflow_counts.get(n, 0)
        if i == 1 and o > 1:
            return "Diverge"
        elif i > 1 and o == 1:
            return "Merge"
        elif i == 1 and o == 1:
            return "Single"
        elif o == 0:
            return "End"
        elif i == 0:
            return "Start"
        else:
            return "Multi"

    nodes_df["Type"] = nodes_df["NodeID"].apply(classify_node)

    # ---- extra node tags (safe lookup) -----------------------------
    def safe_series(tag):
        """Return Series if tag exists else NA-filled Series matching nodes_df."""
        if tag in nodes_gdf:
            return nodes_gdf[tag].apply(_first)
        return pd.Series(pd.NA, index=nodes_df.index)

    nodes_df["TrafficCtrl"]     = safe_series("highway").where(
                                      safe_series("highway")
                                      .isin(["traffic_signals", "crossing", "stop"])
                                 )
    nodes_df["Elevation"]       = safe_series("ele")
    nodes_df["Barrier"]         = safe_series("barrier")
    nodes_df["Amenity"]         = safe_series("amenity")
    nodes_df["PublicTransport"] = safe_series("public_transport")
    nodes_df["Railway"]         = safe_series("railway")

    # ---------- links ------------------------------------------------
    edges_reset = edges_gdf.reset_index()

    def safe_edge(tag):
        return edges_reset[tag].apply(_first) if tag in edges_reset \
               else pd.Series(pd.NA, index=edges_reset.index)

    edges_reset["Class"]       = safe_edge("highway")
    edges_reset["Lanes"]       = edges_reset["lanes"].apply(parse_lanes) \
                                 if "lanes" in edges_reset else 1
    edges_reset["LinkLength"]  = edges_reset.geometry.length.round(1)
    edges_reset["Curvature"] = edges_reset.geometry.apply(_curvature)
    edges_reset["SpeedLimit"]  = edges_reset["maxspeed"].apply(parse_maxspeed) \
                                 if "maxspeed" in edges_reset else pd.NA

    # new raw tags ----------------------------------------------------
    edges_reset["Oneway"]     = safe_edge("oneway")
    edges_reset["Surface"]    = safe_edge("surface")
    edges_reset["Bridge"]     = safe_edge("bridge")
    edges_reset["Tunnel"]     = safe_edge("tunnel")
    edges_reset["Access"]     = safe_edge("access")
    edges_reset["Vehicle"]    = safe_edge("vehicle")
    edges_reset["Cycleway"]   = safe_edge("cycleway")
    edges_reset["Sidewalk"]   = safe_edge("sidewalk")
    edges_reset["Busway"]     = safe_edge("busway")
    edges_reset["MaxHeight"]  = safe_edge("maxheight")
    edges_reset["MaxWeight"]  = safe_edge("maxweight")
    edges_reset["MaxWidth"]   = safe_edge("maxwidth")
    edges_reset["Roundabout"] = safe_edge("junction") == "roundabout"
    edges_reset["Lit"]        = safe_edge("lit")

    links_df = (
        edges_reset
        .rename(columns={"u": "StartNode", "v": "EndNode"})
        .assign(LinkID=lambda df: df.index)
        [[
          "LinkID", "StartNode", "EndNode", "Class",
          "Lanes", "LinkLength", "Curvature", "SpeedLimit",
          "Oneway", "Surface", "Bridge", "Tunnel",
          "Access", "Vehicle", "Cycleway", "Sidewalk", "Busway",
          "MaxHeight", "MaxWeight", "MaxWidth",
          "Roundabout", "Lit"
        ]]
    )

    # -- remove columns that are completely NA -----------------------
    nodes_df = nodes_df.dropna(axis=1, how="all")
    links_df = links_df.dropna(axis=1, how="all")
    # 1. send the node geometries back to WGS-84
    nodes_latlon = nodes_gdf.to_crs("EPSG:4326")

    # 2. replace X,Y (in metres) with Longitude, Latitude (in deg)
    # Get WGS84 graph from projected graph
    nodes_wgs = ox.graph_to_gdfs(ox.project_graph(G_proj, to_crs="EPSG:4326"),
                                 nodes=True, edges=False)

    # Merge lat/lon directly from OSM geometries
    lonlat = nodes_wgs.reset_index()[["osmid", "geometry"]]
    lonlat["Longitude"] = lonlat.geometry.x.round(7)
    lonlat["Latitude"]  = lonlat.geometry.y.round(7)

    nodes_df = nodes_df.drop(columns=["X", "Y"])
    nodes_df = pd.merge(nodes_df, lonlat[["osmid", "Longitude", "Latitude"]],
                        left_on="NodeID", right_on="osmid", how="left").drop(columns="osmid")

    
    nodes_df = nodes_df.loc[:, ~nodes_df.columns.duplicated()]

    return nodes_df, links_df

# ----------------------------------------------------------------------
# 2. Visualisation helper ----------------------------------------------
# ----------------------------------------------------------------------
def plot_network_area(edges_proj, gdf_det, hull, title, line_color="black"):
    """Plot road links, snapped detectors, and the final hull boundary."""
    crs_proj = edges_proj.crs

    fig, ax = plt.subplots(figsize=(8, 8))
    edges_proj.plot(ax=ax, linewidth=1, color="grey", label="roads")
    gdf_det.to_crs(crs_proj).plot(ax=ax, markersize=25, color="red",
                                  zorder=3, label="detectors")
    gpd.GeoSeries([hull], crs="EPSG:4326").to_crs(crs_proj).boundary.plot(
        ax=ax, color=line_color, linewidth=2, zorder=2, label="hull"
    )

    ax.set_axis_off(); ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(); plt.tight_layout(); plt.show()

# ----------------------------------------------------------------------
# 3. Build hull & snap detectors ---------------------------------------
# ----------------------------------------------------------------------
def detectors_to_polygon(detector_path,
                         concave=False, alpha=None,
                         buffer_m=200,
                         chunk_size=50_000):
    """Return buffered hull & snapped detectors for any CSV/XLSX detector file."""
    
    net_pad_m = buffer_m

    # 1. read detectors ----------------------------------------------
    ext = os.path.splitext(detector_path)[1].lower()
    if ext == ".csv":
        df_det = pd.read_csv(detector_path)
    else:
        df_det = pd.read_excel(detector_path, usecols=["Latitude", "Longitude"])

    df_det = (df_det
              .dropna(subset=["Latitude", "Longitude"])
              .rename(columns={"Longitude": "lon", "Latitude": "lat"}))

    gdf_det = gpd.GeoDataFrame(df_det,
                               geometry=gpd.points_from_xy(df_det.lon, df_det.lat),
                               crs="EPSG:4326")

    # 2. project to local CRS ----------------------------------------
    gdf_det_proj = ox.projection.project_gdf(gdf_det)
    crs_proj     = gdf_det_proj.crs

    # 3. TEMP snap polygon -------------------------------------------
    hull_proj_tmp  = MultiPoint(gdf_det_proj.geometry).convex_hull.buffer(net_pad_m)
    hull_tmp_wgs84 = gpd.GeoSeries([hull_proj_tmp], crs=crs_proj)\
                         .to_crs("EPSG:4326").iloc[0]

    G_tmp = ox.graph_from_polygon(hull_tmp_wgs84, network_type="all", truncate_by_edge=True, retain_all=True)
    G_tmp_proj  = ox.project_graph(G_tmp, to_crs=crs_proj)
    edges_proj  = ox.graph_to_gdfs(G_tmp_proj, nodes=False, edges=True,
                                   fill_edge_geometry=True)
    edges_proj  = edges_proj.explode(index_parts=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. snap  (tolerant, non-destructive, STRtree-accelerated)
    # ------------------------------------------------------------------
    MAX_SNAP_DIST = 40          # metres – tweak if needed

    # Build the spatial index **once**
    edge_geoms = [
        geom for geom in edges_proj.geometry
        if geom.is_valid and not geom.is_empty
    ]
    edge_tree  = STRtree(edge_geoms)       # ~O(M log M) build, then fast queries

    def snap(pt):
        """
        Return the nearest point on the nearest edge **iff** it lies
        within MAX_SNAP_DIST; otherwise leave detector where it is.
        """
        res = edge_tree.nearest(pt)               # may be GEOM (Shapely ≥ 2) *or* INT (PyGEOS)
        nearest_edge = (
            edge_geoms[res]                       # PyGEOS → int index
            if isinstance(res, (int, np.integer)) # covers plain int + numpy scalars
            else res                              # Shapely → geometry already
        )

        if nearest_edge is None:                  # ultra-rare, but safe
            return pt

        dist = pt.distance(nearest_edge)
        if dist > MAX_SNAP_DIST:
            return pt

        return nearest_points(pt, nearest_edge)[1]

    # Vectorised apply – still runs per-point, but the lookup is ~100× faster
    snapped_geom = gdf_det_proj.geometry.apply(snap)


    # 5. hull ---------------------------------------------------------
    snapped_pts = list(snapped_geom)
    if chunk_size is None or chunk_size <= 0 or chunk_size >= len(snapped_pts):
        # one-shot
        if concave:
            import alphashape
            a = alpha if alpha is not None \
                    else alphashape.optimizealpha(snapped_pts)
            combined_hull = alphashape.alphashape(snapped_pts, a)
        else:
            combined_hull = MultiPoint(snapped_pts).convex_hull
    else:
        combined_hull = None
        for idx in range(0, len(snapped_pts), chunk_size):
            chunk = MultiPoint(snapped_pts[idx: idx + chunk_size])
            if concave:
                import alphashape
                ca = alpha if alpha is not None else alphashape.optimizealpha(chunk)
                chunk_h = alphashape.alphashape(chunk, ca)
            else:
                chunk_h = chunk.convex_hull
            combined_hull = chunk_h if combined_hull is None \
                                     else combined_hull.union(chunk_h)

    # 6. buffer & back-project ---------------------------------------
    hull_proj    = combined_hull.buffer(buffer_m)
    hull_wgs84   = gpd.GeoSeries([hull_proj], crs=crs_proj)\
                        .to_crs("EPSG:4326").iloc[0]
    gdf_det_out = gdf_det_proj.copy()
    gdf_det_out["geometry"] = snapped_geom # apply snaps only now
    gdf_det_out = gdf_det_out.to_crs("EPSG:4326")
    gdf_det_out["lon"] = gdf_det_out.geometry.x
    gdf_det_out["lat"] = gdf_det_out.geometry.y

    return hull_wgs84, gdf_det_out

# ----------------------------------------------------------------------
# 4. File writers ------------------------------------------------------
# ----------------------------------------------------------------------
def save_hull(city, hull_polygon, directory):
    out_path = os.path.join(directory, f"hull_{city}.geojson")
    gpd.GeoSeries([hull_polygon], crs="EPSG:4326")\
        .to_file(out_path, driver="GeoJSON")
    print(f"  → hull saved to {out_path}")


def save_snapped_points(city, gdf_det, directory):
    out_path = os.path.join(directory, f"snapped_{city}.csv")
    gdf_det[["lat", "lon"]].to_csv(out_path, index=False)
    print(f"  → snapped points saved to {out_path}")

def save_topology(city, nodes_df, links_df, directory):
    nodes_path = os.path.join(directory, f"nodes_{city}.csv")
    links_path = os.path.join(directory, f"links_{city}.csv")
    nodes_df.to_csv(nodes_path, index=False)
    links_df.to_csv(links_path, index=False)
    print(f"  → nodes saved to {nodes_path}")
    print(f"  → links saved to {links_path}")

# ----------------------------------------------------------------------
# helper: do all expected output files already exist?
# ----------------------------------------------------------------------
def outputs_complete(city, directory) -> bool:
    """Return True if every artefact for *city* is already on disk."""
    expected = [
        f"nodes_{city}.csv",
        f"links_{city}.csv",
        f"hull_{city}.geojson",
        f"snapped_{city}.csv",
        f"missing_attrs_{city}.csv",
    ]
    return all(os.path.isfile(os.path.join(directory, f)) for f in expected)


# ----------------------------------------------------------------------
# 5. Batch processing --------------------------------------------------
# ----------------------------------------------------------------------
def process_all_detector_files(directory,
                               buffer_m=800,
                               concave=False,
                               chunk_size=50_000):
    """Loop over every detectors_*.csv in *directory* and persist artefacts."""
    pattern = os.path.join(directoryrectory, "detectors_*.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"No detector files found in {directory} matching 'detectors_*.csv'.")
        return

    for fpath in files:
        city = os.path.splitext(os.path.basename(fpath))[0].split("detectors_")[-1]

        # ✱✱ NEW: skip if everything is already there -----------------
        if outputs_complete(city, directory):
            print(f"✓ All outputs present for {city} – skipping.")
            continue

        print(f"\nProcessing {city} …")

        # 1. hull & snapped dets -------------------------------------
        poly, gdf_det = detectors_to_polygon(
            fpath, concave=concave, buffer_m=buffer_m,
            chunk_size=chunk_size
        )

        save_hull(city, poly, directory)
        save_snapped_points(city, gdf_det, directory)

        # 2. network -------------------------------------------------
        G = ox.graph_from_polygon(poly, network_type="all", truncate_by_edge=True, retain_all=True)
        G_proj   = ox.project_graph(G)                                 # local CRS

        nodes_df, links_df = build_topology(G_proj)

        # >>> NEW: flag links that contain (snapped) detectors <<<
        # ------------------------------------------------------------
        edges_full = ox.graph_to_gdfs(
            G_proj, nodes=False, edges=True, fill_edge_geometry=True
        ).reset_index(drop=True)

        # NEW: initialize list columns for each edge
        edges_full["detector_relpos_list"] = [[] for _ in range(len(edges_full))]

        # project snapped detector points into the same CRS
        det_proj = gdf_det.to_crs(edges_full.crs)

        for pt in det_proj.geometry:
            idx  = edges_full.distance(pt).idxmin()        # nearest link row
            line = edges_full.loc[idx, "geometry"]
            rel  = line.project(pt) / line.length if line.length else -1
            if rel >= 0:
                edges_full.at[idx, "detector_relpos_list"].append(rel)
        
        edges_full["num_detectors"] = edges_full["detector_relpos_list"].apply(len)
        edges_full["avg_detector_pos"] = edges_full["detector_relpos_list"].apply(
            lambda lst: round(sum(lst)/len(lst), 6) if lst else -1.0
        )

        links_df = links_df.join(
            edges_full[["num_detectors", "avg_detector_pos"]]
        )


        # --- NEW: add regional polygon attributes -------------------
        geo_base_dir = os.path.join(directory, "Geographical Data")
        nodes_df     = enrich_nodes_with_geodata(nodes_df, G, poly, geo_base_dir)

        # ---- drop asdf_id column (kept only for counting later) -----
        if "asdf_id" in nodes_df.columns:
            nodes_df = nodes_df.drop(columns="asdf_id")

        # 3. persist outputs -----------------------------------------
        save_topology(city, nodes_df, links_df, directory)
        save_missing_attrs(city, nodes_df, links_df, directory)

        # 3. quick plot ----------------------------------------------
        edges_proj = ox.graph_to_gdfs(
            G_proj, nodes=False, edges=True, fill_edge_geometry=True
        )
        edges_proj = edges_proj.explode(index_parts=False).reset_index(drop=True)

        plot_title = f"{city.title()} – drivable network within hull"
        plot_network_area(edges_proj, gdf_det, poly, title=plot_title)
        
# ----------------------------------------------------------------------
# ++ NEW helpers -------------------------------------------------------
# ----------------------------------------------------------------------
COUNTRY_FOLDERS = [
    "United States", "United Kingdom", "Germany", "Lithuania", "Taiwan",
    "Netherlands", "Spain", "Switzerland", "France", "Canada",
    "Italy", "Austria"
]

# ------------------------------------------------------------------
# helper: tidy + rename geo-attribute columns ----------------------
# ------------------------------------------------------------------
def _tidy_geo_columns(df):
    """
    • Drop unneeded metadata columns
    • Rename / collapse columns following the rules given by the user
    • Keep all other columns unchanged
    """

    # ----- 1. drop metadata we never need ---------------------------
    meta_drop = {
        "Level", "Shape_Area", "Shape_Leng", "gqid", "id",
        "shapeGroup", "shapeID", "shapeName", "shapeType"
    }
    df = df.drop(columns=[c for c in meta_drop if c in df], errors="ignore")

    # ----- 2. rename pattern-based columns --------------------------
    rename_map   = {}
    drop_these   = []              # keep track of duplicates / _count col
    singleton_ok = set()           # helper to avoid double columns

    for col in list(df.columns):

        # distance / elevation / slope --------------------------------
        if col.startswith("distance_to_coast"):
            new = "distance_to_coast"
        elif col.startswith("dist_to_water"):
            new = "dist_to_water"
        elif col.startswith("srtm_slope"):
            new = "slope"
        elif col.startswith("srtm_elevation"):
            new = "elevation"

        # vegetation / wind -------------------------------------------
        elif col.startswith("ltdr"):
            new = "difference_vegetation_index"
        elif col.startswith("globalwindatlas"):
            new = "wind_speed_potential"

        # MODIS LST (24 cols) -----------------------------------------
        elif col.startswith("modis_lst"):
            m = re.search(r"(\d{6})", col)       # YYYYMM
            month = m.group(1)[4:] if m else "??"
            daynight = "day" if "_day_" in col else "night"
            new = f"{month}_{daynight}_st"

        # CRU precipitation (12 cols) ---------------------------------
        elif col.startswith("cru_"):
            m = re.search(r"(\d{6})", col)
            month = m.group(1)[4:] if m else "??"
            new = f"{month}_precip"

        # MODIS land-cover (20 cols; drop *_count) --------------------
        elif col.startswith("modis_landcover"):
            if col.endswith("_count"):
                drop_these.append(col)           # discard the *_count column
                continue
            idx = col.find("categorical_")
            new = col[idx + len("categorical_") :] if idx != -1 else col

        elif col.startswith("gdp") or col.startswith("gpw"):
            drop_these.append(col)
            continue
        else:
            continue                            # leave other columns unchanged

        # -- handle duplicates (e.g., multiple distance_to_coast cols)
        if new in singleton_ok:
            drop_these.append(col)              # we already kept one
        else:
            rename_map[col] = new
            singleton_ok.add(new)

    # apply drops + renames
    df = df.drop(columns=drop_these, errors="ignore").rename(columns=rename_map)

    # ----- 3. collapse any remaining duplicate names (e.g., two “03_night_st”
    #          columns from different years → keep first non-NA) ------
    df = (
        df.T.groupby(level=0)
             .first()          # first non-NA per duplicate group
             .T
    )
    return df

def enrich_nodes_with_geodata(nodes_df,
                              G_wgs84,
                              hull_wgs84,
                              base_geo_dir):
    """
    Add every attribute available in the country-level CSV to *nodes_df*
    by (a) figuring out which of the 12 countries the hull sits in and
    (b) spatial-joining nodes to the polygons identified by 'asdf_id'.
    """
    # ------------------------------------------------------------------
    # 1. Detect intersecting countries *and* collect their data --------
    # ------------------------------------------------------------------
    hull_geom   = hull_wgs84                    # already EPSG:4326
    countries   = []
    poly_parts  = []
    csv_list    = []                            # ⬅️  now built here

    for c in COUNTRY_FOLDERS:
        try:
            poly, csv = _load_country_geodata(c, base_geo_dir)  # ← cached
        except IndexError:
            continue                             # no files for that country

        if unary_union(poly.geometry).intersects(hull_geom):
            countries.append(c)
            poly_parts.append(poly)
            csv_list.append(csv)                # keep tidied CSV


    if not countries:
        raise RuntimeError("Hull does not intersect any configured country polygons.")

    # Merge every intersecting country’s polygons into one GDF
    poly_gdf        = pd.concat(poly_parts, ignore_index=True)
    csv_attrs_raw   = pd.concat(csv_list,  ignore_index=True)   # ← already tidy
    country_str     = ", ".join(countries)

    if "asdf_id" not in poly_gdf.columns:
        raise KeyError("'asdf_id' column missing in the GeoJSON for " + country_str)

    # ------------------------------------------------------------------
    # 3. Build a GeoDataFrame of network *nodes* in WGS84 --------------
    # ------------------------------------------------------------------
    gnodes = ox.graph_to_gdfs(G_wgs84, nodes=True, edges=False)
    gnodes = (
        gnodes.reset_index()
              .rename(columns={"osmid": "NodeID"})[["NodeID", "geometry"]]
    )
    gnodes = gpd.GeoDataFrame(gnodes, geometry="geometry", crs="EPSG:4326")

    # ------------------------------------------------------------------
    # 4. Spatial join → pick up 'asdf_id' for each node ----------------
    # ------------------------------------------------------------------
    joined = (
        gpd.sjoin(gnodes,
                  poly_gdf[["asdf_id", "geometry"]],
                  how="left",
                  predicate="within")
        .drop(columns=["index_right"])
    )

    # ------------------------------------------------------------------
    # 5. Merge full CSV attributes on 'asdf_id' ------------------------
    # ------------------------------------------------------------------
    csv_attrs = csv_attrs_raw
    if "asdf_id" not in csv_attrs.columns:
        raise KeyError("'asdf_id' column missing in the CSV for " + country_str)

    attribs = joined.merge(csv_attrs, on="asdf_id", how="left")

    # ------------------------------------------------------------------
    # 6. Fuse attributes back into the numerical nodes_df --------------
    # ------------------------------------------------------------------
    nodes_aug = nodes_df.merge(
        attribs.drop(columns="geometry"),  # keep nodes_df tidy
        on="NodeID",
        how="left"
    )

    return nodes_aug

def parse_width(val):
    """
    Parse OSM `width` tag to metres (float) or None.
    Accepts '7.5', '24 ft', '330cm', '120"', etc.
    """
    if val is None or (isinstance(val, numbers.Number) and np.isnan(val)):
        return None
    if isinstance(val, (list, tuple, np.ndarray)):
        for item in val:
            v = parse_width(item)
            if v is not None:
                return v
        return None

    s = str(val)
    m = re.search(r"\d+(?:\.\d+)?", s)
    if not m:
        return None
    num = float(m.group())

    s_low = s.lower()
    if "ft" in s_low:
        num *= 0.3048          # feet → m
    elif '"' in s_low or "inch" in s_low:
        num *= 0.0254          # inches → m
    elif "cm" in s_low:
        num /= 100.0           # cm → m
    # otherwise assume metres
    return round(num, 2)

def estimate_lane_width(row):
    """Return lane width in metres using width tag or class defaults."""
    if pd.notnull(row["RoadWidth"]):
        return round(row["RoadWidth"] / max(row["Lanes"], 1), 2)

    hw = row["highway"]
    if isinstance(hw, list):
        hw = hw[0]
        
# ----------------------------------------------------------------------
# helper: write a list of completely-missing attributes ---------------
# ----------------------------------------------------------------------
def save_missing_attrs(city, nodes_df, links_df, directory):
    exp_node = ["TrafficCtrl", "Elevation", "Barrier",
                "Amenity", "PublicTransport", "Railway"]
    exp_link = ["Oneway", "Surface", "Bridge", "Tunnel",
                "Access", "Vehicle", "Cycleway", "Sidewalk", "Busway",
                "MaxHeight", "MaxWeight", "MaxWidth",
                "Roundabout", "Lit"]

    missing = []
    for col in exp_node:
        if col not in nodes_df or nodes_df[col].isna().all():
            missing.append({"Level": "node",  "Attribute": col})
    for col in exp_link:
        if col not in links_df or links_df[col].isna().all():
            missing.append({"Level": "link",  "Attribute": col})

    out_path = os.path.join(directory, f"missing_attrs_{city}.csv")
    pd.DataFrame(missing).to_csv(out_path, index=False)
    print(f"  → missing-attribute list saved to {out_path}")
    
# ------------------------------------------------------------------
# helper: length / chord ratio for a LineString --------------------
# ------------------------------------------------------------------
def _curvature(line):
    """Return link-length / straight-line distance; NaN if degenerate."""
    if line is None or line.is_empty:
        return pd.NA
    try:
        start, end = line.coords[0], line.coords[-1]
        chord      = ((start[0]-end[0])**2 + (start[1]-end[1])**2)**0.5
        if chord == 0:
            return 1.0
        return round(line.length / chord, 3)
    except Exception:
        return pd.NA

# ----------------------------------------------------------------------
# 6. Script entry ------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    DET_DIR = os.path.join(RAW_ROOT, "detector_lists")
    process_all_detector_files(
        DET_DIR,
        buffer_m=2700,
        concave=False,
        chunk_size=50_000     # “big-only” chunking threshold
    )

# %% cell 4 of 4
import pandas as pd
from sklearn.cluster import KMeans
import os
ncl=5
# Define the file path
input_file = os.path.join(RAW_ROOT, "maps/detectors_seattle.csv")

# Load the CSV file
df = pd.read_csv(input_file)

# Ensure Latitude and Longitude columns exist
if "Latitude" not in df.columns or "Longitude" not in df.columns:
    raise ValueError("CSV must contain 'Latitude' and 'Longitude' columns.")

# Perform KMeans clustering
kmeans = KMeans(n_clusters=ncl, random_state=42)
df["Cluster"] = kmeans.fit_predict(df[["Latitude", "Longitude"]])

# Output directory (same as input file)
output_dir = os.path.dirname(input_file)

# Save each cluster into a separate CSV file
for cluster_id in range(ncl):
    cluster_df = df[df["Cluster"] == cluster_id]
    output_file = os.path.join(output_dir, f"detectors_cluster_{cluster_id + 1}.csv")
    cluster_df.to_csv(output_file, index=False)

print("Clustering complete. Files saved.")

