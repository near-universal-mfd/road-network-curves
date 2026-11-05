"""Write the summary tables in data/:

    networks.csv            one row per network: source, place, period, detectors, density method, lanes, factor, speed correction
    network_summary.csv     one row per network: scales, map inputs, urbanization index, better form, and each form's error
    outcomes_7point.csv     the seven summaries of each curve, per lane and as each source reports them
    feature_dictionary.csv  each candidate attribute: what it is, and whether it survived the pruning
    splits/pls_folds.csv    the statistics' outer folds, for the five fold seeds
    splits/model_cv_folds.csv  the model's 20 cross-validation folds
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import mfd_uncertainty as M
from paths import AS_REPORTED_OUTPUTS, DATA, MAP_METRICS, MFD_DIR, NODES_DIR, OUTPUTS_DIR, PER_LANE, SEED, SHIPPED_RESULTS, SOURCE_GROUPS

R = SHIPPED_RESULTS
COUNTRY = {"augsburg": "Germany", "basel": "Switzerland", "bern": "Switzerland", "birmingham": "United Kingdom", "bolton": "United Kingdom", "bordeaux": "France", "bremen": "Germany", "cagliari": "Italy",
           "constance": "Germany", "darmstadt": "Germany", "essen": "Germany", "frankfurt": "Germany", "graz": "Austria", "groningen": "Netherlands", "hamburg": "Germany", "innsbruck": "Austria", "kassel": "Germany",
           "london": "United Kingdom", "losangeles": "United States", "luzern": "Switzerland", "madrid": "Spain", "manchester": "United Kingdom", "marseille": "France", "munich": "Germany", "paris": "France",
           "rotterdam": "Netherlands", "santander": "Spain", "speyer": "Germany", "strasbourg": "France", "stuttgart": "Germany", "taipeh": "Taiwan", "torino": "Italy", "toronto": "Canada", "toulouse": "France",
           "vilnius": "Lithuania", "wolfsburg": "Germany", "zurich": "Switzerland"}
CONTINENT = {"United States": "North America", "Canada": "North America", "Taiwan": "Asia"}
DISPLAY = {"losangeles": "Los Angeles", "sanbernardino": "San Bernardino", "SanFrancisco": "San Francisco", "constance": "Konstanz", "taipeh": "Taipei", "torino": "Turin", "luzern": "Lucerne",
           "London2": "London (WebTRIS)", "Birmingham2": "Birmingham (WebTRIS)", "Manchester2": "Manchester (WebTRIS)", "london": "London (UTD19)", "birmingham": "Birmingham (UTD19)", "manchester": "Manchester (UTD19)"}
PERIOD_FROM_SOURCE = {"SanFrancisco": ("2018-01", "2018-02"), "sanbernardino": ("2016-07", "2016-08")}      # the files number their time steps: months from the PEMS04 and PEMS08 documentation
MINUTES = {"WebTRIS": 15, "PEMS": 5, "Portland": 5, "Seattle": 5}


def outputs(folder, c):
    with open(os.path.join(folder, f"outputs_{c}.txt")) as fh:
        return [float(v) for v in fh.read().split(",")]


def networks():
    S = pd.read_csv(os.path.join(PER_LANE, "network_scales.csv"))
    groups = pd.read_csv(SOURCE_GROUPS).set_index("city").group
    maps = pd.read_csv(MAP_METRICS).set_index("city").map_name
    det = pd.read_csv(os.path.join(DATA, "detectors.csv"), usecols=["network"], low_memory=False).network.value_counts()
    st = pd.read_csv(os.path.join(R, "diagnostics", "network_structure.csv")).set_index("city")
    rows = []
    for r in S.itertuples():
        c = r.city
        d = pd.read_csv(os.path.join(MFD_DIR, f"data_{c}_grouped.csv"))
        if "timestamp" in d:
            t = pd.to_datetime(d.timestamp)
            start, end = t.min().strftime("%Y-%m-%d"), t.max().strftime("%Y-%m-%d")
            step = float(t.sort_values().diff().dt.total_seconds().median() / 60)
        elif "day" in d:
            start, end = str(d.day.min()), str(d.day.max())
            step = float(d.sort_values(["day", "interval"]).groupby("day").interval.diff().median() / 60)
        else:
            start, end = PERIOD_FROM_SOURCE[c]
            step = 5.0
        src = groups[c]
        country = COUNTRY.get(c, "United States" if src in ("PEMS", "Portland", "Seattle") else "United Kingdom")
        note = ""
        if r.speed_multiple != 1:
            note = "speed in mph, converted" if src == "Seattle" else "detectors reporting no speed left out of the network speed"
        rows.append(dict(network=c, map_files=maps[c], name=DISPLAY.get(c, c.replace("_", " ")), country=country, continent=CONTINENT.get(country, "Europe"), source=src,
                         period_start=start, period_end=end, interval_min=MINUTES.get(src, round(step, 2)), n_time_steps=len(d), n_detectors_or_sites=int(det.get(c, 0)),
                         detector_share_motorway=st.det_share_motorway.get(c, np.nan), density_method=r.density_method, occupancy_factor=r.factor, lanes=r.lanes, lane_source=r.lane_source, speed_multiple=r.speed_multiple, speed_correction=note,
                         partition_of="seattle" if src == "Seattle" else ""))
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "networks.csv"), index=False)
    return pd.DataFrame(rows)


def summary():
    S = pd.read_csv(os.path.join(PER_LANE, "network_scales.csv")).set_index("city")[["source", "density_method", "qc", "kc", "uf", "qc_as_reported", "kc_as_reported", "uf_as_reported"]]
    S = S.rename(columns={"qc": "q_c", "kc": "K_c", "uf": "u_fe", "qc_as_reported": "q_c_as_reported", "kc_as_reported": "K_c_as_reported", "uf_as_reported": "u_fe_as_reported"})
    m = pd.read_csv(MAP_METRICS).set_index("city")[["avg_link_length", "avg_speed_limit", "node_connectivity"]]
    u = pd.read_csv(os.path.join(R, "MFDs_Curve_uncertainty", "per_network_UI.csv")).set_index("city")[["n_obs", "UI", "urbanized_UI_le_0"]]
    f = pd.read_csv(os.path.join(R, "Fig2_uf_index", "fig2_per_network.csv")).set_index("city")[["nrmse_cubic_branch", "nrmse_exp_branch", "better_branch", "index_says", "correct"]]
    e = pd.read_csv(os.path.join(R, "MFDs_FINAL", "per_city_errors_normalized.csv")).set_index("city").drop(columns=["uf", "kce", "qce"]).rename(columns={"alpha_uf": "alpha"})
    t = S.join(m).join(u).join(f).join(e)
    t["better_standard_form"] = np.where(t.RMSE_Underwood < t.RMSE_Cubic, "Underwood", "cubic")
    t.index.name = "network"
    t.to_csv(os.path.join(DATA, "network_summary.csv"))


def outcomes():
    names = ["q_c", "K_c", "q_Kc_6", "q_Kc_3", "q_Kc_2", "q_2Kc_3", "q_5Kc_6"]
    rows = []
    for p in sorted(glob.glob(os.path.join(OUTPUTS_DIR, "outputs_*.txt"))):
        c = os.path.basename(p)[8:-4]
        rows.append(dict(network=c, **dict(zip(names, outputs(OUTPUTS_DIR, c))), **{n + "_as_reported": v for n, v in zip(names, outputs(AS_REPORTED_OUTPUTS, c))}))
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "outcomes_7point.csv"), index=False)


COVERS = {"barren_or_sparsely_vegetated": "barren or sparsely vegetated", "closed_shrublands": "closed shrublands", "cropland_natural_vegetation_mosaic": "cropland and natural vegetation mosaic", "croplands": "croplands",
          "deciduous_broadleaf_forest": "deciduous broadleaf forest", "deciduous_needleleaf_forest": "deciduous needleleaf forest", "evergreen_broadleaf_forest": "evergreen broadleaf forest", "evergreen_needleleaf_forest": "evergreen needleleaf forest",
          "fill_value": "no land-cover value", "grasslands": "grasslands", "mixed_forest": "mixed forest", "open_shrublands": "open shrublands", "permanent_wetlands": "permanent wetlands", "savannas": "savannas", "snow_and_ice": "snow and ice",
          "unclassified": "unclassified", "urban_and_builtup": "urban and built-up", "water": "water", "woody_savannas": "woody savannas"}
VARS = {"difference_vegetation_index": ("vegetation index (NDVI)", "GeoQuery: NASA LTDR v5 AVHRR, 2020"), "dist_to_water": ("distance to water", "GeoQuery"), "distance_to_coast": ("distance to coast", "GeoQuery: World Vector Shorelines v2.3.6"),
        "elevation": ("elevation", "GeoQuery: SRTM v4.1"), "slope": ("slope", "GeoQuery: SRTM v4.1"), "wind_speed_potential": ("wind speed potential at 50 m", "GeoQuery: Global Wind Atlas v2"),
        "InflowCount": ("links entering a node", "OpenStreetMap"), "OutflowCount": ("links leaving a node", "OpenStreetMap")}
LINK = {"Lanes": "lanes (OSM tag)", "num_detectors": "detectors on a link", "LinkLength": "link length (m)", "Curvature": "link curvature (length / straight-line distance)", "SpeedLimit": "tagged speed limit (km/h)", "avg_detector_pos": "mean position of a link's detectors along it"}
CITY = {"Density (ppl/km^2)": ("city population divided by the area of the detector hull", "compiled by hand, 2025; hull"), "nb_of_polygons": ("administrative polygons overlapping the detector hull", "geoBoundaries"), "Area (km^2)": ("area of the detector hull", "OpenStreetMap hull"),
        "GDP (Million US$)": ("gridded GDP, area-weighted", "GeoQuery: GDP grid"), "Pop Density (ppkm2)": ("city population divided by city area", "compiled by hand, 2025"), "City Area (km^2)": ("city area", "compiled by hand, 2025"), "Income (USD/yr)": ("median individual income", "compiled by hand, 2025"),
        "City Population": ("city population", "compiled by hand, 2025"), "Employment Rate (%)": ("employment rate", "compiled by hand, 2025"), "Longitude": ("longitude", "hull"), "Latitude": ("latitude", "hull"),
        "kept_rows_classes": ("links kept by the road-class filter", "OpenStreetMap"), "kept_prop_classes": ("share of links kept by the road-class filter", "OpenStreetMap"), "kept_prop_access": ("share of links kept by the access filter", "OpenStreetMap")}
AGG = {"mean": "mean", "std": "standard deviation", "min": "minimum", "max": "maximum", "q50": "median"}
MONTH = dict(day_st="daytime land surface temperature", night_st="nighttime land surface temperature", precip="precipitation")
SRC_MONTH = dict(day_st="GeoQuery: MODIS MOD11C3 v6.1, 2023", night_st="GeoQuery: MODIS MOD11C3 v6.1, 2023", precip="GeoQuery: CRU TS v4.07, 2022")


def describe(f):
    """(level, what, source) of a feature, from its name."""
    m = re.fullmatch(r"link_Class_(\w+)", f)
    if m:
        return "link", f"share of links of class {m.group(1)}", "OpenStreetMap"
    m = re.fullmatch(r"link_Access_(\w+)", f)
    if m:
        return "link", f"share of links with access tag {m.group(1)}", "OpenStreetMap"
    m = re.fullmatch(r"link_(Bridge|Tunnel|Oneway|Roundabout)_frac", f)
    if m:
        return "link", f"share of links tagged {m.group(1).lower()}", "OpenStreetMap"
    if f == "link_SpeedLimit_missing_frac":
        return "link", "share of links with no speed-limit tag", "OpenStreetMap"
    m = re.fullmatch(r"link_(\w+?)_(mean|std|min|max|q50)", f)
    if m and m.group(1) in LINK:
        return "link", f"{AGG[m.group(2)]} over links of {LINK[m.group(1)]}", "OpenStreetMap"
    m = re.fullmatch(r"node_(\d\d)_(day_st|night_st|precip)_(mean|std|min|max)", f)
    if m:
        return "node", f"{AGG[m.group(3)]} over nodes of {MONTH[m.group(2)]}, month {m.group(1)}", SRC_MONTH[m.group(2)]
    m = re.fullmatch(r"node_(\w+?)_(mean|std|min|max)", f)
    if m and m.group(1) in COVERS:
        return "node", f"{AGG[m.group(2)]} over nodes of land-cover share: {COVERS[m.group(1)]}", "GeoQuery: MODIS MCD12Q1 v5.1, 2012"
    if m and m.group(1) in VARS:
        return "node", f"{AGG[m.group(2)]} over nodes of {VARS[m.group(1)][0]}", VARS[m.group(1)][1]
    m = re.fullmatch(r"(\d\d)_(day_st|night_st|precip)", f)
    if m:
        return "network", f"{MONTH[m.group(2)]}, month {m.group(1)}", SRC_MONTH[m.group(2)]
    if f in COVERS:
        return "network", f"land-cover share: {COVERS[f]}", "GeoQuery: MODIS MCD12Q1 v5.1, 2012"
    if f in VARS:
        return "network", VARS[f][0], VARS[f][1]
    if f in CITY:
        return "network", CITY[f][0], CITY[f][1]
    m = re.fullmatch(r"([FM])(\d+)", f)
    if m:
        return "network", f"{'women' if m.group(1) == 'F' else 'men'} aged {m.group(2)} (lower edge of the age group), summed over the hull", "WorldPop 2020 constrained"
    raise ValueError(f)


def features():
    a = pd.read_csv(os.path.join(R, "StatResults", "predictors_all_cities_raw.csv"), index_col=0, nrows=1).columns
    kept = set(pd.read_csv(os.path.join(R, "StatResults", "modeling_X_clean.csv"), index_col=0, nrows=1).columns)
    dg = os.path.join(R, "StatResults", "diagnostics")
    step = {}
    for nm, label in (("removed_by_variance", "near-zero variance"), ("removed_by_deduplicate", "exact duplicate"), ("removed_by_correlation", "correlation above 0.95"), ("removed_by_vif", "variance inflation factor above 10")):
        for v in pd.read_csv(os.path.join(dg, nm + ".csv")).iloc[:, 0].astype(str):
            step[v] = label
    rows = []
    for f in a:
        lv, what, src = describe(f)
        rows.append(dict(feature=f, level=lv, description=what, source=src, retained_in_74=f in kept, removed_by="" if f in kept else step.get(f, "not among the 428 candidates: no values or held fixed before pruning")))
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "feature_dictionary.csv"), index=False)


def splits():
    os.makedirs(os.path.join(DATA, "splits"), exist_ok=True)
    idx = pd.read_csv(os.path.join(R, "StatResults", "modeling_Y_multi.csv"), index_col=0).index
    rows = []
    for seed in (42, 1, 2, 3, 4):
        for k, (_, te) in enumerate(KFold(5, shuffle=True, random_state=seed).split(idx)):
            rows += [dict(network=idx[i], seed=seed, fold=k) for i in te]
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "splits", "pls_folds.csv"), index=False)
    prep = M.build_prepared(MFD_DIR, OUTPUTS_DIR, NODES_DIR, keep_exact=False, quiet=True)
    rows = []
    for rep, f, _, te in M.make_cv_folds(prep, 5, 4, SEED, "uf"):
        rows += [dict(network=prep.cities[i], repeat=rep, fold=f) for i in te]
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "splits", "model_cv_folds.csv"), index=False)


if __name__ == "__main__":
    n = networks()
    print(n.groupby("source").agg(networks=("network", "size"), detectors=("n_detectors_or_sites", "sum"), steps=("n_time_steps", "median"), interval=("interval_min", "median")).to_string())
    print(n.continent.value_counts().to_dict())
    summary()
    outcomes()
    features()
    splits()
    print("wrote the tables in", DATA)
