"""The road class and lane count of the links that carry each network's detectors, from the OpenStreetMap files.
Writes diagnostics/network_structure.csv.
"""
import os
import re

import numpy as np
import pandas as pd

from mfd_uncertainty import MAP_NAME
from paths import RAW_MAPS, RESULTS

GROUPS = {"motorway": ["motorway", "motorway_link"], "trunk": ["trunk", "trunk_link"], "primary": ["primary", "primary_link"],
          "secondary": ["secondary", "secondary_link"], "tertiary": ["tertiary", "tertiary_link"], "residential": ["residential"],
          "other": None}


def tags(col):
    """The mean of the numbers in each cell: OSM can list several, like "['3', '2']"."""
    vals = {}
    for v in col.dropna().unique():
        nums = [float(x) for x in re.findall(r"\d+\.?\d*", str(v))]
        vals[v] = float(np.mean(nums)) if nums else np.nan
    return col.map(vals)


def main():
    rows = []
    for c in pd.read_csv(os.path.join(RESULTS, "MFDs_Curve_uncertainty", "per_network_UI.csv")).city:
        L = pd.read_csv(os.path.join(RAW_MAPS, f"links_{MAP_NAME.get(c, c)}.csv"), low_memory=False)
        cls = L.Class.astype(str).str.extract(r"([a-z_]+)")[0]
        L["group"] = "other"
        for g, members in GROUPS.items():
            if members:
                L.loc[cls.isin(members), "group"] = g
        lanes = tags(L.Lanes)
        L["lanes_dir"] = np.where(L.Oneway.astype(str).str.upper().eq("TRUE"), lanes, lanes / 2)    # a two-way road's tag counts both directions
        D = L[pd.to_numeric(L.num_detectors, errors="coerce").fillna(0) > 0]
        nd = pd.to_numeric(D.num_detectors, errors="coerce")
        rec = {"city": c, **{f"det_share_{g}": float(nd[D.group == g].sum() / nd.sum()) if nd.sum() else np.nan for g in GROUPS}}
        ok = D.lanes_dir.notna()
        rec["det_lanes"] = float(np.average(D.lanes_dir[ok], weights=nd[ok])) if ok.any() else np.nan
        rows.append(rec)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "diagnostics", "network_structure.csv"), index=False)


if __name__ == "__main__":
    main()
