"""The occupancy factor per lane where flow, occupancy and speed are all reported: per detector, the median of density from flow and speed
divided by occupancy (occupancy 1-50%), then the median over detectors. Writes diagnostics/occupancy_factor_measured.csv.
"""
import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 200)
from paths import RESULTS, RAW_TRAFFIC, RAW_UTD19  # noqa: E402
det = pd.read_csv(os.path.join(RAW_UTD19, "detectors_public.csv"))
det["detid"] = det.detid.astype(str)
T = pd.read_csv(os.path.join(RESULTS, "diagnostics", "lanes_by_network.csv"), index_col=0)
rows = []


def per_detector(d, name, source, lanes=None, divide_by=1.0):
    d = d[(d.occ > 0.01) & (d.occ < 0.5) & (d.speed > 5) & (d.flow > 0)].copy()
    d["f"] = d.flow / d.speed / d.occ
    per = d.groupby("detid").f.median()
    if lanes is not None:
        per = (per / lanes.reindex(per.index)).dropna()
    per = per / divide_by
    rows.append(dict(network=name, source=source, n_detectors=len(per), factor_per_lane=per.median(), q25=per.quantile(.25), q75=per.quantile(.75), lanes_divided_by=divide_by))


for part, name in ((9, "constance"), (11, "essen")):
    d = pd.read_csv(os.path.join(RAW_UTD19, f"data{part}_cor.csv"), usecols=["detid", "flow", "occ", "speed"], low_memory=False)
    for c in ("flow", "occ", "speed"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[(d.speed <= 105) & (d.occ <= 1)].dropna()
    d["detid"] = d.detid.astype(str)
    per_detector(d, name, "UTD19", det[det.citycode == name].drop_duplicates("detid").set_index("detid").lanes)

for fn, name in (("traffic04.csv", "SanFrancisco"), ("traffic08.csv", "sanbernardino")):
    d = pd.read_csv(os.path.join(RAW_TRAFFIC, fn))
    e = pd.DataFrame({"detid": d.location.astype(str), "flow": 12 * d.flow, "occ": d.occupancy, "speed": 0.95 * 1.60934 * d.speed})
    per_detector(e, name, "PEMS", divide_by=float(T.lanes[name]))

parts, n = [], 0
for ch in pd.read_csv(os.path.join(RAW_TRAFFIC, "Portland.csv"), usecols=["detectorid", "volume", "speed", "occupancy", "status"], chunksize=2_000_000, dtype={"detectorid": str}):
    parts.append(ch[ch.status <= 3])
    n += len(parts[-1])
    if n > 8_000_000:
        break
d = pd.concat(parts).dropna(subset=["volume", "speed", "occupancy"])
per_detector(pd.DataFrame({"detid": d.detectorid, "flow": 180.0 * d.volume, "occ": d.occupancy / 100.0, "speed": 0.95 * 1.60934 * d.speed}), "portland", "Portland")

# Seattle's file has no header: column 6 is occupancy, volumes are per lane, speeds are in mph
parts, n = [], 0
for ch in pd.read_csv(os.path.join(RAW_TRAFFIC, "seattle.csv"), header=None, names=["t", "det", "dir", "speed", "vol", "occ"], usecols=["det", "dir", "speed", "vol", "occ"], chunksize=2_000_000, dtype={"det": str, "dir": str}):
    parts.append(ch)
    n += len(ch)
    if n > 8_000_000:
        break
d = pd.concat(parts)
per_detector(pd.DataFrame({"detid": d.det + d["dir"], "flow": 12.0 * d.vol, "occ": d.occ, "speed": 0.95 * 1.60934 * d.speed}), "seattle", "Seattle")

R = pd.DataFrame(rows)
R["L_eff_m"] = 1000 / R.factor_per_lane
R.to_csv(os.path.join(RESULTS, "diagnostics", "occupancy_factor_measured.csv"), index=False)
print(R.round(1).to_string())
print("median over networks:", round(R.factor_per_lane.median(), 1))
