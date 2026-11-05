"""Put every network on a per-lane scale with one occupancy factor (1000 / 6, and 300 for Toulouse), correct Seattle's and Portland's speeds,
and write data/per_lane/MFDs/, data/per_lane/Outputs/ and data/per_lane/network_scales.csv. With --update-cache, also the responses in
data/features_per_network.
Usage: python build_per_lane_data.py [--update-cache]
"""
import os
import sys

import numpy as np
import pandas as pd

from paths import RESULTS, PER_LANE, AS_REPORTED_OUTPUTS, FEATURES_PER_NETWORK, RAW_TRAFFIC, RAW_CURVES, RAW_PORTLAND_STATIONS  # noqa: E402
F = 1000 / 6
FACTOR_EXCEPTION = {"toulouse": 300.0}
MPH = 1.60934
SPEED_MULTIPLE = {"seattle1": MPH, "seattle5": MPH}       # Portland's comes from portland_corrected()


def read_y(folder, city):
    with open(os.path.join(folder, f"outputs_{city}.txt")) as fh:
        return np.array([float(v) for v in fh.read().split(",")])


def portland_corrected(d, kc):
    """Portland's network speed from the detectors that report one. Returns the curve and K_c."""
    five = pd.read_csv(os.path.join(RAW_TRAFFIC, "Portland_5min.csv"), dtype={"detectorid": str}, parse_dates=["starttime"])
    stations = pd.read_csv(RAW_PORTLAND_STATIONS, usecols=lambda c: c.strip().casefold().replace(" ", "_") in {"detectorid", "stationid"}, dtype=str).rename(columns=str.lower)
    five = five.merge(stations, on="detectorid", how="left").dropna(subset=["stationid"])
    five["speed_ok"] = five.speed.where(five.speed > 0)
    st = five.groupby(["stationid", "starttime"], sort=False).agg(flow=("flow", "mean"), speed=("speed", "mean"), speed_ok=("speed_ok", "mean")).reset_index()
    net = st.groupby("starttime", sort=False).agg(flow=("flow", "mean"), speed=("speed", "mean"), speed_ok=("speed_ok", "mean")).reset_index()
    assert len(net) == len(d) and np.allclose(net.flow, d.flow) and np.allclose(net.speed.fillna(0), d.speed.fillna(0))
    old = np.where(net.speed > 0, net.flow / net.speed, 0.0)
    assert np.allclose(old, d.density)
    new = np.where(net.speed_ok > 0, net.flow / net.speed_ok, 0.0)
    near = np.abs(old / kc - 1) < .1
    assert near.sum() >= 100
    multiple = float(np.median(old[near] / new[near]))
    d = d.copy()
    d["speed"] = net.speed_ok.to_numpy()
    d["density"] = new
    return d, kc / multiple, multiple


def main():
    T = pd.read_csv(os.path.join(RESULTS, "diagnostics", "lanes_by_network.csv"), index_col=0)
    assert len(T) == 141 and T.lanes.notna().all() and (T.lanes >= 1).all() and (T.density_method == "occupancy").sum() == 28
    os.makedirs(os.path.join(PER_LANE, "MFDs"), exist_ok=True)
    os.makedirs(os.path.join(PER_LANE, "Outputs"), exist_ok=True)
    rows = []
    for c, r in T.iterrows():
        y = read_y(AS_REPORTED_OUTPUTS, c)
        occ = r.density_method == "occupancy"
        div_q = float(r.lanes)
        factor = FACTOR_EXCEPTION.get(c, F)
        div_k = 500.0 / factor if occ else float(r.lanes)
        new = y.copy()
        new[[0, 2, 3, 4, 5, 6]] = y[[0, 2, 3, 4, 5, 6]] / div_q
        new[1] = y[1] / div_k
        d = pd.read_csv(os.path.join(RAW_CURVES, f"data_{c}_grouped.csv"))
        assert {"flow", "density"} <= set(d.columns), (c, list(d.columns))
        d["flow"] = d.flow / div_q
        d["density"] = d.density / div_k
        multiple = SPEED_MULTIPLE.get(c, 1.0)
        d["density"] = d.density / multiple
        new[1] = new[1] / multiple
        if c == "portland":
            assert div_q == 1 and div_k == 1
            d, new[1], multiple = portland_corrected(d, new[1])
        with open(os.path.join(PER_LANE, "Outputs", f"outputs_{c}.txt"), "w") as fh:
            fh.write(",".join(repr(float(v)) for v in new))
        if "speed" in d.columns and occ:      # the column is derived from flow and occupancy in these networks
            d["speed"] = d.speed * div_k / div_q
        d.to_csv(os.path.join(PER_LANE, "MFDs", f"data_{c}_grouped.csv"), index=False)
        uf_old = 6 * y[2] / y[1]
        uf_new = 6 * new[2] / new[1]
        assert abs((new[3] / (2 * new[2])) - (y[3] / (2 * y[2]))) < 1e-9
        rows.append(dict(city=c, source=r.source, density_method=r.density_method, lanes=r.lanes, lane_source=r.lane_source, factor=factor if occ else np.nan, flow_divisor=div_q, density_divisor=div_k, speed_multiple=multiple,
                         qc_as_reported=y[0], qc=new[0], kc_as_reported=y[1], kc=new[1], uf_as_reported=uf_old, uf=uf_new))
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(PER_LANE, "network_scales.csv"), index=False)
    pd.set_option("display.width", 200)
    print(S.groupby(["source", "density_method"])[["lanes", "qc_as_reported", "qc", "kc_as_reported", "kc", "uf_as_reported", "uf"]].median().round(1).to_string())
    print(S[S.speed_multiple != 1][["city", "speed_multiple", "qc", "kc_as_reported", "kc", "uf_as_reported", "uf"]].round(2).to_string())
    print("networks whose u_fe changes:", int((np.abs(S.uf / S.uf_as_reported - 1) > 1e-9).sum()), "| u_fe range", round(S.uf.min(), 1), round(S.uf.max(), 1), "| q_c ratio max/min", round(S.qc.max() / S.qc.min(), 1), "was", round(S.qc_as_reported.max() / S.qc_as_reported.min(), 1))
    if "--update-cache" in sys.argv:
        cache = FEATURES_PER_NETWORK
        n = 0
        for c in T.index:
            y = read_y(os.path.join(PER_LANE, "Outputs"), c)
            rp = os.path.join(cache, f"{c}__responses.csv")
            if not os.path.exists(rp):
                continue
            res = pd.read_csv(rp)
            for k, v in (("qc", y[0]), ("kc", y[1]), ("q_kc/6", y[2]), ("q_kc/3", y[3]), ("q_kc/2", y[4]), ("q_2kc/3", y[5]), ("q_5kc/6", y[6])):
                assert k in res.columns, (c, k)
                res[k] = v
            res.to_csv(rp, index=False)
            old = np.load(os.path.join(cache, f"{c}__flow.npy"))
            assert old.size == 6, (c, old.size)
            np.save(os.path.join(cache, f"{c}__flow.npy"), np.array([y[2], y[3], y[4], y[5], y[6], y[0]]))
            n += 1
        print("cache updated:", n, "networks")


if __name__ == "__main__":
    main()
