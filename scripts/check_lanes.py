"""The lane count behind each network's flows: UTD19's detector table, weighted by records; OpenStreetMap for WebTRIS and PEMS (mainline
links with 2+ lanes, and slip roads in WebTRIS); none for Portland and Seattle, which are per lane. Also checks the counts against peak flows.
Writes diagnostics/lanes_by_network.csv, lanes_utd19_detectors.csv and lanes_webtris_sites.csv.
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from paths import RESULTS, SOURCE_GROUPS, AS_REPORTED_OUTPUTS, RAW_TRAFFIC, RAW_UTD19, RAW_CLEANED, RAW_MAPS  # noqa: E402
DG = os.path.join(RESULTS, "diagnostics")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_aggregation import UTD, TOWNS  # noqa: E402

MAP_NAME = {"london": "London2", "London2": "London", "birmingham": "Birmingham2", "Birmingham2": "Birmingham", "manchester": "Manchester2", "Manchester2": "Manchester"}
SPEED_BASED = {"birmingham", "bolton", "constance", "essen", "groningen", "innsbruck", "manchester", "rotterdam", "torino"}


def link_lanes(city):
    """Mean lanes per direction at a network's detectors: mainline links with 2+ lanes, and slip roads."""
    L = pd.read_csv(os.path.join(RAW_MAPS, f"links_{MAP_NAME.get(city, city)}.csv"), usecols=["Class", "Lanes", "Oneway", "num_detectors"], low_memory=False)
    L = L[pd.to_numeric(L.num_detectors, errors="coerce").fillna(0) > 0].copy()
    L["n"] = pd.to_numeric(L.num_detectors)
    L["l"] = [np.mean([float(x) for x in re.findall(r"\d+\.?\d*", str(v))]) if re.search(r"\d", str(v)) else np.nan for v in L.Lanes]
    L["l"] = np.where(L.Oneway.astype(str).str.upper().eq("TRUE"), L.l, L.l / 2)
    cls = L.Class.astype(str).str.extract(r"([a-z_]+)")[0]
    slip = cls.str.endswith("_link")
    main = (~slip) & (L.l >= 2)
    avg = lambda m: float(np.average(L.l[m], weights=L.n[m])) if m.any() else np.nan  # noqa: E731
    return avg(main), avg(slip & L.l.notna()), float(L.n[(~slip) & (L.l < 2)].sum() / L.n.sum())


def utd19():
    det = pd.read_csv(os.path.join(RAW_UTD19, "detectors_public.csv"))
    det["detid"] = det.detid.astype(str)
    rows, per_det = [], []
    for corre, net in [*UTD.items(), ("bolton", "bolton")]:
        code = {"losangeles": "losanageles"}.get(net, net)
        lan = det[det.citycode == code].drop_duplicates("detid").set_index("detid").lanes
        tot, samples = [], []
        cnt = {}
        if net == "bolton":       # its file was made with another rule: rebuild it from the part file
            b = pd.read_csv(os.path.join(RAW_UTD19, "data5_cor.csv"), usecols=["day", "interval", "detid", "flow", "speed"], dtype={"detid": str}, low_memory=False)
            for c_ in ("flow", "speed"):
                b[c_] = pd.to_numeric(b[c_], errors="coerce")
            b = b[(b.speed > 0) & (b.speed <= 105) & (b.flow <= 3000)].assign(density=lambda x: x.flow / x.speed)
            chunks = [b[["day", "interval", "detid", "flow", "density"]]]
        else:
            chunks = pd.read_csv(os.path.join(RAW_CLEANED, f"data_{corre}_corre.csv"), usecols=["day", "interval", "detid", "flow", "density"], chunksize=2_000_000,
                                 dtype={"detid": str}, low_memory=False)
        for ch in chunks:
            ch["l"] = ch.detid.map(lan)
            for k, v in ch.detid.value_counts().items():
                cnt[k] = cnt.get(k, 0) + v
            ch = ch.dropna(subset=["l"])
            ch["q_lane"] = ch.flow / ch.l
            tot.append(ch.groupby(["day", "interval"]).agg(q=("flow", "sum"), ql=("q_lane", "sum"), k=("density", "sum"), n=("flow", "size")))
            samples.append(ch.loc[ch.flow > 0, ["detid", "flow", "l"]].sample(frac=0.25, random_state=0))
        a = pd.concat(tot).groupby(level=[0, 1]).sum()
        Q, Ql, K = a.q / a.n, a.ql / a.n, a.k / a.n
        s = pd.concat(samples)
        q95 = s.groupby("detid").agg(q95=("flow", lambda v: v.quantile(.95)), l=("l", "first"))
        q95["network"] = net
        per_det.append(q95.reset_index())
        cn = pd.Series(cnt)
        cl = cn.index.map(lan)
        known = ~pd.isna(cl)
        L_rec = float(np.average(np.asarray(cl[known], float), weights=cn.values[known]))
        L_q = float(Q.sum() / Ql.sum())
        # the same curve? mean ratio Q / Q_lane across bins of density
        bins = pd.qcut(K.rank(method="first"), 20, labels=False)
        ratio = (Q.groupby(bins).mean() / Ql.groupby(bins).mean())
        rows.append(dict(city=net, lanes_detectors=float(lan.reindex(cn.index).mean()), lanes=L_rec, L_flow=L_q, unmatched=float(1 - known.mean()), n_detectors=len(cn),
                         curve_ratio_min=float(ratio.min()), curve_ratio_max=float(ratio.max()), curve_deviation=float((ratio / L_q - 1).abs().max())))
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows).set_index("city")
    q = pd.concat(per_det)
    q.to_csv(os.path.join(DG, "lanes_utd19_detectors.csv"), index=False)
    m = q[q.l.isin([1, 2, 3, 4, 5])]
    print("median 95th-percentile flow by lanes, all UTD19 detectors:", m.groupby("l").q95.agg(["count", "median"]).round(0).to_dict("index"))
    within = []
    for n, g in m.groupby("network"):
        if g.l.nunique() >= 2 and (g.l.value_counts() >= 5).sum() >= 2:
            f = stats.linregress(np.log(g.l), np.log(g.q95))
            within.append(dict(network=n, slope=f.slope, se=f.stderr, n=len(g)))
    print(pd.DataFrame(within).round(2).to_string())
    print("median within-network slope of log peak flow on log lanes:", round(pd.DataFrame(within).slope.median(), 2))
    return t


def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / np.maximum(dx * dx + dy * dy, 1e-12), 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def webtris_sites():
    rows = []
    for town in TOWNS:
        d = pd.read_csv(os.path.join(RAW_TRAFFIC, "Engl_cities", f"England_15min_cities_{town}.csv"), usecols=["Site Name", "flow"], low_memory=False)
        d = d[d.flow > 0]
        peak = d.groupby("Site Name").flow.quantile(.995)
        site = pd.read_csv(os.path.join(RAW_MAPS, f"detectors_{town}.csv")).drop_duplicates("Name").set_index("Name")
        nodes = pd.read_csv(os.path.join(RAW_MAPS, f"nodes_{town}.csv"), usecols=["NodeID", "Longitude", "Latitude"]).set_index("NodeID")
        L = pd.read_csv(os.path.join(RAW_MAPS, f"links_{town}.csv"), usecols=["StartNode", "EndNode", "Class", "Lanes", "Oneway", "num_detectors"], low_memory=False)
        L = L[pd.to_numeric(L.num_detectors, errors="coerce").fillna(0) > 0]
        L["lanes_raw"] = [np.mean([float(x) for x in re.findall(r"\d+\.?\d*", str(v))]) if re.search(r"\d", str(v)) else np.nan for v in L.Lanes]
        L["lanes_dir"] = np.where(L.Oneway.astype(str).str.upper().eq("TRUE"), L.lanes_raw, L.lanes_raw / 2)
        ax, ay = L.StartNode.map(nodes.Longitude).to_numpy(), L.StartNode.map(nodes.Latitude).to_numpy()
        bx, by = L.EndNode.map(nodes.Longitude).to_numpy(), L.EndNode.map(nodes.Latitude).to_numpy()
        for name, q in peak.items():
            if name not in site.index:
                continue
            py, px = site.Latitude[name], site.Longitude[name]
            c = np.cos(np.radians(py))
            dist = seg_dist(px * c, py, ax * c, ay, bx * c, by) * 111_000
            i = int(np.nanargmin(dist))
            rows.append(dict(town=town, site=name, q995=float(q), lanes_osm=float(L.lanes_dir.iloc[i]), cls=str(L.Class.iloc[i]), dist_m=float(dist[i])))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(DG, "lanes_webtris_sites.csv"), index=False)
    ok = t[t.dist_m <= 60].dropna(subset=["lanes_osm"])
    print("WebTRIS sites matched:", len(ok), "of", len(t), "| classes:", ok.cls.value_counts().head(6).to_dict())
    print("by OSM lanes: median peak flow", ok.groupby("lanes_osm").q995.agg(["count", "median"]).round(0).to_dict("index"))
    print("Spearman peak vs lanes:", stats.spearmanr(ok.q995, ok.lanes_osm))
    f = stats.linregress(np.log(ok.lanes_osm), np.log(ok.q995))
    print(f"log-log slope {f.slope:.2f} (se {f.stderr:.2f}), r {f.rvalue:.2f}")
    per_lane = (ok.q995 / ok.lanes_osm).median()
    implied = (ok.q995 / per_lane).round()
    print(f"median peak per OSM lane {per_lane:.0f} veh/h; lanes deduced from the peak agree with OSM within one lane at {((implied - ok.lanes_osm).abs() <= 1).mean():.0%} of sites, exactly (to the nearest lane) at {((implied - ok.lanes_osm.round()).abs() < 0.5).mean():.0%}")
    print("sites whose peak exceeds 2,400 veh/h per OSM lane:", int((ok.q995 / ok.lanes_osm > 2400).sum()), "of", len(ok), "| with 1 lane on a motorway:", int(((ok.lanes_osm == 1) & ok.cls.str.fullmatch("motorway")).sum()))
    print("motorway and trunk mainline, peak per lane by lanes:", ok[ok.cls.isin(["motorway", "trunk"])].assign(pl=lambda x: x.q995 / x.lanes_osm).groupby("lanes_osm").pl.median().round(0).to_dict())
    return t


def main():
    st = pd.read_csv(os.path.join(DG, "network_structure.csv")).set_index("city")
    g = pd.read_csv(SOURCE_GROUPS).set_index("city").group.reindex(st.index)
    y = {}
    for c in st.index:
        with open(os.path.join(AS_REPORTED_OUTPUTS, f"outputs_{c}.txt")) as fh:
            y[c] = [float(v) for v in fh.read().split(",")]
    qc = pd.Series({c: v[0] for c, v in y.items()})
    u = utd19()
    webtris_sites()
    T = pd.DataFrame({"source": g, "qc_as_reported": qc.reindex(st.index), "lanes_osm_all_detector_links": st.det_lanes})
    T["lanes"] = np.nan
    T["lane_source"] = ""
    per_site = g.isin(["WebTRIS", "PEMS"])
    ll = pd.DataFrame({c: link_lanes(c) for c in st.index[per_site]}, index=["main", "slip", "share_mainline_one_lane"]).T
    for c in st.index[g == "WebTRIS"]:
        names = pd.read_csv(os.path.join(RAW_MAPS, f"detectors_{MAP_NAME.get(c, c)}.csv")).Name.astype(str)
        ll.loc[c, "share_slip_sites"] = names.str.extract(r"/\d+([A-Z]+)\d*$")[0].isin(list("JKLM")).mean()
    wt = ll.index[g[ll.index] == "WebTRIS"]
    ll.loc[wt, "main_filled"] = ll.main[wt].fillna(ll.main[wt].median())
    ll.loc[wt, "slip_filled"] = ll.slip[wt].fillna(ll.slip[wt].median())
    print("WebTRIS: no mainline tag:", ll.index[(g[ll.index] == "WebTRIS") & ll.main.isna()].tolist(), "| no slip-road tag:", int(ll.slip[wt].isna().sum()), "networks")
    T.loc[wt, "lanes"] = ((1 - ll.share_slip_sites[wt]) * ll.main_filled[wt] + ll.share_slip_sites[wt] * ll.slip_filled[wt]).to_numpy()
    T.loc[wt, "lane_source"] = "OpenStreetMap lanes of the detector links: mainline with 2+ lanes and slip roads, weighted by site names"
    pm = ll.index[g[ll.index] == "PEMS"]
    T.loc[pm, "lanes"] = ll.main[pm].to_numpy()
    T.loc[pm, "lane_source"] = "OpenStreetMap lanes of the detector links: mainline with 2+ lanes"
    T = T.join(ll[["main", "slip", "share_slip_sites", "share_mainline_one_lane"]])
    T.loc[g.isin(["Portland", "Seattle"]), "lanes"] = 1.0
    T.loc[g == "Portland", "lane_source"] = "already per lane: the authors' code averages each station's lane detectors"
    T.loc[g == "Seattle", "lane_source"] = "treated as per lane: no lane field, capacity matches the per-lane capacity of other freeway sources"
    utd = g == "UTD19"
    assert set(st.index[utd]) == set(u.index), set(st.index[utd]) ^ set(u.index)
    T.loc[utd, "lanes"] = u.lanes.reindex(st.index[utd]).to_numpy()
    T.loc[utd, "lane_source"] = "UTD19 detector table, weighted by records"
    T["density_method"] = np.where(utd & ~st.index.str.lower().isin(sorted(SPEED_BASED)), "occupancy", "flow/speed")
    T = T.join(u[["L_flow", "curve_ratio_min", "curve_ratio_max", "curve_deviation", "unmatched", "n_detectors"]])
    T["qc_per_lane"] = T.qc_as_reported / T.lanes
    T.to_csv(os.path.join(DG, "lanes_by_network.csv"))
    assert T.lanes.notna().all() and (T.density_method == "occupancy").sum() == 28
    print(T.groupby("source")[["lanes", "qc_as_reported", "qc_per_lane"]].median().round(2).to_string())
    print("UTD19: lanes against flow-weighted lanes, largest gap", round((u.L_flow / u.lanes - 1).abs().max(), 3), "| curve deviation, largest", round(u.curve_deviation.max(), 3), "median", round(u.curve_deviation.median(), 3))
    print(u.sort_values("curve_deviation", ascending=False).head(6)[["lanes", "L_flow", "curve_ratio_min", "curve_ratio_max", "curve_deviation"]].round(3).to_string())
    w = T[T.source == "WebTRIS"]
    f = stats.linregress(np.log(w.lanes), np.log(w.qc_as_reported))
    print(f"WebTRIS networks: log q_c on log lanes: slope {f.slope:.2f} (se {f.stderr:.2f}), r {f.rvalue:.2f}; lanes range {w.lanes.min():.2f}-{w.lanes.max():.2f}")
    f0 = stats.linregress(np.log(w.lanes_osm_all_detector_links), np.log(w.qc_as_reported))
    print(f"  with every tag taken at face value: slope {f0.slope:.2f} (se {f0.stderr:.2f}), r {f0.rvalue:.2f}")
    print("per-lane capacity spread: WebTRIS CV", round(w.qc_per_lane.std() / w.qc_per_lane.mean(), 2), "as reported CV", round(w.qc_as_reported.std() / w.qc_as_reported.mean(), 2))
    uu = T[utd]
    fu = stats.linregress(np.log(uu.lanes), np.log(uu.qc_as_reported))
    print(f"UTD19 networks: log q_c on log lanes: slope {fu.slope:.2f} (se {fu.stderr:.2f}), r {fu.rvalue:.2f}; CV {uu.qc_as_reported.std() / uu.qc_as_reported.mean():.2f} as reported, {uu.qc_per_lane.std() / uu.qc_per_lane.mean():.2f} per lane")


if __name__ == "__main__":
    main()
