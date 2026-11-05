"""Each occupancy network's factor from its least delayed detectors: the slope of flow per lane on occupancy at a high percentile, divided by the
network's usual speed limit; and the factor measured directly where speed is reported too. With --summary-only, it summarizes the shipped detector
table. Writes diagnostics/factor_from_free_flow_detectors.csv and diagnostics/factor_from_free_flow.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
from paths import RESULTS, PER_LANE, RAW_UTD19  # noqa: E402
PARTS = {1: "augsburg", 2: "basel", 3: "bern", 4: "birmingham", 5: "bolton", 6: "bordeaux", 7: "bremen", 8: "cagliari", 9: "constance", 10: "darmstadt", 11: "essen", 12: "frankfurt", 13: "graz",
         14: "groningen", 15: "hamburg", 16: "innsbruck", 17: "kassel", 18: "london", 19: "losangeles", 20: "luzern", 21: "madrid", 23: "manchester", 24: "marseille", 25: "munich", 26: "paris",
         27: "rotterdam", 28: "santander", 29: "speyer", 30: "strasbourg", 31: "stuttgart", 32: "taipeh", 33: "torino", 34: "toronto", 35: "toulouse", 37: "vilnius", 38: "wolfsburg", 39: "zurich"}
CODE = {"losangeles": "losanageles"}        # the spelling in UTD19's detector table
TRUE_OCC = {"constance", "essen"}           # flow/speed networks with a true occupancy column
MIN_RECORDS = 200
MIN_AT_LIMIT = 20
PCT = (50, 75, 90, 95)
DET = os.path.join(RESULTS, "diagnostics", "factor_from_free_flow_detectors.csv")


def detector_sums(path, occ_max, has_speed):
    """Sums over each detector's low-occupancy records."""
    cols = ["detid", "flow", "occ"] + (["speed"] if has_speed else [])
    parts = []
    for ch in pd.read_csv(path, usecols=cols, chunksize=3_000_000, dtype={"detid": str}, low_memory=False):
        for c in cols[1:]:
            ch[c] = pd.to_numeric(ch[c], errors="coerce")
        ch = ch[(ch.flow > 0) & (ch.occ > 0) & (ch.occ <= occ_max)].copy()
        if has_speed:
            ch = ch[ch.speed > 0].copy()
            ch["k"] = ch.flow / ch.speed
        g = ch.groupby("detid").agg(n=("flow", "size"), flow=("flow", "sum"), occ=("occ", "sum"), **({"k": ("k", "sum")} if has_speed else {}))
        parts.append(g)
    return pd.concat(parts).groupby(level=0).sum()


def summarize(g, c, method, uf):
    """The factor read off one network's detectors, with an interval from resampling them."""
    rng = np.random.default_rng(20261103)
    lim = g.limit.dropna()
    mode = float(lim.round(-1).mode().iloc[0]) if len(lim) else np.nan
    on = g[g.limit.round(-1) == mode]
    known = len(on) >= MIN_AT_LIMIT
    use = on if known else g                  # few limits known: take the network's usual limit for all its detectors
    den = float(on.limit.median()) if len(on) else np.nan
    rec = {"city": c, "method": method, "detectors": len(g), "limit": den, "share_limit_known": float(g.limit.notna().mean()), "detectors_used": len(use), "limit_known_for_those_used": known, "uf_at_167": uf,
           "pos_median_m": float(g.pos.median() * 1000)}
    for q in PCT:
        rec[f"F_p{q}"] = float(np.percentile(use.slope, q) / den) if den == den else np.nan
    if den == den:
        v = use.slope.to_numpy()
        for q, nm in ((50, "p50"), (90, "p90")):
            b = np.percentile(rng.choice(v, (2000, len(v))), q, axis=1) / den
            rec[f"F_{nm}_lo"], rec[f"F_{nm}_hi"] = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    if "factor_measured" in g and g.factor_measured.notna().any():
        rec["factor_measured_median"] = float(g.factor_measured.median())
        top = use[use.slope >= np.percentile(use.slope, 90)]
        rec["speed_top_decile"] = float(top.speed.median())
        rec["speed_median"] = float(g.speed.median())
    pk = g.dropna(subset=["pos"])
    if len(pk) >= 20:
        rec["spearman_pos_slope"] = float(pk.pos.corr(pk.slope, method="spearman"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--occ-max", type=float, default=0.06)
    ap.add_argument("--summary-only", action="store_true", help="read no part files; summarize the detectors read so far")
    args = ap.parse_args()
    S = pd.read_csv(os.path.join(PER_LANE, "network_scales.csv")).set_index("city")
    if not args.summary_only:
        D = pd.read_csv(os.path.join(RAW_UTD19, "detectors_public.csv"), low_memory=False)
        D["code"] = D.citycode.astype(str).str.lower()
        D["detid"] = D.detid.astype(str)
        D["limit"] = pd.to_numeric(D["limit"], errors="coerce").where(lambda v: v > 0)
        D["lanes"] = pd.to_numeric(D.lanes, errors="coerce").clip(lower=1)
    done = pd.read_csv(DET, dtype={"detid": str}) if os.path.exists(DET) else pd.DataFrame(columns=["city"])
    for part, c in ([] if args.summary_only else PARTS.items()):
        method = S.loc[c, "density_method"]
        if (method != "occupancy" and c not in TRUE_OCC) or c in set(done.city) or c == "london":     # UTD19's table has no speed limits for London
            continue
        p = os.path.join(RAW_UTD19, f"data{part}_cor.csv")
        has_speed = c in TRUE_OCC
        g = detector_sums(p, args.occ_max, has_speed)
        t = D[D.code == CODE.get(c, c)].drop_duplicates("detid").set_index("detid")[["lanes", "limit", "fclass", "pos", "length"]]
        g = g.join(t, how="left")
        g = g[(g.n >= MIN_RECORDS) & g.lanes.notna()].copy()
        g["slope"] = (g.flow / g.lanes) / g.occ            # veh/h/lane per unit of occupancy: factor * speed
        if has_speed:
            g["speed"] = g.flow / g.k                      # flow-weighted harmonic mean, the speed in k = q/v
            g["factor_measured"] = g.slope / g.speed
        g["city"] = c
        g.index.name = "detid"
        done = pd.concat([done, g.reset_index()], ignore_index=True)
        done.to_csv(DET, index=False)
        print(c, "read:", len(g), "detectors", flush=True)
    uf167 = S.uf * (S.factor / (1000 / 6)).fillna(1.0)      # a network with its own factor, back at the common one
    rows = [summarize(g, c, S.loc[c, "density_method"], uf167[c]) for c, g in done.groupby("city")]
    R = pd.DataFrame(rows).sort_values("F_p90", ascending=False)
    R.to_csv(os.path.join(RESULTS, "diagnostics", "factor_from_free_flow.csv"), index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 60)
    print(R.round(2).to_string())


if __name__ == "__main__":
    main()
