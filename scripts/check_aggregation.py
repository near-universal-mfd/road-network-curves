"""Rebuild network files from the detector files and compare every time step they share (UTD19, PEMS and six WebTRIS towns), and compare
the hand-read K_c with an automatic rule. Writes diagnostics/aggregation_check.csv and diagnostics/kc_hand_vs_auto.csv.
"""
import os

import numpy as np
import pandas as pd

from paths import RESULTS, PER_LANE, SOURCE_GROUPS, RAW_TRAFFIC, RAW_UTD19, RAW_CLEANED, RAW_CURVES  # noqa: E402
A_CUBIC = 0.157


def auto_kc(k, q, nb=30):
    """The density of the bin with the highest mean flow."""
    edges = np.linspace(np.nanpercentile(k, 0.5), np.nanpercentile(k, 99.5), nb + 1)
    g = pd.DataFrame({"i": np.clip(np.digitize(k, edges) - 1, 0, nb - 1), "k": k, "q": q}).groupby("i").agg(k=("k", "mean"), q=("q", "mean"), n=("q", "size"))
    g = g[g.n >= max(5, 0.002 * len(k))]
    return float(g.k.loc[g.q.idxmax()])


def score(k, q, kc):
    near = lambda c: (k > c * 0.85) & (k < c * 1.15)  # noqa: E731
    uf = 6 * q[near(kc / 6)].mean() / kc
    qc = q[near(kc)].mean()
    keep = k <= 1.1 * kc
    x, y, c = k[keep] / kc, q[keep] / qc, uf * kc / qc
    a = A_CUBIC
    cubic = x * (1 + a * x - (1 + 2 * a) / 3 * x ** 2)
    under = x * np.exp(-x)
    rc, ru = float(np.sqrt(np.mean((y - c * cubic) ** 2))), float(np.sqrt(np.mean((y - c * under) ** 2)))
    return dict(Kc=kc, uf=uf, qc=float(qc), rmse_cubic=rc, rmse_underwood=ru, better="exponential" if ru < rc else "cubic")


UTD = {"augsburg": "augsburg", "basel": "basel", "bern": "bern", "birmingham": "birmingham", "bordeaux": "bordeaux",
       "bremen": "bremen", "cagliari": "cagliari", "konstanz": "constance", "darmstadt": "darmstadt", "essen": "essen",
       "frankfurt": "frankfurt", "graz": "graz", "groningen": "groningen", "hamburg": "hamburg", "innsbruck": "innsbruck",
       "kassel": "kassel", "london": "london", "losangeles": "losangeles", "lucerne": "luzern", "madrid": "madrid",
       "manchester": "manchester", "marseille": "marseille", "munich": "munich", "paris": "paris", "rotterdam": "rotterdam",
       "santander": "santander", "speyer": "speyer", "strasbourg": "strasbourg", "stuttgart": "stuttgart", "taipei": "taipeh",
       "turin": "torino", "toronto": "toronto", "toulouse": "toulouse", "vilnius": "vilnius", "wolfsburg": "wolfsburg",
       "zurich": "zurich"}
TOWNS = ["Ashfield", "Winchester", "Wanborough", "Barnsley", "Bristol", "Salford"]


def compare(name, source, mine, ref, keys):
    j = mine.merge(ref, on=keys, suffixes=("_mine", "_file"))
    close = lambda a, b: (np.abs(a - b) <= 1e-6 * (1 + np.abs(b)))  # noqa: E731
    return dict(network=name, source=source, steps_rebuilt=len(mine), steps_in_file=len(ref), matched=len(j),
                flow_exact=float(close(j.flow_mine, j.flow_file).mean()) if len(j) else np.nan,
                density_exact=float(close(j.density_mine, j.density_file).mean()) if len(j) else np.nan)


def mean_by(path_or_df, keys, usecols=None, chunks=2_000_000):
    it = pd.read_csv(path_or_df, usecols=usecols, chunksize=chunks)
    parts = [c.groupby(keys).agg(f=("flow", "sum"), k=("density", "sum"), n=("flow", "size")) for c in it]
    a = pd.concat(parts).groupby(level=list(range(len(keys)))).sum()
    return pd.DataFrame({"flow": a.f / a.n, "density": a.k / a.n}).reset_index()


def main():
    rows = []
    for corre, name in UTD.items():
        ref = pd.read_csv(os.path.join(RAW_CURVES, f"data_{name}_grouped.csv"))
        mine = mean_by(os.path.join(RAW_CLEANED, f"data_{corre}_corre.csv"), ["day", "interval"], ["day", "interval", "flow", "density"])
        rows.append(compare(name, "UTD19", mine, ref, ["day", "interval"]))
        print(rows[-1], flush=True)
    # Bolton's file was made with another rule (raw_processing/R/dataD.xlsx, part 5): rebuild it from its part file
    d = pd.read_csv(os.path.join(RAW_UTD19, "data5_cor.csv"), usecols=["day", "interval", "flow", "speed"], low_memory=False)
    for c in ("flow", "speed"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[(d.speed > 0) & (d.speed <= 105) & (d.flow <= 3000)].assign(density=lambda x: x.flow / x.speed)
    mine = d.groupby(["day", "interval"])[["flow", "density"]].mean().reset_index()
    rows.append(compare("bolton", "UTD19, from flow/speed", mine, pd.read_csv(os.path.join(RAW_CURVES, "data_bolton_grouped.csv")), ["day", "interval"]))
    for fn, name in (("traffic04.csv", "SanFrancisco"), ("traffic08.csv", "sanbernardino")):
        d = pd.read_csv(os.path.join(RAW_TRAFFIC, fn))
        d = d[d.speed > 0]
        g = d.groupby("timestep").agg(flow=("flow", "mean"), spd=("speed", "mean")).reset_index()
        g["flow"] = 12 * g.flow
        g["density"] = g.flow / (0.95 * 1.60934 * g.spd)
        rows.append(compare(name, "PEMS", g[["timestep", "flow", "density"]], pd.read_csv(os.path.join(RAW_CURVES, f"data_{name}_grouped.csv")), ["timestep"]))
    for town in TOWNS:
        p = os.path.join(RAW_TRAFFIC, "Engl_cities", f"England_15min_cities_{town}.csv")
        if not os.path.exists(p):
            print("no town file", town)
            continue
        d = pd.read_csv(p, usecols=["Report Date", "Time Period Ending", "flow", "speed"], low_memory=False)
        d = d[d.flow > 0]
        d["timestamp"] = pd.to_datetime(d["Report Date"]).dt.strftime("%Y-%m-%d") + " " + d["Time Period Ending"]
        mine = d.groupby("timestamp").agg(flow=("flow", "mean"), spd=("speed", "mean")).reset_index()
        mine["density"] = mine.flow / (0.95 * mine.spd)
        rows.append(compare(town, "WebTRIS", mine, pd.read_csv(os.path.join(RAW_CURVES, f"data_{town}_grouped.csv")), ["timestamp"]))
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RESULTS, "diagnostics", "aggregation_check.csv"), index=False)
    pd.set_option("display.width", 200)
    print(t.round(4).to_string())

    ui = pd.read_csv(os.path.join(RESULTS, "MFDs_Curve_uncertainty", "per_network_UI.csv"))
    scales = pd.read_csv(os.path.join(PER_LANE, "network_scales.csv")).set_index("city")
    for col in ("kc", "uf", "qc"):       # the curves below are in each source's own units
        ui[col] = ui.city.map(scales[col + "_as_reported"])
    err = pd.read_csv(os.path.join(RESULTS, "MFDs_FINAL", "per_city_errors_normalized.csv")).set_index("city")
    grp = pd.read_csv(SOURCE_GROUPS).set_index("city").group
    rows = []
    for r in ui.itertuples():
        g = pd.read_csv(os.path.join(RAW_CURVES, f"data_{r.city}_grouped.csv"), usecols=["flow", "density"]).dropna()
        k, q = g.density.to_numpy(float), g.flow.to_numpy(float)
        ok = (k > 0) & (q >= 0)
        k, q = k[ok], q[ok]
        try:
            ka = auto_kc(k, q)
            s = score(k, q, ka)
        except Exception as exc:                                        # too few points near K_c/6 for the rule
            print("no automatic K_c for", r.city, exc)
            s = dict(Kc=np.nan, uf=np.nan, better=None, rmse_cubic=np.nan, rmse_underwood=np.nan, qc=np.nan)
        hand_exp = err.loc[r.city, "RMSE_Underwood"] < err.loc[r.city, "RMSE_Cubic"]
        rows.append(dict(city=r.city, source=grp[r.city], kc_hand=r.kc, kc_auto=s["Kc"], ratio=s["Kc"] / r.kc, uf_hand=r.uf, uf_auto=s["uf"],
                         qc_hand=r.qc, qc_auto=s["qc"], form_hand="exponential" if hand_exp else "cubic", form_auto=s["better"]))
    a = pd.DataFrame(rows)
    a.to_csv(os.path.join(RESULTS, "diagnostics", "kc_hand_vs_auto.csv"), index=False)
    v = a.dropna(subset=["ratio"])
    print("networks with an automatic K_c:", len(v), "median ratio", round(v.ratio.median(), 3), "within 10%:", round(float(((v.ratio > .9) & (v.ratio < 1.1)).mean()), 3),
          "within 25%:", round(float(((v.ratio > .75) & (v.ratio < 1.25)).mean()), 3), "same form:", round(float((v.form_hand == v.form_auto).mean()), 3))
    print(v.groupby("source")[["ratio", "form_hand", "form_auto"]].apply(lambda x: pd.Series({"n": len(x), "median_ratio": x.ratio.median(), "within25": ((x.ratio > .75) & (x.ratio < 1.25)).mean(), "same_form": (x.form_hand == x.form_auto).mean()})).round(3).to_string())


if __name__ == "__main__":
    main()
