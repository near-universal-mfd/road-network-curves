"""The model's errors as the paper reports them: paired tests against each benchmark, the error distribution, the trend share of
each network's squared error, the all-cubic baselines, and where the networks and the urbanized ones come from.
Writes diagnostics/model_errors.json.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import mfd_uncertainty as M
from paths import MFD_DIR, NODES_DIR, OUTPUTS_DIR, RESULTS, SOURCE_GROUPS

BENCH = ["Cubic", "Nonagonal", "Pentagonal", "Drake", "Greenshields", "Underwood"]
NOT_EUROPE = {"losangeles": "North America", "toronto": "North America", "SanFrancisco": "North America", "sanbernardino": "North America",
              "portland": "North America", "seattle1": "North America", "seattle5": "North America", "taipeh": "Asia"}


def trend_shares():
    """Per network: the share of its squared error that is departure of the binned means from the model, rather than scatter."""
    M.set_fixed_params({"s_u": 0.2344, "s_K": 0.1343, "w_L": 0.0, "w_SL": 0.0, "w_C": 0.0})
    prep = M.build_prepared(MFD_DIR, OUTPUTS_DIR, NODES_DIR, quiet=True)
    with open(os.path.join(RESULTS, "MFDs_Curve_uncertainty", "point_estimate.json")) as fh:
        th = json.load(fh)["hybrid_params"]
    theta = np.array([th[k] for k in M.HYBRID_NAMES], float)
    alpha = M._dec_sigmoid(M.ui_summary(prep, theta, np.ones(prep.n_cities))["_ui"] / theta[1])
    shares = {}
    for i, c in enumerate(prep.cities):
        m = prep.exact_city == i
        x, y = prep.exact_x[m], prep.exact_y[m]
        pred = prep.c_scale[i] * M.f_model(x, np.full(x.size, alpha[i]), theta[0], theta[3], theta[2])
        b = np.clip(np.digitize(x, np.linspace(0, 1.1, 23)) - 1, 0, 21)
        g = pd.DataFrame({"b": b, "y": y, "p": pred, "r2": (y - pred) ** 2})
        gm = g.groupby("b").agg(n=("y", "size"), y=("y", "mean"), p=("p", "mean"))
        total = float(g.r2.mean())
        shares[c] = float(((gm.y - gm.p) ** 2 * gm.n).sum() / gm.n.sum()) / total if total > 0 else float("nan")
    return pd.Series(shares)


def main():
    d = (pd.read_csv(os.path.join(RESULTS, "Fig2_uf_index", "fig2_per_network.csv"))
         .merge(pd.read_csv(os.path.join(RESULTS, "MFDs_FINAL", "per_city_errors_normalized.csv")).drop(columns=["uf"]), on="city")
         .merge(pd.read_csv(SOURCE_GROUPS), on="city"))
    assert len(d) == 141 and set(NOT_EUROPE) <= set(d.city)
    utd, urb, h = d.group == "UTD19", d.UI <= 0, d.RMSE_Hybrid
    paired = {g: {b: dict(p=float(stats.wilcoxon((d["RMSE_" + b] - h)[m]).pvalue), hybrid_better=int((d["RMSE_" + b] > h)[m].sum()), n=int(m.sum()))
                  for b in BENCH} for g, m in (("all", np.ones(len(d), bool)), ("urbanized", urb.to_numpy()))}
    R = dict(continents=d.city.map(NOT_EUROPE).fillna("Europe").value_counts().to_dict(), n_urbanized=int(urb.sum()), urbanized_outside_utd19=int((urb & ~utd).sum()),
             all_cubic=dict(branches=float(1 - (d.better_branch == "exponential").mean()), standard_forms=float(1 - (d.RMSE_Underwood < d.RMSE_Cubic).mean())),
             paired=paired, median_nrmse=float(h.median()), n_over_0p2=int((h > .2).sum()),
             trend_share_median=float(np.median(trend_shares().reindex(d.city).to_numpy())))
    with open(os.path.join(RESULTS, "diagnostics", "model_errors.json"), "w") as fh:
        json.dump(R, fh, indent=1)
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
