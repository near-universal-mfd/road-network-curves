"""Refit every bootstrap replicate whose index offset b ends on its bound (|b| <= 6) with both bounds widened fivefold (|b| <= 30, |w| <= 50),
and recompute the reported intervals with them. Writes diagnostics/b_bound_widened_replicates.csv and diagnostics/b_bound_intervals.csv.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from paths import RESULTS  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mfd_uncertainty as M  # noqa: E402
from paths import MFD_DIR, OUTPUTS_DIR, NODES_DIR, SEED  # noqa: E402

FIXED = {"s_u": 0.2344, "s_K": 0.1343, "w_L": 0.0, "w_SL": 0.0, "w_C": 0.0}
BOUNDS = {"bootstrap": (6.0, 10.0), "widened": (30.0, 50.0)}      # (|b| <=, |w| <=)
KEYS = ["objective", "a", "d", "b", "w_uf", "frac_urbanized", "class_agreement",
        "dq_S", "dq_cap_sparse", "dq_cap_urban", "dq_x_jam_sparse", "dq_wave_sparse", "dq_wave_urban"]
REPORTED = ["a", "d", "dq_S", "dq_cap_sparse", "dq_cap_urban", "dq_x_jam_sparse", "dq_wave_sparse",
            "dq_wave_urban", "class_agreement"]
_P = {}


def _init(prep, theta, ref):
    _P["prep"], _P["theta"], _P["ref"] = prep, theta, ref


def refit(job):
    r, label, w = job
    b_hi, w_hi = BOUNDS[label]
    M.set_fixed_params(FIXED)
    M.set_bounds(b=(-b_hi, b_hi), w=(-w_hi, w_hi))
    prep = _P["prep"]
    th, ob = M.fit_hybrid(prep, w, x0=_P["theta"], restarts=2, rng=np.random.default_rng(100_000 + r),
                          binned=True, use_global=False, maxfev=3000)
    d = M.ui_summary(prep, th, w)
    rec = dict(replicate=r, bounds=label, objective=ob, a=th[0], d=th[3], b=th[4], w_uf=th[5],
               frac_urbanized=d["frac_urbanized"],
               class_agreement=float(np.mean((d["_ui"] <= M.U_TH_FIXED) == _P["ref"])))
    rec.update({f"dq_{k}": v for k, v in M.derived_quantities(th[0], th[3], th[2]).items()})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--draws", default=os.path.join(RESULTS, "MFDs_Curve_uncertainty_full", "bootstrap_draws.csv"))
    args = ap.parse_args()
    M.set_fixed_params(FIXED)
    M.set_bounds(b=(-6.0, 6.0), w=(-10.0, 10.0))
    prep = M.build_prepared(MFD_DIR, OUTPUTS_DIR, NODES_DIR, nbins=400, kmax_factor=M.KMAX_FACTOR_DEFAULT,
                            keep_exact=False)
    full = os.path.join(RESULTS, "MFDs_Curve_uncertainty_full")
    pe = json.load(open(os.path.join(full, "point_estimate.json")))
    theta = np.array([pe["hybrid_params"][n] for n in M.HYBRID_NAMES], float)
    ref = M.ui_summary(prep, theta, np.ones(prep.n_cities))["_ui"] <= M.U_TH_FIXED
    rng = np.random.default_rng(SEED)
    W = [np.bincount(rng.integers(0, prep.n_cities, size=prep.n_cities), minlength=prep.n_cities)
         .astype(float) for _ in range(1000)]
    boot = pd.read_csv(args.draws).set_index("replicate")
    at_bound = boot[boot.b > 5.999]
    print(f"{len(at_bound)} of {len(boot)} replicates end on b = +6; none of them has |w_uf| at 10: "
          f"{bool((at_bound.w_uf.abs() < 9.999).all())}")
    all_at_bound(args, prep, theta, ref, W, boot, at_bound, full)


def all_at_bound(args, prep, theta, ref, W, boot, at_bound, full):
    jobs = [(r, "widened", W[r]) for r in at_bound.index.astype(int)]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(prep, theta, ref)) as ex:
        new = pd.DataFrame(list(ex.map(refit, jobs))).set_index("replicate")
    new.to_csv(os.path.join(RESULTS, "diagnostics", "b_bound_widened_replicates.csv"))
    b_hi = BOUNDS["widened"][0]
    print(f"widened refits: {int((new.b > b_hi - 1e-3).sum())} of {len(new)} end on b = +{b_hi:g}; "
          f"objective lower in {int((new.objective < at_bound.objective - 1e-12).sum())}, "
          f"largest fall {(at_bound.objective - new.objective).max():.4f}; "
          f"networks urbanized changed in {int((new.frac_urbanized != at_bound.frac_urbanized).sum())}")

    iv = pd.read_csv(os.path.join(full, "interval_summary.csv")).set_index("parameter")
    jack = pd.read_csv(os.path.join(full, "jackknife_influence.csv"))
    point = iv["estimate"].to_dict()
    same = M.summarize(point, boot.reset_index(), jack, 0.95, {}).set_index("parameter")
    assert np.allclose(same.loc[REPORTED, ["bca_lo", "bca_hi", "pct_lo", "pct_hi"]],
                       iv.loc[REPORTED, ["bca_lo", "bca_hi", "pct_lo", "pct_hi"]], rtol=0, atol=1e-12), \
        "summarize() does not reproduce interval_summary.csv"
    mod = boot.copy()
    mod.loc[new.index, KEYS] = new[KEYS]
    s = M.summarize(point, mod.reset_index(), jack, 0.95, {}).set_index("parameter")
    rows = []
    for p_ in REPORTED:
        kind = "pct" if p_ == "class_agreement" else "bca"
        rows.append(dict(parameter=p_, interval=kind, estimate=point[p_],
                         lo=iv.loc[p_, f"{kind}_lo"], hi=iv.loc[p_, f"{kind}_hi"],
                         lo_widened=s.loc[p_, f"{kind}_lo"], hi_widened=s.loc[p_, f"{kind}_hi"],
                         boot_mean=iv.loc[p_, "boot_mean"], boot_mean_widened=s.loc[p_, "boot_mean"]))
    cmp = pd.DataFrame(rows)
    cmp["largest_change"] = np.maximum((cmp.lo_widened - cmp.lo).abs(), (cmp.hi_widened - cmp.hi).abs())
    cmp.to_csv(os.path.join(RESULTS, "diagnostics", "b_bound_intervals.csv"), index=False)
    pd.set_option("display.width", 250)
    print(cmp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
