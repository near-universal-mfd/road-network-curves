"""The model's flow-density curve for a network, from its effective free-flow speed and critical density, with the paper's constants.

    python scripts/predict_curve.py --u_fe 45 --k_c 30
    python scripts/predict_curve.py --u_fe 45 --k_c 30 --observations my_network.csv --out my_network

--observations: a CSV with columns density (veh/km/lane) and flow (veh/h/lane). The script then also gives the normalized error on K <= 1.1 K_c,
with the capacity from --q_c or else from the prediction.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import mfd_uncertainty as M
from paths import SHIPPED_RESULTS


def constants(path=None):
    with open(path or os.path.join(SHIPPED_RESULTS, "MFDs_Curve_uncertainty", "point_estimate.json")) as fh:
        return json.load(fh)


def urbanization_index(u_fe, pe):
    """Equation (6). The other three inputs carry a weight of zero, so they sit at their sample means here."""
    hp, mom = pe["hybrid_params"], pe["ui_moments"]
    assert hp["w_L"] == hp["w_SL"] == hp["w_C"] == 0.0, "expected the free-flow-speed-only index"
    z = np.clip((u_fe - mom["metric_mean"][0]) / mom["metric_std"][0], -2.0, 2.0)
    return hp["b"] + hp["w_uf"] * (np.exp(z) - mom["phi_mean"][0]) / mom["phi_std"][0]


def alpha_of(u_fe, pe):
    return float(M._dec_sigmoid((urbanization_index(u_fe, pe) - pe["fixed"]["u_th"]) / pe["hybrid_params"]["s_u"]))


def flow(density, u_fe, k_c, pe):
    """Equation (5): q(K) = u_fe K_c f(K / K_c; alpha)."""
    hp = pe["hybrid_params"]
    x = np.asarray(density, float) / k_c
    return u_fe * k_c * M.f_model(x, np.full_like(x, alpha_of(u_fe, pe)), hp["a"], hp["d"], hp["s_K"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--u_fe", type=float, required=True, help="effective free-flow speed, km/h")
    ap.add_argument("--k_c", type=float, required=True, help="critical density, veh/km/lane")
    ap.add_argument("--q_c", type=float, default=None, help="observed capacity, veh/h/lane: normalizes the errors")
    ap.add_argument("--observations", default=None)
    ap.add_argument("--out", default="predicted_curve", help="prefix of the files written")
    args = ap.parse_args()
    pe = constants()
    al = alpha_of(args.u_fe, pe)
    k = np.linspace(0.0, 1.8 * args.k_c, 181)
    q = flow(k, args.u_fe, args.k_c, pe)
    pd.DataFrame({"density_veh_km_lane": k, "flow_veh_h_lane": q}).to_csv(args.out + ".csv", index=False)
    print(f"alpha = {al:.3f} ({'urbanized' if al >= 0.5 else 'cubic'} side of the switch); predicted capacity {q.max():.0f} veh/h/lane at {k[q.argmax()]:.1f} veh/km/lane; curve in {args.out}.csv")
    if args.observations:
        d = pd.read_csv(args.observations)
        d = d[(d.density >= 0) & (d.density <= 1.1 * args.k_c) & (d.flow >= 0)].dropna(subset=["density", "flow"])
        scale = args.q_c or float(flow(np.array([args.k_c]), args.u_fe, args.k_c, pe)[0])
        res = (d.flow.to_numpy(float) - flow(d.density.to_numpy(float), args.u_fe, args.k_c, pe)) / scale
        print(f"{len(d)} observations with K <= 1.1 K_c: normalized RMSE {np.sqrt(np.mean(res ** 2)):.3f}, normalized MAE {np.mean(np.abs(res)):.3f}")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        ax.scatter(d.density, d.flow, s=5, alpha=0.2, c="gray", label="Observations")
        ax.plot(k, q, lw=2, label="Equation (5)")
        ax.set_xlabel("Density (veh/km/lane)")
        ax.set_ylabel("Flow (veh/h/lane)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out + ".png")
        print("figure in", args.out + ".png")


if __name__ == "__main__":
    main()
