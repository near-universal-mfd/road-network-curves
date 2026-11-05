"""The demo: the model with the paper's constants, and refitted, on four networks in demo/data/ (Cagliari, Wanborough, Bordeaux, Konstanz).
Writes errors.csv, constants.json and fits.png to demo/output/ and compares them with demo/expected_output/.
Usage: python scripts/run_demo.py [--write-expected]
"""
import json
import os
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import predict_curve as P
from paths import ROOT

D = os.path.join(ROOT, "demo")
DATA, OUT, EXPECTED = os.path.join(D, "data"), os.path.join(D, "output"), os.path.join(D, "expected_output")
NETWORKS = {"cagliari": "Cagliari", "Wanborough": "Wanborough", "bordeaux": "Bordeaux", "constance": "Konstanz"}
A_CUBIC = 0.157         # equation (2): scripts/cubic_factors.py
FORMS = {"Cubic": lambda k, u, kc: u * (k + A_CUBIC * k ** 2 / kc - (2 * A_CUBIC + 1) * k ** 3 / (3 * kc ** 2)),
         "Greenshields": lambda k, u, kc: u * k * (1 - k / (2 * kc)),
         "Underwood": lambda k, u, kc: u * k * np.exp(-k / kc),
         "Drake": lambda k, u, kc: u * k * np.exp(-0.5 * (k / kc) ** 2)}


def network(c):
    with open(os.path.join(DATA, "Outputs", f"outputs_{c}.txt")) as fh:
        qc, kc, q6 = [float(v) for v in fh.read().split(",")][:3]
    d = pd.read_csv(os.path.join(DATA, "MFDs", f"data_{c}_grouped.csv"))
    d = d[(d.density >= 0) & (d.density <= 1.1 * kc) & (d.flow >= 0)].dropna(subset=["density", "flow"])
    return qc, kc, 6 * q6 / kc, d.density.to_numpy(float), d.flow.to_numpy(float)


def errors(pe, label):
    rows = []
    for c, nm in NETWORKS.items():
        qc, kc, uf, k, q = network(c)
        for form, fn in {**FORMS, label: lambda k_, u, kc_: P.flow(k_, u, kc_, pe)}.items():
            r = (q - fn(k, uf, kc)) / qc
            rows.append(dict(network=nm, form=form, RMSE=np.sqrt(np.mean(r ** 2)), MAE=np.mean(np.abs(r))))
    return rows


def figure(paper, refit, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=130)
    for ax, (c, nm) in zip(axes.flat, NETWORKS.items()):
        qc, kc, uf, k, q = network(c)
        g = np.linspace(0, 1.8 * kc, 300)
        ax.scatter(k, q, s=4, alpha=0.15, c="gray", label="Observations")
        for form in ("Cubic", "Greenshields", "Underwood", "Drake"):
            ax.plot(g, FORMS[form](g, uf, kc), lw=1.2, label=form)
        ax.plot(g, P.flow(g, uf, kc, paper), lw=2.4, c="k", label="Equation (5), paper's constants")
        ax.plot(g, P.flow(g, uf, kc, refit), lw=1.6, c="k", ls="--", label="Equation (5), refitted on the four")
        ax.set_xlim(0, 1.8 * kc)
        ax.set_ylim(0, 1.25 * max(qc, np.percentile(q, 99.5)))
        ax.set_title(f"{nm} ($u_{{fe}}$ = {uf:.0f} km/h, $\\alpha$ = {P.alpha_of(uf, paper):.2f})", fontsize=10)
        ax.set_xlabel("Density (veh/km/lane)")
        ax.set_ylabel("Flow (veh/h/lane)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)


def main():
    os.makedirs(OUT, exist_ok=True)
    fit_dir = os.path.join(OUT, "fit")
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(OUT, "fit.log"), "w", encoding="utf-8") as fh:
        subprocess.check_call([sys.executable, os.path.join(here, "mfd_uncertainty.py"), "--mfd-dir", os.path.join(DATA, "MFDs"), "--outputs-dir", os.path.join(DATA, "Outputs"),
                               "--nodes-dir", os.path.join(DATA, "map_metrics.csv"), "--results-dir", fit_dir, "--point-only", "--global-search", "--b-bounds=-6,6", "--w-bounds=-10,10",
                               "--fix", "w_L=0", "--fix", "w_SL=0", "--fix", "w_C=0"], stdout=fh, stderr=subprocess.STDOUT, cwd=here, env=dict(os.environ, PYTHONUTF8="1", PYTHONWARNINGS="ignore"))
    paper = P.constants()
    with open(os.path.join(fit_dir, "point_estimate.json")) as fh:
        refit = json.load(fh)
    rows = errors(paper, "Equation (5), paper's constants") + [r for r in errors(refit, "Equation (5), refitted on the four") if r["form"].startswith("Equation")]
    err = pd.DataFrame(rows)
    err.to_csv(os.path.join(OUT, "errors.csv"), index=False, float_format="%.6f")
    keep = ("a", "d", "b", "w_uf", "s_u", "s_K")
    cons = {"paper": {k: paper["hybrid_params"][k] for k in keep}, "refitted_on_the_four": {k: refit["hybrid_params"][k] for k in keep},
            "objective_on_the_four": refit["objective_hybrid"], "networks": list(NETWORKS.values())}
    with open(os.path.join(OUT, "constants.json"), "w") as fh:
        json.dump(cons, fh, indent=1)
    figure(paper, refit, os.path.join(OUT, "fits.png"))
    print(err.pivot(index="network", columns="form", values="RMSE").round(3).to_string())
    print(json.dumps(cons, indent=1))
    if "--write-expected" in sys.argv:
        os.makedirs(EXPECTED, exist_ok=True)
        for f in ("errors.csv", "constants.json", "fits.png"):
            shutil.copy2(os.path.join(OUT, f), EXPECTED)
        print("wrote", EXPECTED)
        return 0
    e0 = pd.read_csv(os.path.join(EXPECTED, "errors.csv"))
    with open(os.path.join(EXPECTED, "constants.json")) as fh:
        c0 = json.load(fh)
    bad = [f"{a.network}, {a.form}" for a, b in zip(e0.itertuples(), err.itertuples()) if abs(a.RMSE - b.RMSE) > 5e-4 or abs(a.MAE - b.MAE) > 5e-4]
    bad += [k for k in keep if abs(c0["refitted_on_the_four"][k] - cons["refitted_on_the_four"][k]) > 5e-3 * max(1, abs(c0["refitted_on_the_four"][k]))]
    print("\nMatches demo/expected_output/ to 3 decimal places." if not bad else f"\nDIFFERS from demo/expected_output/: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
