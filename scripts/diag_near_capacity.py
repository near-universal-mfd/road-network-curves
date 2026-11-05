"""Each form's error per network and loading bin, with its bias and the scatter of the observations, using notebook 05's definitions.
Writes diagnostics/near_capacity_bins.csv.
"""
import os

import nbformat
import numpy as np
import pandas as pd

from paths import RESULTS, NOTEBOOKS  # noqa: E402
src = nbformat.read(os.path.join(NOTEBOOKS, "05_models_and_figures.ipynb"), as_version=4).cells[1].source
head = src[:src.index("rows = []")]                 # constants and model definitions, no main loop
os.chdir(NOTEBOOKS)
ns = {"__name__": "nb05"}
exec(compile(head, "nb05_head", "exec"), ns)
g = ns
import glob  # noqa: E402

rows = []
for data_fp in sorted(glob.glob(g["data_pattern"])):
    city = os.path.basename(data_fp)[5:-12]
    outs = g["read_outputs_for_city"](city)
    if outs is None:
        continue
    qc, kc, q_kc6 = outs[0], outs[1], outs[2]
    uf = 6.0 * q_kc6 / kc
    met = g["get_city_network_metrics"](city)
    ui = g["compute_ui_from_metrics"](uf=uf, avg_link_length=met[0], avg_speed_limit=met[1],
                                      node_connectivity=met[2])
    df = pd.read_csv(data_fp)
    dens = pd.to_numeric(df["density"], errors="coerce")
    flow = pd.to_numeric(df["flow"], errors="coerce")
    m = dens.ge(0) & dens.le(1.1 * kc) & flow.ge(0) & dens.notna() & flow.notna()
    k, q = dens[m].to_numpy(float), flow[m].to_numpy(float)
    models = {
        "Hybrid": g["q_hybrid"](k, uf, ui, kc, u_th=g["U_TH"], s_u=g["S_U"], s_k=g["S_K"], a_h=g["alph"]),
        "Cubic": g["q_cubic"](k, uf, kc, g["alp"]),
        "Greenshields": g["q_greenshields"](k, uf, 2.0 * kc),
        "Drake": g["q_drake"](k, uf, kc),
        "Underwood": g["q_underwood"](k, uf, kc),
    }
    x = k / kc
    edges = g["BIN_EDGES"]
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        mb = (x >= lo) & (x < hi) if b < len(edges) - 2 else (x >= lo) & (x <= hi + 1e-9)
        if not mb.any():
            continue
        qo = q[mb] / qc
        rec = dict(city=city, urbanized=bool(ui <= g["U_TH"]), bin=b, n=int(mb.sum()),
                   obs_mean=qo.mean(), obs_sd=qo.std(), uf_kc_over_qc=uf * kc / qc)
        for name, qm in models.items():
            d = qm[mb] / qc - qo
            rec[f"bias_{name}"] = d.mean()
            rec[f"rmse_{name}"] = np.sqrt(np.mean(d ** 2))
        rows.append(rec)

out = pd.DataFrame(rows)
os.makedirs(os.path.join(RESULTS, "diagnostics"), exist_ok=True)
out.to_csv(os.path.join(RESULTS, "diagnostics", "near_capacity_bins.csv"), index=False)

pd.set_option("display.width", 200)
names = ["Hybrid", "Cubic", "Greenshields", "Drake", "Underwood"]
print("Mean over networks, by loading bin (errors normalized by each network's q_c):")
agg = out.groupby("bin").agg(n_networks=("city", "size"), obs_mean=("obs_mean", "mean"), obs_sd=("obs_sd", "mean"),
                             **{f"rmse_{n}": (f"rmse_{n}", "mean") for n in names},
                             **{f"bias_{n}": (f"bias_{n}", "mean") for n in names})
print(agg.round(3).to_string())
last = out[out["bin"] == out["bin"].max()]
print("\nBin nearest capacity, split by the index's class:")
print(last.groupby("urbanized")[["obs_mean", "obs_sd", "uf_kc_over_qc"] + [f"rmse_{n}" for n in names]
                                + [f"bias_{n}" for n in names]].mean().round(3).T.to_string())
