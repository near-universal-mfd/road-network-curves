"""Extended Data Fig. 1c: Cagliari's curve before screening, per lane. Writes figures/panels/edfig1c_cagliari_raw.csv."""
import os

import numpy as np
import pandas as pd

from paths import RESULTS, PER_LANE, MFD_DIR, RAW_CLEANED  # noqa: E402
S = pd.read_csv(os.path.join(PER_LANE, "network_scales.csv")).set_index("city").loc["cagliari"]
parts = [c.groupby(["day", "interval"]).agg(f=("flow", "sum"), k=("density", "sum"), n=("flow", "size"))
         for c in pd.read_csv(os.path.join(RAW_CLEANED, "data_cagliari_corre.csv"), usecols=["day", "interval", "flow", "density"], chunksize=2_000_000)]
a = pd.concat(parts).groupby(level=[0, 1]).sum()
out = pd.DataFrame({"flow": a.f / a.n / S.flow_divisor, "density": a.k / a.n / S.density_divisor}).reset_index()
ref = pd.read_csv(os.path.join(MFD_DIR, "data_cagliari_grouped.csv"))
j = out.merge(ref, on=["day", "interval"], suffixes=("", "_f"))
assert len(j) == len(ref) and np.allclose(j.flow, j.flow_f, rtol=1e-6) and np.allclose(j.density, j.density_f, rtol=1e-6), "does not reproduce the network file"
out.to_csv(os.path.join(RESULTS, "figures", "panels", "edfig1c_cagliari_raw.csv"), index=False)
print("time steps before screening:", len(out), "after:", len(ref), "| max density", round(out.density.max(), 1), "max flow", round(out.flow.max(), 1))
