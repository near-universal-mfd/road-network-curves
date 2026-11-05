"""The mean capacity ratio q_c / (u_fe K_c) over the networks, 0.719, hence the cubic's a = 3 x 0.719 - 2 = 0.157. Writes cubic_factors_summary.csv."""
import glob
import math
import os

import numpy as np
import pandas as pd

from paths import OUTPUTS_DIR, RESULTS

rows = []
for path in glob.glob(os.path.join(OUTPUTS_DIR, "outputs_*.txt")):
    city = os.path.basename(path)[len("outputs_"):-len(".txt")]
    with open(path) as fh:
        qc, kc, q_kc_6, q_kc_3, q_kc_2, q_2kc_3, q_5kc_6 = [float(v) for v in fh.read().split(",")]
    slope = 6.0 * q_kc_6 / kc               # u_fe: the slope from the origin to (K_c / 6, q(K_c / 6))
    rows.append({"city": city, "factor_q(kc/3)": q_kc_3 / (slope * kc / 3), "factor_q(kc/2)": q_kc_2 / (slope * kc / 2), "factor_q(2kc/3)": q_2kc_3 / (slope * 2 * kc / 3),
                 "factor_q(5kc/6)": q_5kc_6 / (slope * 5 * kc / 6), "factor_qc": qc / (slope * kc)})
df = pd.DataFrame(rows).set_index("city").sort_index()
n = len(df)
s = pd.DataFrame({"mean": df.mean(), "std": df.std(ddof=1)})
margin = 1.96 * s["std"] / math.sqrt(n)
s["ci95_lower"], s["ci95_upper"] = s["mean"] - margin, s["mean"] + margin
s.index.name = "factor"
s.to_csv(os.path.join(RESULTS, "cubic_factors_summary.csv"))
print(s.round(4).to_string())
cap = float(s.loc["factor_qc", "mean"])
a = 3 * cap - 2
print(f"{n} networks; capacity ratio {cap:.3f}, so a = 3 x {cap:.3f} - 2 = {a:.3f}")
assert np.isclose(round(cap, 3), 0.719) and np.isclose(round(a, 3), 0.157), "the paper gives a capacity ratio of 0.719 and a = 0.157"
