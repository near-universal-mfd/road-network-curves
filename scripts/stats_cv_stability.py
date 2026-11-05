"""Notebook 01's nested PLS cross-validation with five fold seeds (42, 1, 2, 3, 4). Writes StatResults/pls2_r2_CV_by_seed.csv."""
import os

import nbformat
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from paths import RESULTS, NOTEBOOKS  # noqa: E402
S = os.path.join(RESULTS, "StatResults")
SEEDS = (42, 1, 2, 3, 4)
NAMES = ["qc", "kc", "q_kc/6", "q_kc/3", "q_kc/2", "q_2kc/3", "q_5kc/6"]

src = nbformat.read(os.path.join(NOTEBOOKS, "01_statistics.ipynb"), as_version=4).cells[1].source
ns = {"np": np, "KFold": KFold, "PLSRegression": PLSRegression, "r2_score": r2_score, "clone": clone}
for fn in ("def select_pls_components_cv", "def nested_cv_r2_pls"):
    i = src.index(fn)
    exec(src[i:src.index("\ndef ", i + 10)], ns)

X = pd.read_csv(os.path.join(S, "modeling_X_normal_scores.csv"), index_col=0).to_numpy(float)
Y = pd.read_csv(os.path.join(S, "modeling_Y_multi.csv"), index_col=0).to_numpy(float)
rows = []
for seed in SEEDS:
    r2, _ = ns["nested_cv_r2_pls"](X, Y, max_comp=10, n_splits=5, random_state=seed)
    rows.append(dict(seed=seed, **dict(zip(NAMES, r2[:7]))))
out = pd.DataFrame(rows)
nb_r2 = pd.read_csv(os.path.join(S, "pls2_all_targets_r2_CV.csv"))["r2_cv_oof"].to_numpy()[:7]
assert np.allclose(out.loc[out.seed == 42, NAMES].to_numpy()[0], nb_r2), "seed 42 does not reproduce the notebook"
out.to_csv(os.path.join(S, "pls2_r2_CV_by_seed.csv"), index=False)
print(out.round(3).to_string(index=False))
