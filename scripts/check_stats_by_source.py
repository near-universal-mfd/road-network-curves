"""Notebook 01's nested PLS cross-validation repeated within WebTRIS and within UTD19, averaged over five fold seeds.
Writes diagnostics/stats_by_source.csv.
"""
import os

import nbformat
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from paths import NOTEBOOKS, RESULTS, SOURCE_GROUPS

S = os.path.join(RESULTS, "StatResults")
NAMES = ["qc", "kc", "q_kc/6", "q_kc/3", "q_kc/2", "q_2kc/3", "q_5kc/6"]

src = nbformat.read(os.path.join(NOTEBOOKS, "01_statistics.ipynb"), as_version=4).cells[1].source
ns = {"np": np, "KFold": KFold, "PLSRegression": PLSRegression, "r2_score": r2_score, "clone": clone}
for fn in ("def select_pls_components_cv", "def nested_cv_r2_pls"):
    i = src.index(fn)
    exec(src[i:src.index("\ndef ", i + 10)], ns)
nested = ns["nested_cv_r2_pls"]

X = pd.read_csv(os.path.join(S, "modeling_X_normal_scores.csv"), index_col=0)
Y = pd.read_csv(os.path.join(S, "modeling_Y_multi.csv"), index_col=0)
groups = pd.read_csv(SOURCE_GROUPS).set_index("city").group.reindex(Y.index)
full = nested(X.to_numpy(float), Y.to_numpy(float), max_comp=10, n_splits=5, random_state=42)[0]
assert np.allclose(full, pd.read_csv(os.path.join(S, "pls2_all_targets_r2_CV.csv"))["r2_cv_oof"].to_numpy()), "does not reproduce the notebook"
rows = []
for g in ("WebTRIS", "UTD19"):
    m = (groups == g).to_numpy()
    r2 = np.array([nested(X.to_numpy(float)[m], Y[NAMES].to_numpy(float)[m], max_comp=10, n_splits=5, random_state=s)[0][:7] for s in (42, 1, 2, 3, 4)]).mean(axis=0)
    rows.append(dict(source=g, n=int(m.sum()), **dict(zip(NAMES, r2))))
pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "diagnostics", "stats_by_source.csv"), index=False)
