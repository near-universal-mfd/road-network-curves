"""Calibration of equations (5) and (6) over all networks, with the network-level bootstrap, the jackknife, profiles and cross-validation.

With x = K / K_c, the model predicts q = u_fe K_c f(x; alpha), where
    f_exp(x) = d x exp(-x)
    f_cub(x) = x + a x^2 - (1 + 2a) / 3 x^3
    S        = f_exp(1) / f_cub(1) = 3 d exp(-1) / (2 + a)
    beta(x)  = 1 / (1 + exp((x - 1) / s_K))
    alpha    = 1 / (1 + exp((UI - u_th) / s_u))
    gamma    = alpha beta(x)
    f(x)     = gamma f_exp + (1 - gamma) (alpha S + 1 - alpha) f_cub
s_u, s_K and u_th are held fixed; a, d and the constants of the index are fitted. The commands behind the paper are in run_all.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import MFD_DIR, NODES_DIR, OUTPUTS_DIR  # noqa: E402

EPS = 1e-9

A_BOUNDS  = (0.01, 0.75)
A_CUBIC_BOUNDS = (-0.50, 0.90)
SU_BOUNDS = (0.005, 5.00)
SK_BOUNDS = (0.005, 0.17)
D_BOUNDS  = (0.20, 2.50)
B_BOUNDS  = (-2.0, 2.0)
W_BOUNDS  = (-3.0, 3.0)

U_TH_FIXED = 0.0

FIXED_PARAMS: Dict[str, float] = {"s_u": 0.2344, "s_K": 0.1343}

START_THETA = np.array([0.3413, 0.2344, 0.1343, 1.2485, 1.8193,
                            1.42980065, 0.50479542, -0.5957572, -0.05015428])


def set_fixed_params(fixed: Optional[Dict[str, float]]) -> None:
    """Replace the set of held-constant parameters (names from HYBRID_NAMES)."""
    global FIXED_PARAMS
    FIXED_PARAMS = dict(fixed or {})
    for nm in FIXED_PARAMS:
        if nm not in HYBRID_NAMES:
            raise SystemExit(f"Cannot fix unknown parameter '{nm}'. "
                             f"Valid names: {', '.join(HYBRID_NAMES)}")


def fixed_indices() -> List[int]:
    return [HYBRID_NAMES.index(n) for n in FIXED_PARAMS]


def apply_fixed(theta: np.ndarray) -> np.ndarray:
    """Overwrite held-constant entries; their optimizer dimensions go inert."""
    t = np.array(theta, float)
    for nm, val in FIXED_PARAMS.items():
        t[HYBRID_NAMES.index(nm)] = float(val)
    return t


def _set_concavity_floor(x_min: float) -> None:
    """Lower edge of the concavity screen, in units of K_c."""
    global CONCAVITY_X_MIN
    CONCAVITY_X_MIN = float(x_min)


A1_FIXED   = 1.0

N_CONCAVITY_SAMPLES = 361
CONCAVITY_PENALTY   = 0.5
CONCAVITY_X_MIN     = 0.30
CONCAVITY_X_MAX     = 1.80

KMAX_FACTOR_DEFAULT = 1.1

METRIC_NAMES = ["uf", "avg_link_length", "avg_speed_limit", "node_connectivity"]

HYBRID_NAMES  = ["a", "s_u", "s_K", "d", "b", "w_uf", "w_L", "w_SL", "w_C"]
HYBRID_BOUNDS = [A_BOUNDS, SU_BOUNDS, SK_BOUNDS, D_BOUNDS,
                 B_BOUNDS, W_BOUNDS, W_BOUNDS, W_BOUNDS, W_BOUNDS]


def set_bounds(a=None, s_u=None, s_K=None, d=None, b=None, w=None,
               a_cubic=None) -> None:
    """Override the optimizer bounds. Worker processes receive them explicitly, because spawned processes don't inherit globals."""
    global A_BOUNDS, SU_BOUNDS, SK_BOUNDS, D_BOUNDS, B_BOUNDS, W_BOUNDS
    global A_CUBIC_BOUNDS, HYBRID_BOUNDS
    if a is not None:       A_BOUNDS = tuple(a)
    if s_u is not None:     SU_BOUNDS = tuple(s_u)
    if s_K is not None:     SK_BOUNDS = tuple(s_K)
    if d is not None:       D_BOUNDS = tuple(d)
    if b is not None:       B_BOUNDS = tuple(b)
    if w is not None:       W_BOUNDS = tuple(w)
    if a_cubic is not None: A_CUBIC_BOUNDS = tuple(a_cubic)
    HYBRID_BOUNDS = [A_BOUNDS, SU_BOUNDS, SK_BOUNDS, D_BOUNDS,
                     B_BOUNDS, W_BOUNDS, W_BOUNDS, W_BOUNDS, W_BOUNDS]


def current_bounds() -> Dict[str, object]:
    d = {"a": A_BOUNDS, "s_u": SU_BOUNDS, "s_K": SK_BOUNDS, "d": D_BOUNDS,
         "b": B_BOUNDS, "w": W_BOUNDS, "a_cubic": A_CUBIC_BOUNDS}
    d["_fixed"] = dict(FIXED_PARAMS)
    d["_cxmin"] = CONCAVITY_X_MIN
    return d


def log(msg: str = "") -> None:
    """Print immediately, never buffered."""
    print(msg, flush=True)


def fmt_duration(seconds: float) -> str:
    """Human-readable duration: 45s / 12m 30s / 2h 05m."""
    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


class Progress:
    """Live counter with elapsed time, throughput and ETA."""

    def __init__(self, total: int, label: str, every: int = 1,
                 min_interval: float = 0.0):
        self.total = max(int(total), 1)
        self.label = label
        self.every = max(int(every), 1)
        self.min_interval = float(min_interval)
        self.n = 0
        self.t0 = time.time()
        self._last_print = 0.0
        log(f"[{self.label}] starting: {self.total} task(s)")

    def update(self, n: int = 1, extra: str = "") -> None:
        self.n += n
        now = time.time()
        due = (self.n % self.every == 0) or (self.n >= self.total)
        if not due or (now - self._last_print) < self.min_interval:
            if self.n < self.total:
                return
        self._last_print = now
        el = now - self.t0
        rate = el / max(self.n, 1)
        eta = rate * (self.total - self.n)
        pct = 100.0 * self.n / self.total
        line = (f"[{self.label}] {self.n:>5d}/{self.total} {pct:5.1f}% | "
                f"elapsed {fmt_duration(el)} | {rate:6.2f}s/task | "
                f"ETA {fmt_duration(eta)}")
        if extra:
            line += f" | {extra}"
        log(line)

    def close(self) -> None:
        el = time.time() - self.t0
        log(f"[{self.label}] finished {self.n}/{self.total} in "
            f"{fmt_duration(el)} ({el / max(self.n, 1):.2f}s/task)")


class Checkpoint:
    """Append each finished replicate to disk, so that an interrupted run can resume."""

    def __init__(self, path: str, resume: bool = False):
        self.path = path
        self.columns: Optional[List[str]] = None
        self.done: List[Dict[str, float]] = []
        if resume and os.path.exists(path):
            # a run stopped mid-write can leave a short last line: keep the complete rows
            with open(path, encoding="utf-8") as fh:
                lines = [ln.rstrip("\n") for ln in fh.readlines() if ln.endswith("\n") and ln.strip()]
            if lines:
                self.columns = lines[0].split(",")
                good = [ln for ln in lines[1:] if len(ln.split(",")) == len(self.columns)]
                for ln in good:
                    self.done.append({c: (float(v) if v != "" else np.nan)
                                      for c, v in zip(self.columns, ln.split(","))})
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("\n".join([lines[0]] + good) + "\n")
        elif os.path.exists(path):
            os.remove(path)

    def append(self, rec: Dict[str, float]) -> None:
        try:
            if self.columns is None:
                self.columns = list(rec.keys())
                with open(self.path, "w", encoding="utf-8") as fh:
                    fh.write(",".join(self.columns) + "\n")
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(",".join(
                    "" if rec.get(c) is None else str(rec.get(c, ""))
                    for c in self.columns) + "\n")
        except Exception as exc:
            log(f"[WARN] checkpoint write failed: {exc}")


@dataclass
class Prepared:
    """Everything a refit needs, precomputed once."""
    cities: List[str]
    uf: np.ndarray            # (N,) effective free-flow speed
    kc: np.ndarray            # (N,) critical density
    qscale: np.ndarray        # (N,) qc, the NRMSE normalizer
    metrics: np.ndarray       # (N,4) [uf, L, S_L, C]
    c_scale: np.ndarray       # (N,) uf*kc/qscale - converts f(x) to normalized q
    bin_city: np.ndarray      # (M,) city index for each occupied bin
    bin_x: np.ndarray         # (M,) mean x within the bin
    bin_n: np.ndarray         # (M,) observation count in the bin
    bin_sy: np.ndarray        # (M,) sum of y = q_obs/qscale in the bin
    bin_m2: np.ndarray        # (M,) sum of (x - xbar)^2 within the bin
    bin_cy: np.ndarray        # (M,) sum of y*(x - xbar) within the bin
    n_obs: np.ndarray         # (N,) observations per city
    A_c: np.ndarray           # (N,) mean(y^2) per city (constant term of MSE)
    exact_city: Optional[np.ndarray] = None
    exact_x: Optional[np.ndarray] = None
    exact_y: Optional[np.ndarray] = None

    @property
    def n_cities(self) -> int:
        return len(self.cities)


def read_outputs_for_city(outputs_dir: str, city: str):
    """outputs_{city}.txt -> (qc, kc, q_kc_6, q_kc_3, q_kc_2, q_2kc_3, q_5kc_6)."""
    path = os.path.join(outputs_dir, f"outputs_{city}.txt")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as fh:
            parts = [p.strip() for p in fh.read().strip().split(",")]
        if len(parts) != 7:
            return None
        return tuple(map(float, parts))
    except Exception:
        return None


# Three pairs of networks carry each other's map names: the map of the curve "london" (UTD19) is London2, and so on.
MAP_NAME = {"london": "London2", "London2": "London", "birmingham": "Birmingham2", "Birmingham2": "Birmingham",
            "manchester": "Manchester2", "Manchester2": "Manchester"}


_MAP_TABLES: Dict[str, pd.DataFrame] = {}


def _map_table(path: str) -> pd.DataFrame:
    if path not in _MAP_TABLES:
        _MAP_TABLES[path] = pd.read_csv(path).set_index("city")
    return _MAP_TABLES[path]


def compute_city_metrics(city: str, outputs_dir: str, nodes_base_dir: str,
                         quiet: bool = False):
    """Return (uf, avg_link_length, avg_speed_limit, node_connectivity, qc, kc)."""
    def warn(msg):
        if not quiet:
            log(f"[WARN] {msg}")

    outs = read_outputs_for_city(outputs_dir, city)
    if outs is None:
        warn(f"Missing/bad outputs for '{city}'.")
        return None

    qc, kc_raw, q_kc6, _q_kc3, _q_kc2, _q_2kc3, _q_5kc6 = outs
    if not (kc_raw and np.isfinite(kc_raw) and kc_raw > 0):
        warn(f"kc invalid for '{city}' (kc={kc_raw}).")
        return None

    uf = 6.0 * q_kc6 / kc_raw
    if not (np.isfinite(uf) and uf > 0):
        warn(f"uf invalid for '{city}' (uf={uf}).")
        return None

    if os.path.isfile(nodes_base_dir):
        t = _map_table(nodes_base_dir)
        if city not in t.index:
            warn(f"'{city}' is not in {nodes_base_dir}")
            return None
        r = t.loc[city]
        return uf, float(r.avg_link_length), float(r.avg_speed_limit), float(r.node_connectivity), qc, kc_raw

    links_fp = os.path.join(nodes_base_dir, f"links_{MAP_NAME.get(city, city)}.csv")
    avg_link_length = float("nan")
    avg_speed_limit = float("nan")
    if os.path.exists(links_fp):
        try:
            links_df = pd.read_csv(links_fp)
            if "LinkLength" in links_df.columns:
                v = pd.to_numeric(links_df["LinkLength"], errors="coerce")
                avg_link_length = float(v.mean()) if v.notna().any() else float("nan")
            else:
                warn(f"'LinkLength' missing for '{city}'.")
            if "SpeedLimit" in links_df.columns:
                v = pd.to_numeric(links_df["SpeedLimit"], errors="coerce")
                avg_speed_limit = float(v.mean()) if v.notna().any() else float("nan")
            else:
                warn(f"'SpeedLimit' missing for '{city}'.")
        except Exception as exc:
            warn(f"Cannot read links file for '{city}': {exc}")
    else:
        warn(f"links file not found for '{city}': {links_fp}")

    nodes_fp = os.path.join(nodes_base_dir, f"nodes_{MAP_NAME.get(city, city)}.csv")
    node_connectivity = float("nan")
    if os.path.exists(nodes_fp):
        try:
            nodes_df = pd.read_csv(nodes_fp)
            if "InflowCount" in nodes_df.columns and "OutflowCount" in nodes_df.columns:
                inflow = pd.to_numeric(nodes_df["InflowCount"], errors="coerce")
                outflow = pd.to_numeric(nodes_df["OutflowCount"], errors="coerce")
                avg_in = float(inflow.mean()) if inflow.notna().any() else float("nan")
                avg_out = float(outflow.mean()) if outflow.notna().any() else float("nan")
                if np.isnan(avg_in) and np.isnan(avg_out):
                    node_connectivity = float("nan")
                elif np.isnan(avg_in):
                    node_connectivity = avg_out
                elif np.isnan(avg_out):
                    node_connectivity = avg_in
                else:
                    node_connectivity = 0.5 * (avg_in + avg_out)
            else:
                warn(f"'InflowCount'/'OutflowCount' missing for '{city}'.")
        except Exception as exc:
            warn(f"Cannot read nodes file for '{city}': {exc}")
    else:
        warn(f"nodes file not found for '{city}': {nodes_fp}")

    vec = np.array([uf, avg_link_length, avg_speed_limit, node_connectivity], float)
    if np.any(~np.isfinite(vec)):
        warn(f"Some metrics NaN/inf for '{city}'.")
        return None

    return uf, avg_link_length, avg_speed_limit, node_connectivity, qc, kc_raw


def build_prepared(mfd_dir: str, outputs_dir: str, nodes_base_dir: str,
                   nbins: int = 400, kmax_factor: float = KMAX_FACTOR_DEFAULT,
                   keep_exact: bool = True, quiet: bool = False) -> Prepared:
    """Load every network and pre-bin its observations in normalized coordinates."""
    pattern = os.path.join(mfd_dir, "data_*_grouped.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        raise SystemExit(f"No data_*_grouped.csv files found at: {pattern}")

    cities, uf_l, kc_l, qs_l, met_l = [], [], [], [], []
    xs_l, ys_l, cid_l = [], [], []

    if not quiet:
        prog_load = Progress(len(csv_files), "LOAD",
                             every=max(1, len(csv_files) // 20))
    for data_fp in csv_files:
        cityfile = os.path.basename(data_fp)
        low = cityfile.lower()
        if low.startswith("data_") and low.endswith("_grouped.csv"):
            city = cityfile[5:-12]
        else:
            city = cityfile.replace("data_", "").replace("_grouped.csv", "")

        got = compute_city_metrics(city, outputs_dir, nodes_base_dir, quiet=quiet)
        if got is None:
            if not quiet:
                prog_load.update(1, f"kept {len(cities)}")
            continue
        uf, L, SL, C, qc, kc = got
        if not np.isfinite(kc) or kc <= 0:
            if not quiet:
                prog_load.update(1, f"kept {len(cities)}")
            continue

        try:
            df = pd.read_csv(data_fp)
        except Exception as exc:
            if not quiet:
                log(f"[WARN] Cannot read {data_fp}: {exc}. Skipping.")
            continue
        if "density" not in df.columns or "flow" not in df.columns:
            if not quiet:
                log(f"[WARN] density/flow columns missing in {data_fp}. Skipping.")
            continue

        dens = pd.to_numeric(df["density"], errors="coerce")
        flow = pd.to_numeric(df["flow"], errors="coerce")
        mask = (dens.ge(0) & dens.le(kmax_factor * kc) & flow.ge(0)
                & dens.replace([np.inf, -np.inf], np.nan).notna()
                & flow.replace([np.inf, -np.inf], np.nan).notna())
        clean = pd.DataFrame({"density": dens[mask], "flow": flow[mask]}).dropna()
        if clean.empty:
            if not quiet:
                log(f"[WARN] No valid rows for '{city}'. Skipping.")
            continue

        k_obs = clean["density"].to_numpy(float)
        q_obs = clean["flow"].to_numpy(float)

        q_scale = qc if (np.isfinite(qc) and qc > 0) else float(np.nanmax(q_obs))
        if not (np.isfinite(q_scale) and q_scale > 0):
            q_scale = 1.0

        if not quiet:
            prog_load.update(1, f"kept {len(cities) + 1}")
        idx = len(cities)
        cities.append(city)
        uf_l.append(uf); kc_l.append(kc); qs_l.append(q_scale)
        met_l.append([uf, L, SL, C])
        xs_l.append(k_obs / kc)
        ys_l.append(q_obs / q_scale)
        cid_l.append(np.full(k_obs.size, idx, dtype=np.int64))

    if not quiet:
        prog_load.n = len(csv_files)
        prog_load.close()
    if not cities:
        raise SystemExit("No valid datasets to calibrate on.")

    uf_a = np.asarray(uf_l, float)
    kc_a = np.asarray(kc_l, float)
    qs_a = np.asarray(qs_l, float)
    met_a = np.asarray(met_l, float)

    x_all = np.concatenate(xs_l)
    y_all = np.concatenate(ys_l)
    c_all = np.concatenate(cid_l)
    n_cities = len(cities)

    n_obs = np.bincount(c_all, minlength=n_cities).astype(float)
    A_c = np.bincount(c_all, weights=y_all ** 2, minlength=n_cities) / np.maximum(n_obs, 1)

    # Per bin: the count, mean x, sum y, and the centred sums of (x - xbar)^2 and y (x - xbar), so that the
    # objective keeps the first-order variation of f within each bin (check_binning_accuracy).
    x_hi = max(float(np.nanmax(x_all)), kmax_factor) * (1.0 + 1e-9)
    edges = np.linspace(0.0, x_hi, nbins + 1)
    b_idx = np.clip(np.digitize(x_all, edges) - 1, 0, nbins - 1)
    flat = c_all * nbins + b_idx
    order_len = n_cities * nbins

    cnt = np.bincount(flat, minlength=order_len)
    sx = np.bincount(flat, weights=x_all, minlength=order_len)
    sxx = np.bincount(flat, weights=x_all ** 2, minlength=order_len)
    sy = np.bincount(flat, weights=y_all, minlength=order_len)
    sxy = np.bincount(flat, weights=x_all * y_all, minlength=order_len)

    occ = cnt > 0
    bin_city = (np.nonzero(occ)[0] // nbins).astype(np.int64)
    bin_n = cnt[occ].astype(float)
    bin_x = sx[occ] / bin_n
    bin_sy = sy[occ]
    bin_m2 = np.maximum(sxx[occ] - bin_n * bin_x ** 2, 0.0)
    bin_cy = sxy[occ] - bin_x * bin_sy

    prep = Prepared(
        cities=cities, uf=uf_a, kc=kc_a, qscale=qs_a, metrics=met_a,
        c_scale=uf_a * kc_a / qs_a,
        bin_city=bin_city, bin_x=bin_x, bin_n=bin_n, bin_sy=bin_sy,
        bin_m2=bin_m2, bin_cy=bin_cy, n_obs=n_obs, A_c=A_c,
    )
    if keep_exact:
        prep.exact_city, prep.exact_x, prep.exact_y = c_all, x_all, y_all

    if not quiet:
        log(f"[INFO] Loaded {n_cities} networks, {int(n_obs.sum()):,} observations, "
              f"{bin_n.size:,} occupied bins.")
    return prep


def _dec_sigmoid(z: np.ndarray) -> np.ndarray:
    """Decreasing logistic 1 / (1 + exp(z)), clipped."""
    return 1.0 / (1.0 + np.exp(np.clip(z, -60.0, 60.0)))


def tail_scale_S(a: float, d: float, a1: float = A1_FIXED) -> float:
    """S = f_exp(1)/f_cub(1) = 3*d*exp(-a1)/(2+a): matches the branches at x=1."""
    return 3.0 * d * np.exp(-a1) / (2.0 + a)


def f_model(x: np.ndarray, alpha: np.ndarray, a: float, d: float, s_K: float,
            a1: float = A1_FIXED) -> np.ndarray:
    """Equation (5) in normalized form: q = u_fe K_c f_model(x, alpha)."""
    x = np.asarray(x, float)
    f_exp = d * x * np.exp(-a1 * x)
    f_cub = x + a * x ** 2 - (1.0 + 2.0 * a) / 3.0 * x ** 3
    beta = _dec_sigmoid((x - 1.0) / max(EPS, s_K))
    S = tail_scale_S(a, d, a1)
    gamma = alpha * beta
    cub_choice = alpha * S * f_cub + (1.0 - alpha) * f_cub
    return gamma * f_exp + (1.0 - gamma) * cub_choice


def f_model_and_deriv(x: np.ndarray, alpha: np.ndarray, a: float, d: float,
                      s_K: float, a1: float = A1_FIXED
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (f, df/dx). The derivative feeds the binning correction."""
    x = np.asarray(x, float)
    e = np.exp(-a1 * x)
    f_exp = d * x * e
    df_exp = d * e * (1.0 - a1 * x)

    f_cub = x + a * x ** 2 - (1.0 + 2.0 * a) / 3.0 * x ** 3
    df_cub = 1.0 + 2.0 * a * x - (1.0 + 2.0 * a) * x ** 2

    sk = max(EPS, s_K)
    beta = _dec_sigmoid((x - 1.0) / sk)
    dbeta = -beta * (1.0 - beta) / sk

    S = tail_scale_S(a, d, a1)
    gamma = alpha * beta
    dgamma = alpha * dbeta

    mix = 1.0 + alpha * (S - 1.0)
    g, dg = mix * f_cub, mix * df_cub

    f = gamma * f_exp + (1.0 - gamma) * g
    df = dgamma * (f_exp - g) + gamma * df_exp + (1.0 - gamma) * dg
    return f, df


def f_cubic_only(x: np.ndarray, a: float) -> np.ndarray:
    """Eq. (2) in normalized form: q = u_fe * K_c * (x + a x^2 - (1+2a)/3 x^3)."""
    x = np.asarray(x, float)
    return x + a * x ** 2 - (1.0 + 2.0 * a) / 3.0 * x ** 3


def df_cubic_only(x: np.ndarray, a: float) -> np.ndarray:
    x = np.asarray(x, float)
    return 1.0 + 2.0 * a * x - (1.0 + 2.0 * a) * x ** 2


def weighted_moments(M: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted column means and (population) standard deviations."""
    w = np.asarray(w, float)
    wsum = w.sum()
    mean = (M * w[:, None]).sum(axis=0) / wsum
    var = (w[:, None] * (M - mean) ** 2).sum(axis=0) / wsum
    std = np.sqrt(np.maximum(var, 0.0))
    return mean, np.where(std > 0, std, 1.0)


def build_ui_features(metrics: np.ndarray, w_city: np.ndarray):
    """The features of equation (6) with replicate weights: Z0 = clip((x - mean) / std, -2, 2), phi = exp(Z0), Z = (phi - phi_mean) / phi_std. Returns (Z, moments)."""
    m_mean, m_std = weighted_moments(metrics, w_city)
    Z0 = np.clip((metrics - m_mean) / m_std, -2.0, 2.0)
    Phi = np.exp(Z0)
    p_mean, p_std = weighted_moments(Phi, w_city)
    Z = (Phi - p_mean) / p_std
    return Z, {"metric_mean": m_mean, "metric_std": m_std,
               "phi_mean": p_mean, "phi_std": p_std}


def ui_from_Z(Z: np.ndarray, b: float, w: np.ndarray) -> np.ndarray:
    return b + Z @ np.asarray(w, float)


def _nrmse_per_city_binned(prep: Prepared, alpha: np.ndarray, a: float, d: float,
                           s_K: float, a1: float = A1_FIXED) -> np.ndarray:
    """Per-network NRMSE from the binned sums."""
    n = prep.n_cities
    cid = prep.bin_city
    fb, dfb = f_model_and_deriv(prep.bin_x, alpha[cid], a, d, s_K, a1)
    B = np.bincount(cid, weights=fb * prep.bin_sy + dfb * prep.bin_cy,
                    minlength=n) / prep.n_obs
    C = np.bincount(cid, weights=fb ** 2 * prep.bin_n + dfb ** 2 * prep.bin_m2,
                    minlength=n) / prep.n_obs
    c = prep.c_scale
    mse = prep.A_c - 2.0 * c * B + c ** 2 * C
    return np.sqrt(np.maximum(mse, 0.0))


def _nrmse_per_city_exact(prep: Prepared, alpha: np.ndarray, a: float, d: float,
                          s_K: float, a1: float = A1_FIXED) -> np.ndarray:
    """Per-network NRMSE from every observation, for the binning check."""
    if prep.exact_x is None:
        raise RuntimeError("Exact arrays were not retained (keep_exact=False).")
    n = prep.n_cities
    cid = prep.exact_city
    f = f_model(prep.exact_x, alpha[cid], a, d, s_K, a1)
    resid = prep.exact_y - prep.c_scale[cid] * f
    mse = np.bincount(cid, weights=resid ** 2, minlength=n) / prep.n_obs
    return np.sqrt(np.maximum(mse, 0.0))


def _concavity_violation(alpha: np.ndarray, a: float, d: float, s_K: float,
                         active: np.ndarray, a1: float = A1_FIXED) -> bool:
    """True if any network's curve is non-concave on [0.3 K_c, 1.8 K_c]."""
    xs = np.linspace(CONCAVITY_X_MIN, CONCAVITY_X_MAX, N_CONCAVITY_SAMPLES)
    al = alpha[active][:, None]
    F = f_model(xs[None, :], al, a, d, s_K, a1)
    d2 = F[:, 2:] - 2.0 * F[:, 1:-1] + F[:, :-2]
    tol = 1e-8 * np.maximum(np.nanmax(np.abs(F), axis=1, keepdims=True), 1e-12)
    return bool(np.any(d2 > tol))


def hybrid_objective(theta: Sequence[float], prep: Prepared, w_city: np.ndarray,
                     Z: np.ndarray, binned: bool = True,
                     penalty: bool = True) -> float:
    """Mean NRMSE over networks, weighted by w_city (the bootstrap multiplicities), plus the concavity penalty."""
    theta = apply_fixed(theta)
    a, s_u, s_K, d, b = theta[0], theta[1], theta[2], theta[3], theta[4]
    w = np.asarray(theta[5:9], float)

    _fixed = set(fixed_indices())
    for j, (val, (lo, hi)) in enumerate(zip(theta, HYBRID_BOUNDS)):
        if j in _fixed:
            continue
        if not (lo <= val <= hi):
            return 1e9

    ui = ui_from_Z(Z, b, w)
    alpha = _dec_sigmoid((ui - U_TH_FIXED) / max(EPS, s_u))

    per_city = (_nrmse_per_city_binned if binned else _nrmse_per_city_exact)(
        prep, alpha, a, d, s_K)
    if not np.all(np.isfinite(per_city)):
        return 1e9

    val = float(np.average(per_city, weights=w_city))
    if penalty:
        active = np.nonzero(w_city > 0)[0]
        if active.size and _concavity_violation(alpha, a, d, s_K, active):
            val += CONCAVITY_PENALTY
    return val


def cubic_objective(theta: Sequence[float], prep: Prepared, w_city: np.ndarray,
                    binned: bool = True, penalty: bool = True) -> float:
    """Eq. (2): the same objective with a single free parameter a."""
    a = float(theta[0])
    if not (A_CUBIC_BOUNDS[0] <= a <= A_CUBIC_BOUNDS[1]):
        return 1e9
    n = prep.n_cities
    if binned:
        cid = prep.bin_city
        fb = f_cubic_only(prep.bin_x, a)
        dfb = df_cubic_only(prep.bin_x, a)
        B = np.bincount(cid, weights=fb * prep.bin_sy + dfb * prep.bin_cy,
                        minlength=n) / prep.n_obs
        C = np.bincount(cid, weights=fb ** 2 * prep.bin_n + dfb ** 2 * prep.bin_m2,
                        minlength=n) / prep.n_obs
        mse = prep.A_c - 2.0 * prep.c_scale * B + prep.c_scale ** 2 * C
    else:
        cid = prep.exact_city
        resid = prep.exact_y - prep.c_scale[cid] * f_cubic_only(prep.exact_x, a)
        mse = np.bincount(cid, weights=resid ** 2, minlength=n) / prep.n_obs
    per_city = np.sqrt(np.maximum(mse, 0.0))
    val = float(np.average(per_city, weights=w_city))

    if penalty:
        xs = np.linspace(CONCAVITY_X_MIN, CONCAVITY_X_MAX, N_CONCAVITY_SAMPLES)
        F = f_cubic_only(xs, a)
        d2 = F[2:] - 2.0 * F[1:-1] + F[:-2]
        if np.any(d2 > 1e-8 * max(np.max(np.abs(F)), 1e-12)):
            val += CONCAVITY_PENALTY
    return val


def _clip_to_bounds(theta: np.ndarray, bounds) -> np.ndarray:
    out = np.array(theta, float)
    for j, (lo, hi) in enumerate(bounds):
        out[j] = float(np.clip(out[j], lo, hi))
    return out


def fit_hybrid(prep: Prepared, w_city: np.ndarray, x0: Optional[np.ndarray] = None,
               restarts: int = 2, rng: Optional[np.random.Generator] = None,
               binned: bool = True, use_global: bool = False,
               de_maxiter: int = 120, de_popsize: int = 12,
               maxfev: int = 3000, verbose: bool = False
               ) -> Tuple[np.ndarray, float]:
    """Differential evolution for the full-sample fit (use_global), then local search from each start."""
    rng = rng or np.random.default_rng(0)
    Z, _ = build_ui_features(prep.metrics, w_city)

    def obj(t):
        return hybrid_objective(t, prep, w_city, Z, binned=binned)

    best_x, best_v = None, np.inf

    if use_global:
        log(f"[FIT] global search: differential evolution, up to {de_maxiter} "
            f"generations x popsize {de_popsize} over {len(HYBRID_BOUNDS)} "
            f"parameters. This is the slowest single step; it runs once.")
        t_de = time.time()
        state = {"it": 0, "last": 0.0}

        def de_callback(xk, convergence=None):
            state["it"] += 1
            now = time.time()
            el = now - t_de
            if state["it"] > 3 and (now - state["last"]) < 5.0:
                return
            state["last"] = now
            frac = min(max(float(convergence or 0.0), 1e-9), 1.0)
            log(f"[FIT] generation {state['it']:>4d}/{de_maxiter} | "
                f"elapsed {fmt_duration(el)} | convergence {frac:.2e} | "
                f"a={xk[0]:.4f} s_u={xk[1]:.4f} s_K={xk[2]:.4f} d={xk[3]:.4f}")

        res = differential_evolution(
            obj, bounds=HYBRID_BOUNDS, strategy="best1bin", maxiter=de_maxiter,
            popsize=de_popsize, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            seed=int(rng.integers(0, 2 ** 31 - 1)), polish=True,
            updating="deferred", workers=1, callback=de_callback)
        best_x, best_v = np.asarray(res.x, float), float(res.fun)
        log(f"[FIT] global search done in {fmt_duration(time.time() - t_de)}: "
            f"objective={best_v:.6f}")

    starts: List[np.ndarray] = []
    if x0 is not None:
        starts.append(_clip_to_bounds(np.asarray(x0, float), HYBRID_BOUNDS))
    if best_x is not None:
        starts.append(best_x.copy())
    for _ in range(max(0, restarts)):
        base = starts[0] if starts else np.array(
            [np.mean(b) for b in HYBRID_BOUNDS], float)
        span = np.array([hi - lo for lo, hi in HYBRID_BOUNDS], float)
        starts.append(_clip_to_bounds(base + rng.normal(0, 0.05, base.size) * span,
                                      HYBRID_BOUNDS))
    if use_global and best_x is not None:
        # b and w_uf have more than one basin, so also start from a few pairs
        for b0, w0 in ((1.0, 0.8), (1.5, 1.2), (2.0, 1.6), (2.5, 2.0), (4.0, 3.5), (5.5, 5.0)):
            s0 = best_x.copy()
            s0[4], s0[5] = b0, w0
            starts.append(_clip_to_bounds(s0, HYBRID_BOUNDS))

    for i_start, s in enumerate(starts, 1):
        t_s = time.time()
        r1 = minimize(obj, s, method="Nelder-Mead",
                      options={"maxfev": int(maxfev), "xatol": 1e-6,
                               "fatol": 1e-9, "adaptive": True})
        cand = _clip_to_bounds(r1.x, HYBRID_BOUNDS)
        v_nm = obj(cand)
        r2 = minimize(obj, cand, method="Powell", bounds=HYBRID_BOUNDS,
                      options={"maxiter": 40, "xtol": 1e-6, "ftol": 1e-9})
        cand_pw = _clip_to_bounds(r2.x, HYBRID_BOUNDS)
        v = obj(cand_pw)
        # a bounded Powell search can end in a worse basin: keep its polish only if it improves
        if v <= v_nm:
            cand = cand_pw
        else:
            v = v_nm
        if verbose:
            log(f"[FIT] local start {i_start}/{len(starts)}: objective={v:.6f} "
                f"({fmt_duration(time.time() - t_s)})"
                + ("  <- best so far" if v < best_v else ""))
        if v < best_v:
            best_x, best_v = cand, v

    if best_x is not None:
        best_x = apply_fixed(best_x)
    return best_x, float(best_v)


def fit_cubic(prep: Prepared, w_city: np.ndarray, binned: bool = True,
              verbose: bool = False) -> Tuple[float, float]:
    """One-dimensional fit of equation (2): a dense scan, then Nelder-Mead."""
    def obj(a):
        return cubic_objective([a], prep, w_city, binned=binned)

    if verbose:
        log("[FIT] cubic form (Eq. 2): scanning a over "
            f"[{A_CUBIC_BOUNDS[0]}, {A_CUBIC_BOUNDS[1]}] ...")
    grid = np.linspace(A_CUBIC_BOUNDS[0], A_CUBIC_BOUNDS[1], 300)
    vals = np.array([obj(g) for g in grid])
    a0 = float(grid[int(np.argmin(vals))])
    res = minimize(lambda t: obj(float(t[0])), [a0], method="Nelder-Mead",
                   options={"maxfev": 400, "xatol": 1e-8, "fatol": 1e-12})
    a_hat = float(np.clip(res.x[0], *A_CUBIC_BOUNDS))
    return a_hat, float(obj(a_hat))


def derived_quantities(a: float, d: float, s_K: float,
                       a1: float = A1_FIXED) -> Dict[str, float]:
    """S, the capacity coefficients, K_j / K_c and the backward wave speeds, computed from f_model."""
    out: Dict[str, float] = {"S": tail_scale_S(a, d, a1)}
    x = np.linspace(1e-6, 3.0, 300_001)

    for tag, alpha in (("sparse", 0.0), ("urban", 1.0)):
        f = f_model(x, np.full_like(x, alpha), a, d, s_K, a1)
        i_pk = int(np.argmax(f))
        out[f"cap_{tag}"] = float(f[i_pk])
        out[f"x_cap_{tag}"] = float(x[i_pk])

        # First zero crossing beyond the peak -> jam density
        tail = f[i_pk:]
        neg = np.nonzero(tail <= 0.0)[0]
        if neg.size:
            j = i_pk + int(neg[0])
            x0, x1 = x[j - 1], x[j]
            f0, f1 = f[j - 1], f[j]
            xj = float(x0 - f0 * (x1 - x0) / (f1 - f0)) if f1 != f0 else float(x1)
            out[f"x_jam_{tag}"] = xj
            h = 1e-5
            fp = f_model(np.array([xj + h]), np.array([alpha]), a, d, s_K, a1)[0]
            fm = f_model(np.array([xj - h]), np.array([alpha]), a, d, s_K, a1)[0]
            out[f"wave_{tag}"] = float((fp - fm) / (2 * h))
        else:
            out[f"x_jam_{tag}"] = float("nan")
            out[f"wave_{tag}"] = float("nan")

    disc = a ** 2 + 4.0 * (1.0 + 2.0 * a) / 3.0
    x_jam_cf = (a + np.sqrt(disc)) / (2.0 * (1.0 + 2.0 * a) / 3.0)
    out["x_jam_closed_form"] = float(x_jam_cf)
    out["cap_sparse_closed_form"] = float(1.0 + a - (1.0 + 2.0 * a) / 3.0)
    return out


def canonicalize(theta: np.ndarray) -> np.ndarray:
    """(b, w, s_u) rescaled so that ||w|| = 1. The objective is invariant to this scale."""
    t = np.array(theta, float)
    nw = float(np.linalg.norm(t[5:9]))
    if nw > EPS:
        t[1] /= nw          # s_u
        t[4] /= nw          # b
        t[5:9] /= nw        # w
    return t


def ui_summary(prep: Prepared, theta: np.ndarray, w_city: np.ndarray) -> Dict[str, float]:
    """UI-scale diagnostics: dispersion, the scale-free s_u, and class shares."""
    Z, mom = build_ui_features(prep.metrics, w_city)
    ui = ui_from_Z(Z, theta[4], theta[5:9])
    sd_ui = float(np.sqrt(np.average((ui - np.average(ui, weights=w_city)) ** 2,
                                     weights=w_city)))
    urbanized = ui <= U_TH_FIXED
    can = canonicalize(theta)
    return {
        "sd_UI": sd_ui,
        "can_s_u": float(can[1]), "can_b": float(can[4]),
        "can_w_uf": float(can[5]), "can_w_L": float(can[6]),
        "can_w_SL": float(can[7]), "can_w_C": float(can[8]),
        "w_norm": float(np.linalg.norm(np.asarray(theta[5:9], float))),
        "s_u_rel": float(theta[1] / sd_ui) if sd_ui > 0 else float("nan"),
        "frac_urbanized": float(np.average(urbanized.astype(float), weights=w_city)),
        "_ui": ui,
        "_moments": mom,
    }


_W: Dict[str, object] = {}


def _init_worker(prep: Prepared, x0_hybrid: np.ndarray, a0_cubic: float,
                 binned: bool, restarts: int, maxfev: int = 3000,
                 bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 ref_class: Optional[np.ndarray] = None) -> None:
    if bounds:
        bounds = dict(bounds)
        set_fixed_params(bounds.pop("_fixed", None))
        _set_concavity_floor(bounds.pop("_cxmin", CONCAVITY_X_MIN))
        set_bounds(**{k: v for k, v in bounds.items() if k != "w"},
                   w=bounds.get("w"))
    _W["ref_class"] = ref_class
    _W["prep"] = prep
    _W["x0"] = np.asarray(x0_hybrid, float)
    _W["a0"] = float(a0_cubic)
    _W["binned"] = bool(binned)
    _W["restarts"] = int(restarts)
    _W["maxfev"] = int(maxfev)


def _one_replicate(job: Tuple[int, np.ndarray]) -> Optional[Dict[str, float]]:
    """Refit both models on one weighted resample of the networks."""
    rep_id, w_city = job
    prep: Prepared = _W["prep"]           # type: ignore[assignment]
    rng = np.random.default_rng(100_000 + rep_id)
    try:
        theta, obj = fit_hybrid(prep, w_city, x0=_W["x0"], restarts=_W["restarts"],
                                rng=rng, binned=_W["binned"], use_global=False,
                                maxfev=int(_W.get("maxfev", 3000)))
        a_cub, obj_cub = fit_cubic(prep, w_city, binned=_W["binned"])
    except Exception as exc:
        log(f"[WARN] replicate {rep_id} failed: {exc}")
        return None

    rec: Dict[str, float] = {"replicate": rep_id, "objective": obj,
                             "n_distinct": int(np.count_nonzero(w_city))}
    rec.update({n: float(v) for n, v in zip(HYBRID_NAMES, theta)})
    rec.update({f"dq_{k}": v for k, v in
                derived_quantities(theta[0], theta[3], theta[2]).items()})
    diag = ui_summary(prep, theta, w_city)
    for k in ("sd_UI", "can_s_u", "can_b", "can_w_uf", "can_w_L",
              "can_w_SL", "can_w_C", "w_norm"):
        rec[k] = diag[k]
    rec["s_u_rel"] = diag["s_u_rel"]
    rec["frac_urbanized"] = diag["frac_urbanized"]
    ref = _W.get("ref_class")
    if ref is not None:
        rec["class_agreement"] = float(np.mean(
            (diag["_ui"] <= U_TH_FIXED) == np.asarray(ref, bool)))
    rec["a_cubic"] = float(a_cub)
    rec["objective_cubic"] = float(obj_cub)
    return rec


def _running_extra(results: List[Dict[str, float]],
                   keys: Sequence[str] = ("a", "d")) -> str:
    """Running medians for the progress line."""
    if len(results) < 3:
        return ""
    bits = []
    for k in keys:
        vals = np.array([r[k] for r in results if k in r and np.isfinite(r[k])])
        if vals.size >= 3:
            bits.append(f"{k}~{np.median(vals):.4f}")
    return " ".join(bits)


def _run_jobs(jobs, prep, x0, a0, binned, restarts, workers, label, maxfev=3000,
              every: int = 1, checkpoint_path: Optional[str] = None,
              bounds: Optional[Dict] = None,
              ref_class: Optional[np.ndarray] = None,
              worker_fn=None, running_keys=("a", "d"), resume: bool = False):
    """Run replicate jobs serially or in a process pool, checkpointing each result. With resume, the records already saved are kept."""
    results: List[Dict[str, float]] = []
    n_failed = 0
    ckpt = Checkpoint(checkpoint_path, resume=resume) if checkpoint_path else None
    if ckpt and ckpt.done:
        done_ids = {int(r["replicate"]) for r in ckpt.done}
        results.extend(ckpt.done)
        jobs = [j for j in jobs if int(j[0]) not in done_ids]
        log(f"[{label}] resuming: {len(done_ids)} done in {os.path.basename(checkpoint_path)}, "
            f"{len(jobs)} to go")
    prog = Progress(len(jobs), label, every=every)
    if ckpt:
        log(f"[{label}] streaming results to {os.path.basename(checkpoint_path)}")

    def handle(rec):
        nonlocal n_failed
        if rec is None:
            n_failed += 1
        else:
            results.append(rec)
            if ckpt:
                ckpt.append({k: v for k, v in rec.items()
                             if not k.startswith("_")})
        prog.update(1, _running_extra(results, running_keys))

    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        log(f"[{label}] using {workers} worker processes")
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(prep, x0, a0, binned, restarts,
                                           maxfev, bounds, ref_class)) as ex:
            futures = [ex.submit(worker_fn or _one_replicate, job)
                       for job in jobs]
            for fut in as_completed(futures):
                try:
                    handle(fut.result())
                except Exception as exc:
                    log(f"[WARN] {label} task raised: {exc}")
                    handle(None)
    else:
        _init_worker(prep, x0, a0, binned, restarts, maxfev, bounds, ref_class)
        fn = worker_fn or _one_replicate
        for job in jobs:
            handle(fn(job))

    prog.close()
    if n_failed:
        log(f"[{label}] {n_failed} task(s) failed and were dropped "
            f"({len(results)} usable).")
    return pd.DataFrame(results)


def run_bootstrap(prep: Prepared, x0: np.ndarray, a0: float, n_boot: int,
                  seed: int, workers: int, binned: bool, restarts: int,
                  maxfev: int = 3000, every: int = 1,
                  checkpoint_path: Optional[str] = None,
                  bounds: Optional[Dict] = None,
                  ref_class: Optional[np.ndarray] = None,
                  resume: bool = False) -> pd.DataFrame:
    """Nonparametric cluster bootstrap: resample the N networks with replacement."""
    rng = np.random.default_rng(seed)
    n = prep.n_cities
    jobs = []
    for r in range(n_boot):
        idx = rng.integers(0, n, size=n)
        w = np.bincount(idx, minlength=n).astype(float)
        jobs.append((r, w))
    log(f"[BOOT] {n_boot} replicates over {n} networks "
        f"(mean distinct per replicate ~ {n * (1 - (1 - 1/n) ** n):.0f})")
    return _run_jobs(jobs, prep, x0, a0, binned, restarts, workers, "BOOT",
                     maxfev, every, checkpoint_path, bounds, ref_class, resume=resume)


def run_jackknife(prep: Prepared, x0: np.ndarray, a0: float, workers: int,
                  binned: bool, restarts: int, maxfev: int = 3000,
                  every: int = 1, checkpoint_path: Optional[str] = None,
                  bounds: Optional[Dict] = None,
                  ref_class: Optional[np.ndarray] = None,
                  resume: bool = False) -> pd.DataFrame:
    """Leave-one-network-out: N refits. Gives BCa acceleration and influence."""
    n = prep.n_cities
    jobs = []
    for i in range(n):
        w = np.ones(n, float)
        w[i] = 0.0
        jobs.append((i, w))
    log(f"[JACK] {n} leave-one-network-out refits")
    df = _run_jobs(jobs, prep, x0, a0, binned, restarts, workers, "JACK",
                   maxfev, every, checkpoint_path, bounds, ref_class, resume=resume)
    if not df.empty:
        df = df.rename(columns={"replicate": "left_out_index"})
        df.insert(1, "left_out_city",
                  [prep.cities[int(i)] for i in df["left_out_index"]])
    return df


def run_profile(prep: Prepared, x0: np.ndarray, which: str, grid: np.ndarray,
                binned: bool) -> pd.DataFrame:
    """Fix one parameter on a grid and refit the others."""
    j = HYBRID_NAMES.index(which)
    w_city = np.ones(prep.n_cities, float)
    Z, _ = build_ui_features(prep.metrics, w_city)
    free = [k for k in range(len(HYBRID_NAMES)) if k != j]
    rows = []
    prog = Progress(len(grid), f"PROFILE {which}", every=1)
    for g in grid:
        def obj(sub):
            full = np.array(x0, float)
            full[free] = sub
            full[j] = g
            return hybrid_objective(full, prep, w_city, Z, binned=binned)

        s0 = np.array(x0, float)[free]
        r = minimize(obj, s0, method="Nelder-Mead",
                     options={"maxfev": 4000, "fatol": 1e-10, "adaptive": True})
        val, sub = float(r.fun), r.x
        full = np.array(x0, float)
        full[free] = sub
        full[j] = g
        row = {which: float(g), "objective": val}
        row.update({f"refit_{n}": float(v) for n, v in zip(HYBRID_NAMES, full)})
        rows.append(row)
        prog.update(1, f"{which}={g:.4f} objective={val:.6f}")
    prog.close()
    return pd.DataFrame(rows)


def make_cv_folds(prep: Prepared, n_folds: int, n_repeats: int, seed: int,
                  stratify: str = "uf") -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    """Repeated K-fold over networks, stratified on quantile bins of u_fe."""
    n = prep.n_cities
    if stratify == "uf":
        edges = np.quantile(prep.uf, np.linspace(0, 1, n_folds + 1)[1:-1])
        strata = np.digitize(prep.uf, edges)
    else:
        strata = np.zeros(n, dtype=int)

    folds = []
    for rep in range(n_repeats):
        rng = np.random.default_rng(seed + 9973 * rep)
        assign = np.empty(n, dtype=int)
        for s_val in np.unique(strata):
            idx = np.nonzero(strata == s_val)[0]
            rng.shuffle(idx)
            assign[idx] = np.arange(idx.size) % n_folds
        for f in range(n_folds):
            test = np.nonzero(assign == f)[0]
            train = np.nonzero(assign != f)[0]
            if test.size and train.size:
                folds.append((rep, f, train, test))
    return folds


def _one_fold(job):
    """Fit on the training networks, score the held-out ones with no refit."""
    fold_id, rep, fold, train, test = job
    prep: Prepared = _W["prep"]              # type: ignore[assignment]
    binned = _W["binned"]
    rng = np.random.default_rng(500_000 + fold_id)

    w_train = np.zeros(prep.n_cities, float)
    w_train[train] = 1.0

    try:
        theta, obj = fit_hybrid(prep, w_train, x0=_W["x0"], restarts=_W["restarts"],
                                rng=rng, binned=binned, use_global=False,
                                maxfev=int(_W.get("maxfev", 3000)))
    except Exception as exc:
        log(f"[WARN] fold {fold_id} failed: {exc}")
        return None

    a, s_u, s_K, d = theta[0], theta[1], theta[2], theta[3]
    # the moments of the index come from the training networks only
    Z, _ = build_ui_features(prep.metrics, w_train)
    ui = ui_from_Z(Z, theta[4], theta[5:9])
    alpha = _dec_sigmoid((ui - U_TH_FIXED) / max(EPS, s_u))

    scorer = _nrmse_per_city_binned if binned else _nrmse_per_city_exact
    per_city = scorer(prep, alpha, a, d, s_K)

    nr_cub = scorer(prep, np.zeros(prep.n_cities), a, d, s_K)
    nr_exp = scorer(prep, np.ones(prep.n_cities), a, d, s_K)
    better_is_exp = nr_exp < nr_cub
    predicted_exp = ui <= U_TH_FIXED

    rec = {
        "fold_id": fold_id, "repeat": rep, "fold": fold,
        "n_train": int(train.size), "n_test": int(test.size),
        "train_nrmse": float(per_city[train].mean()),
        "test_nrmse": float(per_city[test].mean()),
        "test_nrmse_median": float(np.median(per_city[test])),
        "objective_train": float(obj),
        "a": float(a), "d": float(d),
        "assign_acc_train": float(np.mean(
            predicted_exp[train] == better_is_exp[train])),
        "assign_acc_test": float(np.mean(
            predicted_exp[test] == better_is_exp[test])),
        "frac_urbanized_train": float(np.mean(predicted_exp[train])),
    }
    rec["_per_network"] = {int(i): float(per_city[i]) for i in test}
    rec["_correct"] = {int(i): bool(predicted_exp[i] == better_is_exp[i])
                       for i in test}
    return rec


def run_cv(prep: Prepared, folds, x0: np.ndarray, workers: int, binned: bool,
           restarts: int, maxfev: int, every: int, results_dir: str,
           label: str = "CV") -> pd.DataFrame:
    jobs = [(i, rep, f, tr, te) for i, (rep, f, tr, te) in enumerate(folds)]
    log(f"[{label}] {len(jobs)} folds; each refits the model and the "
        f"urbanization index on its training networks only")
    df = _run_jobs(jobs, prep, x0, 0.0, binned, restarts, workers, label,
                   maxfev, every,
                   os.path.join(results_dir, f"{label.lower()}_folds_partial.csv"),
                   current_bounds(), None,
                   worker_fn=_one_fold,
                   running_keys=("test_nrmse", "assign_acc_test"))
    return df


def summarize_cv(df: pd.DataFrame, prep: Prepared, results_dir: str,
                 label: str = "CV") -> None:
    """Print the out-of-sample numbers and write the per-network breakdown."""
    if df.empty:
        log(f"[{label}] no folds completed.")
        return

    tr = df["train_nrmse"].to_numpy(float)
    te = df["test_nrmse"].to_numpy(float)
    at = df["assign_acc_test"].to_numpy(float)

    log("\n" + "=" * 74)
    log(f"  {label}: OUT-RAW_UTD19-SAMPLE PERFORMANCE".center(74))
    log("=" * 74)
    log(f"  folds                       {len(df)}")
    log(f"  networks held out per fold  {df['n_test'].min()}-{df['n_test'].max()}")
    log(f"  in-sample  NRMSE            {tr.mean():.4f} "
        f"(sd {tr.std(ddof=1):.4f})")
    log(f"  OUT-RAW_UTD19-SAMPLE NRMSE         {te.mean():.4f} "
        f"(sd {te.std(ddof=1):.4f}, range {te.min():.4f}-{te.max():.4f})")
    gap = 100.0 * (te.mean() / tr.mean() - 1.0)
    log(f"  generalization gap          {gap:+.1f}%")
    log(f"  form assignment, in-sample  "
        f"{100 * df['assign_acc_train'].mean():.1f}%")
    log(f"  FORM ASSIGNMENT, HELD OUT   {100 * at.mean():.1f}% "
        f"(sd {100 * at.std(ddof=1):.1f}, min {100 * at.min():.1f}%)")
    log(f"  a  across folds             {df['a'].mean():.4f} "
        f"(sd {df['a'].std(ddof=1):.4f})")
    log(f"  d  across folds             {df['d'].mean():.4f} "
        f"(sd {df['d'].std(ddof=1):.4f})")
    log("=" * 74)
    if gap > 15:
        log("  [!] A gap this large suggests the index is not transferring; "
            "inspect the worst networks below.")
    log("")

    acc: Dict[int, List[float]] = {}
    cor: Dict[int, List[bool]] = {}
    for _, r in df.iterrows():
        for k, v in (r.get("_per_network") or {}).items():
            acc.setdefault(int(k), []).append(v)
        for k, v in (r.get("_correct") or {}).items():
            cor.setdefault(int(k), []).append(bool(v))
    if acc:
        rows = [{"city": prep.cities[i],
                 "oof_nrmse_mean": float(np.mean(v)),
                 "times_held_out": len(v),
                 "form_assigned_correctly":
                     float(np.mean(cor.get(i, [np.nan])))}
                for i, v in sorted(acc.items())]
        per = pd.DataFrame(rows).sort_values("oof_nrmse_mean", ascending=False)
        per.to_csv(os.path.join(results_dir, f"{label.lower()}_per_network.csv"),
                   index=False)
        log("  Worst-transferring networks (mean out-of-fold NRMSE):")
        for _, r in per.head(5).iterrows():
            log(f"      {r['city']:<28s} {r['oof_nrmse_mean']:.4f}   "
                f"form correct {100 * r['form_assigned_correctly']:.0f}%")
        log("")

    df.drop(columns=[c for c in df.columns if c.startswith("_")]).to_csv(
        os.path.join(results_dir, f"{label.lower()}_folds.csv"), index=False)
    partial = os.path.join(results_dir, f"{label.lower()}_folds_partial.csv")
    if os.path.exists(partial):
        os.remove(partial)


def percentile_ci(draws: np.ndarray, level: float = 0.95) -> Tuple[float, float]:
    lo = 100 * (1 - level) / 2
    return (float(np.nanpercentile(draws, lo)),
            float(np.nanpercentile(draws, 100 - lo)))


def basic_ci(draws: np.ndarray, point: float, level: float = 0.95) -> Tuple[float, float]:
    """Basic (reverse-percentile) interval: 2*theta_hat - percentiles."""
    lo, hi = percentile_ci(draws, level)
    return (2 * point - hi, 2 * point - lo)


def bca_ci(draws: np.ndarray, point: float, jack: Optional[np.ndarray],
           level: float = 0.95) -> Tuple[float, float]:
    """Bias-corrected and accelerated interval, or the percentile interval if the jackknife is missing or degenerate."""
    draws = np.asarray(draws, float)
    draws = draws[np.isfinite(draws)]
    if draws.size < 20:
        return (float("nan"), float("nan"))
    if jack is None or not np.all(np.isfinite(jack)) or jack.size < 3:
        return percentile_ci(draws, level)

    try:
        from scipy.stats import norm
    except Exception:
        return percentile_ci(draws, level)

    prop = float(np.mean(draws < point))
    prop = min(max(prop, 1e-6), 1 - 1e-6)
    z0 = norm.ppf(prop)

    jm = float(np.mean(jack))
    num = float(np.sum((jm - jack) ** 3))
    den = 6.0 * (float(np.sum((jm - jack) ** 2)) ** 1.5)
    if den == 0 or not np.isfinite(den):
        return percentile_ci(draws, level)
    acc = num / den

    alpha = (1 - level) / 2
    out = []
    for q in (alpha, 1 - alpha):
        zq = norm.ppf(q)
        adj = z0 + (z0 + zq) / (1 - acc * (z0 + zq))
        p = float(norm.cdf(adj)) * 100
        out.append(float(np.nanpercentile(draws, min(max(p, 0.0), 100.0))))
    return (out[0], out[1])


def summarize(point: Dict[str, float], boot: pd.DataFrame,
              jack: Optional[pd.DataFrame], level: float,
              bounds_map: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Assemble the reporting table: estimate, SE, three interval flavours."""
    rows = []
    for name, est in point.items():
        if name not in boot.columns:
            continue
        draws = boot[name].to_numpy(float)
        draws = draws[np.isfinite(draws)]
        if draws.size == 0:
            continue
        jvals = (jack[name].to_numpy(float)
                 if (jack is not None and name in jack.columns) else None)

        p_lo, p_hi = percentile_ci(draws, level)
        b_lo, b_hi = basic_ci(draws, est, level)
        c_lo, c_hi = bca_ci(draws, est, jvals, level)

        row = {
            "parameter": name,
            "estimate": est,
            "boot_mean": float(np.mean(draws)),
            "boot_bias": float(np.mean(draws) - est),
            "boot_se": float(np.std(draws, ddof=1)),
            "pct_lo": p_lo, "pct_hi": p_hi,
            "basic_lo": b_lo, "basic_hi": b_hi,
            "bca_lo": c_lo, "bca_hi": c_hi,
            "n_draws": int(draws.size),
        }
        if jvals is not None and np.all(np.isfinite(jvals)) and jvals.size > 1:
            n = jvals.size
            row["jack_se"] = float(np.sqrt((n - 1) / n * np.sum((jvals - jvals.mean()) ** 2)))
        if name in bounds_map:
            lo, hi = bounds_map[name]
            tolr = 1e-4 * (hi - lo)
            row["frac_at_bound"] = float(np.mean((draws <= lo + tolr) |
                                                 (draws >= hi - tolr)))
        rows.append(row)
    return pd.DataFrame(rows)


def check_binning_accuracy(prep: Prepared, theta: np.ndarray) -> float:
    """Relative difference between the binned and the exact objective at theta."""
    if prep.exact_x is None:
        return float("nan")
    w = np.ones(prep.n_cities, float)
    Z, _ = build_ui_features(prep.metrics, w)
    v_bin = hybrid_objective(theta, prep, w, Z, binned=True, penalty=False)
    v_exact = hybrid_objective(theta, prep, w, Z, binned=False, penalty=False)
    rel = abs(v_bin - v_exact) / max(abs(v_exact), 1e-12)
    log(f"[CHECK] binned objective={v_bin:.8f}  exact={v_exact:.8f}  "
          f"relative difference={rel:.2e}")
    return rel


def report_point_estimate(prep: Prepared, theta: np.ndarray, obj: float,
                          a_cub: float, obj_cub: float) -> Dict[str, float]:
    log("\n=================== FULL-SAMPLE POINT ESTIMATE ===================")
    for n, v in zip(HYBRID_NAMES, theta):
        log(f"  {n:>6s} = {v: .6f}")
    log(f"  {'a_cubic':>6s} = {a_cub: .6f}")
    log(f"  objective (hybrid) = {obj:.6f}   |   objective (cubic) = {obj_cub:.6f}")

    dq = derived_quantities(theta[0], theta[3], theta[2])
    log("  derived: " + ", ".join(f"{k}={v:.4f}" for k, v in dq.items()
                                    if not k.startswith("x_cap")))

    pinned = []
    for n, v, (lo, hi) in zip(HYBRID_NAMES, theta, HYBRID_BOUNDS):
        if n in FIXED_PARAMS:
            continue
        rng_ = hi - lo
        if rng_ > 0 and (v <= lo + 1e-3 * rng_ or v >= hi - 1e-3 * rng_):
            pinned.append((n, float(v), lo, hi))
    if pinned:
        log("\n[!!] PARAMETER PINNED TO AN OPTIMIZER BOUND")
        for n, v, lo, hi in pinned:
            log(f"      {n} = {v:.6f} sits on [{lo}, {hi}]")
        log("    The objective is still improving against this wall, so the "
            "value is set by the bound rather than by the data, and any "
            "interval computed for it would just measure the wall. Widen the "
            "bound and refit before bootstrapping, e.g.")
        log("      --sk-bounds 0.005,0.8   (or --su-bounds / --a-bounds / "
            "--d-bounds as appropriate)")
        log("    then check whether the optimum moves to an interior point.")

    log("==================================================================\n")

    point = {n: float(v) for n, v in zip(HYBRID_NAMES, theta)}
    point.update({f"dq_{k}": v for k, v in dq.items()})
    point["a_cubic"] = float(a_cub)
    diag = ui_summary(prep, theta, np.ones(prep.n_cities, float))
    for k in ("sd_UI", "can_s_u", "can_b", "can_w_uf", "can_w_L",
              "can_w_SL", "can_w_C", "w_norm"):
        point[k] = diag[k]
    point["s_u_rel"] = diag["s_u_rel"]
    point["class_agreement"] = 1.0
    log("  canonical UI form (||w|| = 1, the comparable parameterization):")
    log(f"      s_u = {diag['can_s_u']:.5f}   b = {diag['can_b']:.4f}   "
        f"w = [{diag['can_w_uf']:.4f}, {diag['can_w_L']:.4f}, "
        f"{diag['can_w_SL']:.4f}, {diag['can_w_C']:.4f}]")
    point["frac_urbanized"] = diag["frac_urbanized"]
    return point


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--mfd-dir", default=MFD_DIR,
                    help="directory holding data_*_grouped.csv")
    ap.add_argument("--outputs-dir",
                    default=OUTPUTS_DIR,
                    help="directory holding outputs_*.txt")
    ap.add_argument("--nodes-dir",
                    default=NODES_DIR,
                    help="CSV with one row per network (data/map_metrics.csv), or a directory holding links_*.csv and nodes_*.csv")
    ap.add_argument("--results-dir", default=None,
                    help="output directory (default: <mfd-dir>/MFDs_Curve_uncertainty)")

    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel processes; 8 turns hours into minutes")
    ap.add_argument("--restarts", type=int, default=2,
                    help="jittered local restarts per replicate")

    ap.add_argument("--point-only", action="store_true",
                    help="fit the full sample and stop (run this first)")
    ap.add_argument("--no-jackknife", action="store_true")
    ap.add_argument("--bca", action="store_true",
                    help="report BCa intervals (implies the jackknife)")
    ap.add_argument("--profile", action="store_true",
                    help="sensitivity profiles for s_u and s_K")
    ap.add_argument("--profile-params", default="s_u,s_K")
    ap.add_argument("--profile-points", type=int, default=15)

    ap.add_argument("--exact", action="store_true",
                    help="disable binning (slower, no approximation)")
    ap.add_argument("--nbins", type=int, default=400)
    ap.add_argument("--kmax-factor", type=float, default=KMAX_FACTOR_DEFAULT)
    ap.add_argument("--warm-start", default=None,
                    help="JSON with a prior fit; skips the global search")
    ap.add_argument("--resume", action="store_true",
                    help="keep the replicates already in bootstrap_draws_partial.csv and "
                         "jackknife_partial.csv and run only the missing ones (same results "
                         "as an uninterrupted run: replicate r depends only on the seed and r)")
    ap.add_argument("--de-maxiter", type=int, default=150)
    ap.add_argument("--de-popsize", type=int, default=14)
    ap.add_argument("--maxfev", type=int, default=3000,
                    help="local-search budget per replicate; the main runtime dial")
    ap.add_argument("--progress-every", type=int, default=1,
                    help="print a progress line every N completed replicates")

    ap.add_argument("--a-bounds", default=None, metavar="LO,HI")
    ap.add_argument("--su-bounds", default=None, metavar="LO,HI")
    ap.add_argument("--sk-bounds", default=None, metavar="LO,HI",
                    help="widen this if s_K pins to its wall (e.g. 0.005,0.8)")
    ap.add_argument("--d-bounds", default=None, metavar="LO,HI")
    ap.add_argument("--w-bounds", default=None, metavar="LO,HI",
                    help="use --w-bounds=-15,15 (the '=' is required when the "
                         "lower limit is negative)")
    ap.add_argument("--b-bounds", default=None, metavar="LO,HI")
    ap.add_argument("--cubic-a-bounds", default=None, metavar="LO,HI")
    ap.add_argument("--fix", action="append", default=None, metavar="NAME=VALUE",
                    help="hold a parameter constant, e.g. --fix s_K=0.1343. "
                         "Repeatable. Defaults: s_u=0.2344, s_K=0.1343")
    ap.add_argument("--free", action="append", default=None, metavar="NAME",
                    help="release a default-fixed parameter, e.g. --free s_K")
    ap.add_argument("--global-search", action="store_true",
                    help="start with differential evolution")
    ap.add_argument("--concavity-x-min", type=float, default=CONCAVITY_X_MIN,
                    help="lower edge of the concavity screen, in units of Kc")
    ap.add_argument("--cv", action="store_true",
                    help="repeated stratified K-fold cross-validation over "
                         "networks; refits everything per fold")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--cv-repeats", type=int, default=4)
    ap.add_argument("--cv-stratify", default="uf", choices=["uf", "none"],
                    help="balance folds on free-flow-speed quantiles (default) "
                         "or not at all")
    ap.add_argument("--profile-range", default=None, metavar="LO,HI",
                    help="explicit grid range for --profile (overrides auto)")
    args = ap.parse_args(argv)

    def _pair(txt):
        if not txt:
            return None
        lo, hi = (float(v) for v in txt.replace(" ", "").split(","))
        return (lo, hi)

    _set_concavity_floor(args.concavity_x_min)

    fixed = dict(FIXED_PARAMS)
    for item in (args.fix or []):
        if "=" not in item:
            raise SystemExit(f"--fix expects NAME=VALUE, got '{item}'")
        nm, val = item.split("=", 1)
        fixed[nm.strip()] = float(val)
    for nm in (args.free or []):
        fixed.pop(nm.strip(), None)
    set_fixed_params(fixed)
    free_names = [n for n in HYBRID_NAMES if n not in fixed]
    log(f"[INFO] fitting {len(free_names)} parameters: {', '.join(free_names)}")
    if fixed:
        log("[INFO] held fixed: "
            + ", ".join(f"{k}={v:g}" for k, v in fixed.items())
            + f"  (plus u_th={U_TH_FIXED:g}, a1={A1_FIXED:g})")
    set_bounds(a=_pair(args.a_bounds), s_u=_pair(args.su_bounds),
               s_K=_pair(args.sk_bounds), d=_pair(args.d_bounds),
               b=_pair(args.b_bounds), w=_pair(args.w_bounds),
               a_cubic=_pair(args.cubic_a_bounds))
    if any([args.a_bounds, args.su_bounds, args.sk_bounds, args.d_bounds,
            args.b_bounds, args.w_bounds, args.cubic_a_bounds]):
        log(f"[INFO] bounds in effect: {current_bounds()}")

    results_dir = args.results_dir or os.path.join(args.mfd_dir,
                                                   "MFDs_Curve_uncertainty")
    os.makedirs(results_dir, exist_ok=True)
    binned = not args.exact

    log("[STEP 1/6] Loading networks ...")
    prep = build_prepared(args.mfd_dir, args.outputs_dir, args.nodes_dir,
                          nbins=args.nbins, kmax_factor=args.kmax_factor,
                          keep_exact=True)

    log("[STEP 2/6] Full-sample fit ...")
    w_full = np.ones(prep.n_cities, float)
    rng = np.random.default_rng(args.seed)
    x0 = apply_fixed(START_THETA)
    if args.warm_start and os.path.exists(args.warm_start):
        with open(args.warm_start) as fh:
            js = json.load(fh)
        hp, ep, up = js.get("hybrid_params", {}), js.get("exp_params", {}), js.get("ui_params", {})
        x0 = np.array([hp.get("a_h", 0.34), hp.get("S_U", 0.23), hp.get("S_K", 0.13),
                       ep.get("EXP_A0", 1.25), up.get("bias_b", 0.0),
                       *(up.get("weights_w", [0, 0, 0, 0]))], float)
        log(f"[INFO] Warm start from {args.warm_start}")

    theta, obj = fit_hybrid(prep, w_full, x0=x0, restarts=4, rng=rng, binned=binned,
                            use_global=args.global_search,
                            de_maxiter=args.de_maxiter,
                            de_popsize=args.de_popsize,
                            maxfev=max(args.maxfev, 8000), verbose=True)
    a_cub, obj_cub = fit_cubic(prep, w_full, binned=binned, verbose=True)

    if binned:
        rel = check_binning_accuracy(prep, theta)
        if np.isfinite(rel) and rel > 1e-4:
            log(f"[ABORT] Binning error {rel:.2e} exceeds 1e-4. Rerun with "
                  f"--nbins {args.nbins * 4} or --exact.")
            return 2

    point = report_point_estimate(prep, theta, obj, a_cub, obj_cub)

    diag = ui_summary(prep, theta, w_full)
    with open(os.path.join(results_dir, "point_estimate.json"), "w") as fh:
        json.dump({
            "objective_hybrid": obj, "objective_cubic": obj_cub,
            "hybrid_params": dict(zip(HYBRID_NAMES, map(float, theta))),
            "fixed": {"u_th": U_TH_FIXED, "a1_EXP_A1": A1_FIXED},
            "a_cubic": float(a_cub),
            "derived": derived_quantities(theta[0], theta[3], theta[2]),
            "ui_moments": {k: np.asarray(v).tolist()
                           for k, v in diag["_moments"].items()},
            "metric_names_order": METRIC_NAMES,
            "n_networks": prep.n_cities,
            "n_observations": int(prep.n_obs.sum()),
        }, fh, indent=2)

    pd.DataFrame({"city": prep.cities, "uf": prep.uf, "kc": prep.kc,
                  "qc": prep.qscale, "n_obs": prep.n_obs,
                  "UI": diag["_ui"],
                  "urbanized_UI_le_0": diag["_ui"] <= U_TH_FIXED}
                 ).to_csv(os.path.join(results_dir, "per_network_UI.csv"), index=False)

    if args.cv:
        folds = make_cv_folds(prep, args.cv_folds, args.cv_repeats,
                              args.seed, args.cv_stratify)
        log(f"[CV] {args.cv_folds}-fold x {args.cv_repeats} repeats, "
            f"stratified on {args.cv_stratify}")
        cvdf = run_cv(prep, folds, theta, args.workers, binned,
                      args.restarts, args.maxfev, args.progress_every,
                      results_dir)
        summarize_cv(cvdf, prep, results_dir)
        if args.point_only:
            log(f"[DONE] Written to {results_dir}")
            return 0

    if args.point_only:
        log(f"[DONE] Point estimate written to {results_dir}")
        return 0

    log("[STEP 3/6] Cluster bootstrap ...")
    boot = run_bootstrap(prep, theta, a_cub, args.n_boot, args.seed,
                         args.workers, binned, args.restarts, args.maxfev,
                         every=args.progress_every,
                         checkpoint_path=os.path.join(results_dir,
                                                      "bootstrap_draws_partial.csv"),
                         bounds=current_bounds(),
                         ref_class=(diag["_ui"] <= U_TH_FIXED),
                         resume=args.resume)
    boot.to_csv(os.path.join(results_dir, "bootstrap_draws.csv"), index=False)
    _partial = os.path.join(results_dir, "bootstrap_draws_partial.csv")
    if os.path.exists(_partial):
        os.remove(_partial)
    log(f"[BOOT] {len(boot)} replicates saved to bootstrap_draws.csv")

    jack = None
    if args.bca or not args.no_jackknife:
        log("[STEP 4/6] Leave-one-network-out jackknife ...")
        jack = run_jackknife(prep, theta, a_cub, args.workers, binned,
                             args.restarts, args.maxfev,
                             every=args.progress_every,
                             checkpoint_path=os.path.join(
                                 results_dir, "jackknife_partial.csv"),
                             bounds=current_bounds(),
                             ref_class=(diag["_ui"] <= U_TH_FIXED),
                             resume=args.resume)
        if not jack.empty:
            for col in ("a", "d", "s_u", "s_K", "a_cubic"):
                if col in jack.columns and col in point:
                    jack[f"delta_{col}"] = jack[col] - point[col]
            jack.to_csv(os.path.join(results_dir, "jackknife_influence.csv"),
                        index=False)
            _jp = os.path.join(results_dir, "jackknife_partial.csv")
            if os.path.exists(_jp):
                os.remove(_jp)
            infl = jack.reindex(jack["delta_a"].abs().sort_values(
                ascending=False).index).head(5)
            log("[JACK] Most influential networks for a:")
            for _, r in infl.iterrows():
                log(f"       {r['left_out_city']:<28s} delta_a = {r['delta_a']:+.4f}")

    log("[STEP 5/6] Intervals ...")
    bounds_map = dict(zip(HYBRID_NAMES, HYBRID_BOUNDS))
    bounds_map["a_cubic"] = A_CUBIC_BOUNDS
    summary = summarize(point, boot, jack, args.level, bounds_map)
    summary.to_csv(os.path.join(results_dir, "interval_summary.csv"), index=False)

    ci_kind = "bca" if (args.bca and jack is not None and not jack.empty) else "pct"
    with pd.option_context("display.width", 160, "display.max_columns", 30):
        cols = ["parameter", "estimate", "boot_se", f"{ci_kind}_lo", f"{ci_kind}_hi"]
        if "frac_at_bound" in summary.columns:
            cols.append("frac_at_bound")
        log("\n" + summary[cols].to_string(index=False, float_format="%.4g") + "\n")

    profiles: Dict[str, pd.DataFrame] = {}
    if args.profile:
        log("[STEP 6/6] Sensitivity profiles ...")
        for name in [p.strip() for p in args.profile_params.split(",") if p.strip()]:
            lo, hi = bounds_map[name]
            est = point[name]
            rng_ = _pair(args.profile_range)
            if rng_ is not None:
                g_lo, g_hi = rng_
            else:
                g_lo, g_hi = max(lo, est * 0.35), min(hi, est * 2.2)
            grid = np.unique(np.clip(
                np.linspace(g_lo, g_hi, args.profile_points), lo, hi))
            df = run_profile(prep, theta, name, grid, binned)
            df.to_csv(os.path.join(results_dir, f"profile_{name}.csv"), index=False)
            profiles[name] = df
            best = df["objective"].min()
            flat = df[df["objective"] <= best * 1.01][name]
            if len(flat):
                log(f"[PROFILE {name}] objective within 1% of best over "
                      f"{name} in [{flat.min():.3f}, {flat.max():.3f}]")

    log(f"[DONE] All artifacts written to {results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
