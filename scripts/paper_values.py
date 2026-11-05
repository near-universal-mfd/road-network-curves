"""Every number in the paper, computed from the results and compared with the values the paper uses (results/expected/paper_values.csv).

Each check() ties a result to a sentence of the text and lists the sentence if the result no longer supports it. With --quick, the numbers that rest on
bootstrap or cross-validation replicates are marked rather than counted as differences.
Writes paper_values_computed.csv and paper_values_comparison.csv in the results folder.
Usage: python paper_values.py [--results FOLDER] [--check] [--quick]
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

import mfd_uncertainty as mu
from paths import OUTPUTS_DIR, PER_LANE, RESULTS as DEFAULT_RESULTS, SHIPPED_RESULTS, SOURCE_GROUPS

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--results", default=DEFAULT_RESULTS)
ap.add_argument("--check", action="store_true")
ap.add_argument("--quick", action="store_true")
ap.add_argument("--write-expected", default=None, metavar="NUMBERS_JSON", help="write results/expected/paper_values.csv from the values the manuscript uses")
args = ap.parse_args()
RESULTS = os.path.abspath(args.results)
FAILED_CHECKS = []


def check(ok, message):
    if not bool(ok):
        FAILED_CHECKS.append(message)


# ================================================================ the model
N = {}
pe = json.load(open(os.path.join(RESULTS, "MFDs_Curve_uncertainty", "point_estimate.json")))
hp = pe["hybrid_params"]
check(hp["w_L"] == hp["w_SL"] == hp["w_C"] == 0.0, "expected the free-flow-speed-only index")
full = os.path.join(RESULTS, "MFDs_Curve_uncertainty_full")
pe_full = json.load(open(os.path.join(full, "point_estimate.json")))
check(max(abs(pe["hybrid_params"][k] - pe_full["hybrid_params"][k]) for k in hp) < 1e-4, "point estimates differ")
iv = pd.read_csv(os.path.join(full, "interval_summary.csv")).set_index("parameter")


def ci(name, kind="bca"):
    return iv.loc[name, f"{kind}_lo"], iv.loc[name, f"{kind}_hi"]


N["a"], N["d"], N["S"] = hp["a"], hp["d"], pe["derived"]["S"]
N["a_ci"], N["d_ci"], N["S_ci"] = ci("a"), ci("d"), ci("dq_S")
N["cap_s"], N["cap_u"] = pe["derived"]["cap_sparse"], pe["derived"]["cap_urban"]
N["cap_s_ci"], N["cap_u_ci"] = ci("dq_cap_sparse"), ci("dq_cap_urban")
N["xj"], N["xj_ci"] = pe["derived"]["x_jam_sparse"], ci("dq_x_jam_sparse")
N["ws"], N["ws_ci"] = pe["derived"]["wave_sparse"], ci("dq_wave_sparse")
N["wu"], N["wu_ci"] = pe["derived"]["wave_urban"], ci("dq_wave_urban")
N["class_agree"] = iv.loc["class_agreement", "boot_mean"]
N["class_agree_ci"] = ci("class_agreement", "pct")
ui = pd.read_csv(os.path.join(RESULTS, "MFDs_Curve_uncertainty", "per_network_UI.csv")).set_index("city")
N["n_urban"] = int(ui["urbanized_UI_le_0"].sum())
N["frac_urban"] = N["n_urban"] / len(ui)
N["wavg"] = N["frac_urban"] * N["cap_u"] + (1 - N["frac_urban"]) * N["cap_s"]
N["b"], N["w_uf"] = hp["b"], hp["w_uf"]
alpha = mu._dec_sigmoid((ui["UI"].to_numpy() - mu.U_TH_FIXED) / hp["s_u"])
urb = ui["urbanized_UI_le_0"].to_numpy()
xs = np.linspace(0.01, 2.0, 4000)
cap_net = np.array([mu.f_model(xs, np.full_like(xs, al), hp["a"], hp["d"], hp["s_K"]).max() for al in alpha])


def wave_full_model(al):
    """Slope of equation (5) at its jam point."""
    x = np.linspace(1e-6, 3.0, 300_001)
    f = mu.f_model(x, np.full_like(x, al), hp["a"], hp["d"], hp["s_K"])
    i = int(np.argmax(f))
    j = i + int(np.nonzero(f[i:] <= 0)[0][0])
    xj = x[j - 1] - f[j - 1] * (x[j] - x[j - 1]) / (f[j] - f[j - 1])
    g = lambda z: mu.f_model(np.array([z]), np.array([al]), hp["a"], hp["d"], hp["s_K"])[0]  # noqa: E731
    return (g(xj + 1e-5) - g(xj - 1e-5)) / 2e-5


wave_net = np.array([wave_full_model(al) if u else np.nan for al, u in zip(alpha, urb)])
check(np.allclose(cap_net[~urb].max(), N["cap_s"], atol=1e-3), "sparse networks should sit at the cubic limit")
N["alpha_urb"] = (float(alpha[urb].min()), float(alpha[urb].max()))
N["cap_urb_net"] = (float(cap_net[urb].min()), float(np.median(cap_net[urb])), float(cap_net[urb].max()))
N["wave_urb_net"] = (float(wave_net[urb].min()), float(wave_net[urb].max()))
check(abs(wave_full_model(0.0) - N["ws"]) < 1e-3 and abs(wave_full_model(1.0) - N["wu"]) < 1e-3, "the network wave speeds should meet the two limits")
mom = pe["ui_moments"]
check(pe["metric_names_order"][0] == "uf", "pe['metric_names_order'][0] == 'uf'")
N["mu_phi"], N["sd_phi"] = mom["phi_mean"][0], mom["phi_std"][0]
N["mu_u"], N["sd_u"] = mom["metric_mean"][0], mom["metric_std"][0]
N["a_cubic_ls"] = pe["a_cubic"]

cv = pd.read_csv(os.path.join(full, "cv_folds.csv"))
N["cv"] = dict(train=cv.train_nrmse.mean(), test=cv.test_nrmse.mean(), a=cv.a.mean(), a_sd=cv.a.std(),
               d=cv.d.mean(), d_sd=cv.d.std(), acc_train=cv.assign_acc_train.mean(),
               acc_test=cv.assign_acc_test.mean())
cvp = pd.read_csv(os.path.join(full, "cv_per_network.csv")).set_index("city")
NAMES = {"losangeles": "Los Angeles", "sanbernardino": "San Bernardino", "SanFrancisco": "San Francisco",
         "constance": "Konstanz"}


def name(c):
    return NAMES.get(c, c.replace("_", " ").title())


worst = cvp.sort_values("oof_nrmse_mean", ascending=False).head(5).index.tolist()
N["worst"] = [name(c) for c in worst]

# ================================================================ the four-input index
F4 = os.path.join(RESULTS, "comparison_four_input", "point")
F4CV = os.path.join(RESULTS, "comparison_four_input", "cv")
F4B = os.path.join(RESULTS, "comparison_four_input", "bootstrap")
pe4 = json.load(open(os.path.join(F4, "point_estimate.json")))
cv4 = pd.read_csv(os.path.join(F4CV, "cv_folds.csv"))
m4 = cv.set_index(["repeat", "fold"]).sort_index().join(cv4.set_index(["repeat", "fold"]).sort_index(),
                                                        rsuffix="_4")
check((m4.n_test == m4.n_test_4).all(), "the two cross-validations used different folds")
p4 = pd.read_csv(os.path.join(F4CV, "cv_per_network.csv")).set_index("city")
dnet = (cvp.oof_nrmse_mean - p4.oof_nrmse_mean.reindex(cvp.index))          # speed-only minus four-input
rng = np.random.default_rng(1)
boot = np.array([rng.choice(dnet.to_numpy(), dnet.size).mean() for _ in range(4000)])
draws = pd.read_csv(os.path.join(F4B, "bootstrap_draws.csv"))
if "warm_start_objective" in draws:
    check((draws.objective <= draws.warm_start_objective + 1e-9).all(), "a four-input replicate is still worse than its start")
flip = {w: float(np.mean(np.sign(draws[w]) != np.sign(pe4["hybrid_params"][w]))) for w in ("w_uf", "w_L", "w_SL", "w_C")}
gains = dnet.sort_values().head(3).index
N["four"] = dict(cv_train=cv4.train_nrmse.mean(), cv_test=cv4.test_nrmse.mean(),
                 acc_train=cv4.assign_acc_train.mean(), acc_test=cv4.assign_acc_test.mean(),
                 folds_worse=int((m4.test_nrmse_4 > m4.test_nrmse).sum()), folds=len(m4),
                 net_median_diff=float(dnet.median()), largest_gains=[name(c) for c in gains],
                 gains_correct4=[float(p4.loc[c, "form_assigned_correctly"]) for c in gains],
                 boot_n=len(draws), sign_flips=flip)
F = N["four"]
check(F["cv_test"] > N["cv"]["test"] and np.percentile(boot, 97.5) < 0, "the speed-only index no longer transfers better")
check(abs(F["net_median_diff"]) < 1e-3 and flip["w_uf"] < 0.05, "the text says the two indices transfer equally well for the median network")
check(max(F["gains_correct4"]) <= 0.25, "the text says the four-input index usually misassigned them")
check(F["acc_test"] - N["cv"]["acc_test"] < 0.01, "the main text says the map inputs did not materially improve held-out form assignment")

# ================================================================ Fig. 2 and the standard forms
f2 = json.load(open(os.path.join(RESULTS, "Fig2_uf_index", "fig2_summary.json")))
check(abs(f2["a"] - N["a"]) < 1e-12, "Fig. 2 was made with another fit")
N["u_star"], N["acc_index"], N["acc_index_std"] = (f2["u_star_kmh"], f2["assignment_index_exact"],
                                                    f2["assignment_index_vs_standard_forms"])
pn = pd.read_csv(os.path.join(RESULTS, "Fig2_uf_index", "fig2_per_network.csv")).set_index("city")
N["worst_slow_cubic"] = [name(c) for c in worst if pn.loc[c, "better_branch"] == "cubic" and pn.loc[c, "index_says"] == "exponential"]
N["worst_fast_exp"] = [name(c) for c in worst if pn.loc[c, "better_branch"] == "exponential" and pn.loc[c, "index_says"] == "cubic"]
N["worst_right"] = [name(c) for c in worst if pn.loc[c, "better_branch"] == pn.loc[c, "index_says"]]
check(all(pn.loc[c, "best_nrmse"] > 0.15 for c in worst if pn.loc[c, "better_branch"] == pn.loc[c, "index_says"]), "the text says the right branch still fits them poorly")
err = pd.read_csv(os.path.join(RESULTS, "MFDs_FINAL", "per_city_errors_normalized.csv"))
und = err.RMSE_Underwood < err.RMSE_Cubic
check(und.sum() == f2["n_underwood_better_standard_forms"], "und.sum() == f2['n_underwood_better_standard_forms']")
N["uf_und_iqr"] = (err.uf[und].quantile(.25), err.uf[und].quantile(.75))
N["uf_cubstd_iqr"] = (err.uf[~und].quantile(.25), err.uf[~und].quantile(.75))

rows = []
for fn in glob.glob(os.path.join(OUTPUTS_DIR, "outputs_*.txt")):
    qc, kc, y3, y4, y5, y6, y7 = [float(v) for v in open(fn).read().split(",")]
    rows.append({"city": os.path.basename(fn)[8:-4], "qc": qc, "kc": kc, "uf": 6 * y3 / kc, "phi13": y4 / (2 * y3)})
sm = pd.DataFrame(rows).set_index("city")
check(abs((sm.qc / (sm.uf * sm.kc)).mean() - 0.719) < 0.001, "the text gives a capacity ratio of 0.719 and a = 0.157")
N["phi13_median"], N["phi13_q25"] = sm.phi13.median(), sm.phi13.quantile(.25)
N["phi13_bordeaux"], N["phi13_konstanz"] = sm.loc["bordeaux", "phi13"], sm.loc["constance", "phi13"]
N["uf_ratio"], N["qc_ratio"] = sm.uf.max() / sm.uf.min(), sm.qc.max() / sm.qc.min()
sc = pd.read_csv(os.path.join(PER_LANE, "network_scales.csv")).set_index("city")
N["fig3_uf"] = dict(cagliari=float(sc.loc["cagliari", "uf"]), wanborough=float(sc.loc["Wanborough", "uf"]))

# ================================================================ statistics
S = os.path.join(RESULTS, "StatResults")
fd = pd.read_csv(os.path.join(S, "diagnostics", "feature_drop_summary.csv")).set_index("step")
N["feat"] = dict(start=int(fd.loc["start", "n_after"]), var=int(fd.loc["variance", "dropped"]),
                 corr=int(fd.loc["correlation", "dropped"]), vif=int(fd.loc["vif", "dropped"]),
                 final=int(fd.loc["final", "n_after"]))
r2 = pd.read_csv(os.path.join(S, "pls2_all_targets_r2_CV.csv"))["r2_cv_oof"].to_numpy()[:7]
check((r2 > -1).all(), "held-out predictions exploded: check the feature transform")
flows = [r2[0]] + list(r2[2:7])
N["r2_flow"], N["r2_kc"] = (min(flows), max(flows)), r2[1]
bys = pd.read_csv(os.path.join(S, "pls2_r2_CV_by_seed.csv"))
fcols = ["qc", "q_kc/6", "q_kc/3", "q_kc/2", "q_2kc/3", "q_5kc/6"]
base = bys.loc[bys.seed == 42, fcols].to_numpy()[0]
check(np.allclose(base, flows), "by-seed file does not match the notebook")
N["r2_seed_maxdev"] = math.ceil(100 * np.abs(bys[fcols].to_numpy() - base).max()) / 100
N["r2_kc_range"] = (bys.kc.min(), bys.kc.max())
sig, strong = set(), set()
for tg in ["qc", "q_kc_6", "q_kc_3", "q_kc_2", "q_2kc_3", "q_5kc_6"]:
    d = pd.read_csv(os.path.join(S, f"corr_predictors_vs_{tg}.csv"))
    s_ = d["is_significant_bh_0.05"].astype(bool)
    sig |= set(d.loc[s_, "predictor"])
    strong |= set(d.loc[s_ & (d.rho.abs() >= 0.3), "predictor"])
N["n_sig_flow"], N["n_strong_flow"] = len(sig), len(strong)
kc_ = pd.read_csv(os.path.join(S, "corr_predictors_vs_kc.csv"))
N["n_sig_kc"] = int(kc_["is_significant_bh_0.05"].astype(bool).sum())
DG = os.path.join(RESULTS, "diagnostics")
sb = pd.read_csv(os.path.join(DG, "stats_by_source.csv")).set_index("source")
N["r2_within_max"] = float(sb[fcols + ["kc"]].to_numpy().max())
check(N["r2_within_max"] <= 0.15, "the text says the attributes predict almost nothing within a source")
yy = pd.read_csv(os.path.join(S, "modeling_Y_multi.csv"), index_col=0)
gg_ = pd.read_csv(SOURCE_GROUPS).set_index("city").group.reindex(yy.index)
ph13_ = yy["q_kc/3"] / (2 * yy["q_kc/6"])
check(abs(ph13_.median() - N["phi13_median"]) < 1e-9, "phi(1/3) differs from the summaries")
N["phi13_by_source"] = dict(utd19=float(ph13_[gg_ == "UTD19"].median()), others=float(ph13_[gg_ != "UTD19"].median()))

# ================================================================ data, model errors and the checks in Methods
me = json.load(open(os.path.join(DG, "model_errors.json")))
N["model_errors"] = dict(continents=me["continents"], urbanized_outside_utd19=me["urbanized_outside_utd19"], all_cubic=me["all_cubic"],
                         paired=dict(all={b: v["p"] for b, v in me["paired"]["all"].items()}, urbanized={b: me["paired"]["urbanized"][b]["p"] for b in ("Greenshields", "Drake")}),
                         median_nrmse=me["median_nrmse"], n_over_0p2=me["n_over_0p2"], trend_share_median=me["trend_share_median"])
check(me["continents"] == {"Europe": 133, "North America": 7, "Asia": 1}, me["continents"])
check(me["n_urbanized"] == N["n_urban"] and me["urbanized_outside_utd19"] == 0, "the text says holding out UTD19 leaves no urbanized network")
check(all(v["p"] < 0.001 and v["hybrid_better"] > v["n"] / 2 for v in me["paired"]["all"].values()), "the text says the model beats every benchmark, P < 0.001")
check(all((v["p"] > 0.05) == (b in ("Greenshields", "Drake")) for b, v in me["paired"]["urbanized"].items()), "update the urbanized-network sentence")
check(0.6 < me["trend_share_median"] < 0.72, "the text says about two-thirds")
nc = pd.read_csv(os.path.join(DG, "near_capacity_bins.csv"))
last = nc[nc["bin"] == nc["bin"].max()]
lu, ls_ = last[last.urbanized].mean(numeric_only=True), last[~last.urbanized]
N["near_capacity"] = dict(obs_share_sparse=float(ls_.obs_mean.mean()), observed_coeff_sparse=float((1 / ls_.uf_kc_over_qc).median()),
                          rmse_urban_hybrid=float(lu.rmse_Hybrid), rmse_urban_greenshields=float(lu.rmse_Greenshields))
check(set(lu[[c for c in lu.index if c.startswith("rmse_")]].nsmallest(2).index) == {"rmse_Hybrid", "rmse_Greenshields"}, "the Fig. 4 legend says these two are the most accurate")
N["n_occupancy"] = int((sc.density_method == "occupancy").sum())
check((sc.source[sc.density_method == "occupancy"] == "UTD19").all(), "the text says the occupancy networks are UTD19's")
N["lanes"] = dict(webtris=float(sc.lanes[sc.source == "WebTRIS"].median()), pems=float(sc.lanes[sc.source == "PEMS"].median()))
wt_ = sc[sc.source == "WebTRIS"]
lf_ = sps.linregress(np.log(wt_.lanes), np.log(wt_.qc_as_reported))
N["lane_check"] = dict(slope=float(lf_.slope), se=float(lf_.stderr))
check(0.8 < lf_.slope < 1.4, "the text says capacity scales with the lane count")
ns_ = pd.read_csv(os.path.join(DG, "network_structure.csv")).set_index("city")
gs_ = pd.read_csv(SOURCE_GROUPS).set_index("city").group.reindex(ns_.index)
N["detectors_motorway"] = dict(webtris=float(ns_.det_share_motorway[gs_ == "WebTRIS"].median()), pems=float(ns_.det_share_motorway[gs_ == "PEMS"].median()))
check((ns_.det_share_motorway + ns_.det_share_trunk)[gs_ == "UTD19"].median() < 0.5, "the text says UTD19's detectors are mostly on smaller roads")
mf_ = pd.read_csv(os.path.join(DG, "occupancy_factor_measured.csv")).set_index("network")
N["factor_measured"] = mf_.factor_per_lane.to_dict()
ff_ = pd.read_csv(os.path.join(DG, "factor_from_free_flow.csv")).set_index("city")
fo_ = ff_[ff_.method == "occupancy"]
tl_ = fo_.loc["toulouse"]
peers_ = fo_[(fo_.pos_median_m >= 100) & (fo_.index != "toulouse")]
pr_ = [float(tl_[c] / peers_[c].median()) for c in ("F_p50", "F_p75", "F_p90")]
sl_ = pd.read_csv(os.path.join(DG, "speed_limits.csv")).set_index("city")
meas_ = sl_.density_method != "occupancy"
top_ = float(sl_.speed_over_limit_at_167[meas_].max())
sm_t = float(sc.loc["toulouse", "factor"])
N["free_flow"] = dict(n=len(fo_), median=float(fo_.F_p90.median()), q25=float(fo_.F_p90.quantile(.25)), q75=float(fo_.F_p90.quantile(.75)), konstanz=float(ff_.loc["constance", "F_p90"]),
                      konstanz_direct=float(ff_.loc["constance", "factor_measured_median"]), toulouse_uf=float(tl_.uf_at_167), toulouse_limit=float(tl_.limit),
                      toulouse_ratio=float(tl_.uf_at_167 / tl_.limit), max_measured_ratio=top_, peers=len(peers_), peer_ratio=(min(pr_), max(pr_)),
                      above_limit=sl_.index[~meas_ & (sl_.speed_over_limit_at_167 > top_)].tolist(), toulouse_factor=sm_t, toulouse_uf_own=float(tl_.uf_at_167 * (1000 / 6) / sm_t))
check(N["free_flow"]["above_limit"] == ["toulouse"], "the text says Toulouse is the one network whose speed rules out the common factor")
ks_ = pd.read_csv(os.path.join(DG, "kc_hand_vs_auto.csv")).dropna(subset=["ratio"])
N["kc_auto"] = dict(within25=float(((ks_.ratio > 0.75) & (ks_.ratio < 1.25)).mean()), same_form=float((ks_.form_hand == ks_.form_auto).mean()))
check(len(ks_) == 141, "the automatic rule should read every network")
draws_ = pd.read_csv(os.path.join(full, "bootstrap_draws.csv"))
N["b_at_bound"] = float((draws_.b >= 6 - 1e-6).mean())
bbi_ = pd.read_csv(os.path.join(DG, "b_bound_intervals.csv"))
N["b_bound_largest_change"] = math.ceil(1000 * float(bbi_.largest_change.max())) / 1000
check(N["b_bound_largest_change"] <= 0.01, "update the sentence on the bound")

# ================================================================ where each number comes from
UNC, FULL, DIAG, STATS = "MFDs_Curve_uncertainty", "MFDs_Curve_uncertainty_full", "diagnostics", "StatResults"
FIG2, ERR, OUT = ("Fig2_uf_index/fig2_summary.json", "notebooks/08_fig2_uf_index.ipynb"), ("MFDs_FINAL/per_city_errors_normalized.csv", "notebooks/05_models_and_figures.ipynb"), ("data/per_lane/Outputs/", "build_per_lane_data.py")
SOURCES = {
    **{k: (f"{UNC}/point_estimate.json", "mfd_uncertainty.py") for k in ("a", "d", "S", "b", "w_uf", "cap_s", "cap_u", "xj", "ws", "wu", "a_cubic_ls", "mu_phi", "sd_phi", "mu_u", "sd_u")},
    **{k: (f"{FULL}/interval_summary.csv", "mfd_uncertainty.py") for k in ("a_ci", "d_ci", "S_ci", "cap_s_ci", "cap_u_ci", "xj_ci", "ws_ci", "wu_ci", "class_agree", "class_agree_ci")},
    **{k: (f"{UNC}/per_network_UI.csv", "mfd_uncertainty.py") for k in ("n_urban", "frac_urban", "wavg", "alpha_urb", "cap_urb_net", "wave_urb_net")},
    "cv": (f"{FULL}/cv_folds.csv", "mfd_uncertainty.py --cv"), "worst": (f"{FULL}/cv_per_network.csv", "mfd_uncertainty.py --cv"), "four": ("comparison_four_input/", "mfd_uncertainty.py"),
    "u_star": FIG2, "acc_index": FIG2, "acc_index_std": FIG2,
    **{k: ("Fig2_uf_index/fig2_per_network.csv", "notebooks/08_fig2_uf_index.ipynb") for k in ("worst_slow_cubic", "worst_fast_exp", "worst_right")},
    "uf_und_iqr": ERR, "uf_cubstd_iqr": ERR,
    **{k: OUT for k in ("phi13_median", "phi13_q25", "phi13_bordeaux", "phi13_konstanz", "uf_ratio", "qc_ratio")},
    **{k: ("data/per_lane/network_scales.csv", "build_per_lane_data.py") for k in ("fig3_uf", "n_occupancy", "lanes", "lane_check")},
    "feat": (f"{STATS}/diagnostics/feature_drop_summary.csv", "notebooks/01_statistics.ipynb"), "r2_flow": (f"{STATS}/pls2_all_targets_r2_CV.csv", "notebooks/01_statistics.ipynb"), "r2_kc": (f"{STATS}/pls2_all_targets_r2_CV.csv", "notebooks/01_statistics.ipynb"),
    "r2_seed_maxdev": (f"{STATS}/pls2_r2_CV_by_seed.csv", "stats_cv_stability.py"), "r2_kc_range": (f"{STATS}/pls2_r2_CV_by_seed.csv", "stats_cv_stability.py"),
    "n_sig_flow": (f"{STATS}/corr_predictors_vs_*.csv", "notebooks/01_statistics.ipynb"), "n_strong_flow": (f"{STATS}/corr_predictors_vs_*.csv", "notebooks/01_statistics.ipynb"), "n_sig_kc": (f"{STATS}/corr_predictors_vs_kc.csv", "notebooks/01_statistics.ipynb"),
    "r2_within_max": (f"{DIAG}/stats_by_source.csv", "check_stats_by_source.py"), "phi13_by_source": (f"{STATS}/modeling_Y_multi.csv", "notebooks/01_statistics.ipynb"),
    "model_errors": (f"{DIAG}/model_errors.json", "check_model_errors.py"), "near_capacity": (f"{DIAG}/near_capacity_bins.csv", "diag_near_capacity.py"),
    "detectors_motorway": (f"{DIAG}/network_structure.csv", "check_network_structure.py"), "factor_measured": (f"{DIAG}/occupancy_factor_measured.csv", "measure_factor.py"),
    "free_flow": (f"{DIAG}/factor_from_free_flow.csv, speed_limits.csv", "check_factor_from_free_flow.py, check_speed_limits.py"), "kc_auto": (f"{DIAG}/kc_hand_vs_auto.csv", "check_aggregation.py"),
    "b_at_bound": (f"{FULL}/bootstrap_draws.csv", "mfd_uncertainty.py"), "b_bound_largest_change": (f"{DIAG}/b_bound_intervals.csv", "check_b_bound.py"),
    "runtime_min": ("the authors' logs", "not recomputed: run times on the authors' laptop"),
}
FEWER_REPLICATES = ("a_ci", "d_ci", "S_ci", "cap_s_ci", "cap_u_ci", "xj_ci", "ws_ci", "wu_ci", "class_agree", "class_agree_ci", "cv", "worst", "worst_slow_cubic", "worst_fast_exp", "worst_right", "four", "b_at_bound", "model_errors.paired")


def flatten(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flatten(v, f"{pre}.{k}" if pre else str(k))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from flatten(v, f"{pre}.{i}")
    else:
        yield pre, o


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


computed = dict(flatten(json.loads(json.dumps(N, default=float))))
pd.DataFrame({"key": list(computed), "value": list(computed.values())}).to_csv(os.path.join(RESULTS, "paper_values_computed.csv"), index=False)
expected_path = os.path.join(SHIPPED_RESULTS, "expected", "paper_values.csv")
if args.write_expected:
    used = dict(flatten(json.load(open(args.write_expected, encoding="utf-8"))))
    rows = []
    for k, v in used.items():
        f, s = SOURCES[k.split(".")[0]]
        rows.append(dict(key=k, value=v, output_file=f, script=s))
    os.makedirs(os.path.dirname(expected_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(expected_path, index=False)
    print("wrote", expected_path, len(rows), "values")
if args.check:
    exp = pd.read_csv(expected_path, dtype={"value": object})
    rows = []
    for r in exp.itertuples():
        key = r.key
        if key.startswith("runtime_min"):
            rows.append(dict(key=key, expected=r.value, computed=None, status="not recomputed"))
            continue
        if key not in computed:
            rows.append(dict(key=key, expected=r.value, computed=None, status="missing"))
            continue
        c = computed[key]
        fe, fc = to_float(r.value), to_float(c)
        if fe is None or fc is None or (isinstance(c, str)):
            st = "identical" if str(r.value) == str(c) else "differs"
        elif np.isnan(fe) and np.isnan(fc):
            st = "identical"
        else:
            d = abs(fe - fc) / max(1e-12, abs(fe))
            st = "identical" if d <= 1e-6 else ("within 0.5%" if d <= 5e-3 else "differs")
        if args.quick and st == "differs" and key.startswith(FEWER_REPLICATES):
            st = "fewer replicates"
        rows.append(dict(key=key, expected=r.value, computed=c, status=st))
    cmp_ = pd.DataFrame(rows)
    cmp_.to_csv(os.path.join(RESULTS, "paper_values_comparison.csv"), index=False)
    print(cmp_.status.value_counts().to_string())
    d_ = cmp_[cmp_.status.isin(["differs", "missing"])]
    if len(d_):
        print("\nThese differ from the paper:")
        print(d_.head(60).to_string(index=False))
    if FAILED_CHECKS:
        print("\nSentences of the paper these results no longer support:" + (" (expected in --quick for those about intervals)" if args.quick else ""))
        for m in FAILED_CHECKS:
            print(" -", m)
    print("\nEvery check of the text holds." if not FAILED_CHECKS else f"\n{len(FAILED_CHECKS)} checks of the text do not hold.")
