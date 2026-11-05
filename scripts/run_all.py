"""Run the pipeline step by step and compare the results with the numbers in the paper.

    python scripts/run_all.py --list         the steps, with what each reads and writes
    python scripts/run_all.py --quick        fewer replicates and no slow steps (about 20 minutes)
    python scripts/run_all.py                every step that runs from the shipped data (about 6 hours with 9 workers)
    python scripts/run_all.py --from-raw     also the steps that read the original downloads (data/RAW_SOURCES.md)
    python scripts/run_all.py --only 08_point_estimate 09_models_and_figures

Results go to work/ (or --out), never to results/. The first run copies results/ there, so a skipped step leaves the shipped output. Running the same
command again continues where it stopped (--restart starts over). Each step's output is in <out>/logs/<step>.log.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

from paths import NOTEBOOKS, ROOT, SHIPPED_RESULTS

FIX = ["--fix", "w_L=0", "--fix", "w_SL=0", "--fix", "w_C=0"]       # the index uses u_fe alone
BND = ["--b-bounds=-6,6", "--w-bounds=-10,10"]


@dataclass
class Step:
    name: str
    args: list                       # after the Python interpreter, or an R script for r=True
    what: str
    writes: str
    raw: bool = False                # reads the original downloads: only with --from-raw
    slow: bool = False               # left out of --quick
    quick_args: list = field(default_factory=list)     # replaces args in --quick
    took: str = ""                   # on the authors' laptop (Windows 11, 10 of its cores)
    r: bool = False


def steps(o, w):
    """o: the output folder. w: worker processes."""
    u0, u1, f = (os.path.join(o, "MFDs_Curve_uncertainty"), os.path.join(o, "MFDs_Curve_uncertainty_full"), os.path.join(o, "comparison_four_input"))
    nb = lambda n: ["run_notebook.py", os.path.join(NOTEBOOKS, n + ".ipynb")]      # noqa: E731
    ws0, ws4 = os.path.join(u0, "warm_start_from_point_estimate.json"), os.path.join(f, "point", "warm_start_from_point_estimate.json")
    W = ["--workers", str(w)]
    return [
        Step("01_lanes", ["check_lanes.py"], "Lane count of every network", "diagnostics/lanes_*.csv", raw=True, took="minutes"),
        Step("02_measure_factor", ["measure_factor.py"], "Occupancy factor where flow, occupancy and speed are all reported", "diagnostics/occupancy_factor_measured.csv", raw=True, took="minutes"),
        Step("02a_detector_slopes", ["check_factor_from_free_flow.py"], "Slopes of flow on occupancy, detector by detector", "diagnostics/factor_from_free_flow*.csv", raw=True, took="1 h"),
        Step("02b_factor_from_free_flow", ["check_factor_from_free_flow.py", "--summary-only"], "The factor from each network's fastest detectors", "diagnostics/factor_from_free_flow.csv", took="seconds"),
        Step("02c_map_metrics", ["make_map_metrics.py"], "Three map averages per network, from the map files", "data/map_metrics.csv", raw=True, took="5 min"),
        Step("03_per_lane_data", ["build_per_lane_data.py", "--update-cache"], "Every network per lane, one occupancy factor, corrected speeds", "data/per_lane/, data/features_per_network/", raw=True, took="under 1 min"),
        Step("04_network_structure", ["check_network_structure.py"], "Road class and lanes of the links that carry the detectors", "diagnostics/network_structure.csv", raw=True, took="10 min"),
        Step("05_statistics", nb("01_statistics"), "Feature pruning (428 to 74) and the statistics", "StatResults/", took="8 min"),
        Step("06_statistics_figures", nb("02_statistics_figures"), "Consensus of the five methods and the statistics panels", "StatResults/diagnostics/, StatResults/SlidesFigures_More/", took="2 min"),
        Step("07_stats_cv_stability", ["stats_cv_stability.py"], "The statistics' R2 over five fold assignments", "StatResults/pls2_r2_CV_by_seed.csv", took="seconds"),
        Step("08_point_estimate", ["mfd_uncertainty.py", "--results-dir", u0, "--point-only", "--global-search", *BND, *FIX], "Global fit of equations (5)-(6)", "MFDs_Curve_uncertainty/", took="5 min"),
        Step("09_models_and_figures", nb("05_models_and_figures"), "The model against the benchmarks; panels of Figs 3-4 and Extended Data Fig. 3", "MFDs_FINAL/", took="4 min"),
        Step("10_cubic_factors", ["cubic_factors.py"], "Capacity ratio 0.719, hence the cubic's a = 0.157", "cubic_factors_summary.csv", took="seconds"),
        Step("11_fig2", nb("08_fig2_uf_index"), "Fig. 2 and the form assignment", "Fig2_uf_index/", took="1 min"),
        Step("12_fig1", nb("07_fig1_collapse"), "Fig. 1", "cone_mean_CI_outputs/", took="1 min"),
        Step("13_warm_start", ["make_warm_start.py", u0], "The point estimate as the start of later fits", "MFDs_Curve_uncertainty/warm_start_from_point_estimate.json", took="seconds"),
        Step("14_cv", ["mfd_uncertainty.py", "--results-dir", u1, "--warm-start", ws0, *FIX, *BND, "--point-only", "--cv", *W], "Cross-validation, 5 folds by 4 repeats", "MFDs_Curve_uncertainty_full/cv_*.csv", quick_args=["mfd_uncertainty.py", "--results-dir", u1, "--warm-start", ws0, *FIX, *BND, "--point-only", "--cv", "--cv-repeats", "1", *W], took="6 min"),
        Step("15_four_point", ["mfd_uncertainty.py", "--results-dir", os.path.join(f, "point"), "--point-only", "--global-search", *BND], "Four-input index: fit", "comparison_four_input/point/", slow=True, took="5 min"),
        Step("16_four_warm_start", ["make_warm_start.py", os.path.join(f, "point")], "Four-input index: start of later fits", "comparison_four_input/point/", slow=True, took="seconds"),
        Step("17_four_cv", ["mfd_uncertainty.py", "--results-dir", os.path.join(f, "cv"), "--warm-start", ws4, *BND, "--point-only", "--cv", *W], "Four-input index: cross-validation", "comparison_four_input/cv/", slow=True, took="9 min"),
        Step("18_model_errors", ["check_model_errors.py"], "Paired tests against the benchmarks, the error distribution and its trend share", "diagnostics/model_errors.json", took="seconds"),
        Step("19_stats_by_source", ["check_stats_by_source.py"], "The statistics within WebTRIS and within UTD19", "diagnostics/stats_by_source.csv", took="seconds"),
        Step("20_aggregation", ["check_aggregation.py"], "Rebuild the network files from the detector files; hand-read against automatic K_c", "diagnostics/aggregation_check.csv, kc_hand_vs_auto.csv", raw=True, took="30 min"),
        Step("30_bootstrap", ["mfd_uncertainty.py", "--results-dir", u1, "--warm-start", ws0, *FIX, *BND, "--bca", "--profile", *W, "--resume"], "Bootstrap of 1,000 replicates, jackknife, profiles", "MFDs_Curve_uncertainty_full/", quick_args=["mfd_uncertainty.py", "--results-dir", u1, "--warm-start", ws0, *FIX, *BND, "--n-boot", "50", "--no-jackknife", *W], took="3 h"),
        Step("31_four_bootstrap", ["mfd_uncertainty.py", "--results-dir", os.path.join(f, "bootstrap"), "--warm-start", ws4, *BND, "--n-boot", "300", *W, "--resume"], "Four-input index: bootstrap of 300 replicates", "comparison_four_input/bootstrap/", slow=True, took="2 h 20 min"),
        Step("32_b_bound", ["check_b_bound.py", *W], "The replicates with b on its bound, refitted with wider bounds", "diagnostics/b_bound_*.csv", slow=True, took="20 min"),
        Step("33_near_capacity", ["diag_near_capacity.py"], "Errors near capacity", "diagnostics/near_capacity_bins.csv", took="4 min"),
        Step("34_edfig_raw", ["edfig1_raw.py"], "Cagliari before screening, for Extended Data Fig. 1c", "figures/panels/edfig1c_cagliari_raw.csv", raw=True, took="seconds"),
        Step("34b_edfig_panels", ["edfig1_panels.R", ROOT], "Panels of Extended Data Fig. 1 (R; skipped if Rscript is not on the PATH, leaving the shipped panels)", "figures/panels/", r=True, took="seconds"),
        Step("35_figures", ["make_manuscript_figures.py"], "Figs 1-4 and Extended Data Figs 1-4", "figures/", took="seconds"),
        Step("36_speed_limits", ["check_speed_limits.py"], "Free-flow speed against the speed limit, UTD19", "diagnostics/speed_limits.csv", raw=True, took="seconds"),
        Step("90_paper_values", ["paper_values.py", "--results", o, "--check"], "Every number in the paper, from the results, against results/expected/paper_values.csv", "paper_values_computed.csv, paper_values_comparison.csv", quick_args=["paper_values.py", "--results", o, "--check", "--quick"], took="seconds"),
    ]


def seed_results(out):
    """Copy the shipped results into the output folder. The per-network figures come from step 09."""
    n = 0
    for dp, dn, fn in os.walk(SHIPPED_RESULTS):
        rel = os.path.relpath(dp, SHIPPED_RESULTS)
        if rel.split(os.sep)[0] == "expected" or os.path.basename(dp).startswith("figures_per_city"):
            continue
        os.makedirs(os.path.join(out, rel), exist_ok=True)
        for f in fn:
            t = os.path.join(out, rel, f)
            if not os.path.exists(t):
                shutil.copy2(os.path.join(dp, f), t)
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "work"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--from-raw", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, metavar="STEP")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    all_steps = steps(out, args.workers)
    if args.list:
        for s in all_steps:
            mark = ("raw " if s.raw else "    ") + ("slow" if s.slow else "    ")
            print(f"{s.name:28s} {mark} {s.took:12s} {s.what}  ->  {s.writes}")
        return 0
    if args.only:
        unknown = set(args.only) - {s.name for s in all_steps}
        if unknown:
            raise SystemExit(f"no such step: {sorted(unknown)}; see --list")
    logs, done = os.path.join(out, "logs"), os.path.join(out, "logs", "done" + ("_quick" if args.quick else ""))
    if args.restart and os.path.isdir(done):
        shutil.rmtree(done)
    os.makedirs(done, exist_ok=True)
    print(f"results go to {out}; {seed_results(out)} shipped files copied there", flush=True)
    env = dict(os.environ, ROAD_NETWORK_RESULTS=out, PYTHONUTF8="1", PYTHONWARNINGS="ignore", MPLBACKEND="Agg")
    here = os.path.dirname(os.path.abspath(__file__))
    t00 = time.time()
    for s in all_steps:
        if args.only is not None:
            if s.name not in args.only:
                continue
        else:
            if s.raw and not args.from_raw:
                print(f"{s.name}: needs the raw data, keeping the shipped output ({s.writes})", flush=True)
                continue
            if s.slow and args.quick:
                print(f"{s.name}: left out of --quick, keeping the shipped output ({s.writes})", flush=True)
                continue
            if os.path.exists(os.path.join(done, s.name)):
                print(f"{s.name}: already done", flush=True)
                continue
        cmd_args = s.quick_args if (args.quick and s.quick_args) else s.args
        if s.r:
            rs = shutil.which("Rscript")
            if rs is None:
                print(f"{s.name}: Rscript is not on the PATH, keeping the shipped panels", flush=True)
                continue
            cmd = [rs, os.path.join(here, cmd_args[0]), *cmd_args[1:], out]
        else:
            cmd = [sys.executable, os.path.join(here, cmd_args[0]), *cmd_args[1:]]
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {s.name}: {s.what}", flush=True)
        with open(os.path.join(logs, s.name + ".log"), "a", encoding="utf-8") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=here)
        if rc != 0:
            print(f"FAILED: {s.name} (exit {rc}); see {os.path.join(logs, s.name + '.log')}", flush=True)
            return 1
        open(os.path.join(done, s.name), "w").close()
        print(f"           finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"done in {(time.time() - t00) / 60:.1f} min. The comparison with the paper is in {os.path.join(out, 'paper_values_comparison.csv')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
