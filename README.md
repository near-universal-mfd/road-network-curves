# Road networks fill with traffic along a near-universal curve

Code and processed data accompanying the manuscript *Road networks fill with traffic along a near-universal curve*. The repository contains the network traffic series, network attributes, fitted models, statistical analyses and figures used in the paper.

The main analysis starts from the processed files in `data/`. The original detector downloads are not redistributed; see [Data sources](data/RAW_SOURCES.md) and [Data notes](data/README_data.md).

## Getting started

The analysis uses Python 3.11. Install the dependencies from the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

The manuscript results were produced on Windows 11. `requirements-lock.txt` records that environment's installed packages; `requirements-raw.txt` lists additional packages needed for the raw-data steps. R is needed for the original UTD19 cleaning scripts and to regenerate panels of Extended Data Fig. 1. Those panels are included in the repository.

Two starting points do not require a long run:

```bash
python scripts/run_demo.py
python scripts/paper_values.py --results results --check
```

The demo uses Cagliari, Wanborough, Bordeaux and Konstanz. It evaluates the fitted model and benchmarks, refits the model using those four networks alone, and compares its outputs with `demo/expected_output/`. Results are written to `demo/output/`.

The second command checks the numerical manuscript values recorded in `results/expected/paper_values.csv` against the supplied results. It does not rerun the underlying analyses.

## Reproducing the analysis

```bash
python scripts/run_all.py --list
python scripts/run_all.py --quick --workers 4
python scripts/run_all.py --workers 9
```

`--list` shows the individual steps and their inputs and outputs. The quick run uses fewer bootstrap replicates and cross-validation folds and skips the slower four-input comparison. The full run from the processed files takes several hours. Both write to `work/` rather than changing the supplied `results/`. A failed or interrupted run can be resumed. Use `--only STEP_NAME` to run a selected step.

The steps are organized as follows:

| Stage | Main code | Outputs |
| --- | --- | --- |
| Per-network attributes and statistical analysis | `notebooks/01_statistics.ipynb`, `02_statistics_figures.ipynb`, `scripts/stats_cv_stability.py` | `results/StatResults/`; Extended Data Fig. 2 |
| Cubic and unified-model calibration | `scripts/cubic_factors.py`, `scripts/mfd_uncertainty.py` | `results/MFDs_Curve_uncertainty/` |
| Model comparisons and scaling plots | `notebooks/05_models_and_figures.ipynb`, `07_fig1_collapse.ipynb`, `08_fig2_uf_index.ipynb` | `results/MFDs_FINAL/`, `Fig2_uf_index/`, `cone_mean_CI_outputs/` |
| Bootstrap and cross-validation | `scripts/mfd_uncertainty.py` | `results/MFDs_Curve_uncertainty_full/` |
| Four-input urbanization index | `scripts/mfd_uncertainty.py` with four inputs | `results/comparison_four_input/` |
| Checks cited in Methods | `scripts/check_*.py`, `diag_near_capacity.py`, `stats_cv_stability.py` | `results/diagnostics/` |
| Paper figures and numerical checks | `scripts/make_manuscript_figures.py`, `paper_values.py` | `results/figures/`, `paper_values_comparison.csv` in the run folder |

The model and its uncertainty calculations are implemented in `scripts/mfd_uncertainty.py`; `f_model` implements equation (5). The fitted constants and index standardization moments are saved in `results/MFDs_Curve_uncertainty/point_estimate.json`. The main figures, Fig. 1–4 and Extended Data Fig. 1–4, are in `results/figures/`. Intermediate tables used for these figures are identified in [Data notes](data/README_data.md).

## Using the calibrated curve

To evaluate equation (5) for a network with an effective free-flow speed and critical density:

```bash
python scripts/predict_curve.py --u_fe 45 --k_c 30
```

To compare with observations, use a CSV with columns `density` (vehicles/km/lane) and `flow` (vehicles/hour/lane):

```bash
python scripts/predict_curve.py --u_fe 45 --k_c 30 --q_c 800 --observations network.csv --out network_fit
```

This writes the predicted curve and, if observations are provided, normalized errors and a plot. In the manuscript, effective free-flow speed and critical density were obtained from each network's observed curve. Performance when these scales are estimated independently remains to be tested.

## Data and upstream processing

- `data/per_lane/` contains the screened traffic series and seven manually read flow-density summaries for each of the 141 networks. `data/features_per_network/` contains their candidate attributes and response vectors.
- `data/networks.csv`, `network_summary.csv`, `screening_decisions.csv` and `feature_dictionary.csv` document the networks, decisions and derived variables.
- `results/` holds the reported calibration, cross-validation, bootstrap, diagnostics and statistical outputs.
- `deposit/` contains compressed OpenStreetMap-derived map tables and GeoQuery request outputs. The map archives are used only for upstream processing, not for refitting from the supplied network series.
- `raw_processing/` contains the earlier R and Python scripts used to clean the original source data, construct networks and extract map attributes.

The initial data-processing stage and the main rerun are **not a single end-to-end command**. The scripts in `raw_processing/` require the original downloads and some manual decisions; they were not rerun as part of packaging this repository. Their inputs and known corrections are described in [Upstream processing](raw_processing/README.md). `python scripts/run_all.py --from-raw` adds the selected raw-data checks and per-lane conversion steps in `run_all.py`; it does not execute every script in `raw_processing/`.

To unpack the included map archives and rerun the map-only checks:

```bash
python scripts/unpack_maps.py
python scripts/run_all.py --only 02c_map_metrics 04_network_structure
```

The original downloaded detector records, some very large geospatial sources, and the short conversion script from the PeMS `.npz` files are not included. See `data/RAW_SOURCES.md` for sources and expected input paths.

## Notes

Network capacity and critical density were read from plots, rather than selected by a maximum-observation rule. These inputs are supplied alongside the network series. The occupancy-to-density conversion and Toulouse exception are described in Methods and in `data/README_data.md`.

The statistical feature-pruning step involves a nearly singular correlation matrix. On processors using different linear algebra kernels, it can retain a slightly different set of 74 attributes and change some statistical summaries; the shipped outputs are those used for the manuscript. The GitHub Actions workflow runs the demo, the numerical checks and a quick rerun on fresh installations.

Code is released under the [MIT licence](LICENSE). Derived data and map files have separate source conditions, described in [data/LICENSES.md](data/LICENSES.md). For questions during anonymous review, contact the authors through the journal.
