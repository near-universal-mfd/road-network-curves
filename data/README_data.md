# The data in this repository

This page describes the inputs and outputs used in the analysis. `LICENSES.md` gives the data terms, and `RAW_SOURCES.md` identifies the original sources and the data needed for the upstream steps.

A network's name is the one its files carry. Most are place names; `constance` is Konstanz, `taipeh` Taipei, `torino` Turin and `luzern` Lucerne, and `london`, `birmingham` and `manchester` (UTD19) are different networks from `London2`, `Birmingham2` and `Manchester2` (WebTRIS). `networks.csv` gives each one's display name.

## Units

All flows are per lane, in vehicles per hour per lane, and all densities in vehicles per kilometre per lane. Speeds are in km/h. `*_as_reported` columns and `outputs_as_reported/` keep each source's own units: flows are totals over the lanes a detector or site covers, and occupancy-based densities use a factor of 500.

## What the pipeline starts from

| File | One row per | Columns | Made by |
|---|---|---|---|
| `per_lane/MFDs/data_<network>_grouped.csv` | time step, after screening | `flow`, `density`, and the time: `day` and `interval` (seconds since midnight), `timestamp`, or `timestep` (PEMS numbers its steps). UTD19 also has `speed`, which in occupancy-based networks is derived from flow and occupancy | `scripts/build_per_lane_data.py` |
| `per_lane/Outputs/outputs_<network>.txt` | network | Seven comma-separated numbers, always in this order: q_c, K_c, and the flow at K_c/6, K_c/3, K_c/2, 2K_c/3 and 5K_c/6. Read by hand from a plot of each curve (`raw_processing/python/read_scales.py`), then put per lane | same |
| `per_lane/network_scales.csv` | network | `city`; `source`; `density_method` (`occupancy` or `flow/speed`); `lanes` and `lane_source`; `factor`, the occupancy factor (empty for flow/speed); `flow_divisor`, `density_divisor`, `speed_multiple`, which turn the reported values into the per-lane ones; and `qc`, `kc`, `uf` (u_fe), each also `_as_reported` | same |
| `outputs_as_reported/outputs_<network>.txt` | network | The same seven numbers in each source's own units, as read by hand | `raw_processing/python/read_scales.py` |
| `map_metrics.csv` | network | `city`; `map_name`, the name on its map files; `avg_link_length` (m), `avg_speed_limit` (km/h, tagged links only), `node_connectivity` (mean of the average in-degree and out-degree). These enter only the four-input comparison | `scripts/make_map_metrics.py` |
| `features_per_network/<network>__predictors.csv` | network | The candidate attributes before pruning (`feature_dictionary.csv`) | `notebooks/01_statistics.ipynb`, `build_predictors_from_original` |
| `features_per_network/<network>__responses.csv`, `__flow.npy` | network | The seven summaries per lane, and the six flows in the order q(K_c/6) … q(5K_c/6), q_c | `scripts/build_per_lane_data.py` |
| `source_groups.csv` | network | `city`, `group`: UTD19, WebTRIS, PEMS, Portland or Seattle | `raw_processing/python/` |

## Summaries

| File | One row per | Columns |
|---|---|---|
| `networks.csv` | network | `network`; `map_files`; `name`; `country`; `continent`; `source`; `period_start` and `period_end`, the first and last day in its curve (for PEMS, the months in the source's documentation); `interval_min`, the reporting interval; `n_time_steps`, after screening; `n_detectors_or_sites`, in its detector list; `detector_share_motorway`, the share of detectors on motorway links; `density_method`; `occupancy_factor`; `lanes` and `lane_source`; `speed_multiple` and `speed_correction`; `partition_of` |
| `network_summary.csv` | network | Scales (`q_c`, `K_c`, `u_fe`, each also `_as_reported`); map inputs; `n_obs`; `UI`, the urbanization index; `urbanized_UI_le_0`; `nrmse_cubic_branch`, `nrmse_exp_branch`, the error of each branch of equation (5) alone; `better_branch`, `index_says`, `correct`; `alpha`; `RMSE_<form>`, `MAE_<form>` for each form, normalized by q_c on K ≤ 1.1 K_c (`Hybrid` is equation 5); `better_standard_form` |
| `outcomes_7point.csv` | network | The seven summaries per lane and `_as_reported` |
| `network_properties.csv` | network, under its map's name | City-level attributes as compiled: hull area, GDP, population density, city area, income, city population, employment rate, population by age and sex (`F0` … `M80`), and covariates. Income, city population and employment rate were compiled by hand in 2025, and their individual sources and years weren't recorded |
| `feature_dictionary.csv` | candidate attribute | `feature`; `level` (link, node, network); `description`; `source`; `retained_in_74`; `removed_by`, the pruning step that removed it |
| `detectors.csv` | detector or site | `network`; coordinates, and a name where the source gives one. These are the positions used to draw each hull and match detectors to links; the map deposit holds the matched links |
| `screening_decisions.csv` | decision | `network_or_dataset`; `stage`; `decision`; `rule`, in the source's own units; `reason`; `source`, the code that applies it. Every judgment call in screening, from dropped datasets to the limits on each curve |
| `splits/pls_folds.csv` | network × seed | `network`, `seed`, `fold`: the outer folds of the statistics, for the fold seeds 42, 1, 2, 3 and 4 |
| `splits/model_cv_folds.csv` | network × repeat | `network`, `repeat`, `fold`: the model's 20 cross-validation folds (seed 20260813, stratified on u_fe) |

## Results

| Folder in `results/` | What | Main files |
|---|---|---|
| `MFDs_Curve_uncertainty/` | The global fit of equations (5)–(6) | `point_estimate.json`: constants, derived quantities, and the moments behind the index. `per_network_UI.csv` |
| `MFDs_Curve_uncertainty_full/` | Bootstrap and cross-validation | `interval_summary.csv`; `bootstrap_draws.csv`, all 1,000 replicates; `jackknife_influence.csv`; `cv_folds.csv`; `cv_per_network.csv`; `profile_s_u.csv`, `profile_s_K.csv` |
| `comparison_four_input/` | The index with the three map inputs added | `point/`, `cv/`, `bootstrap/` |
| `MFDs_FINAL/` | Each form against each network | `per_city_errors_normalized.csv`; `errors_by_Kratio_bins_across_cities.csv`; `errors_summary_across_cities.csv`; a figure per network in `figures_per_city/` |
| `Fig2_uf_index/` | The form assignment | `fig2_per_network.csv`, `fig2_summary.json` |
| `cone_mean_CI_outputs/` | Fig. 1 | `factors_cone_curve.csv` |
| `StatResults/` | The statistics | `predictors_all_cities_raw.csv`, the candidate attributes; `modeling_X_clean.csv`, the 74 retained; `modeling_Y_multi.csv`, the outcomes; `corr_predictors_vs_<outcome>.csv`, Spearman with Benjamini–Hochberg; `pls2_all_targets_*.csv`; `elasticnet_all_targets_*.csv`; `rf_permutation_importance_<outcome>.csv`; `lasso_ols_<outcome>.txt`; `pls2_r2_CV_by_seed.csv`; in `diagnostics/`, the pruning log (`feature_drop_summary.csv`, `removed_by_*.csv`, `correlation_pruned_pairs.csv`, `final_vif.csv`) and `consensus_*.csv` |
| `diagnostics/` | The checks the text cites | See below |
| `figures/` | Figs 1–4 and Extended Data Figs 1–4 | `panels/` holds the parts, with `edfig2d_groups.csv`, the grouping behind Extended Data Fig. 2d |
| `expected/` | The numbers in the paper | `paper_values.csv`: `key`, `value`, `output_file`, `script` |

| File in `results/diagnostics/` | Question it answers | Script |
|---|---|---|
| `lanes_by_network.csv`, `lanes_utd19_detectors.csv`, `lanes_webtris_sites.csv` | How many lanes does each network's flow cover? | `check_lanes.py` |
| `occupancy_factor_measured.csv` | What is the factor where flow, occupancy and speed are all reported? | `measure_factor.py` |
| `factor_from_free_flow.csv`, `factor_from_free_flow_detectors.csv` | What is it from the fastest detectors and the speed limit? | `check_factor_from_free_flow.py` |
| `stats_by_source.csv` | Do the attributes predict flows within WebTRIS and within UTD19? | `check_stats_by_source.py` |
| `network_structure.csv` | On what road classes, and how many lanes, are each network's detectors? | `check_network_structure.py` |
| `speed_limits.csv` | How does each UTD19 network's free-flow speed compare with its speed limit? | `check_speed_limits.py` |
| `model_errors.json` | How does the model compare with each benchmark, and how are its errors distributed? | `check_model_errors.py` |
| `aggregation_check.csv`, `kc_hand_vs_auto.csv` | Do the network files follow from the detector files? How does hand-read K_c compare with a rule? | `check_aggregation.py` |
| `b_bound_intervals.csv`, `b_bound_widened_replicates.csv` | Does the bound on b matter? | `check_b_bound.py` |
| `near_capacity_bins.csv` | Why do simpler forms match the model near capacity? | `diag_near_capacity.py` |

## Definitions and settings

**Screening of detector records.** Rows are removed if flow and speed are both 0, if any value is negative, or if two or more of flow, occupancy and speed are missing. Limits per UTD19 city are in `raw_processing/R/dataD.xlsx`: flow ≤ 3,000 veh/h, occupancy ≤ 1, speed ≤ 105 km/h, where they apply. Rules for individual curves, after inspection, are in `screening_decisions.csv`. Places in WebTRIS need at least 20 sites.

**From detectors to a network.** Unweighted means across detectors per time step, then divided by the network's lane count. The rule per source, as `scripts/check_aggregation.py` confirms it:
- UTD19: means of flow and density per day and interval.
- WebTRIS: rows with flow above 0; density = mean flow / (0.95 × mean speed); 15-minute Total Volume × 4 is the flow.
- PEMS: 12 × mean flow; density = that flow / (0.95 × 1.60934 × mean speed), because PEMS speeds are in mph.
- Portland: 20-second records to 5 minutes, means per station, then per time step; detectors reporting no speed left out of the speed. Seattle: 5-minute records; means over each cluster's detectors; mph converted. Both use 0.95 × speed. They are already per lane.

**Occupancy to density.** Where occupancy was reported without speed (28 UTD19 networks), density per lane = 167 × occupancy (1000 / 6, a 6 m effective vehicle), and 300 × occupancy for Toulouse. Methods gives the evidence.

**Lanes.** UTD19: the detector table's lanes, weighted by records. WebTRIS: (1 − s) × (OpenStreetMap mainline lanes, on links with at least 2) + s × (slip-road lanes), where s is the share of sites whose names end in J, K, L or M. PEMS: mainline lanes, on links with at least 2.

**Scales.** q_c, K_c and the five flows were read by hand from a plot of each curve. u_fe = 6 q(K_c/6) / K_c. `results/diagnostics/kc_hand_vs_auto.csv` compares the hand-read K_c with an automatic rule.

**Errors.** RMSE or MAE of flow, divided by the network's q_c, on observations with 0 ≤ K ≤ 1.1 K_c, then averaged with equal weight per network. By-bin values, on six equal bins of K / K_c in [0, 1.1], average each network's error within the bin.

**Best-fitting form.** Per network, the lower normalized RMSE of the cubic (a = 0.157) or Underwood's form; or, for the model's own branches, of its cubic or exponential branch. In this repository's documents "exponential" means only the model's branch d x exp(−x); Underwood's form is called Underwood's.

**Fig. 1.** The line is the mean across networks, and the band a 75% normal-theory prediction interval.

**The index with four inputs.** The three map inputs are unweighted means over each network's OpenStreetMap links and nodes, all road classes. Speed limits are averaged over tagged links only, and connectivity is the mean of in-degree and out-degree.

**Calibration.** Differential evolution (150 iterations, population 14), then Nelder–Mead and a bounded Powell polish from several starts, including six (b, w) pairs, keeping the best; a polish is kept only if it improves the fit. Bootstrap and cross-validation refits use 2 jittered restarts and 3,000 evaluations from the full-sample estimate. The objective uses 400 bins of x (`check_binning_accuracy` confirms they agree with the exact objective) and K ≤ 1.1 K_c, with a penalty of 0.5 if any network's curve is non-concave on [0.3, 1.8] K_c. Bounds: |b| ≤ 6, |w| ≤ 10. s_u = 0.2344, s_K = 0.1343, u_th = 0 and a_1 = 1 are held fixed in every refit; the moments in the index are re-derived on the training networks of each fold and on each bootstrap sample.

**Statistics.** Accuracy is out-of-fold R² from nested fivefold cross-validation over networks, seed 42, with the number of components chosen inside each training fold; `pls2_r2_CV_by_seed.csv` repeats it with four other seeds. Importance fits use a 121/20 split, seed 42. Feature pruning, median imputation, the flow PCA and the normal-score transform were applied to all 141 networks before splitting; none of them uses outcome values. A feature counts as important by each method if: VIP ≥ 1; elastic-net |coefficient| > 1e-8; Spearman q ≤ 0.05 and |ρ| ≥ 0.30; random-forest mean permutation importance > 0 over 20 repeats; selected by LASSO. LASSO: 30 alphas from 0.001 to 10, 5-fold; the chosen alpha sits at the top of this grid for most targets. Elastic net: l1 ratios 0.1, 0.5, 0.7, 0.9, 0.95 and 1, 100 alphas, 5-fold. Random forest: 300 trees.

**Seeds.** Model: 20260813 (bootstrap), and 20260813 + 9973 × repeat (cross-validation folds). Statistics and Seattle's k-means: 42.

## What isn't here

- The original downloads (`RAW_SOURCES.md`).
- Detector-level time series, including UTD19's. For UTD19, `detectors.csv` and the `day` values in each network's file identify what to rebuild from your own registered download.
- The full GeoQuery boundary files and WorldPop rasters (27 GB). The smaller GeoQuery response tables are included under `deposit/geoquery/`; the extracted attributes are provided in the network and map tables.
- Separate source-data files for every figure panel. The underlying values are included in the analysis tables described above, together with selected panel files under `results/figures/panels/`.
