# From the original downloads to the network curves

This folder holds the steps before `data/per_lane`, as the authors ran them. They need the original downloads (`data/RAW_SOURCES.md`), so they were **not rerun for this package**, and `run_all.py` doesn't call them. Their outputs are what the rest of the pipeline starts from, and the rebuild in `scripts/check_aggregation.py` confirms the link between them: means over detectors per time step reproduce every shared time step of all 37 UTD19 network files, both PEMS files and the six WebTRIS places tested.

The only changes from what was run are the locations, which now start from `ROAD_NETWORK_RAW` (default `raw/`, from the repository's root), and the corrections listed below. Each Python file is the cells of one notebook, in the order they were run.

Besides the folders in `data/RAW_SOURCES.md`, this code uses `raw/detector_tables/` (site and detector tables as downloaded, and Seattle's clusters), `raw/detector_lists/` (one detector list per network, with `Geographical Data/` holding the GeoQuery and WorldPop files), `raw/maps_project/Links/` (the filter's input), and `raw/outputs_manual/` (the hand-read summaries as typed).

| Stage | File | What it does | Writes |
|---|---|---|---|
| 00 | `R/2_accessing_data.R` | Splits UTD19's `utd19_u.csv` into parts, one per city | `data<part>.csv` |
| 00 | `python/pems_portland_seattle_webtris.py` | Seattle clusters and curves; Portland's 5-minute records and curve; WebTRIS download, places and curves | network curves |
| 01 | `R/1_data_availability.R`, `R/3_clean.R`, `R/5_clean_neg.R`, `R/6m_clean_final.R`, `R/7_post_cleaning_observations.R` | Screening of detector records, by the limits in `R/dataD.xlsx`; density from occupancy or from flow and speed | `data<part>_cor.csv`, `data_<city>_corre.csv` |
| 02 | `R/rename_grouped_files.R`, `R/8_grouping.R` | Names each network file from a table of parts and cities; plots each curve for inspection | `data_<network>_grouped.csv`, plots |
| 02 | `python/postprocessing.py` | Screening of individual curves after visual inspection (`data/screening_decisions.csv`) | the same files |
| 03 | `python/read_scales.py` | q_c, K_c and five flows, read by hand from a plot of each curve | `outputs_<network>.txt` (`data/outputs_as_reported`) |
| 04 | `python/map_attributes.py`, `python/filter_classes.py` | Links, nodes and hulls from OpenStreetMap; GeoQuery covariates; WorldPop sums; road-class and access filters | the map deposit, `data/network_properties.csv` |

After these come `scripts/check_lanes.py`, `scripts/measure_factor.py`, `scripts/check_factor_from_free_flow.py` and `scripts/build_per_lane_data.py`, which put every network per lane, with one occupancy factor and corrected speeds. Those are in `run_all.py`, marked `raw`.

## Corrections made for this package

- `rename_grouped_files.R` names each file from a table of parts and cities. Its first version paired the numbered files with an alphabetical list by position, which gave 12 UTD19 networks another city's name; the paper's analysis uses the corrected names.
- `postprocessing.py`: two rules first keyed to `essen` and `toronto` are keyed to `darmstadt` and `torino`, whose data those two labels held when the rules were written.
- `R/dataD.xlsx`: the rows are named, using the same table, and part 5 (Bolton) now says flow and speed. It first said occupancy, but the network file the paper uses is built from flow and speed (`scripts/check_aggregation.py` reproduces it exactly).

`R/6m_clean_final.R` still sets density = 500 × occupancy, the factor when the curves were built. `scripts/build_per_lane_data.py` replaces it with 167 per lane (300 for Toulouse).

## What wasn't kept

The script that flattened PEMS04 and PEMS08 from `.npz` to `traffic04.csv` and `traffic08.csv` wasn't found, and neither was the first script that aggregated detectors to network files; `scripts/check_aggregation.py` reconstructs the rule and checks it.
