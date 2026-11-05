"""Where everything is, relative to this repository. No other file names a location.

data/      what the pipeline starts from, shipped with the repository
results/   the results behind the paper, as shipped. scripts/run_all.py never writes here
work/      where scripts/run_all.py writes a rerun (set ROAD_NETWORK_RESULTS to change it)
raw/       only for the steps marked "raw" in scripts/run_all.py: the original downloads, laid out
           as data/RAW_SOURCES.md says (set ROAD_NETWORK_RAW if they live elsewhere)
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
NOTEBOOKS = os.path.join(ROOT, "notebooks")
SHIPPED_RESULTS = os.path.join(ROOT, "results")
RESULTS = os.environ.get("ROAD_NETWORK_RESULTS", SHIPPED_RESULTS)

# Inputs of the pipeline
PER_LANE = os.path.join(DATA, "per_lane")                      # the 141 network curves and summaries, per lane
MFD_DIR = os.path.join(PER_LANE, "MFDs")                       # data_<network>_grouped.csv
OUTPUTS_DIR = os.path.join(PER_LANE, "Outputs")                # outputs_<network>.txt
AS_REPORTED_OUTPUTS = os.path.join(DATA, "outputs_as_reported")  # the same summaries in each source's own units
MAP_METRICS = os.path.join(DATA, "map_metrics.csv")            # three map averages per network, in place of the map files
NODES_DIR = MAP_METRICS                                        # what mfd_uncertainty.py's --nodes-dir receives
SOURCE_GROUPS = os.path.join(DATA, "source_groups.csv")        # which data source each network comes from
FEATURES_PER_NETWORK = os.path.join(DATA, "features_per_network")  # attributes and summaries per network, read by notebook 01

SEED = 20260813                                                # mfd_uncertainty.py's default --seed

# The original downloads: only the steps marked "raw" read these
RAW = os.environ.get("ROAD_NETWORK_RAW", os.path.join(ROOT, "raw"))
RAW_TRAFFIC = os.path.join(RAW, "traffic")                     # PEMS, Portland, Seattle and the WebTRIS towns
RAW_UTD19 = os.path.join(RAW, "utd19")                         # detectors_public.csv and data<part>_cor.csv
RAW_CLEANED = os.path.join(RAW, "cleaned")                     # data_<city>_corre.csv (raw_processing/R/6m_clean_final.R)
RAW_CURVES = os.path.join(RAW, "curves_as_reported")           # data_<network>_grouped.csv in each source's own units
RAW_MAPS = os.path.join(RAW, "maps")                           # links_, nodes_, detectors_ and hull_ files per network
RAW_LINKS = os.path.join(RAW, "links_filtered")                # the links_ files after raw_processing/python/filter_classes.py
RAW_PORTLAND_STATIONS = os.path.join(RAW, "freeway_detectors.csv")  # PORTAL's detector-to-station table
