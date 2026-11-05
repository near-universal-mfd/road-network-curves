# Map files and GeoQuery files

Nothing in `results/` needs these unless you rerun the steps marked `raw`. `data/features_per_network/` and `data/map_metrics.csv` hold what the pipeline takes from them.

## `maps/<map>.zip`: one per network

`<map>` is the `map_files` column of `data/networks.csv`. `python scripts/unpack_maps.py` unpacks them all into `raw/` (about 5 GB). Each holds:

| File | One row per | Columns |
|---|---|---|
| `maps/nodes_<map>.csv` | node in the network's hull, from OpenStreetMap through OSMnx | `NodeID`, `Longitude`, `Latitude`; monthly daytime and nighttime land surface temperature and precipitation (`01_day_st` … `12_precip`); land-cover shares; elevation, slope, distance to water and to the coast, wind speed potential, vegetation index; `InflowCount`, `OutflowCount` |
| `maps/links_<map>.csv` | link | `LinkID`, `StartNode`, `EndNode`, `Class`, `Lanes`, `LinkLength` (m), `Curvature` (length over straight-line distance), `SpeedLimit` (km/h, where tagged), `Oneway`, `Bridge`, `Tunnel`, `Access`, `Roundabout`, `num_detectors`, `avg_detector_pos` |
| `links_filtered/links_<map>.csv` | link kept by `raw_processing/python/filter_classes.py` | The same columns. The statistics' link attributes come from these |
| `maps/detectors_<name>.csv` | detector or site | Coordinates, and an identifier where the source gives one |
| `maps/snapped_<name>.csv` | detector or site | Coordinates after snapping to the nearest link |
| `maps/hull_<name>.geojson` | | The detectors' hull |
| `maps/missing_attrs_<name>.csv` | attribute | OpenStreetMap attributes with no values in this network |

Los Angeles's detector, snapped, hull and missing-attribute files keep the spelling `losanageles` from UTD19's table.

© OpenStreetMap contributors, under the Open Database Licence. Covariates extracted from GeoQuery are under the terms of each layer (`data/RAW_SOURCES.md`).

## `geoquery/<country>.zip`: one per country

The covariates as GeoQuery returned them: `results.csv`, one row per administrative unit, and `request_details.json`, naming the boundary file and layers requested, with the requester's email and the request's identifier removed. The boundary files (1.8 GB) and WorldPop's rasters (27 GB) aren't here. Both are public (`data/RAW_SOURCES.md`).
