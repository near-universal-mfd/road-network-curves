# The original sources

These aren't redistributed here, because of their size and terms. `results/` and `data/` reproduce every number in the paper without them. You only need them for the steps marked `raw` in `python scripts/run_all.py --list`, and for `raw_processing/`.

| Source | Provider and link | Version or period | Terms | Subset used | Read by |
|---|---|---|---|---|---|
| UTD19 | ETH Zurich, [utd19.ethz.ch](https://utd19.ethz.ch) (free registration) | `utd19_u.csv`, `detectors_public.csv`, `links.csv` | Academic and non-commercial use; cite the DOI | 37 of the 39 parts into which the scripts split the file. Parts 22 and 36 (Melbourne, Utrecht) report flow alone, so no density can be computed | `raw_processing/R/` |
| PEMS04, PEMS08 | [ASTGCN repository](https://github.com/wanhuaiyu/ASTGCN), from Caltrans [PeMS](https://pems.dot.ca.gov) | Bay Area, 307 detectors, Jan–Feb 2018; San Bernardino, 170 detectors, Jul–Aug 2016; 5 min | Attribute | All detectors | `traffic04.csv`, `traffic08.csv` (flattened from the `.npz` files; that short script wasn't kept) |
| Portland | [PORTAL](https://new.portal.its.pdx.edu) | I-205 northbound and southbound, 15 Sep – 15 Nov 2011; 20 s, aggregated to 5 min | None stated | All detectors in `freeway_detectors.csv` | `raw_processing/python/pems_portland_seattle_webtris.py` |
| Seattle | [Seattle Loop Data](https://github.com/zhiyongc/Seattle-Loop-Data), University of Washington STAR Lab; collected by WSDOT | 2015; 5 min; speeds in mph | Attribute | About 700 detectors, in five k-means clusters; clusters 1 and 5 kept | same |
| WebTRIS | [National Highways](https://webtris.nationalhighways.co.uk), API v1.0 | MIDAS sites, daily reports, 15 min, 1 Dec 2024 – 28 Feb 2025 | Free and open; Open Government Licence for related data | Sites grouped into places by reverse geocoding; 99 places with at least 20 sites | same |
| OpenStreetMap | [planet.openstreetmap.org](https://planet.openstreetmap.org), through OSMnx 2.0.4 (`graph_from_polygon`, all road types) | 2025; files dated 25 Jun – 2 Jul 2025 | ODbL | Each network's detector hull plus a 2,700 m buffer | `raw_processing/python/map_attributes.py` |
| GeoQuery | [geo.aiddata.org](https://geo.aiddata.org) | Requests made on 11–12 May 2025, one per country (12) | Cite GeoQuery and each layer | geoBoundaries v4; GPW v4.11 (2020); Global Wind Atlas v2; SRTM v4.1; World Vector Shorelines v2.3.6 and distance to water; NASA LTDR v5 AVHRR NDVI (2020); MODIS MCD12Q1 v5.1 (2012); MODIS MOD11C3 v6.1 (2023; for Taiwan's night values, v6.0, 2020); CRU TS v4.07 (2022); gridded GDP (about 2006) | same |
| WorldPop | [worldpop.org](https://www.worldpop.org) | 2020 constrained, by age and sex: [doi:10.5258/SOTON/WP00699](https://doi.org/10.5258/SOTON/WP00699), and for the United States [doi:10.5258/SOTON/WP00695](https://doi.org/10.5258/SOTON/WP00695) | CC BY 4.0 | 36 rasters per country | same |
| City statistics | Public online sources, including Excel's data types | Compiled by hand in 2025; individual sources and years weren't recorded | — | City area, population, employment rate, median income, in `data/network_properties.csv` | same |

The GeoQuery request for Austria returned monthly temperature in place of precipitation, so Graz's and Innsbruck's precipitation is treated as missing (`notebooks/01_statistics.ipynb`).

## The map deposit

The node and link tables per network are about 5 GB, so they're kept apart from this repository. Nothing in `results/` needs them unless you rerun the steps marked `raw`, because `data/features_per_network/` and `data/map_metrics.csv` hold what the pipeline takes from them. See `deposit/README.md`.

## Where the raw steps look

Put the files under `raw/` in the repository's root, or set `ROAD_NETWORK_RAW` to another folder:

```
raw/
  traffic/                traffic04.csv, traffic08.csv, Portland.csv, Portland_5min.csv, seattle.csv, webtris_15min_all_sites.csv, Engl_cities/England_15min_cities_<place>.csv
  utd19/                  utd19_u.csv, detectors_public.csv, links.csv, data<part>.csv, data<part>_cor.csv
  cleaned/                data_<city>_corre.csv
  curves_as_reported/     data_<network>_grouped.csv, in each source's own units
  maps/                   links_<map>.csv, nodes_<map>.csv, detectors_<map>.csv, hull_<map>.geojson, Network_Properties.csv
  links_filtered/         links_<map>.csv after raw_processing/python/filter_classes.py
  freeway_detectors.csv   PORTAL's table of detectors and stations
```

`<map>` is the name in the `map_files` column of `data/networks.csv`. It differs from the network's name for London, Birmingham and Manchester, whose WebTRIS and UTD19 versions carry each other's map names, and UTD19's detector table spells Los Angeles `losanageles`.
