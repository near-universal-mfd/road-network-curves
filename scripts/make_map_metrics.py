"""The three map averages of the four-input index per network (mean link length, mean tagged speed limit, mean node connectivity), from the
map files. Writes data/map_metrics.csv.
"""
import glob
import os

import pandas as pd

import mfd_uncertainty as M
from paths import MAP_METRICS, MFD_DIR, OUTPUTS_DIR, RAW_MAPS

rows = []
for p in sorted(glob.glob(os.path.join(MFD_DIR, "data_*_grouped.csv"))):
    city = os.path.basename(p)[5:-12]
    got = M.compute_city_metrics(city, OUTPUTS_DIR, RAW_MAPS)
    assert got is not None, city
    _uf, L, SL, C, _qc, _kc = got
    rows.append(dict(city=city, map_name=M.MAP_NAME.get(city, city), avg_link_length=L, avg_speed_limit=SL, node_connectivity=C))
    print(city, flush=True)
pd.DataFrame(rows).to_csv(MAP_METRICS, index=False, float_format="%.17g")
print("wrote", MAP_METRICS)
