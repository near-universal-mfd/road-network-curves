"""Each UTD19 network's usual speed limit (the median over its detectors) against its effective free-flow speed at the common
occupancy factor of 167. Writes diagnostics/speed_limits.csv.
"""
import os

import pandas as pd

from paths import PER_LANE, RAW_UTD19, RESULTS

S = pd.read_csv(os.path.join(PER_LANE, "network_scales.csv")).set_index("city")
D = pd.read_csv(os.path.join(RAW_UTD19, "detectors_public.csv"), low_memory=False)
D["city"] = D.citycode.astype(str).str.lower().replace({"losanageles": "losangeles"})
D["limit"] = pd.to_numeric(D["limit"], errors="coerce").where(lambda v: v > 0)      # 0 means not known
T = S[S.source == "UTD19"][["density_method"]].join(D.groupby("city").limit.median().rename("limit_median"))
T["uf_at_167"] = S.uf * (S.factor / (1000 / 6)).fillna(1.0)
T["speed_over_limit_at_167"] = T.uf_at_167 / T.limit_median
T.to_csv(os.path.join(RESULTS, "diagnostics", "speed_limits.csv"))
