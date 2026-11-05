"""Write a point estimate in the format of mfd_uncertainty.py --warm-start. Usage: python make_warm_start.py [FOLDER]"""
import json
import os
import sys

from paths import RESULTS  # noqa: E402
FOLDER = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RESULTS, "MFDs_Curve_uncertainty")
pe = json.load(open(os.path.join(FOLDER, "point_estimate.json")))
hp = pe["hybrid_params"]
ws = {"hybrid_params": {"a_h": hp["a"], "S_U": hp["s_u"], "S_K": hp["s_K"]},
      "exp_params": {"EXP_A0": hp["d"]},
      "ui_params": {"bias_b": hp["b"], "weights_w": [hp["w_uf"], hp["w_L"], hp["w_SL"], hp["w_C"]]}}
out = os.path.join(FOLDER, "warm_start_from_point_estimate.json")
json.dump(ws, open(out, "w"), indent=1)
print("wrote", out, ws)
