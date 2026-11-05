"""Unpack the map files in deposit/maps to where the raw steps look for them: raw/maps and raw/links_filtered (about 5 GB; set ROAD_NETWORK_RAW to put them elsewhere).

Usage: python unpack_maps.py [network ...]   (default: all 141; names as in the map_files column of data/networks.csv)
"""
import glob
import os
import shutil
import sys
import zipfile

from paths import DATA, RAW, RAW_MAPS, ROOT

names = sys.argv[1:]
files = [os.path.join(ROOT, "deposit", "maps", n + ".zip") for n in names] or sorted(glob.glob(os.path.join(ROOT, "deposit", "maps", "*.zip")))
for i, p in enumerate(files, 1):
    with zipfile.ZipFile(p) as z:
        z.extractall(RAW)
    print(f"{i}/{len(files)} {os.path.basename(p)}", flush=True)
shutil.copyfile(os.path.join(DATA, "network_properties.csv"), os.path.join(RAW_MAPS, "Network_Properties.csv"))
print("unpacked into", RAW)
