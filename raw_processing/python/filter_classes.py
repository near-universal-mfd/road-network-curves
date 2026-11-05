"""Filtering the links

Keeps the road classes and access values used in the attributes, and records the shares kept.

Run by the authors as cells of a notebook, in this order; each "cell" below is one of those cells. It needs the original downloads, so it was not rerun for this package. The only
changes are the locations, which now start from ROAD_NETWORK_RAW (default: raw/, from the repository's root).
"""
import os

RAW_ROOT = os.environ.get("ROAD_NETWORK_RAW", "raw")


# %% cell 1 of 2
from pathlib import Path
import re
import pandas as pd

# ===================== CONFIG =====================
# Root of your project
ROOT = Path(os.path.join(RAW_ROOT, "maps_project"))

# Input links directory (where the raw links_*.csv live)
LINKS_DIR = ROOT / "Links"

# Where to write filtered copies
FILTERED_DIR = LINKS_DIR / "Links"

# Per-file summary CSV (kept/removed counts per links_*.csv)
SUMMARY_CSV = FILTERED_DIR / "links_filter_summary.csv"

# Network properties file to update with the 3 columns
NETWORK_PROPS_PATH = ROOT / "Nodes" / "Network_Properties.csv"

# File pattern for input link files
INPUT_PATTERN = "links_*.csv"

# CSV I/O settings
READ_KW  = {"encoding": "utf-8-sig"}
WRITE_KW = {"encoding": "utf-8-sig", "index": False}

# Which Class prefixes to keep (case-insensitive, leading/trailing spaces ignored)
KEEP_PREFIXES = ("motorway", "trunk", "primary", "secondary", "tertiary", "residential")

# City name extraction: strip "link"/"links" + separators from start of filename stem
_PREFIX_RE = re.compile(r"^(links?|links)([_\-\s]+)", flags=re.IGNORECASE)

# ==================================================


def _file_to_city(fn: str) -> str:
    stem = Path(fn).stem  # e.g., "links_amsterdam"
    cleaned = _PREFIX_RE.sub("", stem)  # remove leading "link(s)" + separator(s)
    return cleaned.replace("_", " ").strip()


def _norm_city(s) -> str:
    return str(s).replace("_", " ").strip().lower()


def filter_links_and_write_summary() -> pd.DataFrame:
    """
    Filters each links_*.csv by Class prefix, writes filtered copies to FILTERED_DIR,
    and returns a summary DataFrame with columns:
        file, total_rows, kept_rows, removed_rows, note
    Also writes SUMMARY_CSV to disk.
    """
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    files = sorted(LINKS_DIR.glob(INPUT_PATTERN))
    if not files:
        print(f"[WARN] No files matched {INPUT_PATTERN} in {LINKS_DIR}")

    for csv_path in files:
        try:
            df = pd.read_csv(csv_path, **READ_KW)
        except Exception as e:
            msg = f"read_error: {e}"
            print(f"[ERROR] {csv_path.name}: {msg}")
            summary_rows.append({
                "file": csv_path.name,
                "total_rows": None,
                "kept_rows": None,
                "removed_rows": None,
                "note": msg
            })
            continue

        total = len(df)
        if "Class" not in df.columns:
            msg = "missing_Class"
            print(f"[SKIP] {csv_path.name}: {msg}")
            kept_df = df.iloc[0:0].copy()  # empty
            kept_df.to_csv(FILTERED_DIR / csv_path.name, **WRITE_KW)
            summary_rows.append({
                "file": csv_path.name,
                "total_rows": int(total),
                "kept_rows": 0,
                "removed_rows": int(total),
                "note": msg
            })
            continue

        cls = df["Class"].astype(str).str.strip().str.lower()
        keep_mask = cls.str.startswith(KEEP_PREFIXES)
        kept_df = df[keep_mask].copy()

        kept = int(len(kept_df))
        removed = int(total - kept)

        # Write filtered copy (same filename) into FILTERED_DIR/
        kept_df.to_csv(FILTERED_DIR / csv_path.name, **WRITE_KW)

        summary_rows.append({
            "file": csv_path.name,
            "total_rows": int(total),
            "kept_rows": kept,
            "removed_rows": removed,
            "note": ""
        })

    summary_df = pd.DataFrame(summary_rows, columns=["file", "total_rows", "kept_rows", "removed_rows", "note"])
    summary_df.to_csv(SUMMARY_CSV, **WRITE_KW)

    print("\n=== Per-file filtering summary ===")
    if not summary_df.empty:
        # Print with safe conversion to string
        print(summary_df.to_string(index=False))
    print(f"\nWrote filtered files to: {FILTERED_DIR}")
    print(f"Wrote summary CSV to:   {SUMMARY_CSV}\n")

    return summary_df


def update_network_properties_with_counts(summary_df: pd.DataFrame) -> None:
    """
    Using the per-file summary_df, compute kept proportion and merge
    (total_rows, kept_rows, kept_prop) into Network_Properties.csv by matching on City.
    """
    if summary_df.empty:
        print("[WARN] Summary is empty; Network_Properties.csv will not be updated.")
        return

    # Prepare summary with City + proportions
    sdf = summary_df.copy()
    required = {"file", "total_rows", "kept_rows"}
    missing = required - set(sdf.columns)
    if missing:
        raise ValueError(f"Internal summary missing required columns: {sorted(missing)}")

    sdf["City"] = sdf["file"].astype(str).map(_file_to_city)
    sdf["kept_prop"] = sdf.apply(
        lambda r: (float(r["kept_rows"]) / float(r["total_rows"])) if pd.notnull(r["total_rows"]) and r["total_rows"] not in (0, None) else None,
        axis=1
    )
    sdf_small = sdf[["City", "total_rows", "kept_rows", "kept_prop"]].copy()
    sdf_small["_key"] = sdf_small["City"].map(_norm_city)

    # Load network properties
    try:
        npdf = pd.read_csv(NETWORK_PROPS_PATH, **READ_KW)
    except Exception as e:
        raise FileNotFoundError(f"Could not read Network_Properties.csv at {NETWORK_PROPS_PATH}: {e}")

    if "City" not in npdf.columns:
        raise ValueError("Network_Properties.csv must contain a 'City' column.")

    npdf["_key"] = npdf["City"].map(_norm_city)

    merged = npdf.merge(sdf_small.drop(columns=["City"]), on="_key", how="left")
    merged.drop(columns=["_key"], inplace=True)

    # Write back
    merged.to_csv(NETWORK_PROPS_PATH, **WRITE_KW)

    # Diagnostics
    unmatched_in_np = sdf_small[~sdf_small["_key"].isin(set(npdf["_key"]))]["City"].sort_values().unique().tolist()
    if unmatched_in_np:
        print("Warning: Cities present in summary but not found in Network_Properties:", unmatched_in_np)

    print("Done. Added/updated columns in Network_Properties.csv: total_rows, kept_rows, kept_prop")
    print(f"Path: {NETWORK_PROPS_PATH}\n")


def main():
    summary_df = filter_links_and_write_summary()
    update_network_properties_with_counts(summary_df)


if __name__ == "__main__":
    main()

# %% cell 2 of 2
from pathlib import Path
import re
import pandas as pd

# ===================== CONFIG =====================
# Root of your project
ROOT = Path(os.path.join(RAW_ROOT, "maps_project"))

# Input links directory (where the raw links_*.csv live)
LINKS_DIR = ROOT / "Links" / "Links"

# Where to write filtered copies
FILTERED_DIR = LINKS_DIR / "Filtered"

# Per-file summary CSV (kept/removed counts per links_*.csv)
SUMMARY_CSV = FILTERED_DIR / "links_filter_summary.csv"

# Network properties file to update with the 3 columns
NETWORK_PROPS_PATH = ROOT / "Nodes" / "Network_Properties.csv"

# File pattern for input link files
INPUT_PATTERN = "links_*.csv"

# CSV I/O settings
READ_KW  = {"encoding": "utf-8-sig"}
WRITE_KW = {"encoding": "utf-8-sig", "index": False}

# Acceptable Access values (case-insensitive, leading/trailing spaces ignored)
ACCESS_ALLOWED = {"yes", "no", "permissive"}

# City name extraction: strip "link"/"links" + separators from start of filename stem
_PREFIX_RE = re.compile(r"^(links?|links)([_\-\s]+)", flags=re.IGNORECASE)

# ==================================================


def _file_to_city(fn: str) -> str:
    stem = Path(fn).stem  # e.g., "links_amsterdam"
    cleaned = _PREFIX_RE.sub("", stem)  # remove leading "link(s)" + separator(s)
    return cleaned.replace("_", " ").strip()


def _norm_city(s) -> str:
    return str(s).replace("_", " ").strip().lower()


def filter_links_and_write_summary() -> pd.DataFrame:
    """
    Filters each links_*.csv by Access (keep only yes/no/permissive), writes filtered copies to FILTERED_DIR,
    and returns a summary DataFrame with columns:
        file, total_rows, kept_rows, removed_rows, note
    Also writes SUMMARY_CSV to disk.
    """
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    files = sorted(LINKS_DIR.glob(INPUT_PATTERN))
    if not files:
        print(f"[WARN] No files matched {INPUT_PATTERN} in {LINKS_DIR}")

    for csv_path in files:
        try:
            df = pd.read_csv(csv_path, **READ_KW)
        except Exception as e:
            msg = f"read_error: {e}"
            print(f"[ERROR] {csv_path.name}: {msg}")
            summary_rows.append({
                "file": csv_path.name,
                "total_rows": None,
                "kept_rows": None,
                "removed_rows": None,
                "note": msg
            })
            continue

        total = len(df)
        if "Access" not in df.columns:
            msg = "missing_Access"
            print(f"[SKIP] {csv_path.name}: {msg}")
            kept_df = df.iloc[0:0].copy()  # empty
            kept_df.to_csv(FILTERED_DIR / csv_path.name, **WRITE_KW)
            summary_rows.append({
                "file": csv_path.name,
                "total_rows": int(total),
                "kept_rows": 0,
                "removed_rows": int(total),
                "note": msg
            })
            continue

        access = df["Access"].astype(str).str.strip().str.lower()
        keep_mask = access.isin(ACCESS_ALLOWED)
        kept_df = df[keep_mask].copy()

        kept = int(len(kept_df))
        removed = int(total - kept)

        # Write filtered copy (same filename) into FILTERED_DIR/
        kept_df.to_csv(FILTERED_DIR / csv_path.name, **WRITE_KW)

        summary_rows.append({
            "file": csv_path.name,
            "total_rows": int(total),
            "kept_rows": kept,
            "removed_rows": removed,
            "note": ""
        })

    summary_df = pd.DataFrame(summary_rows, columns=["file", "total_rows", "kept_rows", "removed_rows", "note"])
    summary_df.to_csv(SUMMARY_CSV, **WRITE_KW)

    print("\n=== Per-file filtering summary ===")
    if not summary_df.empty:
        # Print with safe conversion to string
        print(summary_df.to_string(index=False))
    print(f"\nWrote filtered files to: {FILTERED_DIR}")
    print(f"Wrote summary CSV to:   {SUMMARY_CSV}\n")

    return summary_df


def update_network_properties_with_counts(summary_df: pd.DataFrame) -> None:
    """
    Using the per-file summary_df, compute kept proportion and merge
    (total_rows, kept_rows, kept_prop) into Network_Properties.csv by matching on City.
    """
    if summary_df.empty:
        print("[WARN] Summary is empty; Network_Properties.csv will not be updated.")
        return

    # Prepare summary with City + proportions
    sdf = summary_df.copy()
    required = {"file", "total_rows", "kept_rows"}
    missing = required - set(sdf.columns)
    if missing:
        raise ValueError(f"Internal summary missing required columns: {sorted(missing)}")

    sdf["City"] = sdf["file"].astype(str).map(_file_to_city)
    sdf["kept_prop"] = sdf.apply(
        lambda r: (float(r["kept_rows"]) / float(r["total_rows"])) if pd.notnull(r["total_rows"]) and r["total_rows"] not in (0, None) else None,
        axis=1
    )
    sdf_small = sdf[["City", "total_rows", "kept_rows", "kept_prop"]].copy()
    sdf_small["_key"] = sdf_small["City"].map(_norm_city)

    # Load network properties
    try:
        npdf = pd.read_csv(NETWORK_PROPS_PATH, **READ_KW)
    except Exception as e:
        raise FileNotFoundError(f"Could not read Network_Properties.csv at {NETWORK_PROPS_PATH}: {e}")

    if "City" not in npdf.columns:
        raise ValueError("Network_Properties.csv must contain a 'City' column.")

    npdf["_key"] = npdf["City"].map(_norm_city)

    merged = npdf.merge(sdf_small.drop(columns=["City"]), on="_key", how="left")
    merged.drop(columns=["_key"], inplace=True)

    # Write back
    merged.to_csv(NETWORK_PROPS_PATH, **WRITE_KW)

    # Diagnostics
    unmatched_in_np = sdf_small[~sdf_small["_key"].isin(set(npdf["_key"]))]["City"].sort_values().unique().tolist()
    if unmatched_in_np:
        print("Warning: Cities present in summary but not found in Network_Properties:", unmatched_in_np)

    print("Done. Added/updated columns in Network_Properties.csv: total_rows, kept_rows, kept_prop")
    print(f"Path: {NETWORK_PROPS_PATH}\n")


def main():
    summary_df = filter_links_and_write_summary()
    update_network_properties_with_counts(summary_df)


if __name__ == "__main__":
    main()

