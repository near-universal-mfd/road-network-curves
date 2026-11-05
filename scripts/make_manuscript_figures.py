"""Figs 1-4 and Extended Data Figs 1-4 from the results. Writes figures/."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from paths import RESULTS  # noqa: E402
FIG = os.path.join(RESULTS, "figures")
PAN = os.path.join(FIG, "panels")
os.makedirs(PAN, exist_ok=True)
W = 2160
LETTER = ImageFont.truetype(font_manager.findfont(
    font_manager.FontProperties(family="DejaVu Sans", weight="bold")), 64)


def crop_to_axes(img, margin=1):
    """Drop a title: cut just above the first row that is mostly dark (the top spine)."""
    g = np.asarray(img.convert("L"))
    dark = (g < 110).mean(axis=1)
    rows = np.nonzero(dark > 0.45)[0]
    top = max(0, int(rows[0]) - margin) if rows.size else 0
    return img.crop((0, top, img.width, img.height))


def trim(img, pad=4):
    g = np.asarray(img.convert("L"))
    ys, xs = np.nonzero(g < 245)
    return img.crop((max(0, xs.min() - pad), max(0, ys.min() - pad),
                     min(img.width, xs.max() + pad), min(img.height, ys.max() + pad)))


def compose(panels, ncols, letters, cell_h=None):
    """Place panels on a white grid, bold letters at each cell's top-left."""
    cw = W // ncols
    lpad, tpad = 86, 20
    rows = []
    for i in range(0, len(panels), ncols):
        row = panels[i:i + ncols]
        scaled = []
        for p in row:
            s = (cw - lpad - 10) / p.width
            scaled.append(p.resize((int(p.width * s), int(p.height * s)), Image.LANCZOS))
        rows.append(scaled)
    heights = [max(p.height for p in r) + tpad + 10 for r in rows]
    canvas = Image.new("RGB", (W, sum(heights)), "white")
    d = ImageDraw.Draw(canvas)
    y, k = 0, 0
    for r, h in zip(rows, heights):
        for j, p in enumerate(r):
            x = j * cw
            canvas.paste(p, (x + lpad, y + tpad))
            d.text((x + 6, y), letters[k], font=LETTER, fill="black")
            k += 1
        y += h
    return canvas


# ---------------------------------------------------------------- Fig. 1
c = pd.read_csv(os.path.join(RESULTS, "cone_mean_CI_outputs", "factors_cone_curve.csv"))
plt.figure(figsize=(8, 5), dpi=270)
plt.fill_between(c["k_over_kc"], c["band_lower"], c["band_upper"], alpha=0.25, linewidth=0)
plt.plot(c["k_over_kc"], c["mean"], linewidth=2)
plt.xlabel("K / Kc")
plt.ylabel("Fraction to scale (density * uf)")
plt.ylim(0, np.nanmax(c["band_upper"]) * 1.05)
plt.xlim(0, 1.0)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "Fig1.png"), bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------- Fig. 2
Image.open(os.path.join(RESULTS, "Fig2_uf_index", "fig2_uf_index.png")).convert("RGB").save(
    os.path.join(FIG, "Fig2.png"))

# ---------------------------------------------------------------- Figs 3 and 4
F = os.path.join(RESULTS, "MFDs_FINAL")
p3 = [crop_to_axes(Image.open(os.path.join(F, "figures_per_city", n)).convert("RGB"))
      for n in ("cagliari_mfd_fits.png", "Wanborough_mfd_fits.png")]
compose(p3, 2, "ab").save(os.path.join(FIG, "Fig3.png"))
p4 = [crop_to_axes(Image.open(os.path.join(F, n)).convert("RGB"))
      for n in ("errors_summary_across_cities_all_models.png", "errors_rmse_vs_Kratio_bins.png")]
compose(p4, 2, "ab").save(os.path.join(FIG, "Fig4.png"))

# ---------------------------------------------------------------- Extended Data Fig. 3
p3e = [crop_to_axes(Image.open(os.path.join(F, n)).convert("RGB"))
       for n in ("errors_summary_across_cities_no_hybrid.png",
                 os.path.join("figures_per_city_no_hybrid", "cagliari_mfd_fits_nohybrid.png"),
                 os.path.join("figures_per_city_no_hybrid", "Wanborough_mfd_fits_nohybrid.png"))]
compose(p3e, 2, "abc").save(os.path.join(FIG, "EDFig3.png"))

# ---------------------------------------------------------------- Extended Data Fig. 1
ed1 = [trim(Image.open(os.path.join(PAN, n)).convert("RGB")) for n in ("edfig1a_bordeaux.png", "edfig1b_konstanz.png", "edfig1c_cagliari.png")]
compose(ed1, 2, "abc").save(os.path.join(FIG, "EDFig1.png"))

# ---------------------------------------------------------------- Extended Data Fig. 4
e = pd.read_csv(os.path.join(F, "per_city_errors_normalized.csv"))
und = (e.RMSE_Underwood < e.RMSE_Cubic).to_numpy()
best = np.where(und, e.RMSE_Underwood, e.RMSE_Cubic)
plt.figure(figsize=(8, 4.87), dpi=270)
plt.scatter(e.uf[~und], best[~und], s=38, c="tab:blue", edgecolors="black", linewidths=0.5, label="Cubic", zorder=3)
plt.scatter(e.uf[und], best[und], s=38, c="tab:green", edgecolors="black", linewidths=0.5, label="Underwood", zorder=3)
plt.xlabel("u_fe (km/h)")
plt.ylabel("Best RMSE (normalized)")
plt.grid(True, alpha=0.3)
plt.legend(title="Selected model", loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "EDFig4.png"), bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------- Extended Data Fig. 2
S = os.path.join(RESULTS, "StatResults")


def volcano(target, path):
    t = pd.read_csv(os.path.join(S, f"corr_predictors_vs_{target}.csv"))
    sig = t["is_significant_bh_0.05"].astype(bool)
    y = -np.log10(t["q_bh"].clip(lower=1e-300))
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.scatter(t.loc[~sig, "rho"].abs(), y[~sig], s=16, alpha=0.7, label="NS")
    ax.scatter(t.loc[sig, "rho"].abs(), y[sig], s=16, alpha=0.9, label="BH q<0.05")
    ax.set_xlabel("|Spearman rho|")
    ax.set_ylabel("-log10(q_BH)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Image.open(path).convert("RGB")


pa = volcano("qc", os.path.join(PAN, "edfig2a_volcano_qc.png"))
pb = volcano("kc", os.path.join(PAN, "edfig2b_volcano_kc.png"))

r2 = pd.read_csv(os.path.join(S, "pls2_all_targets_r2_CV.csv")).iloc[:7]
labels = ["qc", "kc", "q(kc/6)", "q(kc/3)", "q(kc/2)", "q(2kc/3)", "q(5kc/6)"]
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
ax.bar(labels, r2["r2_cv_oof"].to_numpy())
ax.axhline(0, color="0.3", linewidth=0.8)
ax.set_ylabel("Cross-validated R²")
ax.set_ylim(min(0.0, r2["r2_cv_oof"].min() - 0.05), 0.8)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
fig.savefig(os.path.join(PAN, "edfig2c_pls_r2.png"))
plt.close(fig)
pc = Image.open(os.path.join(PAN, "edfig2c_pls_r2.png")).convert("RGB")

GROUPS = {
    "link_LinkLength_max": "Link Length",
    "link_SpeedLimit_min": "Speed Limit",
    "link_Tunnel_frac": "Fraction of Tunnels",
    "link_num_detectors_max": "Detector Distribution",
    "link_num_detectors_mean": "Detector Distribution",
    "link_avg_detector_pos_std": "Detector Distribution",
    "GDP (Million US$)": "GDP",
    "Pop Density (ppkm2)": "Population Density",
    "cropland_natural_vegetation_mosaic": "Natural Vegetation",
    "deciduous_broadleaf_forest": "Natural Vegetation",
    "kept_prop_classes": "Link Class (primary, secondary,...)",
    "link_Class_primary_link": "Link Class (primary, secondary,...)",
    "link_Class_secondary": "Link Class (primary, secondary,...)",
    "link_Class_tertiary_link": "Link Class (primary, secondary,...)",
    "link_Class_trunk": "Link Class (primary, secondary,...)",
    "link_Lanes_q50": "Lane Count",
    "link_SpeedLimit_missing_frac": "Speed Limit Availability",
    "node_01_day_st_std": "Surface Temperature",
    "node_12_day_st_std": "Surface Temperature",
    "link_Access_customers": "Access Restrictions",
    "link_Access_destination": "Access Restrictions",
    "link_Access_permit": "Access Restrictions",
    "node_08_precip_min": "Precipitation",
    "node_InflowCount_std": "Node Connectivity",
    "node_distance_to_coast_min": "Distance to Coast",
}
cons = pd.read_csv(os.path.join(S, "diagnostics", "consensus_overall.csv"))
top = cons[cons["count_methods"] >= 4].copy()
missing = sorted(set(top["feature"]) - set(GROUPS))
if missing:
    raise SystemExit(f"attributes with >= 4 methods but no group label: {missing}")
top["group"] = top["feature"].map(GROUPS)
top.to_csv(os.path.join(PAN, "edfig2d_groups.csv"), index=False)
grp = top.groupby("group")["count_methods"].max().reset_index()
grp = grp.sort_values(["count_methods", "group"], ascending=[True, False])
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
ax.barh(grp["group"], grp["count_methods"], height=0.45, color="#1f5f7a")
ax.set_xlim(0, 5)
ax.set_xlabel("Number of Methods Agreeing")
ax.grid(axis="x", color="0.85")
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(PAN, "edfig2d_consensus.png"))
plt.close(fig)
pd_ = Image.open(os.path.join(PAN, "edfig2d_consensus.png")).convert("RGB")
compose([pa, pb, pc, pd_], 2, "abcd").save(os.path.join(FIG, "EDFig2.png"))
print("groups in panel d:", len(grp), grp.values.tolist())
print("done:", sorted(os.listdir(FIG)))
