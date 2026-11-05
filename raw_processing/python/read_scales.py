"""The seven summaries of each curve

First an automatic pass; then a prompt at which q_c, K_c and the five flows are read by hand from a plot of each curve. The values typed in are shipped in data/outputs_as_reported.

Run by the authors as cells of a notebook, in this order; each "cell" below is one of those cells. It needs the original downloads, so it was not rerun for this package. The only
changes are the locations, which now start from ROAD_NETWORK_RAW (default: raw/, from the repository's root).
"""
import os

RAW_ROOT = os.environ.get("ROAD_NETWORK_RAW", "raw")


# %% cell 1 of 2
import os
import pandas as pd
import glob
import math

# =========================
# Starting neighborhood threshold
initial_threshold = 0.75  # <--- You can change this value
threshold_increment = 0.25  # <--- Increase step if NaNs found
# =========================

# Define input and output paths
input_folder = os.path.join(RAW_ROOT, "curves_as_reported")
output_folder = os.path.join(RAW_ROOT, "outputs_as_reported")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to process one city with a given threshold and return output + nan_count + list of values
def process_city(df, kc, city_name, neighborhood_threshold):
    # Recalculate qc based on neighborhood of kc
    qc_neighborhood = df[
        (df["density"] >= kc - neighborhood_threshold) &
        (df["density"] <= kc + neighborhood_threshold)
    ]
    qc = qc_neighborhood["flow"].mean() if not qc_neighborhood.empty else float('nan')

    # Define the 5 neighborhood points
    k_points = [kc * frac for frac in [1/6, 1/3, 1/2, 2/3, 5/6]]
    avg_flows = []

    # Debugging: count NaNs
    nan_count = 0

    for k in k_points:
        neighborhood = df[
            (df["density"] >= k - neighborhood_threshold) &
            (df["density"] <= k + neighborhood_threshold)
        ]
        avg_flow = neighborhood["flow"].mean() if not neighborhood.empty else float('nan')
        if math.isnan(avg_flow):
            nan_count += 1
        avg_flows.append(avg_flow)

    if math.isnan(qc):
        nan_count += 1

    output_values = [qc, kc] + avg_flows
    output_string = ",".join(f"{v:.6g}" for v in output_values)

    return output_string, nan_count, output_values


# Process each CSV file
problematic_cities = []

for filepath in glob.glob(os.path.join(input_folder, "data_*_grouped.csv")):
    filename = os.path.basename(filepath)
    city_name = filename.replace("data_", "").replace("_grouped.csv", "")
    print(f"\n[INFO] Processing {city_name}...")

    # Read data
    df = pd.read_csv(filepath, usecols=["flow", "density"])

    # Find tentative qc and kc
    idx_qc_tentative = df["flow"].idxmax()
    kc = df.loc[idx_qc_tentative, "density"]

    # Start with the initial threshold and iterate until no NaNs
    current_threshold = initial_threshold
    final_values = None
    while True:
        output_string, nan_count, output_values = process_city(df, kc, city_name, current_threshold)
        print(f"[DEBUG] {city_name}: Threshold={current_threshold} -> NaNs={nan_count}")
        if nan_count == 0:
            # Save for later progression check
            final_values = output_values
            # Write output to file
            output_path = os.path.join(output_folder, f"outputs_{city_name}.txt")
            with open(output_path, "w") as f:
                f.write(output_string)
            break
        else:
            current_threshold += threshold_increment

    # --- Flow progression check ---
    if final_values is not None:
        qc = final_values[0]
        # indices: 0=qc, 1=kc, 2=flow at 1/6, 3=flow at 1/3, 4=flow at 1/2, 5=flow at 2/3, 6=flow at 5/6
        f1 = final_values[2]
        f2 = final_values[3]
        f3 = final_values[4]
        f4 = final_values[5]
        f5 = final_values[6]
        fmax = final_values[0]

        deltas = [
            f1 - 0,
            f2 - f1,
            f3 - f2,
            f4 - f3,
            f5 - f4,
            fmax - f5
        ]

        # Check monotonic decreasing with 10% leeway
        progression_ok = True
        for i in range(len(deltas) - 1):
            if not (deltas[i] >= 0.9 * deltas[i+1]):
                progression_ok = False
                break

        if not progression_ok:
            problematic_cities.append(city_name)
            print(f"[WARNING] {city_name} flagged for problematic flow progression!")

# Summary of problematic cities
if problematic_cities:
    print("\n[SUMMARY] Cities with problematic flow progression:")
    for c in problematic_cities:
        print(" -", c)
else:
    print("\n[SUMMARY] No cities with problematic flow progression.")

print("\nProcessing completed.")


# %% cell 2 of 2
###### import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

# =========================
# Define input and output paths
input_folder = (
    os.path.join(RAW_ROOT, "curves_as_reported")
)
output_folder = (
    os.path.join(RAW_ROOT, "outputs_manual")
)

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
# =========================


def beautify_axes(ax, x_range=None, y_range=None):
    """
    Make x and y labels (tick labels) a little more crowded (but not too dense)
    by increasing the maximum allowed number of major ticks and adding minor ticks.
    """
    # Determine a reasonable upper bound for major ticks (about 8–10)
    # Use data range if provided (helps matplotlib choose better tick spacing).
    if x_range is None:
        x_min, x_max = ax.get_xlim()
    else:
        x_min, x_max = x_range
        ax.set_xlim(x_min, x_max)

    if y_range is None:
        y_min, y_max = ax.get_ylim()
    else:
        y_min, y_max = y_range
        ax.set_ylim(y_min, y_max)

    # Set up major ticks: allow up to 10 (usually results in 7–10 actually used)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))

    # Add minor ticks at roughly 2 subdivisions between majors
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Slightly reduce tick label size for readability if crowded
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.tick_params(
        axis="both", which="minor", length=3, labelsize=0
    )  # no labels for minor ticks

    # Optional: light grid for minors, solid for majors
    ax.grid(True, which="major", linestyle="-", alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", alpha=0.3)


def base_scatter(df, city_name, title):
    fig, ax = plt.subplots()
    ax.scatter(df["density"], df["flow"], s=10)
    ax.set_title(title)
    ax.set_xlabel("Density")
    ax.set_ylabel("Flow")
    beautify_axes(ax)
    return fig, ax


def plot_and_get_input(df, city_name):
    # Step 1: scatter plot
    fig, ax = base_scatter(
        df, city_name, f"{city_name}: Flow vs Density\nStep 1: Input max flow"
    )
    plt.show(block=True)

    max_flow = float(input(f"[INPUT] {city_name}: Enter max flow (qc): "))

    # Plot with horizontal line
    fig, ax = base_scatter(
        df, city_name, f"{city_name}: Step 2: Input density at that flow"
    )
    ax.axhline(y=max_flow, color="red", linestyle="--", label=f"Max flow = {max_flow}")
    ax.legend()
    plt.show(block=True)

    density_at_max = float(
        input(f"[INPUT] {city_name}: Enter density at max flow (kc): ")
    )

    # Plot with vertical line
    fig, ax = base_scatter(
        df, city_name, f"{city_name}: Calculating 5 intermediate density points"
    )
    ax.axhline(y=max_flow, color="red", linestyle="--")
    ax.axvline(x=density_at_max, color="red", linestyle="--")
    plt.show(block=False)  # Just show for info; continue below

    # Compute intermediate density points
    factors = [1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6]
    intermediate_densities = [density_at_max * f for f in factors]

    intermediate_flows = []
    for i, kd in enumerate(intermediate_densities, start=1):
        fig, ax = base_scatter(
            df, city_name, f"{city_name}: Step {i+2}: Input flow at density {kd:.2f}"
        )
        ax.axhline(y=max_flow, color="red", linestyle="--")
        ax.axvline(x=density_at_max, color="red", linestyle="--")
        ax.axvline(x=kd, color="green", linestyle="--", label=f"Step {i}: Density={kd:.2f}")
        ax.legend()
        plt.show(block=True)

        flow_at_k = float(input(f"[INPUT] {city_name}: Enter flow at density {kd:.2f}: "))
        intermediate_flows.append(flow_at_k)

    # Combine outputs
    output_values = [max_flow, density_at_max] + intermediate_flows
    return output_values


# =========================
# Main processing loop
# =========================
for filepath in glob.glob(os.path.join(input_folder, "data_*_grouped.csv")):
    filename = os.path.basename(filepath)
    city_name = filename.replace("data_", "").replace("_grouped.csv", "")
    output_path = os.path.join(output_folder, f"outputs_{city_name}.txt")

    # --- NEW CHECK -----------------------------------------------------------
    # Skip processing if the output already exists
    if os.path.exists(output_path):
        print(f"[INFO] Output for {city_name} already exists at {output_path}. Skipping.")
        continue
    # ------------------------------------------------------------------------

    print(f"\n[INFO] Processing {city_name} (manual mode)...")

    # Read data
    df = pd.read_csv(filepath, usecols=["flow", "density"])

    # Get values interactively
    output_values = plot_and_get_input(df, city_name)

    # Save to output file
    output_string = ",".join(f"{v:.6g}" for v in output_values)
    with open(output_path, "w") as f:
        f.write(output_string)

    print(f"[INFO] Saved manual output for {city_name} to {output_path}")

print("\n[INFO] Manual selection processing completed.")

