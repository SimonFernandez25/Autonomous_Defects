
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines


# Resolve project root
PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "Utilities":
    PROJECT_ROOT = PROJECT_ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Adjust path for execution context if running from scripts/Utilities
if str(Path.cwd()).endswith("Utilities"):
    PROJECT_ROOT = Path.cwd().parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ML import ftir_utils, dataset_assembly, ml_models

# Styling
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.grid": False
})

# Color palette (Colorblind safe)
COLORS = {
    "Good": "#d9d9d9",       # Light Gray
    "Missing": "#ffffff",    # White (Empty)
    "Collapsed": "#d62728",  # Red
    "Irregular": "#ff7f0e",  # Orange
    "Stitching": "#1f77b4",  # Blue
    "Measurement": "#2ca02c" # Green (Overlay)
}

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ml_outputs", "physics_visuals")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Output directory: {OUTPUT_DIR}")

# --- 0. Data Loading ---

# Load Pre-computed Regression Dataset
dataset_path = os.path.join(PROJECT_ROOT, "ml_outputs", "regression_dataset.csv")

# FORCE REBUILD to ensure we match the codebase logic and not some stale file
print("Forcing dataset rebuild...")
FTIR_DIR = os.path.join(PROJECT_ROOT, "data")
CSV_PATH = os.path.join(PROJECT_ROOT, "results", "meta_atoms_classified.csv")
array_data = ftir_utils.load_all_arrays(FTIR_DIR)
classifications = ftir_utils.load_classifications(CSV_PATH)

df, _ = dataset_assembly.build_master_dataset(array_data, classifications)
# Fill NaNs for distance
dist_fill = {
    'min_defect_dist': 999.0,
    'mean_defect_dist': 999.0,
    'sum_1_over_dist': 0.0
}
df.fillna(dist_fill, inplace=True)
print(f"Rebuilt dataset: {len(df)} rows")

# Load Raw Classifications to Map
CSV_PATH = os.path.join(PROJECT_ROOT, "results", "meta_atoms_classified.csv")
classifications = ftir_utils.load_classifications(CSV_PATH)
print(f"Loaded classifications for {list(classifications.keys())}")


# --- 1. Measurement Window Visualization ---


# ---------------------------------------------------------------------------
# Image Handling Utilities
# ---------------------------------------------------------------------------
import cv2
import matplotlib.patches as patches
import matplotlib.lines as mlines

def load_pillar_tile(array_name, row, col):
    """
    Load individual pillar tile from data/Meta_Atoms.
    Path format: data/Meta_Atoms/{ArrayName}/{ArrayName}_{Row},{Col}.bmp
    """
    # Map Array_1.0 -> Array_1Crop
    folder_name = ftir_utils.DPT_TO_CSV_NAME.get(array_name, array_name)
    
    # Construct path
    filename = f"{folder_name}_{row},{col}.bmp"
    path = os.path.join(PROJECT_ROOT, "data", "Meta_Atoms", folder_name, filename)
    
    if os.path.exists(path):
        # Read as grayscale
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return img
    else:
        # Fallback: Create a blank placeholder
        return np.full((32, 32), 200, dtype=np.uint8)

def stitch_window_image(df_row, pillar_map, border_size=2):
    """
    Construct a 9x9 montage of real pillar images.
    Returns:
        montage: The stitched image.
        overlays: List of (x, y, color) for defect highlighting.
    """
    center_row = int(df_row["global_row"])
    center_col = int(df_row["global_col"])
    array_name = df_row["array"]
    
    window_half = 4
    
    # We'll determine tile size dynamically from the first valid load
    sample = load_pillar_tile(array_name, center_row, center_col)
    h, w = sample.shape
    
    # Create canvas
    # 9 rows, 9 cols
    montage_h = (h + border_size) * 9 + border_size
    montage_w = (w + border_size) * 9 + border_size
    montage = np.full((montage_h, montage_w), 255, dtype=np.uint8)
    
    rects = [] # For drawing colored borders later
    
    for dr in range(-window_half, window_half + 1):
        for dc in range(-window_half, window_half + 1):
            r, c = center_row + dr, center_col + dc
            
            # Grid indices (0..8)
            grid_r = window_half - dr # Top is row 0
            grid_c = dc + window_half
            
            # Load
            tile = load_pillar_tile(array_name, r, c)
            if tile.shape != (h, w):
                tile = cv2.resize(tile, (w, h))
            
            # Paste coords
            y0 = border_size + grid_r * (h + border_size)
            x0 = border_size + grid_c * (w + border_size)
            
            montage[y0:y0+h, x0:x0+w] = tile
            
            # Defect info
            defect_type = pillar_map.get((r, c), "Good")
            if defect_type != "Good":
                color = COLORS.get(defect_type, "red")
                rects.append((x0, y0, w, h, color, defect_type))

    return montage, rects

def plot_window_schematic(ax, center_row, center_col, pillar_map, title=""):
    """Draws the 9x9 pillar grid centered at (center_row, center_col)."""
    window_half = 4
    pitch = 12.0 # um
    
    # Grid extent (relative microns)
    extent = [-4.5 * pitch, 4.5 * pitch, -4.5 * pitch, 4.5 * pitch]
    
    # Draw pillars
    for dr in range(-window_half, window_half + 1):
        for dc in range(-window_half, window_half + 1):
            r, c = center_row + dr, center_col + dc
            
            # Determine color
            defect_type = pillar_map.get((r, c), "Good")
            
            # Relative coords
            x = dc * pitch
            y = dr * pitch
            
            color = COLORS.get(defect_type, "gray")
            
            # Draw circle
            if defect_type == "Missing":
                # Empty circle
                circ = patches.Circle((x, y), radius=3.5, facecolor="white", edgecolor="#bbbbbb", linewidth=1)
            else:
                circ = patches.Circle((x, y), radius=3.5, facecolor=color, edgecolor="none")
            ax.add_patch(circ)

    # Measurement footprint (approximate circle or box)
    rect = patches.Rectangle((-4.5*pitch, -4.5*pitch), 9*pitch, 9*pitch, 
                             linewidth=1.5, edgecolor="#555555", facecolor="none", linestyle="--")
    ax.add_patch(rect)
    
    # Crosshair
    ax.plot([0], [0], "+", color="k", markersize=10, markeredgewidth=1.5)
    
    ax.set_xlim(extent[0] - 5, extent[1] + 5)
    ax.set_ylim(extent[2] - 5, extent[3] + 5)
    ax.set_aspect("equal")
    ax.axis("off")  # Turn off axes for clean look
    ax.set_title(title, fontsize=11, fontweight="bold")

# -- Select Representative Windows --
# Strategy: Find "Most Deviated" and "Nominal"
dataset_vis = df.copy()
dataset_vis["abs_deviation"] = dataset_vis["delta_peak_mag"].abs()

# 1. Nominal: Low deviation, fewest defects
# Sort by defects (asc) then deviation (asc) to find best "clean" example
nominal = dataset_vis.sort_values(["n_total_defects", "abs_deviation"], ascending=[True, True]).iloc[0]

# 2. Most Deviated (Physical Impact)
most_deviated = dataset_vis.sort_values("abs_deviation", ascending=False).iloc[0]

# 3. High Defect Density (Visual Chaos)
dense = dataset_vis.sort_values("n_total_defects", ascending=False).iloc[0]

examples = [nominal, most_deviated, dense]
labels = [
    f"Nominal Reference\n(0 Defects, Dev: {nominal['delta_peak_mag']:.3f})",
    f"High Spectral Impact\n({int(most_deviated['n_total_defects'])} Defects, Dev: {most_deviated['delta_peak_mag']:.3f})",
    f"Max Defect Density\n({int(dense['n_total_defects'])} Defects, Dev: {dense['delta_peak_mag']:.3f})"
]

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

for ax, row, label in zip(axes, examples, labels):
    array_name = row["array"]
    map_name = ftir_utils.DPT_TO_CSV_NAME.get(array_name, None)
    pillar_map = classifications.get(map_name, {})
    
    # Generate Stitch
    montage, overlays = stitch_window_image(row, pillar_map)
    
    # Plot
    ax.imshow(montage, cmap="gray")
    
    # Draw Borders/Overlays
    for (x, y, w, h, color, dtype) in overlays:
        # Border
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
    
    # Measurement Footprint (Circle overlay)
    cy, cx = montage.shape[0] // 2, montage.shape[1] // 2
    radius = min(montage.shape) * 0.45
    circ = patches.Circle((cx, cy), radius=radius, edgecolor="lime", facecolor="none", 
                          linewidth=1.5, linestyle="--", alpha=0.8)
    ax.add_patch(circ)
    
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.axis("off")

# Visualization Legend
legend_elements = [
    patches.Patch(facecolor='none', edgecolor=COLORS['Missing'], label='Missing', linewidth=2),
    patches.Patch(facecolor='none', edgecolor=COLORS['Collapsed'], label='Collapsed', linewidth=2),
    patches.Patch(facecolor='none', edgecolor=COLORS['Irregular'], label='Irregular', linewidth=2),
    patches.Patch(facecolor='none', edgecolor=COLORS['Stitching'], label='Stitching', linewidth=2),
    mlines.Line2D([0], [0], color='lime', linestyle='--', label='FTIR Beam'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.02), frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig1_Measurement_Windows_Real.png"), bbox_inches="tight", dpi=300)
print("Saved Fig1 (Real)")



# --- 2. Synthetic Thought Experiment ---

# Base features for model training (must match dataset_assembly.INPUT_COLS or what we use)
features = dataset_assembly.INPUT_COLS
target = "delta_peak_mag"

X = df[features]
y = df[target]
rf_model = ml_models.train_rf(X.values, y.values)
print("RF Model trained.")

def create_synthetic_features(defect_coords, center_row=11, center_col=11):
    """Compute feature vector for a synthetic defect arrangement."""
    feat_dict = {col: 0.0 for col in features}
    
    n_defects = len(defect_coords)
    feat_dict["n_missing"] = n_defects 
    feat_dict["n_total_defects"] = n_defects
    
    # Geometry
    dists = []
    pitch = 12.0
    for dr, dc in defect_coords:
        d = np.sqrt((dr*pitch)**2 + (dc*pitch)**2)
        dists.append(d)
    
    dists = np.array(dists)
    feat_dict["min_defect_dist"] = np.min(dists) if len(dists) > 0 else 999.0
    feat_dict["mean_defect_dist"] = np.mean(dists) if len(dists) > 0 else 999.0
    feat_dict["sum_1_over_dist"] = np.sum(1.0 / np.maximum(dists, 1e-6)) if len(dists) > 0 else 0.0
    
    return pd.Series(feat_dict)

configs = [
    {
        "name": "Clustered",
        "coords": [(0, 0), (0, 1), (1, 0)], # Center + neighbors
        "desc": "High local impact"
    },
    {
        "name": "Aligned",
        "coords": [(0, -2), (0, 0), (0, 2)], # Line through center
        "desc": "Directional"
    },
    {
        "name": "Dispersed",
        "coords": [(-3, -3), (3, 3), (-3, 3)], # Corners
        "desc": "Low interaction"
    }
]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Plot Nominal Reference First
plot_window_schematic(axes[0], 11, 11, {}, "Nominal Reference")
axes[0].text(0, -50, "Baseline", ha="center", color="gray")

feature_rows = []
for ax, config in zip(axes[1:], configs):
    # Create map
    syn_map = {}
    for dr, dc in config["coords"]:
        syn_map[(11+dr, 11+dc)] = "Missing"
    
    # Visualization
    plot_window_schematic(ax, 11, 11, syn_map, config["name"])
    
    # Compute features & Predict
    feats = create_synthetic_features(config["coords"])
    feature_rows.append(feats)

# Batch Predict
syn_df = pd.DataFrame(feature_rows)
# Fill missing
for col in features:
    if col not in syn_df.columns:
        syn_df[col] = df[col].mean()

preds = rf_model["model"].predict(syn_df[features])

# Annotate
for i, (ax, config) in enumerate(zip(axes[1:], configs)):
    impact = preds[i]
    color = "red" if abs(impact) > 0.05 else "orange"
    ax.text(0, -60, f"Pred Impact: {impact:.3f}", 
            ha="center", fontweight="bold", color=color, fontsize=10)
    ax.text(0, -75, config["desc"], ha="center", fontsize=9, style="italic")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig2_Defect_Replacement.png"), bbox_inches="tight", dpi=300)
print("Saved Fig2")



# --- 3. Radial Influence Plot ---
fig, ax = plt.subplots(figsize=(7, 5))

subset = df[df["n_total_defects"] > 0]
x = subset["sum_1_over_dist"]
y = subset["delta_peak_mag"]

# sns.regplot replacement
ax.scatter(x, y, alpha=0.3, s=15, color="#1f77b4")
# Simple linear fit for the line
if len(x) > 1:
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m*x + b, color="#d62728", linewidth=2)

ax.set_xlabel(r"Defect Proximity $\sum (1/r)$  [$1/\mu m$]")
ax.set_ylabel("Peak Magnitude Deviation")
ax.set_title("Radial Influence: Closer Defects Cause Larger Pertubations")

ax.text(0.05, 0.95, "Strong Local Influence", transform=ax.transAxes, 
        verticalalignment='top', fontsize=10, fontweight='bold', color='#d62728')
ax.text(0.8, 0.1, "Weak Distant Influence", transform=ax.transAxes, 
        verticalalignment='bottom', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig3_Radial_Influence.png"), bbox_inches="tight", dpi=300)
print("Saved Fig3")



# --- 4. Window-Averaged Defect Field ---

def plot_array_heatmap(ax, df_sub, value_col, title, cmap="viridis", vmin=None, vmax=None, center_diverging=False):
    grid = np.full((13, 13), np.nan)
    for _, row in df_sub.iterrows():
        i, j = int(row['local_i']), int(row['local_j'])
        grid[12-i, j] = row[value_col]
        
    if center_diverging:
        max_abs = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)))
        if vmin is None: vmin = -max_abs
        if vmax is None: vmax = max_abs
        
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, extent=[0, 21, 0, 21])
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    return im

target_array = "Array_1.0"
sub_df = df[df["array"] == target_array]
if len(sub_df) == 0:
    target_array = df["array"].iloc[0] # Fallback
    sub_df = df[df["array"] == target_array]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

im1 = plot_array_heatmap(axes[0], sub_df, "n_total_defects", "Defect Count", cmap="Reds")
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

im2 = plot_array_heatmap(axes[1], sub_df, "delta_peak_mag", "Peak Mag Deviation", 
                         cmap="RdBu_r", center_diverging=True)
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

im3 = plot_array_heatmap(axes[2], sub_df, "delta_peak_wl", "Peak Shift (um)", 
                         cmap="RdBu_r", center_diverging=True)
plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

fig.suptitle(f"Macroscopic Field Maps: {target_array}", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig4_Array_Fields.png"), bbox_inches="tight", dpi=300)
print("Saved Fig4")



# --- 5. Minimal ML Parity Plot ---
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Use the same features/target as before
X_val = df[features].values
y_val = df[target].values

# We need a pipeline because we scale inside the CV
# ml_models doesn't expose a predict_cv utility directly easier than this
kf = KFold(n_splits=5, shuffle=True, random_state=42)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1))
])

print("Running Cross-Validation for Parity Plot...")
y_pred = cross_val_predict(pipe, X_val, y_val, cv=kf)
y_true = y_val

from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_true, y_pred, alpha=0.3, color="#2ca02c", edgecolors='none', s=20)

min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], "--k", linewidth=1.5, alpha=0.6)

ax.text(0.05, 0.9, f"$R^2 = {r2:.2f}$", transform=ax.transAxes, 
        fontsize=14, fontweight="bold", color="#333333")

ax.set_xlabel("Measured Deviation")
ax.set_ylabel("Predicted Deviation (Physics-Features)")
ax.set_title("Model Validation")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig5_ML_Parity.png"), bbox_inches="tight", dpi=300)
print("Saved Fig5")

