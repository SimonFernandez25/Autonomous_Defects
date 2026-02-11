
import json
import os

NOTEBOOK_PATH = r"C:\Users\srfdyz\OneDrive - University of Missouri\Desktop\Defects\Autonomous_Defects\notebooks\defect_to_spectrum_physics_visuals.ipynb"

NEW_CODE = r"""
# ---------------------------------------------------------------------------
# Image Handling Utilities
# ---------------------------------------------------------------------------
import cv2
import matplotlib.patches as patches
import matplotlib.lines as mlines

def load_pillar_tile(array_name, row, col):
    # Load individual pillar tile from data/Meta_Atoms.
    # Path format: data/Meta_Atoms/{ArrayName}/{ArrayName}_{Row},{Col}.bmp
    
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
    # Construct a 9x9 montage of real pillar images.
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
    # Draws the 9x9 pillar grid centered at (center_row, center_col).
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

# 1. Nominal: Low deviation, fewest defects (robust sort)
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
plt.show()
"""

def patch_notebook():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
        

    found = False
    for cell in nb_data["cells"]:
        if cell["cell_type"] == "code":
            source_blob = "".join(cell["source"])
            if "def stitch_window_image" in source_blob or "def plot_window_schematic" in source_blob:
                # Found the cell to replace
                print("Found target cell. Replacing...")
                # Split NEW_CODE into lines for JSON format
                cell["source"] = [line + "\n" for line in NEW_CODE.splitlines()]
                found = True
                break
    
    if found:
        with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
            json.dump(nb_data, f, indent=1)
        print("Notebook patched successfully.")
    else:
        print("Could not find target cell to patch.")

if __name__ == "__main__":
    patch_notebook()
