import csv
import numpy as np
import os
import math
import matplotlib.pyplot as plt

BASE_DIR = r"c:\Users\srfdyz\Downloads\Defect\Meta_Atoms"
OUTPUT_DIR = r"c:\Users\srfdyz\Downloads\Defect\debug_output"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# CONFIG
PITCH = 12.0
ROWS = 21
COLS = 21
MEAS_GRID_SIZE = 13
MEAS_STEP = 12.0
MEAS_START_OFFSET = (5.0, 5.0) # (x, y) rel to Bottom-Left Pillar (21, 1)
WINDOW_X_RANGE = (-50.0, 80.0)  # [min, max] rel to MP
WINDOW_Y_RANGE = (-50.0, 100.0) # [min, max] rel to MP

def load_data(csv_path):
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = int(row['Row'])
            c = int(row['Col'])
            cat = row['Category']
            data[(r,c)] = cat
    return data

def get_pillar_xy_global(r, c):
    # Origin (0,0) is Pillar (21, 1) -> Bottom-Left
    # X = (Col - 1) * Pitch
    # Y = (21 - Row) * Pitch
    x = (c - 1) * PITCH
    y = (ROWS - r) * PITCH
    return x, y

def get_meas_point_global(mr, mc):
    # mr, mc in 1..13
    # Start offset (5, 5) rel to (0,0)
    # Step 12.0 right (mc) and up (mr)
    x = MEAS_START_OFFSET[0] + (mc - 1) * MEAS_STEP
    y = MEAS_START_OFFSET[1] + (mr - 1) * MEAS_STEP
    return x, y

def process_array(array_name):
    csv_path = os.path.join(BASE_DIR, f"{array_name}_results.csv")
    if not os.path.exists(csv_path):
        print(f"Skipping {array_name}")
        return

    print(f"Grouping {array_name}...")
    pillar_data = load_data(csv_path)
    
    output_rows = []
    
    # Store points for visualization
    vis_data = {} # meas_id -> [(rel_x, rel_y, cat), ...]

    meas_id = 0
    # "starting from the bottom left to right and up"
    # Outer loop = Y (Up), Inner loop = X (Right) ??
    # Usually "Scan lines" logic.
    # Grid Row 1 (Bottom) -> Col 1..13. Then Row 2..
    for mr in range(1, MEAS_GRID_SIZE + 1):
        for mc in range(1, MEAS_GRID_SIZE + 1):
            meas_id += 1
            mp_x, mp_y = get_meas_point_global(mr, mc)
            
            # Find Pillars in Window
            # Window Global Bounds:
            w_min_x = mp_x + WINDOW_X_RANGE[0]
            w_max_x = mp_x + WINDOW_X_RANGE[1]
            w_min_y = mp_y + WINDOW_Y_RANGE[0]
            w_max_y = mp_y + WINDOW_Y_RANGE[1]
            
            # Convert Bounds to Indices (Virtual)
            # x = (c-1)*P => c = x/P + 1
            # y = (21-r)*P => r = 21 - y/P
            
            # We must iterate integer indices that cover these ranges.
            # Safety margin included by floor/ceil
            c_min = math.floor(w_min_x / PITCH) + 1
            c_max = math.ceil(w_max_x / PITCH) + 1
            
            # Y is inverted relative to Row
            # min_y corresponds to max_r, max_y to min_r
            r_max_f = ROWS - (w_min_y / PITCH)
            r_min_f = ROWS - (w_max_y / PITCH)
            
            # Indices
            r_min = math.floor(r_min_f)
            r_max = math.ceil(r_max_f)
            
            w_points = []
            
            for r in range(r_min, r_max + 1):
                for c in range(c_min, c_max + 1):
                    # Calc global pos
                    px, py = get_pillar_xy_global(r, c)
                    
                    # Rel Pos
                    rel_x = px - mp_x
                    rel_y = py - mp_y
                    
                    # Check Exact Window
                    in_x = WINDOW_X_RANGE[0] <= rel_x <= WINDOW_X_RANGE[1]
                    in_y = WINDOW_Y_RANGE[0] <= rel_y <= WINDOW_Y_RANGE[1]
                    
                    if in_x and in_y:
                        dist = math.sqrt(rel_x**2 + rel_y**2)
                        
                        # Determine Category
                        # inside Array (1..21)
                        if 1 <= r <= ROWS and 1 <= c <= COLS:
                            cat = pillar_data.get((r,c), "Missing") 
                        else:
                            # OOB Handling
                            # Top Rows (r < 1) or Right Cols (c > 21) -> Missing
                            if r < 1 or c > COLS:
                                cat = "Missing"
                            else:
                                # Bottom Rows (r > 21) or Left Cols (c < 1) -> Good
                                cat = "Good"
                        
                        output_rows.append({
                            'Measurement_ID': meas_id,
                            'Meas_Grid_Row': mr,
                            'Meas_Grid_Col': mc,
                            'Pillar_Row': r,
                            'Pillar_Col': c,
                            'Rel_X': rel_x,
                            'Rel_Y': rel_y,
                            'Distance': dist,
                            'Category': cat
                        })
                        
                        w_points.append((rel_x, rel_y, cat))
            
            vis_data[meas_id] = w_points

    # Save CSV
    out_csv = os.path.join(BASE_DIR, f"{array_name}_ftir_groups.csv")
    fieldnames = ['Measurement_ID','Meas_Grid_Row','Meas_Grid_Col','Pillar_Row','Pillar_Col','Rel_X','Rel_Y','Distance','Category']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Saved {out_csv}")
    
    # Plot Verification Heatmaps (Corner Points)
    vis_data_heatmap = {}
    for mid, points in vis_data.items():
        # Convert list of points to grid for imshow? No, scatter squares is easier for irregular grid
        vis_data_heatmap[mid] = points
        
    plot_ids = [1, 13, 157, 169]
    plot_window_heatmap(vis_data_heatmap, plot_ids, array_name)
    
    # Plot Full Array Map (Overlay)
    plot_full_array_map(pillar_data, array_name)

def plot_window_heatmap(vis_data, ids, array_name):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    # Map categories to int for colormap or just colors
    color_map = {
        'Good': 'green', 'Missing': 'red', 'Irregular': 'orange',
        'Collapsed': 'black', 'Stitching': 'blue'
    }
    
    for ax, mid in zip(axes, ids):
        if mid not in vis_data: continue
        points = vis_data[mid]
        
        # Unpack
        rxs = [p[0] for p in points]
        rys = [p[1] for p in points]
        cats = [p[2] for p in points]
        colors = [color_map.get(c, 'gray') for c in cats]
        
        # Plot as dense squares
        ax.scatter(rxs, rys, c=colors, marker='s', s=150, edgecolors='none', alpha=0.9)
        
        # Meas Point
        ax.plot(0, 0, 'wx', markersize=10, markeredgewidth=2, label='MP')
        
        # Boundary
        ax.axvline(WINDOW_X_RANGE[0], color='k', linestyle='--')
        ax.axvline(WINDOW_X_RANGE[1], color='k', linestyle='--')
        ax.axhline(WINDOW_Y_RANGE[0], color='k', linestyle='--')
        ax.axhline(WINDOW_Y_RANGE[1], color='k', linestyle='--')
        
        ax.set_title(f"Meas {mid} (Array {array_name})")
        ax.set_aspect('equal')
        ax.grid(False) # Clean look
        ax.set_facecolor('#f0f0f0')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='s', color='w', markerfacecolor=c, label=k, markersize=15)
                       for k, c in color_map.items()]
    fig.legend(handles=legend_elements, loc='upper center', ncol=5)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_img = os.path.join(OUTPUT_DIR, f"{array_name}_ftir_verify_heatmap.png")
    plt.savefig(out_img)
    # print(f"Saved {out_img}")

def plot_full_array_map(pillar_data, array_name):
    # Visualize Global Layout
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. Plot all pillars
    pxs, pys, cs = [], [], []
    color_map = {'Good': 'green', 'Missing': 'red', 'Irregular': 'orange', 'Collapsed': 'black', 'Stitching': 'blue'}
    
    for r in range(1, ROWS+1):
        for c in range(1, COLS+1):
            x, y = get_pillar_xy_global(r, c)
            cat = pillar_data.get((r,c), 'Gray') # Gray if missing data
            pxs.append(x)
            pys.append(y)
            cs.append(color_map.get(cat, 'gray'))
            
    ax.scatter(pxs, pys, c=cs, s=20, marker='o', alpha=0.5, label='Pillars')
    
    # 2. Plot Measurement Points & Windows (Corners only to avoid clutter)
    # Corners: 1, 13, 157, 169
    corner_ids = [1, 13, 157, 169]
    meas_count = 0
    
    for mr in range(1, MEAS_GRID_SIZE + 1):
        for mc in range(1, MEAS_GRID_SIZE + 1):
            meas_count += 1
            mx, my = get_meas_point_global(mr, mc)
            
            # Plot center
            ax.plot(mx, my, 'k+', markersize=5, alpha=0.3)
            
            # Plot Box for corners
            if meas_count in corner_ids:
                rect = plt.Rectangle((mx + WINDOW_X_RANGE[0], my + WINDOW_Y_RANGE[0]),
                                     WINDOW_X_RANGE[1] - WINDOW_X_RANGE[0],
                                     WINDOW_Y_RANGE[1] - WINDOW_Y_RANGE[0],
                                     edgecolor='purple', facecolor='none', linewidth=2, label=f'Win {meas_count}')
                ax.add_patch(rect)
                ax.text(mx, my, str(meas_count), color='purple', fontweight='bold')

    # Draw Array Bounds
    # (0,0) is (21,1) Bottom Left
    # Bounds: X [0, 240], Y [0, 240]
    ax.add_patch(plt.Rectangle((0, 0), (COLS-1)*PITCH, (ROWS-1)*PITCH, 
                               edgecolor='blue', facecolor='none', linewidth=3, linestyle='--', label='Array Bounds'))

    ax.set_title(f"Full Array Layout: {array_name}\nBlue Box = Array, Purple Boxes = Corner Windows")
    ax.set_xlabel("Global X (um)")
    ax.set_ylabel("Global Y (um)")
    ax.axis('equal')
    ax.legend(loc='upper right')
    
    out_img = os.path.join(OUTPUT_DIR, f"{array_name}_full_map.png")
    plt.savefig(out_img)
    # print(f"Saved {out_img}")

def main():
    arrays = ["Array_1Crop", "Array_2Crop", "Array_3Crop"]
    for arr in arrays:
        process_array(arr)

if __name__ == "__main__":
    main()
