"""
Dataset assembly for spectral perturbation regression.

Features:
- Inputs: defect counts, defect geometry, global spatial context, AND TOPOLOGY
- Targets: 
    - delta_peak_wl (Local Deviation)
    - delta_peak_mag (Local Deviation + Spatial Residual)
    - peak_prominence (Relative metric)
- No spectral similarity metrics, no leakage.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from . import ftir_utils
from . import feature_extraction
from . import topology_features


def compute_local_baseline_and_deltas(df, value_col, result_col_name):
    """
    Compute local delta = value - mean(neighbors).
    Neighbors are defined as 3x3 grid around (local_i, local_j).
    """
    # Create grid for fast lookup
    # 13x13 grid. local_i=0..12, local_j=0..12
    # Map (local_i, local_j) -> value
    # We use a padded grid to handle boundaries easily (NaN padding)
    grid = np.full((15, 15), np.nan)
    
    # Map row indices to grid coordinates. 
    # local_i, local_j are 0-based.
    # We'll map local_i to grid row i+1, local_j to col j+1
    indices = df[['local_i', 'local_j']].values.astype(int)
    values = df[value_col].values
    
    # Fill grid
    for (r, c), v in zip(indices, values):
        grid[r+1, c+1] = v
        
    # Compute local deltas
    deltas = []
    baselines = []
    
    for (r, c), v in zip(indices, values):
        # Extract 3x3 neighborhood from padded grid
        # Center is at [r+1, c+1]
        # Neighbors are [r:r+3, c:c+3]
        # Exclude center
        neighborhood = grid[r:r+3, c:c+3].flatten()
        # Remove center (index 4 in 0..8 flat array)
        neighbors = np.delete(neighborhood, 4)
        
        # Mean of non-NaN neighbors
        valid_neighbors = neighbors[~np.isnan(neighbors)]
        
        if len(valid_neighbors) > 0:
            baseline = np.mean(valid_neighbors)
            delta = v - baseline
        else:
            # No neighbors (isolated point? shouldn't happen in full grid)
            baseline = np.nan
            delta = np.nan
            
        deltas.append(delta)
        baselines.append(baseline)
        
    return np.array(deltas), np.array(baselines)


def remove_spatial_field(df, target_col, coords_cols=['global_row', 'global_col']):
    """
    Fit a 2D polynomial (degree=2) spatial field and subtract it.
    Returns the residuals.
    """
    # Drop rows with NaN target
    mask = df[target_col].notna()
    if mask.sum() < 10:
        return df[target_col] # Too few points
    
    valid_df = df[mask]
    X = valid_df[coords_cols].values
    y = valid_df[target_col].values
    
    # Polynomial features (degree 2: 1, x, y, x^2, xy, y^2)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predict for ALL rows
    X_all = df[coords_cols].values
    X_all_poly = poly.transform(X_all)
    y_pred = model.predict(X_all_poly)
    
    # Residuals
    residuals = df[target_col].values - y_pred
    return residuals


def build_array_dataset(wavelengths, spectra, array_stem, classifications):
    """
    Build the regression dataset for one array with advanced targets.
    """
    compositions = ftir_utils.build_all_window_compositions(
        classifications, array_stem
    )
    if compositions is None:
        return None, None

    df = pd.DataFrame(compositions)
    
    # --- Step 0: Extract Topology Features ---
    # compositions now has 'defect_coords'
    if 'defect_coords' in df.columns:
        topo_records = df['defect_coords'].apply(topology_features.extract_all_topology).tolist()
        topo_df = pd.DataFrame(topo_records)
        df = pd.concat([df, topo_df], axis=1)
        # Drop raw coords to keep dataframe clean/serializable
        df.drop(columns=['defect_coords'], inplace=True)
    
    # --- Step 1: Extract Raw Peaks ---
    peak_records = []
    # spectra shape (N_points, 169)
    # compositions likely sorted by meas_idx 1..169.
    # df has 'meas_idx' column.
    
    # We trust that dataset columns 0..168 correspond to meas_idx 1..169
    # ftir_utils loads them in order.
    
    for col_idx in range(spectra.shape[1]):
        wl, mag, prom = feature_extraction.extract_dominant_peak(
            wavelengths, spectra[:, col_idx]
        )
        peak_records.append({
            'meas_idx': col_idx + 1,
            'raw_peak_wl': wl,
            'raw_peak_mag': mag,
            'peak_prominence': prom
        })
        
    peak_df = pd.DataFrame(peak_records)
    
    # Merge peak data into main df
    df = pd.merge(df, peak_df, on='meas_idx')
    
    # --- Step 2: Local Baselines (Fix 1) ---
    # Compute delta = val - mean(neighbors)
    
    # Wavelength
    delta_wl, base_wl = compute_local_baseline_and_deltas(
        df, 'raw_peak_wl', 'delta_peak_wl'
    )
    df['delta_peak_wl'] = delta_wl
    
    # Magnitude
    delta_mag, base_mag = compute_local_baseline_and_deltas(
        df, 'raw_peak_mag', 'delta_peak_mag_raw'
    )
    df['delta_peak_mag_raw'] = delta_mag
    
    # --- Step 3: Spatial Field Removal (Fix 2) ---
    # Fit and subtract spatial field from the *local delta* of magnitude
    # This cleans up any remaining global trend that neighbors didn't catch
    # (or maybe user meant fit to raw mag? But local delta logic suggests layering)
    # We'll apply it to the logical "target" candidate.
    
    df['delta_peak_mag'] = remove_spatial_field(
        df, 'delta_peak_mag_raw', ['global_row', 'global_col']
    )
    
    # Also apply spatial removal to raw prominence for a clean relative metric?
    # User: "Use peak prominence... Instead of raw delta_mag"
    # We'll provide raw prominence.
    
    # --- Combine ---
    array_id = int(array_stem.split("_")[1].split(".")[0])
    df["array_id"] = array_id
    df["array"] = array_stem

    # Normalized coordinates (-1 to 1)
    df["norm_row"] = (df["global_row"] - 11.0) / 8.0
    df["norm_col"] = (df["global_col"] - 11.0) / 6.0
    
    # Drop rows with NaNs in targets (edges where neighbors might be missing?)
    # With 3x3 grid, only isolated points fail. 
    # But remove_spatial_field handles NaNs.
    
    # Baseline info (Legacy for notebook compatibility, but empty)
    baseline_info = {
        "ideal_spectrum": np.zeros_like(wavelengths), # Dummy
        "ideal_peak_wl": 0.0,
        "ideal_peak_mag": 0.0,
        "n_zero_defect": 0,
    }

    return df, baseline_info


def build_master_dataset(array_data, classifications):
    """
    Build the merged dataset across arrays 1-3 (training/validation).
    """
    frames = []
    baselines = {}

    for stem in ["Array_1.0", "Array_2.0", "Array_3.0"]:
        if stem not in array_data:
            continue
        wl, sp = array_data[stem]
        print(f"  Assembling {stem} ...")
        df, bl = build_array_dataset(wl, sp, stem, classifications)
        if df is not None:
            frames.append(df)
            baselines[stem] = bl
            print(f"    Processed {len(df)} measurements.")

    master = pd.concat(frames, ignore_index=True)
    return master, baselines


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

BASELINE_INPUT_COLS = [
    # Defect counts
    "n_missing", "n_collapsed", "n_stitching", "n_irregular",
    "n_total_defects",
    # Defect geometry
    "min_defect_dist", "mean_defect_dist", "sum_1_over_dist",
    # Global spatial context
    "global_row", "global_col", "dist_from_center",
    "norm_row", "norm_col",
]

TOPOLOGY_COLS = [
    # Clustering
    "n_clusters_dbscan", "largest_cluster_size", "mean_cluster_size", "cluster_density",
    # Alignment
    "defect_principal_axis_ratio", "defect_principal_axis_angle", "alignment_score",
    # Spatial Topology
    "max_defect_free_chord", "defect_graph_components", "avg_graph_degree",
    # Symmetry
    "defect_centroid_offset", "quadrant_imbalance"
]

INPUT_COLS = BASELINE_INPUT_COLS + TOPOLOGY_COLS

TARGET_COLS = [
    "delta_peak_wl",
    "delta_peak_mag",   # This is now (Mag - Neighbors - SpatialField)
    "peak_prominence",  # New relative metric
]
