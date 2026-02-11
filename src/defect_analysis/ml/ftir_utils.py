"""
FTIR data loading, spatial mapping, and 9x9 window defect composition.

Handles .dpt file I/O, wavenumber-to-wavelength conversion,
spatial grid mapping (21x21 pillar arrays, 13x13 measurement grids),
and defect composition analysis for each measurement window.
"""

import numpy as np
import pandas as pd
import os
import math


# ---------------------------------------------------------------------------
# Array geometry constants
# ---------------------------------------------------------------------------
PITCH = 12.0          # um between adjacent pillars
ROWS = 21
COLS = 21
MEAS_GRID_SIZE = 13   # inner measurement grid dimension
WINDOW_HALF = 4       # 9x9 window: center +/- 4 pillars

# Mapping from .dpt filename stems to classification CSV array names
DPT_TO_CSV_NAME = {
    "Array_1.0": "Array_1Crop",
    "Array_2.0": "Array_2Crop",
    "Array_3.0": "Array_3Crop",
    "Array_4.0": None,           # no classification labels
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dpt(filepath):
    """
    Load a .dpt FTIR file (tab-delimited, no header).

    Returns
    -------
    wavenumbers : np.ndarray, shape (N,)
    spectra : np.ndarray, shape (N, 169)
    """
    data = np.loadtxt(filepath, delimiter="\t")
    return data[:, 0], data[:, 1:]


def wavenumber_to_wavelength(wavenumbers):
    """Convert wavenumber (cm-1) to wavelength (um)."""
    return 10000.0 / wavenumbers


def load_and_convert(filepath):
    """
    Load .dpt and convert to ascending-wavelength domain.

    Returns
    -------
    wavelengths : np.ndarray, shape (N,)  -- ascending
    spectra : np.ndarray, shape (N, 169)
    """
    wn, sp = load_dpt(filepath)
    wl = wavenumber_to_wavelength(wn)
    idx = np.argsort(wl)
    return wl[idx], sp[idx, :]


def load_all_arrays(base_dir):
    """Load all 4 .dpt files. Returns dict {stem: (wavelengths, spectra)}."""
    result = {}
    for stem in ["Array_1.0", "Array_2.0", "Array_3.0", "Array_4.0"]:
        path = os.path.join(base_dir, f"{stem}.dpt")
        if os.path.isfile(path):
            result[stem] = load_and_convert(path)
    return result


# ---------------------------------------------------------------------------
# Spatial mapping
# ---------------------------------------------------------------------------

def meas_index_to_local(idx):
    """
    1-based measurement index (1..169) -> 0-based (i, j) in 13x13.
    Scan order: row-major, bottom-left to right, then up.
    """
    idx0 = idx - 1
    return idx0 // MEAS_GRID_SIZE, idx0 % MEAS_GRID_SIZE


def local_to_global(i, j):
    """
    Map 0-based 13x13 local coords to 1-based 21x21 global (row, col).

    Inner 13x13 maps to rows 5-17, cols 5-17.
    Measurement (i=0, j=0) = pillar (row=17, col=5)  [bottom-left]
    Measurement (i=12, j=12) = pillar (row=5, col=17) [top-right]
    """
    return 17 - i, 5 + j


def get_window_pillars(center_row, center_col):
    """
    Return list of (row, col) for all pillars in the 9x9 window
    centered on (center_row, center_col).

    All positions are clipped to 1..21.
    """
    pillars = []
    for dr in range(-WINDOW_HALF, WINDOW_HALF + 1):
        for dc in range(-WINDOW_HALF, WINDOW_HALF + 1):
            r = center_row + dr
            c = center_col + dc
            if 1 <= r <= ROWS and 1 <= c <= COLS:
                pillars.append((r, c))
    return pillars


def pillar_distance_to_center(pillar_row, pillar_col, center_row, center_col):
    """Euclidean distance in pillar-index space (units of PITCH)."""
    dr = (pillar_row - center_row) * PITCH
    dc = (pillar_col - center_col) * PITCH
    return math.sqrt(dr * dr + dc * dc)


# ---------------------------------------------------------------------------
# Classification loading
# ---------------------------------------------------------------------------

def load_classifications(csv_path):
    """
    Load defect classifications.
    Returns dict: {array_csv_name: {(row, col): defect_type}}
    """
    df = pd.read_csv(csv_path)
    result = {}
    for _, row in df.iterrows():
        arr = row["array"]
        r, c = int(row["row"]), int(row["col"])
        dt = row["defect_type"]
        if arr not in result:
            result[arr] = {}
        result[arr][(r, c)] = dt
    return result


# ---------------------------------------------------------------------------
# Window defect composition
# ---------------------------------------------------------------------------

DEFECT_TYPES = ["Missing", "Collapsed", "Stitching", "Irregular"]


def compute_window_composition(center_row, center_col, pillar_map):
    """
    For a 9x9 window centered on (center_row, center_col), compute:
    - defect counts by type
    - defect distance geometry features

    Parameters
    ----------
    center_row, center_col : int
        1-based global pillar coords of measurement center.
    pillar_map : dict
        {(row, col): defect_type} for the array.

    Returns
    -------
    dict with keys:
        n_missing, n_collapsed, n_stitching, n_irregular,
        n_total_defects,
        min_defect_dist, mean_defect_dist, sum_1_over_dist,
        is_zero_defect (bool)
    """
    pillars = get_window_pillars(center_row, center_col)

    counts = {dt: 0 for dt in DEFECT_TYPES}
    defect_dists = []
    defect_coords = []

    for r, c in pillars:
        dt = pillar_map.get((r, c), "Good")
        if dt != "Good" and dt in counts:
            counts[dt] += 1
            dist = pillar_distance_to_center(r, c, center_row, center_col)
            defect_dists.append(dist)
            # Store relative coordinates (row_offset, col_offset)
            defect_coords.append((r - center_row, c - center_col))

    n_total = sum(counts.values())

    result = {
        "n_missing": counts["Missing"],
        "n_collapsed": counts["Collapsed"],
        "n_stitching": counts["Stitching"],
        "n_irregular": counts["Irregular"],
        "n_total_defects": n_total,
        "is_zero_defect": n_total == 0,
        "defect_coords": defect_coords,
    }

    if len(defect_dists) > 0:
        defect_dists = np.array(defect_dists)
        result["min_defect_dist"] = float(np.min(defect_dists))
        result["mean_defect_dist"] = float(np.mean(defect_dists))
        # For distance = 0 (defect at center), use small epsilon
        safe_dists = np.maximum(defect_dists, PITCH * 0.1)
        result["sum_1_over_dist"] = float(np.sum(1.0 / safe_dists))
    else:
        result["min_defect_dist"] = np.nan
        result["mean_defect_dist"] = np.nan
        result["sum_1_over_dist"] = 0.0

    return result


def build_all_window_compositions(classifications, array_dpt_stem):
    """
    Build window compositions for all 169 measurements of one array.

    Returns list of 169 dicts (one per measurement index 1..169).
    """
    csv_name = DPT_TO_CSV_NAME.get(array_dpt_stem)
    if csv_name is None or csv_name not in classifications:
        return None

    pillar_map = classifications[csv_name]
    compositions = []

    for idx in range(1, 170):
        i, j = meas_index_to_local(idx)
        grow, gcol = local_to_global(i, j)
        comp = compute_window_composition(grow, gcol, pillar_map)
        comp["meas_idx"] = idx
        comp["local_i"] = i
        comp["local_j"] = j
        comp["global_row"] = grow
        comp["global_col"] = gcol
        comp["dist_from_center"] = math.sqrt(
            (grow - 11.0) ** 2 + (gcol - 11.0) ** 2
        )
        compositions.append(comp)

    return compositions
