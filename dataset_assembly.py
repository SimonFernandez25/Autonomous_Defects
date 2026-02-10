"""
Dataset assembly for spectral perturbation regression.

Builds the ML-ready dataset with strict input/target separation:
- Inputs: defect counts, defect geometry, global spatial context
- Targets: delta_peak_wl, delta_peak_mag
- No spectral similarity metrics, no baseline-relative inputs, no leakage.
"""

import numpy as np
import pandas as pd

from . import ftir_utils
from . import feature_extraction


def build_array_dataset(wavelengths, spectra, array_stem, classifications):
    """
    Build the regression dataset for one array.

    Steps:
    1. Compute 9x9 window defect compositions for all 169 measurements
    2. Identify zero-defect windows -> compute ideal baseline spectrum
    3. Extract dominant peak per measurement in 6-8 um band
    4. Compute delta targets relative to ideal

    Parameters
    ----------
    wavelengths : np.ndarray, shape (N,)
    spectra : np.ndarray, shape (N, 169)
    array_stem : str
        e.g. "Array_1.0"
    classifications : dict
        Output of ftir_utils.load_classifications().

    Returns
    -------
    df : pd.DataFrame or None
        One row per measurement. None if no classification data.
    baseline_info : dict or None
        {ideal_spectrum, ideal_peak_wl, ideal_peak_mag, n_zero_defect}
    """
    compositions = ftir_utils.build_all_window_compositions(
        classifications, array_stem
    )
    if compositions is None:
        return None, None

    comp_df = pd.DataFrame(compositions)

    # --- Step 1: Identify zero-defect windows ---
    zero_mask = comp_df["is_zero_defect"].values
    zero_indices = np.where(zero_mask)[0]  # 0-based column indices into spectra

    if len(zero_indices) == 0:
        raise ValueError(
            f"No zero-defect windows in {array_stem}. "
            "Cannot compute baseline."
        )

    # --- Step 2: Compute ideal baseline ---
    ideal_spectrum, ideal_peak_wl, ideal_peak_mag = \
        feature_extraction.compute_ideal_baseline(
            wavelengths, spectra, zero_indices
        )

    # --- Step 3 & 4: Compute targets per measurement ---
    target_records = []
    for col_idx in range(spectra.shape[1]):
        targets = feature_extraction.compute_targets(
            wavelengths, spectra[:, col_idx],
            ideal_peak_wl, ideal_peak_mag
        )
        target_records.append(targets)

    target_df = pd.DataFrame(target_records)

    # --- Combine ---
    array_id = int(array_stem.split("_")[1].split(".")[0])
    comp_df["array_id"] = array_id
    comp_df["array"] = array_stem

    # Normalized coordinates (-1 to 1)
    comp_df["norm_row"] = (comp_df["global_row"] - 11.0) / 8.0
    comp_df["norm_col"] = (comp_df["global_col"] - 11.0) / 6.0

    df = pd.concat([comp_df, target_df], axis=1)

    baseline_info = {
        "ideal_spectrum": ideal_spectrum,
        "ideal_peak_wl": ideal_peak_wl,
        "ideal_peak_mag": ideal_peak_mag,
        "n_zero_defect": int(len(zero_indices)),
    }

    return df, baseline_info


def build_master_dataset(array_data, classifications):
    """
    Build the merged dataset across arrays 1-3 (training/validation).

    Returns
    -------
    master_df : pd.DataFrame
    baselines : dict {array_stem: baseline_info}
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
            print(f"    Zero-defect windows: {bl['n_zero_defect']}/169")
            print(f"    Ideal peak: {bl['ideal_peak_wl']:.4f} um, "
                  f"mag={bl['ideal_peak_mag']:.6f}")

    master = pd.concat(frames, ignore_index=True)
    return master, baselines


# ---------------------------------------------------------------------------
# Column definitions for strict input/target separation
# ---------------------------------------------------------------------------

INPUT_COLS = [
    # Defect counts
    "n_missing", "n_collapsed", "n_stitching", "n_irregular",
    "n_total_defects",
    # Defect geometry
    "min_defect_dist", "mean_defect_dist", "sum_1_over_dist",
    # Global spatial context
    "global_row", "global_col", "dist_from_center",
    "norm_row", "norm_col",
]

TARGET_COLS = [
    "delta_peak_wl",
    "delta_peak_mag",
]
