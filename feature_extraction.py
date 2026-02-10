"""
Peak extraction in the 6-8 um band for FTIR transmission spectra.

Extracts the dominant peak within the specified wavelength band
and computes deviation targets relative to a per-array baseline.
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter


# ---------------------------------------------------------------------------
# Band of interest
# ---------------------------------------------------------------------------
BAND_MIN = 6.0   # um
BAND_MAX = 8.0   # um


def get_band_mask(wavelengths, band_min=BAND_MIN, band_max=BAND_MAX):
    """Boolean mask for wavelengths within the analysis band."""
    return (wavelengths >= band_min) & (wavelengths <= band_max)


def extract_dominant_peak(wavelengths, spectrum, band_min=BAND_MIN,
                          band_max=BAND_MAX, smooth_window=15, polyorder=3):
    """
    Find the dominant (highest magnitude) peak within a wavelength band.

    Parameters
    ----------
    wavelengths : np.ndarray
    spectrum : np.ndarray
    band_min, band_max : float
        Wavelength bounds in um.
    smooth_window : int
        Savitzky-Golay smoothing window (must be odd).
    polyorder : int
        Polynomial order for smoothing.

    Returns
    -------
    peak_wl : float or np.nan
        Wavelength of the dominant peak.
    peak_mag : float or np.nan
        Magnitude (transmission value) of the dominant peak.
    """
    mask = get_band_mask(wavelengths, band_min, band_max)
    wl_band = wavelengths[mask]
    sp_band = spectrum[mask]

    if len(sp_band) < smooth_window:
        # Not enough points for smoothing; use raw
        smoothed = sp_band
    else:
        smoothed = savgol_filter(sp_band, smooth_window, polyorder)

    # Find peaks in the smoothed band
    peaks, properties = find_peaks(smoothed, prominence=0.001)

    if len(peaks) == 0:
        # Fallback: take the global maximum in the band
        max_idx = np.argmax(smoothed)
        return float(wl_band[max_idx]), float(sp_band[max_idx])

    # Select the tallest peak
    best = peaks[np.argmax(smoothed[peaks])]
    return float(wl_band[best]), float(sp_band[best])


def compute_ideal_baseline(wavelengths, spectra, zero_defect_indices):
    """
    Compute the ideal spectrum by averaging zero-defect windows,
    then extract the reference peak in the 6-8 um band.

    Parameters
    ----------
    wavelengths : np.ndarray, shape (N,)
    spectra : np.ndarray, shape (N, 169)
    zero_defect_indices : list of int
        0-based column indices of zero-defect measurement windows.

    Returns
    -------
    ideal_spectrum : np.ndarray, shape (N,)
    ideal_peak_wl : float
    ideal_peak_mag : float
    """
    if len(zero_defect_indices) == 0:
        raise ValueError("No zero-defect windows available for baseline.")

    ideal_spectrum = np.mean(spectra[:, zero_defect_indices], axis=1)
    ideal_peak_wl, ideal_peak_mag = extract_dominant_peak(wavelengths, ideal_spectrum)

    return ideal_spectrum, ideal_peak_wl, ideal_peak_mag


def compute_targets(wavelengths, spectrum, ideal_peak_wl, ideal_peak_mag):
    """
    Compute regression targets for a single measurement.

    Returns
    -------
    dict with keys: measured_peak_wl, measured_peak_mag,
                    delta_peak_wl, delta_peak_mag
    """
    meas_wl, meas_mag = extract_dominant_peak(wavelengths, spectrum)

    return {
        "measured_peak_wl": meas_wl,
        "measured_peak_mag": meas_mag,
        "delta_peak_wl": meas_wl - ideal_peak_wl,
        "delta_peak_mag": meas_mag - ideal_peak_mag,
    }
