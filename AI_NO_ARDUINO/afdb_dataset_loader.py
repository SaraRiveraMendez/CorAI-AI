"""
afdb_features.py

Simplified Pan-Tompkins R-peak detector and advanced ECG feature extractors:
- RR interval features
- QRS morphology features
- Wavelet decomposition features
- extract_all_features(signal, fs) returns a dict of features for a signal block

Dependencies:
    numpy, scipy, pywt
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import pywt


# -----------------------
# Utilities
# -----------------------
def bandpass(
    signal: np.ndarray,
    fs: float,
    lowcut: float = 5.0,
    highcut: float = 15.0,
    order: int = 3,
) -> np.ndarray:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return x
    kernel = np.ones(width) / width
    return np.convolve(x, kernel, mode="same")


# -----------------------
# Pan-Tompkins (simplified)
# -----------------------
def pan_tompkins_detect_rpeaks(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Simplified Pan-Tompkins implementation:
      - Bandpass (5-15 Hz default)
      - Derivative (approximate)
      - Squaring
      - Moving window integration
      - Peak picking on integrated signal, then refine on original filtered signal

    Returns:
      rpeaks : array of sample indices of detected R peaks
    """
    # 1) Bandpass filter
    sig_f = bandpass(signal, fs, lowcut=5.0, highcut=15.0, order=3)

    # 2) Derivative (five-point derivative could be used; use simple diff)
    diff_sig = np.ediff1d(sig_f, to_begin=0)

    # 3) Squaring
    squared = diff_sig**2

    # 4) Moving window integration - window length ~ 0.12 to 0.15 s
    mwi_width = int(0.12 * fs)
    if mwi_width < 1:
        mwi_width = 1
    integrated = moving_average(squared, mwi_width)

    # 5) Peak picking on integrated signal
    # distance: at least 200 ms between peaks
    min_distance = int(0.2 * fs)
    height = np.percentile(integrated, 75)  # adaptive threshold
    peaks, _ = find_peaks(integrated, distance=min_distance, height=height)

    # 6) Refine peak positions: for each integrated-peak, search nearby window in filtered signal for local maxima
    rpeaks = []
    search_radius = int(0.03 * fs)  # +/-30 ms
    for p in peaks:
        left = max(p - search_radius, 0)
        right = min(p + search_radius, len(sig_f) - 1)
        window = sig_f[left : right + 1]
        if window.size == 0:
            continue
        local_max = np.argmax(np.abs(window))
        r_idx = left + local_max
        rpeaks.append(int(r_idx))
    # Remove duplicates and sort
    if len(rpeaks) == 0:
        return np.array([], dtype=int)
    rpeaks = np.unique(rpeaks)
    return rpeaks.astype(int)


# -----------------------
# RR interval features
# -----------------------
def extract_rr_features(rpeaks: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Given rpeaks as sample indices, compute RR features in seconds.
    Returns NaN for features if insufficient peaks.
    """
    if rpeaks is None or len(rpeaks) < 2:
        return {
            "rr_count": 0,
            "rr_mean": np.nan,
            "rr_std": np.nan,
            "rr_min": np.nan,
            "rr_max": np.nan,
            "rr_rmssd": np.nan,
            "rr_sdnn": np.nan,
            "rr_pnn50": np.nan,
        }

    rr = np.diff(rpeaks) / float(fs)  # seconds
    rr_count = len(rr)
    rr_mean = float(np.mean(rr))
    rr_std = float(np.std(rr))
    rr_min = float(np.min(rr))
    rr_max = float(np.max(rr))
    rr_sdnn = float(np.std(rr, ddof=1)) if rr.size > 1 else float(np.nan)
    rr_rmssd = (
        float(np.sqrt(np.mean(np.diff(rr) ** 2))) if rr.size > 1 else float(np.nan)
    )
    rr_pnn50 = float(np.sum(np.abs(np.diff(rr)) > 0.05) / max(1, (len(rr) - 1)))

    return {
        "rr_count": rr_count,
        "rr_mean": rr_mean,
        "rr_std": rr_std,
        "rr_min": rr_min,
        "rr_max": rr_max,
        "rr_sdnn": rr_sdnn,
        "rr_rmssd": rr_rmssd,
        "rr_pnn50": rr_pnn50,
    }


# -----------------------
# QRS morphology features
# -----------------------
def extract_qrs_features(
    signal: np.ndarray, rpeaks: np.ndarray, fs: float
) -> Dict[str, float]:
    """
    For each R-peak, measure a small window around the peak to estimate:
      - QRS width (seconds) via crossing half-amplitude heuristic
      - QRS amplitude (peak-to-trough in window)
      - QRS energy (integral of absolute values)
    Returns averaged statistics across beats.
    """
    if rpeaks is None or len(rpeaks) == 0:
        return {
            "qrs_count": 0,
            "qrs_width_mean": np.nan,
            "qrs_width_std": np.nan,
            "qrs_amp_mean": np.nan,
            "qrs_amp_std": np.nan,
            "qrs_area_mean": np.nan,
        }

    widths = []
    amps = []
    areas = []
    half_window = int(0.06 * fs)  # 60 ms each side

    for r in rpeaks:
        left = max(r - half_window, 0)
        right = min(r + half_window, len(signal) - 1)
        seg = signal[left : right + 1]
        if seg.size == 0:
            continue
        peak_val = signal[r]
        trough_val = np.min(seg)
        amp = float(np.max(seg) - np.min(seg))
        amps.append(amp)
        area = float(np.trapz(np.abs(seg)))
        areas.append(area)

        # half amplitude threshold relative to peak-trough
        half_amp = trough_val + 0.5 * (np.max(seg) - trough_val)
        # find left crossing
        l_idx = r
        while l_idx > left and signal[l_idx] > half_amp:
            l_idx -= 1
        # find right crossing
        ri = r
        while ri < right and signal[ri] > half_amp:
            ri += 1
        width_sec = float((ri - l_idx) / float(fs))
        widths.append(width_sec)

    if len(widths) == 0:
        return {
            "qrs_count": 0,
            "qrs_width_mean": np.nan,
            "qrs_width_std": np.nan,
            "qrs_amp_mean": np.nan,
            "qrs_amp_std": np.nan,
            "qrs_area_mean": np.nan,
        }

    return {
        "qrs_count": len(widths),
        "qrs_width_mean": float(np.mean(widths)),
        "qrs_width_std": float(np.std(widths)),
        "qrs_amp_mean": float(np.mean(amps)),
        "qrs_amp_std": float(np.std(amps)),
        "qrs_area_mean": float(np.mean(areas)),
    }


# -----------------------
# Wavelet features
# -----------------------
def extract_wavelet_features(
    signal: np.ndarray, wavelet: str = "db4", level: int = 4
) -> Dict[str, float]:
    """
    Compute wavelet decomposition and return energy and std for each coefficient level.
    Returns a dictionary with features like wave_energy_L0, wave_std_L0, ...
    """
    # Guard for very short signals
    if len(signal) < 8:
        return {}

    coeffs = pywt.wavedec(signal, wavelet, level=level)
    feats = {}
    for i, c in enumerate(coeffs):
        energy = float(np.sum(np.array(c) ** 2))
        std = float(np.std(c))
        mean = float(np.mean(c))
        feats[f"wave_energy_L{i}"] = energy
        feats[f"wave_std_L{i}"] = std
        feats[f"wave_mean_L{i}"] = mean
    return feats


# -----------------------
# Top-level feature combination
# -----------------------
def extract_all_features(signal: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Extract a combined set of features for one block of ECG signal.
    Returns a flat dictionary suitable to append as a DataFrame row.
    """
    out: Dict[str, float] = {}

    # Basic statistical features (keep names consistent with your previous pipeline)
    out["mean"] = float(np.mean(signal)) if signal.size > 0 else float("nan")
    out["std"] = float(np.std(signal)) if signal.size > 0 else float("nan")
    out["median"] = float(np.median(signal)) if signal.size > 0 else float("nan")
    out["min"] = float(np.min(signal)) if signal.size > 0 else float("nan")
    out["max"] = float(np.max(signal)) if signal.size > 0 else float("nan")
    out["rms"] = float(np.sqrt(np.mean(signal**2))) if signal.size > 0 else float("nan")
    out["zcr"] = float(
        ((signal[:-1] * signal[1:]) < 0).sum() / max(1, (len(signal) - 1))
    )

    # R-peak detection (Pan-Tompkins simplified)
    try:
        rpeaks = pan_tompkins_detect_rpeaks(signal, fs)
    except Exception:
        rpeaks = np.array([], dtype=int)

    # RR features
    rr_feats = extract_rr_features(rpeaks, fs)
    out.update(rr_feats)

    # QRS features
    qrs_feats = extract_qrs_features(signal, rpeaks, fs)
    out.update(qrs_feats)

    # Wavelet features
    wave_feats = extract_wavelet_features(signal, wavelet="db4", level=4)
    out.update(wave_feats)

    # Number of detected peaks
    out["n_rpeaks"] = int(len(rpeaks))
    out["n_samples"] = int(len(signal))
    out["fs"] = float(fs)

    return out
