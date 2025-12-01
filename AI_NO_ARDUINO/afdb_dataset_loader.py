"""
afdb_dataset_loader.py

Simplified Pan-Tompkins R-peak detector and advanced ECG feature extractors:
- RR interval features
- QRS morphology features
- Wavelet decomposition features
- extract_all_features(signal, fs) returns a dict of features for a signal block

Dependencies:
    numpy, scipy, pywt, wfdb, pandas
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import pywt
import pandas as pd
import os
import wfdb


# ======================================================================
#  AUTO-DOWNLOAD AND DIRECTORY CHECK
# ======================================================================


def ensure_afdb_downloaded(pn_dir: str):
    """
    Ensures the AFDB database is available locally.
    If pn_dir == "auto": use wfdb’s built-in PhysioNet downloader.
    If a custom folder is provided and it’s empty → download full AFDB there.

    Returns the resolved directory ("afdb" or a local folder).
    """

    # Mode 1 — auto → use WFDB’s own DB downloader
    if pn_dir.lower() == "auto":
        print("[INFO] Using automatic PhysioNet downloader: ~/.wfdb/afdb/")
        return "afdb"

    # Mode 2 — local directory
    if not os.path.exists(pn_dir):
        print(f"[INFO] Creating directory: {pn_dir}")
        os.makedirs(pn_dir)

    # If directory empty → download full AFDB
    if len(os.listdir(pn_dir)) == 0:
        print(f"[INFO] Directory '{pn_dir}' is empty. Downloading AFDB dataset...")
        wfdb.dl_database("afdb", dl_dir=pn_dir)
        print("[INFO] AFDB download complete.")

    return pn_dir


# ======================================================================
#  MAIN FEATURE EXTRACTOR FOR MULTIPLE RECORDS
# ======================================================================


def extract_features_for_records(records, pn_dir, block_sec=60):
    """
    Load multiple AFDB records, split each signal into blocks, and extract
    advanced ECG features via extract_all_features().

    Parameters
    ----------
    records : list of record IDs (e.g. ["04015"])
    pn_dir  : path to AFDB directory, or "auto"
    block_sec : duration of each analysis block

    Returns
    -------
    df : pandas DataFrame with all blocks and features
    """

    # Ensure db is downloaded or accessible
    pn_dir = ensure_afdb_downloaded(pn_dir)

    features_list = []

    for rec in records:
        print(f"[INFO] Loading record {rec}")

        try:
            # Auto mode — PhysioNet
            if pn_dir == "afdb":
                record = wfdb.rdrecord(rec, pn_dir="afdb")

            # Local directory
            else:
                rec_path = os.path.join(pn_dir, rec)
                record = wfdb.rdrecord(rec_path)

        except Exception as e:
            print(f"[ERROR] Could not load {rec}: {e}")
            continue

        fs = record.fs
        n_samples = record.sig_len
        signals = record.p_signal

        block_size = int(block_sec * fs)
        n_blocks = n_samples // block_size

        print(f"[INFO] {rec}: {n_blocks} blocks, fs={fs}")

        for ch_idx, ch_name in enumerate(record.sig_name):
            signal = signals[:, ch_idx]

            # Loop through blocks
            for blk in range(n_blocks):
                start = blk * block_size
                end = start + block_size
                block = signal[start:end]

                feats = extract_all_features(block, fs)

                # Metadata
                feats["record"] = rec
                feats["channel"] = ch_name
                feats["block_idx"] = blk
                feats["t_start_sec"] = blk * block_sec
                feats["t_end_sec"] = (blk + 1) * block_sec

                features_list.append(feats)

    if len(features_list) == 0:
        raise RuntimeError("No valid record blocks processed.")

    return pd.DataFrame(features_list)


# ======================================================================
#  UTILITIES
# ======================================================================


def bandpass(signal, fs, lowcut=5.0, highcut=15.0, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def moving_average(x, width):
    if width <= 1:
        return x
    kernel = np.ones(width) / width
    return np.convolve(x, kernel, mode="same")


# ======================================================================
#  PAN-TOMPKINS R-PEAK DETECTION
# ======================================================================


def pan_tompkins_detect_rpeaks(signal: np.ndarray, fs: float) -> np.ndarray:
    """Simplified Pan-Tompkins R-peak detector."""
    sig_f = bandpass(signal, fs)
    diff_sig = np.ediff1d(sig_f, to_begin=0)
    squared = diff_sig**2

    mwi_width = int(0.12 * fs)
    if mwi_width < 1:
        mwi_width = 1
    integrated = moving_average(squared, mwi_width)

    min_distance = int(0.2 * fs)
    height = np.percentile(integrated, 75)
    peaks, _ = find_peaks(integrated, distance=min_distance, height=height)

    rpeaks = []
    search_radius = int(0.03 * fs)

    for p in peaks:
        left = max(p - search_radius, 0)
        right = min(p + search_radius, len(sig_f) - 1)
        window = sig_f[left : right + 1]
        if window.size == 0:
            continue
        local_max = np.argmax(np.abs(window))
        r_idx = left + local_max
        rpeaks.append(int(r_idx))

    return np.unique(rpeaks).astype(int)


# ======================================================================
#  RR FEATURES
# ======================================================================


def extract_rr_features(rpeaks, fs):
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

    rr = np.diff(rpeaks) / float(fs)

    return {
        "rr_count": len(rr),
        "rr_mean": float(np.mean(rr)),
        "rr_std": float(np.std(rr)),
        "rr_min": float(np.min(rr)),
        "rr_max": float(np.max(rr)),
        "rr_sdnn": float(np.std(rr, ddof=1)) if rr.size > 1 else np.nan,
        "rr_rmssd": (
            float(np.sqrt(np.mean(np.diff(rr) ** 2))) if rr.size > 1 else np.nan
        ),
        "rr_pnn50": float(np.sum(np.abs(np.diff(rr)) > 0.05) / max(1, len(rr) - 1)),
    }


# ======================================================================
#  QRS FEATURES
# ======================================================================


def extract_qrs_features(signal, rpeaks, fs):
    if rpeaks is None or len(rpeaks) == 0:
        return {
            "qrs_count": 0,
            "qrs_width_mean": np.nan,
            "qrs_width_std": np.nan,
            "qrs_amp_mean": np.nan,
            "qrs_amp_std": np.nan,
            "qrs_area_mean": np.nan,
        }

    widths, amps, areas = [], [], []
    half_window = int(0.06 * fs)

    for r in rpeaks:
        left = max(r - half_window, 0)
        right = min(r + half_window, len(signal) - 1)
        seg = signal[left : right + 1]

        if seg.size == 0:
            continue

        amp = float(np.max(seg) - np.min(seg))
        amps.append(amp)
        areas.append(float(np.trapz(np.abs(seg))))

        trough_val = np.min(seg)
        half_amp = trough_val + 0.5 * amp

        l_idx = r
        while l_idx > left and signal[l_idx] > half_amp:
            l_idx -= 1

        ri = r
        while ri < right and signal[ri] > half_amp:
            ri += 1

        widths.append(float((ri - l_idx) / fs))

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


# ======================================================================
#  WAVELET FEATURES
# ======================================================================


def extract_wavelet_features(signal, wavelet="db4", level=4):
    if len(signal) < 8:
        return {}

    coeffs = pywt.wavedec(signal, wavelet, level=level)
    feats = {}

    for i, c in enumerate(coeffs):
        arr = np.asarray(c, dtype=float)
        feats[f"wave_energy_L{i}"] = float(np.sum(arr**2))
        feats[f"wave_std_L{i}"] = float(np.std(arr))
        feats[f"wave_mean_L{i}"] = float(np.mean(arr))

    return feats


# ======================================================================
#  FULL FEATURE SET
# ======================================================================


def extract_all_features(signal, fs):
    signal = np.asarray(signal).astype(float)

    out = {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "median": float(np.median(signal)),
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "rms": float(np.sqrt(np.mean(signal**2))),
        "zcr": float(((signal[:-1] * signal[1:]) < 0).sum() / max(1, len(signal) - 1)),
    }

    # R-peaks
    try:
        rpeaks = pan_tompkins_detect_rpeaks(signal, fs)
    except:
        rpeaks = np.array([], dtype=int)

    # RR + QRS + Wavelet
    out.update(extract_rr_features(rpeaks, fs))
    out.update(extract_qrs_features(signal, rpeaks, fs))
    out.update(extract_wavelet_features(signal))

    out["n_rpeaks"] = int(len(rpeaks))
    out["n_samples"] = int(len(signal))
    out["fs"] = float(fs)

    return out
