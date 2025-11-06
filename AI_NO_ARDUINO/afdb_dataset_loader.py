"""
afdb_dataset_loader.py

Efficient and memory-safe loader for the MIT-BIH Atrial Fibrillation Database (AFDB) from PhysioNet.

This script combines:
- Streaming feature extraction from ECG records.
- Automatic MLII (Lead II) channel detection.
- Basic data cleaning and normalization.
- In-memory dataset creation (no CSV export).

It’s designed to integrate directly with machine learning workflows,
so that the resulting DataFrame can be used immediately for training or inference.

Usage example:
    from afdb_dataset_loader import load_afdb_dataset

    df = load_afdb_dataset(
        records=["04015", "04043"],
        pn_dir="afdb/1.0.0",
        block_sec=60.0,
        normalize=True
    )

    print(df.head())
"""

from __future__ import annotations
import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------
def setup_logging(level: str = "INFO") -> None:
    """Configure basic console logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# ---------------------------------------------------------------------
# SIGNAL PROCESSING UTILITIES
# ---------------------------------------------------------------------
def to_float32(x: np.ndarray) -> np.ndarray:
    """Convert array to float32 for memory efficiency."""
    return x.astype(np.float32, copy=False)


def bandpass_signal(
    sig: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0, order: int = 3
) -> np.ndarray:
    """Apply a Butterworth band-pass filter to the signal."""
    nyq = 0.5 * fs
    lown, highn = low / nyq, high / nyq
    b, a = butter(order, [lown, highn], btype="band")
    return to_float32(filtfilt(b, a, sig))


# ---------------------------------------------------------------------
# FEATURE EXTRACTION FUNCTIONS
# ---------------------------------------------------------------------
def extract_basic_time_features(block: np.ndarray) -> dict:
    """Compute simple statistical features from a block of ECG signal."""
    x = block
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "rms": float(np.sqrt(np.mean(x * x))),
        "zcr": float(((x[:-1] * x[1:]) < 0).sum()) / max(1, x.size - 1),
    }


def estimate_hr_from_peaks(block: np.ndarray, fs: float) -> tuple[float, int]:
    """Estimate heart rate (in bpm) based on R-peak detection."""
    if block.size < int(0.5 * fs):
        return float("nan"), 0
    try:
        sig_f = bandpass_signal(block, fs, 5.0, 25.0, 2)
    except Exception:
        sig_f = block
    peaks, _ = find_peaks(sig_f, distance=int(0.2 * fs))
    if len(peaks) <= 1:
        return float("nan"), len(peaks)
    rr = np.diff(peaks) / fs
    hr = 60.0 / np.mean(rr)
    return float(hr), len(peaks)


def extract_features_block(block: np.ndarray, fs: float) -> dict:
    """Combine basic statistics and HR estimation for a signal block."""
    feats = extract_basic_time_features(block)
    hr, n = estimate_hr_from_peaks(block, fs)
    feats.update({"hr_bpm": hr, "n_peaks": n})
    return feats


# ---------------------------------------------------------------------
# AUTOMATIC MLII CHANNEL DETECTION
# ---------------------------------------------------------------------
def detect_mlii_channel(record_name: str, pn_dir: str) -> Optional[int]:
    """
    Automatically detect the index of the MLII (Lead II) channel from PhysioNet record metadata.

    Returns:
        Index of the MLII channel (int) or None if not found.
    """
    try:
        header = wfdb.rdheader(record_name, pn_dir=pn_dir)
        sig_names = [s.lower() for s in getattr(header, "sig_name", [])]
        for i, name in enumerate(sig_names):
            if "mlii" in name or "ii" in name:
                logging.info(
                    f"Detected MLII channel for {record_name}: index {i} ({header.sig_name[i]})"
                )
                return i
    except Exception as e:
        logging.warning(f"Could not detect MLII channel for {record_name}: {e}")
    return None


# ---------------------------------------------------------------------
# STREAMING RECORD PROCESSING
# ---------------------------------------------------------------------
def process_record_streaming(
    record_name: str,
    pn_dir: str,
    channels: List[int],
    block_sec: float = 60.0,
    force_fs: Optional[float] = None,
) -> pd.DataFrame:
    """
    Stream and process one AFDB record into feature DataFrame blocks.
    """
    header = wfdb.rdheader(record_name, pn_dir=pn_dir)
    fs = float(force_fs) if force_fs else float(getattr(header, "fs", 250.0))
    sig_len = int(getattr(header, "sig_len", 0))
    block_samples = int(block_sec * fs)

    samp_start, block_idx = 0, 0
    rows = []

    logging.info(f"Processing record {record_name} (fs={fs}, sig_len={sig_len})")

    while samp_start < sig_len:
        samp_end = min(samp_start + block_samples, sig_len)
        rec = wfdb.rdrecord(
            record_name,
            pn_dir=pn_dir,
            sampfrom=samp_start,
            sampto=samp_end,
            channels=channels,
        )
        p_signal = to_float32(rec.p_signal)

        for ch_idx, ch in enumerate(channels):
            sig = p_signal[:, ch_idx] if p_signal.ndim == 2 else p_signal
            feats = extract_features_block(sig, fs)
            rows.append(
                {
                    "record": record_name,
                    "channel": ch,
                    "block_idx": block_idx,
                    "t_start": samp_start / fs,
                    "t_end": samp_end / fs,
                    "n_samples": len(sig),
                    "fs": fs,
                    **feats,
                }
            )
        samp_start = samp_end
        block_idx += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# DATA CLEANING AND NORMALIZATION
# ---------------------------------------------------------------------
def clean_and_normalize(df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
    """
    Clean invalid rows (NaNs, unrealistic HR) and normalize numeric columns.
    """
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Remove unrealistic HR values
    if "hr_bpm" in df.columns:
        df = df[(df["hr_bpm"] > 30) & (df["hr_bpm"] < 220)]

    if normalize:
        feature_cols = [
            "mean",
            "std",
            "median",
            "min",
            "max",
            "rms",
            "zcr",
            "hr_bpm",
            "n_peaks",
        ]
        feature_cols = [c for c in feature_cols if c in df.columns]
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        logging.info(f"Normalized features: {feature_cols}")

    logging.info(f"Cleaned dataset shape: {df.shape}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# MAIN PUBLIC FUNCTION
# ---------------------------------------------------------------------
def load_afdb_dataset(
    records: List[str],
    pn_dir: str = "afdb/1.0.0",
    block_sec: float = 60.0,
    normalize: bool = True,
    log_level: str = "INFO",
) -> pd.DataFrame:
    """
    Load, extract, and preprocess the AFDB dataset directly from PhysioNet.

    Args:
        records (list[str]): List of record IDs to load (e.g., ["04015", "04043"]).
        pn_dir (str): Path or PhysioNet dataset identifier (default: "afdb/1.0.0").
        block_sec (float): Duration (in seconds) per processing block.
        normalize (bool): Whether to apply feature normalization.
        log_level (str): Logging verbosity ("INFO", "DEBUG", etc.).

    Returns:
        pd.DataFrame: Cleaned and ready-to-use feature dataset.
    """
    setup_logging(log_level)
    all_dfs = []

    if not records:
        raise ValueError("No record IDs specified.")

    for rec in records:
        ch = detect_mlii_channel(rec, pn_dir)
        if ch is None:
            logging.warning(f"Skipping {rec}: MLII not found.")
            continue
        df_rec = process_record_streaming(rec, pn_dir, [ch], block_sec)
        all_dfs.append(df_rec)

    if not all_dfs:
        raise RuntimeError("No valid records were processed.")

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_clean = clean_and_normalize(df_all, normalize=normalize)
    logging.info("Dataset successfully loaded and preprocessed.")
    return df_clean
