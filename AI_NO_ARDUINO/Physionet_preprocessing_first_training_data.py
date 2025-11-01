"""
afdb_feature_extraction.py

Lightweight, memory-efficient feature extraction pipeline for the
MIT-BIH Atrial Fibrillation Database (AFDB) hosted on PhysioNet.

Designed to be run in VS Code / local Python and versioned on GitHub.
Reads records directly from PhysioNet (no full download), processes
signals in time-blocks to limit memory usage, and writes extracted
features incrementally to disk (CSV) for later use in ML training.

Requirements
------------
- Python 3.8+
- wfdb          (pip install wfdb)
- numpy         (pip install numpy)
- pandas        (pip install pandas)
- scipy         (pip install scipy)

Usage examples
--------------
# Process a single record and channel 0 in 60s blocks
python afdb_feature_extraction.py --records 04015 --channels 0 --block-sec 60

# Process multiple comma-separated records, output to features.csv
python afdb_feature_extraction.py --records 04015,04043 --out features.csv

# Try to process all records (requires wfdb to fetch list from PhysioNet)
python afdb_feature_extraction.py --all-records

Notes
-----
- The script attempts to use wfdb to read headers and to stream samples
  using sampfrom/sampto to avoid loading whole signals.
- Features are computed per block and appended to the CSV, so memory
  usage stays bounded regardless of dataset size.

"""

from __future__ import annotations
import argparse
import logging
import math
import os
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, find_peaks

# --------------------------- Configuration ---------------------------
DEFAULT_PN_DIR = "afdb/1.0.0"  # PhysioNet path

# --------------------------- Utilities -------------------------------


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Memory-efficient AFDB feature extraction (streaming by blocks)."
    )
    p.add_argument(
        "--records",
        type=str,
        default="",
        help=(
            "Comma-separated record names (e.g. 04015,04043). "
            "If omitted and --all-records is set, the script will attempt to fetch the list from wfdb."
        ),
    )
    p.add_argument(
        "--all-records",
        action="store_true",
        help="Try to process every record in the remote PN_DIR via wfdb.get_record_list().",
    )
    p.add_argument(
        "--pn-dir",
        default=DEFAULT_PN_DIR,
        help="PhysioNet directory (pn_dir for wfdb).",
    )
    p.add_argument(
        "--channels",
        default="0",
        help="Comma-separated channel indices to process (0-based).",
    )
    p.add_argument(
        "--block-sec",
        type=float,
        default=60.0,
        help="Block length in seconds (default 60s).",
    )
    p.add_argument(
        "--out",
        default="afdb_features.csv",
        help="Output CSV file (appended incrementally).",
    )
    p.add_argument(
        "--force-fs",
        type=float,
        default=0.0,
        help="Force sampling frequency override if header is missing/incorrect.",
    )
    p.add_argument("--log", default="INFO", help="Logging level (DEBUG/INFO/WARNING).")
    return p.parse_args()


# --------------------------- Signal helpers -------------------------


def to_float32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32, copy=False)


def bandpass_signal(
    sig: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0, order: int = 3
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter applied to 1D signal.

    Returns filtered signal as float32.
    """
    nyq = 0.5 * fs
    lown = low / nyq
    highn = high / nyq
    b, a = butter(order, [lown, highn], btype="band")
    filtered = filtfilt(b, a, sig)
    return to_float32(filtered)


# --------------------------- Feature extraction ----------------------


def extract_basic_time_features(block: np.ndarray) -> dict:
    """Compute lightweight time-domain features from a 1D numpy array block.

    The block is assumed to be float32 and shape (n_samples,).
    """
    x = block
    features = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=0)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "rms": float(np.sqrt(np.mean(x * x))),
        "zcr": float(((x[:-1] * x[1:]) < 0).sum()) / max(1, x.size - 1),
    }
    return features


def estimate_hr_from_peaks(
    block: np.ndarray, fs: float, height: Optional[float] = None
) -> Tuple[float, int]:
    """Estimate instantaneous heart rate (bpm) from R-peak detections in the block.

    Returns (mean_hr_bpm, n_peaks).
    This is a simple approach: find_peaks on bandpassed signal. It is not
    a clinical-grade detector but useful for coarse feature extraction.
    """
    if block.size < int(0.5 * fs):
        return float("nan"), 0

    # bandpass to emphasize QRS-like activity (adult HR energy concentrated >5Hz)
    try:
        sig_f = bandpass_signal(block, fs, low=5.0, high=25.0, order=2)
    except Exception:
        # fallback: use raw block
        sig_f = block

    # Use median absolute deviation to set a robust threshold if height not provided
    if height is None:
        mad = np.median(np.abs(sig_f - np.median(sig_f)))
        if mad == 0:
            height = None
        else:
            height = float(np.median(sig_f) + 0.6 * mad)

    # find peaks with minimum distance ~ 200 ms (max ~300 bpm)
    min_distance = int(0.2 * fs)
    peaks, props = find_peaks(sig_f, height=height, distance=min_distance)
    n_peaks = peaks.size
    if n_peaks <= 1:
        return float("nan"), int(n_peaks)

    # compute RR intervals and HR in bpm
    rr_samples = np.diff(peaks)
    rr_sec = rr_samples / fs
    rr_mean = float(np.mean(rr_sec))
    if rr_mean == 0:
        return float("nan"), int(n_peaks)
    hr_bpm = 60.0 / rr_mean
    return float(hr_bpm), int(n_peaks)


def extract_features_block(block: np.ndarray, fs: float) -> dict:
    """Combine features for a single block (1D array) into a flat dict."""
    feats = extract_basic_time_features(block)
    hr, n_peaks = estimate_hr_from_peaks(block, fs)
    feats.update({"hr_bpm": hr, "n_peaks": n_peaks})
    return feats


# --------------------------- Main pipeline ---------------------------


def process_record_streaming(
    record_name: str,
    pn_dir: str,
    channels: List[int],
    block_sec: float,
    out_csv: str,
    force_fs: Optional[float] = None,
) -> None:
    """Process a single record by streaming blocks and appending features to CSV.

    The CSV will contain one row per block per channel with metadata.
    """
    logging.info("Processing record %s (pn_dir=%s)", record_name, pn_dir)

    # read header to get fs and length
    try:
        header = wfdb.rdheader(record_name, pn_dir=pn_dir)
    except Exception as e:
        logging.error("Failed to read header for %s: %s", record_name, e)
        raise

    fs = (
        float(force_fs)
        if force_fs and force_fs > 0
        else float(getattr(header, "fs", 250.0))
    )
    sig_len = int(getattr(header, "sig_len", 0))
    if sig_len == 0:
        logging.warning(
            "Header has sig_len=0 for %s; attempting to read record to infer length.",
            record_name,
        )
        temp_rec = wfdb.rdrecord(record_name, pn_dir=pn_dir)
        sig_len = int(temp_rec.p_signal.shape[0])
        del temp_rec

    logging.info(
        " Record header: fs=%s Hz, length=%s samples (%.1f s)",
        fs,
        sig_len,
        sig_len / fs,
    )

    block_samples = int(block_sec * fs)
    if block_samples <= 0:
        raise ValueError("block_sec too small or fs invalid")

    # prepare CSV header if file does not exist
    write_header = not os.path.exists(out_csv)

    # iterate over time blocks
    samp_start = 0
    rows = []
    total_blocks = math.ceil(sig_len / block_samples)
    block_idx = 0
    while samp_start < sig_len:
        samp_end = min(samp_start + block_samples, sig_len)
        # read only requested channels and the sample range
        try:
            rec = wfdb.rdrecord(
                record_name,
                pn_dir=pn_dir,
                sampfrom=samp_start,
                sampto=samp_end,
                channels=channels,
            )
        except Exception as e:
            logging.error(
                "Failed to read samples %d-%d for %s: %s",
                samp_start,
                samp_end,
                record_name,
                e,
            )
            raise

        # rec.p_signal shape: (n_samples, n_channels)
        p_signal = rec.p_signal
        if p_signal is None:
            logging.error("p_signal is None for block %d of %s", block_idx, record_name)
            break

        # ensure float32 and process channel-wise
        p_signal = to_float32(p_signal)
        n_samples_block = p_signal.shape[0]
        t0 = samp_start / fs
        t1 = samp_end / fs

        for ch_idx, ch in enumerate(channels):
            # if only one channel was read, p_signal may be shape (n_samples,1)
            sig_col = p_signal[:, ch_idx] if p_signal.ndim == 2 else p_signal
            feats = extract_features_block(sig_col, fs)
            row = {
                "record": record_name,
                "channel": int(ch),
                "block_idx": int(block_idx),
                "t_start": float(t0),
                "t_end": float(t1),
                "n_samples": int(n_samples_block),
                "fs": float(fs),
            }
            row.update(feats)
            rows.append(row)

        # append rows to CSV to keep memory bounded
        if rows:
            df_block = pd.DataFrame(rows)
            if write_header:
                df_block.to_csv(out_csv, mode="a", index=False)
                write_header = False
            else:
                df_block.to_csv(out_csv, mode="a", index=False, header=False)
            # free memory
            del df_block
            rows.clear()

        # free rec and p_signal
        del rec, p_signal
        samp_start = samp_end
        block_idx += 1
        logging.info(
            "Processed block %d/%d for record %s (%.1f-%.1f s)",
            block_idx,
            total_blocks,
            record_name,
            t0,
            t1,
        )

    logging.info(
        "Finished processing record %s. Features appended to %s", record_name, out_csv
    )


# --------------------------- Record list helper ----------------------


def get_remote_record_list(pn_dir: str) -> List[str]:
    """Attempt to fetch record list from wfdb. Falls back to empty list on failure."""
    try:
        # wfdb.get_record_list exists in wfdb; if not, this will raise
        recs = wfdb.get_record_list(pn_dir)
        if isinstance(recs, (list, tuple)):
            return list(recs)
    except Exception as e:
        logging.warning(
            "Could not fetch remote record list via wfdb.get_record_list('%s'): %s",
            pn_dir,
            e,
        )
    return []


# --------------------------- Entrypoint ------------------------------


def main() -> None:
    args = parse_args()
    setup_logging(args.log)

    # determine records to process
    if args.all_records:
        recs = get_remote_record_list(args.pn_dir)
        if not recs:
            logging.error(
                "--all-records requested but couldn't fetch list from remote. Provide --records manually."
            )
            return
    elif args.records:
        recs = [r.strip() for r in args.records.split(",") if r.strip()]
    else:
        logging.error("No records specified. Use --records or --all-records.")
        return

    channels = [int(c) for c in args.channels.split(",") if c.strip()]
    if not channels:
        logging.error("No channels specified (use --channels).")
        return

    logging.info("Starting processing for %d records", len(recs))
    for rec in recs:
        try:
            process_record_streaming(
                rec,
                pn_dir=args.pn_dir,
                channels=channels,
                block_sec=args.block_sec,
                out_csv=args.out,
                force_fs=args.force_fs,
            )
        except Exception as e:
            logging.exception("Error processing record %s: %s", rec, e)

    logging.info("All done.")


if __name__ == "__main__":
    main()
