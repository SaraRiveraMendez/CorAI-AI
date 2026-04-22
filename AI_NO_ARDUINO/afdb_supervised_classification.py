"""
afdb_supervised_classification.py

Supervised classification using real AFDB rhythm labels, extended to also
incorporate new JSON ECG signals from Google Drive for retraining.

Classes:
  AFIB : Atrial Fibrillation
  AFL  : Atrial Flutter
  J    : AV Junctional Rhythm
  N    : Other rhythms (primarily normal sinus rhythm)

Changes from original:
  1. Added --data-dir argument to point at the JSON signals folder.
  2. Added --json-fs argument for the JSON signal sampling frequency.
  3. load_json_signals() loads all JSON files, extracts features, assigns
     class labels from folder names, and tags each sample with noise_type.
  4. Training uses AFDB blocks + clean JSON signals combined.
  5. Evaluation is run separately on all JSON signals broken down by
     noise_type (clean, Muscular, Respiracion).
  6. col_medians and feature_columns are saved as extra .joblib artifacts
     so that future inference scripts can impute NaN correctly.

Usage example (AFDB only, original behaviour):
    python afdb_supervised_classification.py \
        --records 04015 04043 04048 \
        --pn-dir afdb \
        --block-sec 60 \
        --channel-idx 1 \
        --save-model

Usage example (AFDB + JSON retraining):
    python afdb_supervised_classification.py \
        --records 04015 04043 04048 \
        --pn-dir afdb \
        --block-sec 60 \
        --channel-idx 1 \
        --save-model \
        --data-dir data \
        --json-fs 200 \
        --results-dir results_supervised
"""

import os
import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
)
from sklearn.decomposition import PCA
import wfdb

from afdb_dataset_loader import extract_all_features

# ---------------------------------------------------------------------------
# Folder-to-class mapping for JSON data (compared case-insensitively)
# ---------------------------------------------------------------------------

_FOLDER_TO_CLASS = {
    "ATRIAL 1": "AFIB",
    "ATRIAL 2": "AFL",
    "ECG NORMAL": "N",
}
_FOLDER_LOOKUP = {k.strip().upper(): v for k, v in _FOLDER_TO_CLASS.items()}

# Noise subfolders expected inside each class folder
_NOISE_SUBFOLDERS = ["Muscular", "Respiración"]

# Metadata columns that must be excluded from the feature matrix
_METADATA_COLS = {
    "record",
    "channel",
    "channel_idx",
    "block_idx",
    "t_start_sec",
    "t_end_sec",
    "rhythm_label",
    # JSON-specific metadata
    "filename",
    "noise_type",
    "source",
}


# ---------------------------------------------------------------------------
# Original AFDB annotation helpers (unchanged)
# ---------------------------------------------------------------------------


def load_afdb_annotations(record_name, pn_dir):
    """
    Load rhythm annotations (.atr) for an AFDB record.

    Returns
    -------
    tuple
        (sample_indices, symbols, aux_notes) or (None, None, None) on error.
    """
    try:
        if pn_dir == "afdb":
            annotation = wfdb.rdann(record_name, "atr", pn_dir="afdb")
        else:
            ann_path = os.path.join(pn_dir, record_name)
            annotation = wfdb.rdann(ann_path, "atr")
        return annotation.sample, annotation.symbol, annotation.aux_note
    except Exception as e:
        logging.error("Could not load annotations for %s: %s", record_name, e)
        return None, None, None


def get_rhythm_label_for_block(start_sample, end_sample, ann_samples, aux_notes):
    """
    Determine the predominant rhythm label for a signal block.

    AFDB annotation convention:
      (AFIB -> AFIB, (AFL -> AFL, (J -> J, (N -> N

    Returns the most frequent label within the block, or the most recent
    label before the block if no annotation falls inside it.
    Returns None if no label can be determined.
    """
    if ann_samples is None or len(ann_samples) == 0:
        return None

    block_annotations = []
    for i, sample in enumerate(ann_samples):
        if start_sample <= sample < end_sample:
            aux = aux_notes[i] if i < len(aux_notes) else ""
            if "(AFIB" in aux:
                block_annotations.append("AFIB")
            elif "(AFL" in aux:
                block_annotations.append("AFL")
            elif "(J" in aux:
                block_annotations.append("J")
            elif "(N" in aux:
                block_annotations.append("N")

    if len(block_annotations) == 0:
        prev_annotations = [i for i, s in enumerate(ann_samples) if s < start_sample]
        if prev_annotations:
            idx = prev_annotations[-1]
            aux = aux_notes[idx] if idx < len(aux_notes) else ""
            if "(AFIB" in aux:
                return "AFIB"
            elif "(AFL" in aux:
                return "AFL"
            elif "(J" in aux:
                return "J"
            elif "(N" in aux:
                return "N"
        return None

    return Counter(block_annotations).most_common(1)[0][0]


def extract_features_with_labels(records, pn_dir, block_sec=60, channel_idx=1):
    """
    Extract features and real rhythm labels from AFDB records (unchanged).

    Returns
    -------
    pd.DataFrame
        Feature rows with a 'rhythm_label' column and source='afdb'.
    """
    features_list = []

    for rec in records:
        logging.info("Loading record %s", rec)
        try:
            record = wfdb.rdrecord(rec, pn_dir=pn_dir)
        except Exception as e:
            logging.error("Could not load record %s: %s", rec, e)
            continue

        ann_samples, ann_symbols, aux_notes = load_afdb_annotations(rec, pn_dir)
        if ann_samples is None:
            logging.warning("No annotations for %s, skipping.", rec)
            continue

        fs = record.fs
        signals = record.p_signal

        if channel_idx >= signals.shape[1]:
            logging.warning("Channel %d not available in %s.", channel_idx, rec)
            continue

        block_size = int(block_sec * fs)
        n_blocks = len(signals) // block_size
        signal = signals[:, channel_idx]

        logging.info("%s: %d blocks, fs=%d", rec, n_blocks, fs)

        for blk in range(n_blocks):
            start = blk * block_size
            end = start + block_size
            block = signal[start:end]
            feats = extract_all_features(block, fs)
            rhythm = get_rhythm_label_for_block(start, end, ann_samples, aux_notes)

            if rhythm is None:
                continue

            feats["record"] = rec
            feats["channel"] = record.sig_name[channel_idx]
            feats["channel_idx"] = channel_idx
            feats["block_idx"] = blk
            feats["t_start_sec"] = blk * block_sec
            feats["t_end_sec"] = (blk + 1) * block_sec
            feats["rhythm_label"] = rhythm
            feats["source"] = "afdb"
            features_list.append(feats)

    if not features_list:
        raise RuntimeError("No valid labeled blocks extracted from AFDB.")

    df = pd.DataFrame(features_list)
    logging.info("Total AFDB blocks: %d", len(df))
    logging.info("AFDB label distribution:\n%s", df["rhythm_label"].value_counts())
    return df


# ---------------------------------------------------------------------------
# NEW: JSON signal loader
# ---------------------------------------------------------------------------

# Common sampling frequency for the entire pipeline.
# AFDB records are natively at 250 Hz (used as-is).
# JSON device signals are resampled to this frequency before feature
# extraction so that all time-dependent features (RR intervals, QRS width,
# wavelet bands) are computed on the same scale.
TARGET_FS = 250.0


def _resample_to_target(signal: np.ndarray, src_fs: float, dst_fs: float) -> np.ndarray:
    """
    Resample a 1-D signal from src_fs to dst_fs using polyphase filtering.

    No-op when src_fs == dst_fs. Uses scipy.signal.resample_poly which applies
    an anti-aliasing FIR filter before decimation to avoid aliasing artifacts.

    Parameters
    ----------
    signal : np.ndarray
        1-D float array of ECG samples at src_fs Hz.
    src_fs : float
        Source sampling frequency in Hz.
    dst_fs : float
        Target sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Resampled signal at dst_fs Hz, same dtype as input.
    """
    from scipy.signal import resample_poly
    from math import gcd

    if src_fs == dst_fs:
        return signal

    src_int = int(round(src_fs))
    dst_int = int(round(dst_fs))
    common = gcd(dst_int, src_int)
    up = dst_int // common
    down = src_int // common

    return resample_poly(signal, up, down).astype(signal.dtype)


def _bandpass_prefilter(
    signal: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to a JSON ECG signal before feature
    extraction.

    This attenuates:
      - Baseline wander below 0.5 Hz (respiratory and motion artifacts)
      - High-frequency EMG/muscular noise above 40 Hz

    The filter is applied with zero-phase (filtfilt) to avoid phase distortion.
    If the signal is too short for the filter order, it is returned unchanged.

    Parameters
    ----------
    signal : np.ndarray
        1-D float array of ECG samples.
    fs : float
        Sampling frequency in Hz.
    lowcut : float
        High-pass cutoff frequency in Hz (default 0.5).
    highcut : float
        Low-pass cutoff frequency in Hz (default 40.0).
    order : int
        Butterworth filter order (default 4).

    Returns
    -------
    np.ndarray
        Filtered signal of the same shape and dtype as input.
    """
    from scipy.signal import butter, filtfilt

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    # filtfilt requires signal length > padlen = 3 * max(len(a), len(b))
    min_len = 3 * (order * 2 + 1) * 2
    if len(signal) < min_len:
        logging.debug(
            "Signal too short for bandpass filter (%d samples). Skipping.", len(signal)
        )
        return signal

    try:
        b, a = butter(order, [low, high], btype="band")
        return filtfilt(b, a, signal).astype(signal.dtype)
    except Exception as exc:
        logging.warning("Bandpass prefilter failed: %s. Using raw signal.", exc)
        return signal


def load_json_signals(data_dir: str, json_fs: float) -> pd.DataFrame:
    """
    Load all JSON ECG signals, apply a 0.5-40 Hz bandpass prefilter, extract
    features, and tag each sample with its class label and noise type.

    The bandpass prefilter is applied to every signal before feature extraction
    to attenuate baseline wander (< 0.5 Hz) and high-frequency muscular noise
    (> 40 Hz), improving feature quality for noisy signals.

    Directory structure expected:
      data_dir/
        <FOLDER_NAME>/          <- class root (clean signals)
          *.json
          Muscular/             <- muscular noise signals
            *.json
          Respiracion/          <- respiratory noise signals
            *.json

    Parameters
    ----------
    data_dir : str
        Root directory containing one subfolder per class.
    json_fs : float
        Sampling frequency of the JSON signals in Hz.

    Returns
    -------
    pd.DataFrame
        Feature rows with 'rhythm_label', 'noise_type', 'filename', 'source'
        columns added.
    """
    rows = []

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"JSON data directory not found: '{data_dir}'")

    class_folders = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )

    logging.info("JSON folders (repr): %s", [repr(f) for f in class_folders])

    for folder_name in class_folders:
        class_label = _FOLDER_LOOKUP.get(folder_name.strip().upper())
        if class_label is None:
            logging.warning("No class mapping for folder '%s'. Skipping.", folder_name)
            continue

        class_path = os.path.join(data_dir, folder_name)
        scan_targets = [("clean", class_path)]

        for noise_name in _NOISE_SUBFOLDERS:
            noise_path = os.path.join(class_path, noise_name)
            if os.path.isdir(noise_path):
                scan_targets.append((noise_name, noise_path))
            else:
                logging.warning(
                    "Noise subfolder '%s' not found inside '%s'. Skipping.",
                    noise_name,
                    class_path,
                )

        for noise_type, folder_path in scan_targets:
            json_files = sorted(
                f for f in os.listdir(folder_path) if f.endswith(".json")
            )
            logging.info(
                "  Class='%s' | Noise='%s' | Files: %d",
                class_label,
                noise_type,
                len(json_files),
            )

            for filename in json_files:
                filepath = os.path.join(folder_path, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    ecg_array = data.get("ecg", [])
                    if not ecg_array:
                        logging.warning("Empty ecg array in '%s'. Skipping.", filepath)
                        continue

                    signal = np.array([s["v_raw"] for s in ecg_array], dtype=np.float32)
                    # Resample from device fs (e.g. 200 Hz) to TARGET_FS (250 Hz)
                    # so all features are computed on the same frequency scale as AFDB
                    signal = _resample_to_target(signal, json_fs, TARGET_FS)
                    # Apply bandpass prefilter (0.5-40 Hz) after resampling
                    signal = _bandpass_prefilter(signal, TARGET_FS)

                except Exception as exc:
                    logging.error("Failed to load '%s': %s", filepath, exc)
                    continue

                # Extract features at TARGET_FS (consistent with AFDB training data)
                feats = extract_all_features(signal, TARGET_FS)
                feats["rhythm_label"] = class_label
                feats["noise_type"] = noise_type
                feats["filename"] = filename
                feats["source"] = "json"
                rows.append(feats)

    if not rows:
        raise RuntimeError("No valid JSON signals were loaded.")

    df = pd.DataFrame(rows)
    logging.info("Total JSON samples: %d", len(df))
    logging.info(
        "JSON label distribution:\n%s",
        df.groupby(["rhythm_label", "noise_type"]).size().to_string(),
    )
    return df


# ---------------------------------------------------------------------------
# Consolidated report writer
# ---------------------------------------------------------------------------


def save_metrics_report(metrics: dict, results_dir: str):
    """
    Save all metrics and evaluation results to two files:
      - metrics.json : machine-readable full metrics (timestamped, always appended)
      - report.txt   : single human-readable narrative report (overwritten each run)

    The TXT report consolidates AFDB internal evaluation, JSON original signal
    evaluation, and augmented signal evaluation in one place.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- metrics.json (timestamped, keeps history across runs) ---
    json_path = os.path.join(results_dir, f"metrics_{timestamp}.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logging.info("Metrics JSON -> %s", json_path)
    except Exception as e:
        logging.error("Failed to save metrics JSON: %s", e)

    # --- report.txt (single file, overwritten each run) ---
    txt_path = os.path.join(results_dir, "report.txt")
    sep = "=" * 64
    sep2 = "-" * 64

    def bar(label: str, value: float, width: int = 30) -> str:
        """Render a simple ASCII progress bar for a 0-1 metric."""
        filled = int(round(value * width))
        return f"[{'#' * filled}{'.' * (width - filled)}] {value * 100:.1f}%"

    try:
        with open(txt_path, "w", encoding="utf-8") as f:

            # Header
            f.write(f"{sep}\n")
            f.write("  AFDB RHYTHM CLASSIFIER — TRAINING REPORT\n")
            f.write(f"{sep}\n")
            f.write(f"  Run timestamp  : {timestamp}\n")
            f.write(
                f"  Records used   : {len(metrics.get('records', []))} AFDB records\n"
            )
            f.write(f"  Channel        : {metrics.get('channel_idx', 'N/A')}\n")
            f.write(f"  Block size     : {metrics.get('block_sec', 'N/A')} s\n")
            f.write(
                f"  Pipeline fs    : {metrics.get('target_fs', 250)} Hz "
                f"(AFDB native; JSON device signals resampled to match)\n"
            )
            f.write(
                f"  Training set   : {metrics.get('n_samples', 'N/A')} samples "
                f"| {metrics.get('n_features', 'N/A')} features "
                f"| {metrics.get('n_classes', 'N/A')} classes\n"
            )

            # Dataset composition breakdown
            n_afdb = metrics.get("n_afdb_samples", "N/A")
            n_json = metrics.get("n_json_samples", "N/A")
            n_aug = metrics.get("n_aug_samples", "N/A")
            f.write(
                f"  Dataset split  : AFDB={n_afdb} | "
                f"JSON (all noise types)={n_json} | "
                f"Augmented (all noise types)={n_aug}\n"
            )
            if n_aug_total != "N/A":
                f.write(
                    f"  Augmented total: {n_aug_total} signals "
                    f"(clean + Muscular + Respiracion — noisy used for eval only)\n"
                )
            f.write(f"{sep}\n\n")

            # Class distribution
            f.write("CLASS DISTRIBUTION (training set)\n")
            f.write(f"{sep2}\n")
            dist = metrics.get("class_distribution", {})
            total = sum(dist.values()) or 1
            for label, count in dist.items():
                pct = count / total * 100
                f.write(f"  {label:<8} {count:>6} samples  ({pct:5.1f}%)\n")
            f.write("\n")

            # Internal AFDB evaluation
            f.write("INTERNAL EVALUATION (AFDB 25% test split)\n")
            f.write(f"{sep2}\n")
            f.write(
                f"  Train accuracy : {metrics.get('train_accuracy', 0):.4f}  "
                f"{bar(metrics.get('train_accuracy', 0))}\n"
            )
            f.write(
                f"  Test accuracy  : {metrics.get('accuracy', 0):.4f}  "
                f"{bar(metrics.get('accuracy', 0))}\n"
            )
            f.write(
                f"  F1 weighted    : {metrics.get('f1_weighted', 0):.4f}  "
                f"{bar(metrics.get('f1_weighted', 0))}\n"
            )
            f.write(
                f"  F1 macro       : {metrics.get('f1_macro', 0):.4f}  "
                f"{bar(metrics.get('f1_macro', 0))}\n"
            )
            f.write(f"  Precision (w)  : {metrics.get('precision', 0):.4f}\n")
            f.write(f"  Recall (w)     : {metrics.get('recall', 0):.4f}\n\n")
            if "classification_report" in metrics:
                f.write("  Per-class breakdown:\n")
                for line in metrics["classification_report"].splitlines():
                    f.write(f"    {line}\n")
            f.write("\n")

            # JSON original signal evaluation
            json_eval = metrics.get("json_evaluation", [])
            aug_eval = metrics.get("aug_evaluation", [])

            if json_eval or aug_eval:
                f.write("NEW SIGNAL EVALUATION (field device data)\n")
                f.write(f"{sep2}\n")
                f.write(
                    f"  {'Source':<22} {'Noise type':<18} {'Accuracy':>10} {'n':>6}\n"
                )
                f.write(f"  {'-'*22} {'-'*18} {'-'*10} {'-'*6}\n")

                for entry in json_eval:
                    acc_bar = bar(entry["accuracy"], width=20)
                    f.write(
                        f"  {'Original (200 Hz)':<22} {entry['noise_type']:<18} "
                        f"{entry['accuracy']:>10.4f} {entry['n_samples']:>6}"
                        f"  {acc_bar}\n"
                    )

                if json_eval and aug_eval:
                    f.write(f"  {'':22} {'':18}\n")  # spacer row

                for entry in aug_eval:
                    label = entry["noise_type"].replace("aug_", "")
                    acc_bar = bar(entry["accuracy"], width=20)
                    f.write(
                        f"  {'Augmented (250 Hz)':<22} {label:<18} "
                        f"{entry['accuracy']:>10.4f} {entry['n_samples']:>6}"
                        f"  {acc_bar}\n"
                    )
                f.write("\n")

            f.write(f"{sep}\n")
            f.write("  Output files in this directory:\n")
            f.write("    metrics_<timestamp>.json  -> full machine-readable metrics\n")
            f.write(
                "    report.txt                -> this report (overwritten each run)\n"
            )
            f.write(
                "    evaluation_summary.csv    -> all evaluation rows in one table\n"
            )
            f.write("    predictions.csv           -> per-signal predictions\n")
            f.write(
                "    confusion_matrix.png      -> AFDB test split confusion matrix\n"
            )
            f.write("    feature_importance.png    -> top 20 feature importances\n")
            f.write("    pca_visualization.png     -> PCA of training feature space\n")
            f.write("    class_distribution.png    -> training class distribution\n")
            f.write(f"{sep}\n")

        logging.info("Report TXT  -> %s", txt_path)
    except Exception as e:
        logging.error("Failed to save report TXT: %s", e)

    return json_path, txt_path


# ---------------------------------------------------------------------------
# Main training pipeline (extended)
# ---------------------------------------------------------------------------


def train_supervised_model(
    records,
    pn_dir,
    block_sec=60,
    channel_idx=1,
    save_model=True,
    results_dir="results_supervised",
    data_dir=None,
    json_fs=200.0,
    augmented_dir=None,
    augmented_fs=250.0,
):
    """
    Supervised classification pipeline using real AFDB rhythm labels,
    optionally extended with original JSON signals and augmented signals
    produced by afdb_augment.py.

    Training set composition:
      - All AFDB blocks (always)
      - Clean original JSON signals from data_dir at json_fs (if provided)
      - Clean augmented signals from augmented_dir at augmented_fs (if provided)
        These are already resampled to match AFDB frequency, reducing the
        domain mismatch between training and new-device signals.

    Evaluation:
      - Original JSON signals broken down by noise_type
      - Augmented JSON signals broken down by noise_type (labelled 'aug_*')

    Parameters
    ----------
    records : list of str
        AFDB record IDs to stream from PhysioNet.
    pn_dir : str
        PhysioNet database name ('afdb') or local directory path.
    block_sec : float
        Block duration in seconds.
    channel_idx : int
        ECG channel index to use from each record.
    save_model : bool
        Whether to save model artifacts to results_dir.
    results_dir : str
        Output directory for all results and artifacts.
    data_dir : str or None
        Root directory of original JSON signals at json_fs Hz.
    json_fs : float
        Sampling frequency of the original JSON signals in Hz.
    augmented_dir : str or None
        Root directory of augmented signals from afdb_augment.py.
        These are already resampled to augmented_fs Hz.
    augmented_fs : float
        Sampling frequency of augmented signals in Hz (default 250, matching AFDB).
    """
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    logging.info("Results directory : %s", results_dir)
    logging.info("Processing channel: %d", channel_idx)

    # Verify write access
    test_file = os.path.join(results_dir, "_test_write.tmp")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        logging.info("Directory is writable.")
    except Exception as e:
        raise RuntimeError(f"Cannot write to {results_dir}: {e}")

    # ------------------------------------------------------------------
    # 1. Load AFDB data
    # ------------------------------------------------------------------
    logging.info("Extracting features from AFDB records...")
    afdb_df = extract_features_with_labels(records, pn_dir, block_sec, channel_idx)
    afdb_df["class_label"] = afdb_df["rhythm_label"]

    # ------------------------------------------------------------------
    # 2. Load original JSON data (optional)
    # ------------------------------------------------------------------
    json_df = None
    if data_dir is not None:
        logging.info(
            "Loading original JSON signals from '%s' (%.0f Hz)...", data_dir, json_fs
        )
        json_df = load_json_signals(data_dir, json_fs)
        json_df["class_label"] = json_df["rhythm_label"]

    # ------------------------------------------------------------------
    # 2b. Load augmented JSON data (optional)
    # ------------------------------------------------------------------
    aug_df = None
    if augmented_dir is not None:
        logging.info(
            "Loading augmented JSON signals from '%s' (%.0f Hz)...",
            augmented_dir,
            augmented_fs,
        )
        aug_df = load_json_signals(augmented_dir, augmented_fs)
        aug_df["class_label"] = aug_df["rhythm_label"]
        # Tag augmented noise types so they are reported separately
        aug_df["noise_type"] = "aug_" + aug_df["noise_type"]
        logging.info(
            "Augmented signal distribution:\n%s",
            aug_df.groupby(["class_label", "noise_type"]).size().to_string(),
        )

    # ------------------------------------------------------------------
    # 3. Build training set
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 3. Build training set
    #
    # Previously: AFDB + clean JSON + clean augmented only.
    # Now: AFDB + ALL JSON signals (all noise types) + ALL augmented signals.
    # The held-out evaluation now comes from the AFDB internal split AND
    # from a dedicated test fraction of the JSON/augmented pool (see below).
    # ------------------------------------------------------------------
    train_parts = [afdb_df]

    if json_df is not None:
        # Include all JSON signals regardless of noise type.
        # Previously only json_df[noise_type == "clean"] was used.
        logging.info(
            "Adding %d JSON samples (all noise types) to training set.", len(json_df)
        )
        train_parts.append(json_df)

    if aug_df is not None:
        # Include all augmented signals regardless of noise type.
        # Previously only aug_df[noise_type == "aug_clean"] was used.
        logging.info(
            "Adding %d augmented samples (all noise types) to training set.",
            len(aug_df),
        )
        train_parts.append(aug_df)

    train_df = pd.concat(train_parts, ignore_index=True)

    # ------------------------------------------------------------------
    # 4. Prepare feature matrix
    # ------------------------------------------------------------------
    feature_cols = sorted(
        c
        for c in train_df.columns
        if c not in _METADATA_COLS
        and train_df[c].dtype in [np.float64, np.int64, float, int]
    )

    X = train_df[feature_cols].values.astype(np.float64)
    y_raw = train_df["class_label"].values

    # Impute NaN with per-column median (computed on training data)
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logging.info("Feature matrix shape : %s", X.shape)
    logging.info("Classes              : %s", list(label_encoder.classes_))
    logging.info("Class distribution   : %s", np.bincount(y).tolist())

    # ------------------------------------------------------------------
    # 5. Train / test split on the training set (for internal metrics)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )
    logging.info("Train samples: %d | Test samples: %d", len(y_train), len(y_test))

    # ------------------------------------------------------------------
    # 6. Train RandomForest
    # ------------------------------------------------------------------
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    logging.info("Training RandomForest classifier...")
    clf.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 7. Internal evaluation (AFDB split)
    # ------------------------------------------------------------------
    y_pred = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)

    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_weighted": float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision": float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
    }
    class_report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0,
    )
    logging.info("Internal test accuracy : %.4f", test_metrics["accuracy"])
    logging.info("Internal train accuracy: %.4f", train_acc)
    logging.info("\n%s", class_report)

    # ------------------------------------------------------------------
    # 8. JSON evaluation broken down by noise_type (new)
    # ------------------------------------------------------------------
    json_eval_rows = []

    if json_df is not None:
        logging.info("Evaluating on JSON signals by noise type...")

        X_json = json_df[feature_cols].values.astype(np.float64)
        nan_json = np.isnan(X_json)
        X_json[nan_json] = np.take(col_medians, np.where(nan_json)[1])

        X_json_scaled = scaler.transform(X_json)
        y_json_enc = label_encoder.transform(json_df["class_label"].values)
        y_json_pred = clf.predict(X_json_scaled)

        # Summary by noise type
        noise_types = ["ALL"] + ["clean"] + _NOISE_SUBFOLDERS

        for noise_type in noise_types:
            if noise_type == "ALL":
                mask = np.ones(len(json_df), dtype=bool)
            else:
                mask = json_df["noise_type"].values == noise_type

            if not mask.any():
                continue

            acc = accuracy_score(y_json_enc[mask], y_json_pred[mask])
            logging.info(
                "JSON | noise_type='%s' | accuracy=%.4f | n=%d",
                noise_type,
                acc,
                mask.sum(),
            )
            json_eval_rows.append(
                {
                    "noise_type": noise_type,
                    "accuracy": round(float(acc), 4),
                    "n_samples": int(mask.sum()),
                }
            )

    # ------------------------------------------------------------------
    # 8b. Augmented data evaluation broken down by noise_type
    # ------------------------------------------------------------------
    aug_eval_rows = []

    if aug_df is not None:
        logging.info("Evaluating on augmented signals by noise type...")

        X_aug = aug_df[feature_cols].values.astype(np.float64)
        nan_aug = np.isnan(X_aug)
        X_aug[nan_aug] = np.take(col_medians, np.where(nan_aug)[1])

        X_aug_scaled = scaler.transform(X_aug)
        y_aug_enc = label_encoder.transform(aug_df["class_label"].values)
        y_aug_pred = clf.predict(X_aug_scaled)

        # Summary by noise type (aug_clean, aug_Muscular, aug_Respiracion)
        aug_noise_types = ["ALL"] + sorted(aug_df["noise_type"].unique())

        for noise_type in aug_noise_types:
            if noise_type == "ALL":
                mask = np.ones(len(aug_df), dtype=bool)
            else:
                mask = aug_df["noise_type"].values == noise_type

            if not mask.any():
                continue

            acc = accuracy_score(y_aug_enc[mask], y_aug_pred[mask])
            logging.info(
                "AUG | noise_type='%s' | accuracy=%.4f | n=%d",
                noise_type,
                acc,
                mask.sum(),
            )
            aug_eval_rows.append(
                {
                    "noise_type": noise_type,
                    "accuracy": round(float(acc), 4),
                    "n_samples": int(mask.sum()),
                }
            )

    # ------------------------------------------------------------------
    # 9. Visualisations (unchanged from original)
    # ------------------------------------------------------------------
    logging.info("Generating plots...")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        cbar_kws={"label": "Count"},
        ax=ax1,
    )
    ax1.set_title(
        "Confusion Matrix - AFDB Rhythm Classification", fontsize=14, fontweight="bold"
    )
    ax1.set_xlabel("Predicted Label", fontsize=12)
    ax1.set_ylabel("True Label", fontsize=12)
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    fig1.tight_layout()
    fig1.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    logging.info("Confusion matrix saved: %s", cm_path)

    # Feature importance
    feat_imp = pd.Series(clf.feature_importances_, index=feature_cols)
    top20 = feat_imp.sort_values(ascending=False).head(20)
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    top20.plot(kind="barh", color="steelblue", ax=ax2)
    ax2.set_title("Top 20 Feature Importances", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Importance", fontsize=12)
    ax2.invert_yaxis()
    fi_path = os.path.join(results_dir, "feature_importance.png")
    fig2.tight_layout()
    fig2.savefig(fi_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    logging.info("Feature importance saved: %s", fi_path)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    for i, label in enumerate(label_encoder.classes_):
        mask = y == i
        ax3.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=label,
            alpha=0.6,
            s=30,
            edgecolor="k",
            linewidth=0.5,
        )
    ax3.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=12
    )
    ax3.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=12
    )
    ax3.set_title("PCA - AFDB Rhythm Classes", fontsize=14, fontweight="bold")
    ax3.legend(title="Rhythm", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax3.grid(alpha=0.3)
    pca_path = os.path.join(results_dir, "pca_visualization.png")
    fig3.tight_layout()
    fig3.savefig(pca_path, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    logging.info("PCA visualization saved: %s", pca_path)

    # Class distribution
    class_counts = pd.Series(y).value_counts()
    class_names = [label_encoder.classes_[i] for i in class_counts.index]
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.bar(class_names, class_counts.values, color="coral", edgecolor="black")
    ax4.set_title("Class Distribution in Training Set", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Rhythm Type", fontsize=12)
    ax4.set_ylabel("Number of Blocks", fontsize=12)
    ax4.tick_params(axis="x", rotation=45)
    ax4.grid(axis="y", alpha=0.3)
    cd_path = os.path.join(results_dir, "class_distribution.png")
    fig4.tight_layout()
    fig4.savefig(cd_path, dpi=300, bbox_inches="tight")
    plt.close(fig4)
    logging.info("Class distribution saved: %s", cd_path)

    # ------------------------------------------------------------------
    # 10. Consolidated output files
    # ------------------------------------------------------------------

    # --- Single predictions.csv with source column ---
    all_pred_parts = []

    if json_df is not None:
        X_json = json_df[feature_cols].values.astype(np.float64)
        nan_mask = np.isnan(X_json)
        X_json[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
        y_json_pred = clf.predict(scaler.transform(X_json))

        pred_json = (
            json_df[["filename", "class_label", "noise_type"]]
            .copy()
            .reset_index(drop=True)
        )
        pred_json["source"] = "original"
        pred_json["predicted_label"] = label_encoder.inverse_transform(y_json_pred)
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(scaler.transform(X_json))
            for i, cls in enumerate(label_encoder.classes_):
                pred_json[f"prob_{cls}"] = proba[:, i]
        all_pred_parts.append(pred_json)

    if aug_df is not None:
        X_aug = aug_df[feature_cols].values.astype(np.float64)
        nan_mask = np.isnan(X_aug)
        X_aug[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
        y_aug_pred = clf.predict(scaler.transform(X_aug))

        pred_aug = (
            aug_df[["filename", "class_label", "noise_type"]]
            .copy()
            .reset_index(drop=True)
        )
        pred_aug["source"] = "augmented"
        pred_aug["noise_type"] = pred_aug["noise_type"].str.replace(
            "aug_", "", regex=False
        )
        pred_aug["predicted_label"] = label_encoder.inverse_transform(y_aug_pred)
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(scaler.transform(X_aug))
            for i, cls in enumerate(label_encoder.classes_):
                pred_aug[f"prob_{cls}"] = proba[:, i]
        all_pred_parts.append(pred_aug)

    if all_pred_parts:
        predictions_path = os.path.join(results_dir, "predictions.csv")
        pd.concat(all_pred_parts, ignore_index=True).to_csv(
            predictions_path, index=False
        )
        logging.info("Consolidated predictions -> '%s'", predictions_path)

    # --- Single evaluation_summary.csv ---
    summary_rows = []

    # AFDB internal split row
    summary_rows.append(
        {
            "source": "AFDB internal",
            "noise_type": "test_split",
            "accuracy": round(test_metrics["accuracy"], 4),
            "f1_weighted": round(test_metrics["f1_weighted"], 4),
            "f1_macro": round(test_metrics["f1_macro"], 4),
            "n_samples": int(len(y_test)),
        }
    )

    # Original JSON rows
    for row in json_eval_rows:
        summary_rows.append(
            {
                "source": "original",
                "noise_type": row["noise_type"],
                "accuracy": row["accuracy"],
                "f1_weighted": None,
                "f1_macro": None,
                "n_samples": row["n_samples"],
            }
        )

    # Augmented rows (strip aug_ prefix for readability)
    for row in aug_eval_rows:
        summary_rows.append(
            {
                "source": "augmented",
                "noise_type": row["noise_type"].replace("aug_", ""),
                "accuracy": row["accuracy"],
                "f1_weighted": None,
                "f1_macro": None,
                "n_samples": row["n_samples"],
            }
        )

    eval_summary_path = os.path.join(results_dir, "evaluation_summary.csv")
    pd.DataFrame(summary_rows).to_csv(eval_summary_path, index=False)
    logging.info("Evaluation summary -> '%s'", eval_summary_path)

    # ------------------------------------------------------------------
    # 11. Metrics JSON + narrative report.txt
    # ------------------------------------------------------------------
    class_dist = {
        cls: int(np.sum(y == i)) for i, cls in enumerate(label_encoder.classes_)
    }

    final_metrics = {
        "timestamp": datetime.now().isoformat(),
        "records": records,
        "channel_idx": channel_idx,
        "block_sec": float(block_sec),
        "data_dir": data_dir,
        "json_fs": float(json_fs),
        "target_fs": TARGET_FS,
        "n_samples": int(len(y)),
        "n_features": int(len(feature_cols)),
        "n_classes": int(len(label_encoder.classes_)),
        "classes": label_encoder.classes_.tolist(),
        "class_distribution": class_dist,
        "train_accuracy": float(train_acc),
        "classification_report": class_report,
        "json_evaluation": json_eval_rows,
        "aug_evaluation": aug_eval_rows,
        # Dataset composition counts
        "n_afdb_samples": int(len(afdb_df)),
        "n_json_samples": int(len(json_df)) if json_df is not None else 0,
        "n_aug_samples": int(len(aug_df)) if aug_df is not None else 0,
    }
    save_metrics_report(final_metrics, results_dir)

    # ------------------------------------------------------------------
    # 12. Save model artifacts
    # ------------------------------------------------------------------
    if save_model:
        logging.info("Saving model artifacts...")

        artifacts = {
            "afdb_rhythm_classifier.joblib": clf,
            "afdb_scaler.joblib": scaler,
            "afdb_label_encoder.joblib": label_encoder,
            "afdb_feature_columns.joblib": feature_cols,
            "afdb_col_medians.joblib": col_medians,
        }

        for filename, obj in artifacts.items():
            path = os.path.join(results_dir, filename)
            try:
                joblib.dump(obj, path)
                logging.info("Saved: %s (%d bytes)", path, os.path.getsize(path))
            except Exception as e:
                logging.error("Failed to save %s: %s", filename, e)

    return clf, final_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    # Complete list of all available AFDB records on PhysioNet.
    # Used when --use-all-records is passed instead of --records.
    AFDB_ALL_RECORDS = [
        "04015",
        "04043",
        "04048",
        "04126",
        "04746",
        "04908",
        "04936",
        "05091",
        "05121",
        "05261",
        "06426",
        "06453",
        "06995",
        "07162",
        "07859",
        "07879",
        "07910",
        "08215",
        "08219",
        "08378",
        "08405",
        "08434",
        "08455",
    ]

    parser = argparse.ArgumentParser(
        description="AFDB Supervised Classification with optional JSON retraining"
    )

    # --records and --use-all-records are mutually exclusive
    records_group = parser.add_mutually_exclusive_group(required=True)
    records_group.add_argument(
        "--records",
        nargs="+",
        help="Space-separated list of AFDB record IDs to use for training.",
    )
    records_group.add_argument(
        "--use-all-records",
        action="store_true",
        help=(
            "Use all 23 available AFDB records instead of specifying them manually. "
            "Equivalent to passing all record IDs in --records."
        ),
    )
    parser.add_argument(
        "--pn-dir",
        default="afdb",
        help="AFDB directory or 'afdb' for PhysioNet streaming",
    )
    parser.add_argument(
        "--block-sec", type=float, default=60, help="Block duration in seconds"
    )
    parser.add_argument(
        "--channel-idx", type=int, default=1, help="ECG channel index (0 or 1)"
    )
    parser.add_argument(
        "--save-model", action="store_true", help="Save trained model artifacts"
    )
    parser.add_argument(
        "--results-dir", default="results_supervised", help="Output directory"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Root directory of original JSON signals at src_fs (optional). "
            "If provided, the model is retrained with AFDB + clean JSON signals "
            "and evaluated on all JSON signals by noise type."
        ),
    )
    parser.add_argument(
        "--json-fs",
        type=float,
        default=200.0,
        help="Sampling frequency of the original JSON signals in Hz (default: 200)",
    )
    parser.add_argument(
        "--augmented-dir",
        default=None,
        help=(
            "Root directory of augmented JSON signals produced by afdb_augment.py "
            "(optional). These signals are already resampled to --augmented-fs and "
            "include synthetic Muscular and Respiracion variants. When provided, "
            "augmented clean signals are added to the training set and augmented "
            "noisy signals are included in the evaluation breakdown."
        ),
    )
    parser.add_argument(
        "--augmented-fs",
        type=float,
        default=250.0,
        help=(
            "Sampling frequency of the augmented JSON signals in Hz (default: 250, "
            "matching AFDB). Must match the --dst-fs used in afdb_augment.py."
        ),
    )
    args = parser.parse_args()

    # Resolve the final record list
    records = AFDB_ALL_RECORDS if args.use_all_records else args.records

    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(results_dir, "training.log")),
        ],
    )

    logging.info("=" * 60)
    logging.info("AFDB SUPERVISED CLASSIFICATION")
    logging.info("Using real rhythm labels from .atr annotations")
    logging.info("Records (%d): %s", len(records), records)
    if args.data_dir:
        logging.info(
            "JSON retraining enabled: %s (fs=%.0f Hz)", args.data_dir, args.json_fs
        )
    if args.augmented_dir:
        logging.info(
            "Augmented data enabled: %s (fs=%.0f Hz)",
            args.augmented_dir,
            args.augmented_fs,
        )
    logging.info("=" * 60)
    logging.info("Python          : %s", sys.version)
    logging.info("Working dir     : %s", os.getcwd())
    logging.info("Results dir     : %s", results_dir)

    try:
        model, metrics = train_supervised_model(
            records=records,
            pn_dir=args.pn_dir,
            block_sec=args.block_sec,
            channel_idx=args.channel_idx,
            save_model=args.save_model,
            results_dir=results_dir,
            data_dir=args.data_dir,
            json_fs=args.json_fs,
            augmented_dir=args.augmented_dir,
            augmented_fs=args.augmented_fs,
        )
        logging.info("=" * 60)
        logging.info("TRAINING COMPLETED SUCCESSFULLY")
        logging.info("Results saved in: %s", results_dir)
        logging.info("=" * 60)

    except Exception as e:
        logging.error("TRAINING FAILED: %s", e, exc_info=True)
        raise
