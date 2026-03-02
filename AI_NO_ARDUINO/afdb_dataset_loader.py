"""
afdb_retrain_and_test.py
------------------------
Retraining and evaluation pipeline for the AFDB rhythm classifier.

This script:
  1. Downloads new test data from Google Drive (gdown).
  2. Loads AFDB training records from PhysioNet via streaming (wfdb).
  3. Extracts features from both AFDB and new JSON signals using the EXACT
     same feature extraction as the original training script (afdb_dataset_loader.py).
  4. Retrains a RandomForest classifier on the combined dataset.
  5. Evaluates on the new JSON signals broken down by noise condition.
  6. Saves new model artifacts (.joblib) and evaluation results (CSV).

Feature vector (39 features, must match training exactly):
  Temporal  ( 7): mean, std, median, min, max, rms, zcr
  RR        ( 8): rr_count, rr_mean, rr_std, rr_min, rr_max, rr_sdnn, rr_rmssd, rr_pnn50
  QRS       ( 6): qrs_count, qrs_width_mean, qrs_width_std, qrs_amp_mean, qrs_amp_std, qrs_area_mean
  Wavelet   (15): energy, std, mean x 5 coefficient arrays (db4, level=4)
  Extra     ( 3): n_rpeaks, n_samples, fs

Folder-to-class mapping for JSON data:
  ATRIAL 1   -> AFIB
  ATRIAL 2   -> AFL
  ECG NORMAL -> N

Requirements:
  pip install gdown wfdb joblib scikit-learn numpy pandas scipy pywt
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
import gdown
import pywt
import wfdb
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Google Drive folder ID for the new JSON test data
GDRIVE_FOLDER_ID = "1pwdN3yN-Y79zNcEMWO09i3C9V3D5g-2q"

# Local directory for downloaded JSON data
DATA_DIR = "data"

# AFDB records to stream from PhysioNet for retraining
# Add or remove record IDs as needed
AFDB_RECORDS = [
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

# PhysioNet database name for streaming
AFDB_PN_DIR = "afdb"

# Block duration in seconds (must match training)
BLOCK_SEC = 60

# ECG channel index to use from AFDB records (must match training)
CHANNEL_IDX = 1

# Sampling frequency of the new JSON signals (Hz)
JSON_FS = 200

# JSON structure keys
JSON_ECG_KEY = "ecg"
JSON_VOLTAGE_KEY = "v_raw"

# Noise subfolders inside each class folder
NOISE_SUBFOLDERS = ["Muscular", "Respiración"]

# Mapping from folder name to model class label (compared case-insensitively)
FOLDER_TO_CLASS = {
    "ATRIAL 1": "AFIB",
    "ATRIAL 2": "AFL",
    "ECG NORMAL": "N",
}

# Wavelet settings (must match training)
WAVELET_NAME = "db4"
WAVELET_LEVELS = 4

# RandomForest hyperparameters (must match training)
RF_N_ESTIMATORS = 200
RF_RANDOM_STATE = 42
RF_CLASS_WEIGHT = "balanced"

# Output directory for retrained model artifacts and results
OUTPUT_DIR = "results_retrained"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Internal normalised lookup for folder names
_FOLDER_LOOKUP = {k.strip().upper(): v for k, v in FOLDER_TO_CLASS.items()}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature extraction (identical to afdb_dataset_loader.py)
# ---------------------------------------------------------------------------


def bandpass(
    signal: np.ndarray,
    fs: float,
    lowcut: float = 5.0,
    highcut: float = 15.0,
    order: int = 3,
) -> np.ndarray:
    """Butterworth bandpass filter used in Pan-Tompkins pre-processing."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    """Simple moving-average integrator."""
    if width <= 1:
        return x
    kernel = np.ones(width) / width
    return np.convolve(x, kernel, mode="same")


def pan_tompkins_detect_rpeaks(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Simplified Pan-Tompkins R-peak detector.

    Steps:
      1. Bandpass filter (5-15 Hz) to isolate QRS energy.
      2. Differentiate and square to enhance steep slopes.
      3. Moving-window integration (120 ms).
      4. Peak detection with 75th-percentile height threshold and 200 ms
         minimum distance.
      5. Refine each peak to the local maximum within a 30 ms search radius.

    Parameters
    ----------
    signal : np.ndarray
        1-D ECG signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Sorted array of R-peak sample indices.
    """
    sig_f = bandpass(signal, fs)
    diff_sig = np.ediff1d(sig_f, to_begin=0)
    squared = diff_sig**2

    mwi_width = max(1, int(0.12 * fs))
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
        rpeaks.append(int(left + local_max))

    return np.unique(rpeaks).astype(int)


def extract_temporal_features(signal: np.ndarray) -> dict:
    """
    Seven time-domain statistical features.

    Returns
    -------
    dict with keys: mean, std, median, min, max, rms, zcr
    """
    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "median": float(np.median(signal)),
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "rms": float(np.sqrt(np.mean(signal**2))),
        "zcr": float(((signal[:-1] * signal[1:]) < 0).sum() / max(1, len(signal) - 1)),
    }


def extract_rr_features(rpeaks: np.ndarray, fs: float) -> dict:
    """
    Eight heart-rate variability features derived from RR intervals.

    Returns NaN for all features if fewer than 2 R-peaks are available.

    Returns
    -------
    dict with keys: rr_count, rr_mean, rr_std, rr_min, rr_max,
                    rr_sdnn, rr_rmssd, rr_pnn50
    """
    if rpeaks is None or len(rpeaks) < 2:
        return {
            "rr_count": 0,
            "rr_mean": np.nan,
            "rr_std": np.nan,
            "rr_min": np.nan,
            "rr_max": np.nan,
            "rr_sdnn": np.nan,
            "rr_rmssd": np.nan,
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


def extract_qrs_features(signal: np.ndarray, rpeaks: np.ndarray, fs: float) -> dict:
    """
    Six QRS morphology features averaged across all detected beats.

    QRS window: 60 ms half-window centred on each R-peak (120 ms total).
    QRS width is measured at half-amplitude using a descent search from the peak.

    Returns
    -------
    dict with keys: qrs_count, qrs_width_mean, qrs_width_std,
                    qrs_amp_mean, qrs_amp_std, qrs_area_mean
    """
    _empty = {
        "qrs_count": 0,
        "qrs_width_mean": np.nan,
        "qrs_width_std": np.nan,
        "qrs_amp_mean": np.nan,
        "qrs_amp_std": np.nan,
        "qrs_area_mean": np.nan,
    }

    if rpeaks is None or len(rpeaks) == 0:
        return _empty

    half_window = int(0.06 * fs)
    widths, amps, areas = [], [], []

    for r in rpeaks:
        left = max(r - half_window, 0)
        right = min(r + half_window, len(signal) - 1)
        seg = signal[left : right + 1]

        if seg.size == 0:
            continue

        amp = float(np.max(seg) - np.min(seg))
        amps.append(amp)
        areas.append(float(np.trapezoid(np.abs(seg))))

        # Half-amplitude width measurement
        trough_val = np.min(seg)
        half_amp = trough_val + 0.5 * amp

        l_idx = r
        while l_idx > left and signal[l_idx] > half_amp:
            l_idx -= 1

        r_idx = r
        while r_idx < right and signal[r_idx] > half_amp:
            r_idx += 1

        widths.append(float((r_idx - l_idx) / fs))

    if not widths:
        return _empty

    return {
        "qrs_count": len(widths),
        "qrs_width_mean": float(np.mean(widths)),
        "qrs_width_std": float(np.std(widths)),
        "qrs_amp_mean": float(np.mean(amps)),
        "qrs_amp_std": float(np.std(amps)),
        "qrs_area_mean": float(np.mean(areas)),
    }


def extract_wavelet_features(signal: np.ndarray) -> dict:
    """
    Wavelet decomposition features: energy, std, mean per coefficient array.

    Uses db4 wavelet at level 4, producing 5 coefficient arrays.
    Total: 15 features (3 stats x 5 arrays).

    Returns
    -------
    dict with keys: wave_energy_L0..L4, wave_std_L0..L4, wave_mean_L0..L4
    """
    if len(signal) < 8:
        return {}

    coeffs = pywt.wavedec(signal, WAVELET_NAME, level=WAVELET_LEVELS)
    feats = {}

    for i, c in enumerate(coeffs):
        arr = np.asarray(c, dtype=float)
        feats[f"wave_energy_L{i}"] = float(np.sum(arr**2))
        feats[f"wave_std_L{i}"] = float(np.std(arr))
        feats[f"wave_mean_L{i}"] = float(np.mean(arr))

    return feats


def extract_all_features(signal: np.ndarray, fs: float) -> dict:
    """
    Build the complete 39-feature vector for one ECG signal block.

    Replicates extract_all_features() from afdb_dataset_loader.py exactly.

    Parameters
    ----------
    signal : np.ndarray
        1-D float array of ECG samples.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    dict
        39-key feature dictionary (NaN for unavailable HRV/QRS features).
    """
    signal = np.asarray(signal, dtype=float)
    out = extract_temporal_features(signal)

    try:
        rpeaks = pan_tompkins_detect_rpeaks(signal, fs)
    except Exception:
        rpeaks = np.array([], dtype=int)

    out.update(extract_rr_features(rpeaks, fs))
    out.update(extract_qrs_features(signal, rpeaks, fs))
    out.update(extract_wavelet_features(signal))

    out["n_rpeaks"] = int(len(rpeaks))
    out["n_samples"] = int(len(signal))
    out["fs"] = float(fs)

    return out


# ---------------------------------------------------------------------------
# Step 1: Download JSON data from Google Drive
# ---------------------------------------------------------------------------


def download_drive_folder(folder_id: str, output_dir: str) -> None:
    """
    Download the Google Drive folder containing new JSON ECG signals.

    Skips download if the directory already contains files.
    """
    if os.path.exists(output_dir) and os.listdir(output_dir):
        logger.info(
            "Directory '%s' already exists and is not empty. Skipping download.",
            output_dir,
        )
        return

    logger.info("Downloading data from Google Drive folder: %s", folder_id)
    os.makedirs(output_dir, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    gdown.download_folder(url=url, output=output_dir, quiet=False, use_cookies=False)
    logger.info("Download complete.")


# ---------------------------------------------------------------------------
# Step 2: Load AFDB training data from PhysioNet
# ---------------------------------------------------------------------------


def load_afdb_features(
    records: list,
    pn_dir: str,
    block_sec: int,
    channel_idx: int,
) -> pd.DataFrame:
    """
    Stream AFDB records from PhysioNet, split into blocks, and extract features.

    Also reads rhythm annotations to assign a class label to each block.
    Blocks with no annotation default to 'N'.

    Parameters
    ----------
    records : list of str
        AFDB record IDs to stream.
    pn_dir : str
        PhysioNet database name ('afdb') or local directory path.
    block_sec : int
        Block duration in seconds.
    channel_idx : int
        ECG channel index to extract from each record.

    Returns
    -------
    pd.DataFrame
        Feature rows with an additional 'class_label' column.
    """
    rows = []

    for rec in records:
        logger.info("Streaming AFDB record: %s", rec)
        try:
            record = wfdb.rdrecord(rec, pn_dir=pn_dir)
        except Exception as exc:
            logger.warning("Could not load record %s: %s. Skipping.", rec, exc)
            continue

        fs = record.fs
        signal = record.p_signal[:, channel_idx]
        block_size = int(block_sec * fs)
        n_blocks = len(signal) // block_size

        # Load rhythm annotations
        try:
            ann = wfdb.rdann(rec, "atr", pn_dir=pn_dir)
            ann_samples = ann.sample
            ann_symbols = ann.aux_note  # rhythm labels like '(AFIB', '(N', etc.

            # Build a lookup: sample index -> rhythm string
            rhythm_map = {}
            for s, sym in zip(ann_samples, ann_symbols):
                cleaned = sym.strip().lstrip("(").strip()
                if cleaned:
                    rhythm_map[s] = cleaned
        except Exception as exc:
            logger.warning("Could not load annotations for %s: %s.", rec, exc)
            rhythm_map = {}

        sorted_ann_samples = sorted(rhythm_map.keys())

        def get_rhythm_label(block_start_sample: int) -> str:
            """Return the rhythm label active at block_start_sample."""
            label = "N"
            for s in sorted_ann_samples:
                if s <= block_start_sample:
                    label = rhythm_map[s]
                else:
                    break
            # Map to the four known classes; anything else -> N
            known = {"AFIB", "AFL", "J", "N"}
            return label if label in known else "N"

        logger.info(
            "  %s: %d blocks, fs=%d Hz, channel=%s",
            rec,
            n_blocks,
            fs,
            record.sig_name[channel_idx],
        )

        for blk in range(n_blocks):
            start = blk * block_size
            end = start + block_size
            block = signal[start:end]

            feats = extract_all_features(block, fs)
            feats["class_label"] = get_rhythm_label(start)
            feats["source"] = "afdb"
            feats["record"] = rec
            rows.append(feats)

    if not rows:
        raise RuntimeError("No AFDB blocks were successfully processed.")

    df = pd.DataFrame(rows)
    logger.info(
        "AFDB blocks loaded: %d | Class distribution:\n%s",
        len(df),
        df["class_label"].value_counts().to_string(),
    )
    return df


# ---------------------------------------------------------------------------
# Step 3: Load new JSON signals
# ---------------------------------------------------------------------------


def load_json_signal(filepath: str) -> np.ndarray | None:
    """
    Load a single ECG signal from a .json file (v_raw field).

    Parameters
    ----------
    filepath : str
        Path to the .json file.

    Returns
    -------
    np.ndarray or None
        1-D float32 array of v_raw samples, or None if loading fails.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if JSON_ECG_KEY not in data:
            logger.warning(
                "Key '%s' not found in '%s'. Skipping.", JSON_ECG_KEY, filepath
            )
            return None

        ecg_array = data[JSON_ECG_KEY]
        if not isinstance(ecg_array, list) or len(ecg_array) == 0:
            logger.warning("Empty ecg array in '%s'. Skipping.", filepath)
            return None

        return np.array([s[JSON_VOLTAGE_KEY] for s in ecg_array], dtype=np.float32)

    except KeyError as exc:
        logger.error("Missing key %s in '%s'. Skipping.", exc, filepath)
        return None
    except Exception as exc:
        logger.error("Failed to load '%s': %s", filepath, exc)
        return None


def load_json_features(data_dir: str) -> pd.DataFrame:
    """
    Walk the JSON data directory and extract features from every signal.

    Returns
    -------
    pd.DataFrame
        Feature rows with 'class_label', 'noise_type', 'filename', 'source' columns.
    """
    rows = []

    class_folders = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )

    logger.info(
        "JSON folders found: %s",
        [repr(f) for f in class_folders],
    )

    for folder_name in class_folders:
        class_label = _FOLDER_LOOKUP.get(folder_name.strip().upper())
        if class_label is None:
            logger.warning("No class mapping for folder '%s'. Skipping.", folder_name)
            continue

        class_path = os.path.join(data_dir, folder_name)
        scan_targets = [("clean", class_path)]

        for noise_name in NOISE_SUBFOLDERS:
            noise_path = os.path.join(class_path, noise_name)
            if os.path.isdir(noise_path):
                scan_targets.append((noise_name, noise_path))
            else:
                logger.warning(
                    "Noise subfolder '%s' not found inside '%s'. Skipping.",
                    noise_name,
                    class_path,
                )

        for noise_type, folder_path in scan_targets:
            json_files = sorted(
                f for f in os.listdir(folder_path) if f.endswith(".json")
            )
            logger.info(
                "  Class='%s' | Noise='%s' | Files: %d",
                class_label,
                noise_type,
                len(json_files),
            )

            for filename in json_files:
                signal = load_json_signal(os.path.join(folder_path, filename))
                if signal is None:
                    continue

                feats = extract_all_features(signal, JSON_FS)
                feats["class_label"] = class_label
                feats["noise_type"] = noise_type
                feats["filename"] = filename
                feats["source"] = "json"
                rows.append(feats)

    if not rows:
        raise RuntimeError("No JSON signals were successfully processed.")

    df = pd.DataFrame(rows)
    logger.info(
        "JSON samples loaded: %d | Class distribution:\n%s",
        len(df),
        df["class_label"].value_counts().to_string(),
    )
    return df


# ---------------------------------------------------------------------------
# Step 4: Retrain
# ---------------------------------------------------------------------------

# Columns that are metadata, not features
NON_FEATURE_COLS = {
    "class_label",
    "noise_type",
    "filename",
    "source",
    "record",
    "channel",
    "block_idx",
    "t_start_sec",
    "t_end_sec",
}


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return sorted list of feature column names (excludes metadata columns)."""
    return sorted(c for c in df.columns if c not in NON_FEATURE_COLS)


def retrain(
    afdb_df: pd.DataFrame,
    json_df: pd.DataFrame,
) -> tuple:
    """
    Combine AFDB and JSON datasets, train a new RandomForest classifier,
    and return the fitted artifacts.

    Training set : all AFDB blocks + clean JSON signals only
                   (noisy signals are held out for evaluation only)
    Test set     : all JSON signals (clean + noisy)

    Parameters
    ----------
    afdb_df : pd.DataFrame
        Feature rows from AFDB streaming.
    json_df : pd.DataFrame
        Feature rows from the new JSON signals.

    Returns
    -------
    tuple
        (classifier, scaler, label_encoder, feature_columns)
    """
    # Hold out all JSON data for evaluation; train only on AFDB + clean JSON
    json_clean = json_df[json_df["noise_type"] == "clean"].copy()
    combined = pd.concat([afdb_df, json_clean], ignore_index=True)

    feature_cols = get_feature_columns(combined)
    logger.info("Training feature count: %d", len(feature_cols))

    X_train = combined[feature_cols].values.astype(np.float64)
    y_train = combined["class_label"].values

    # Replace NaN with column median (same strategy as training)
    col_medians = np.nanmedian(X_train, axis=0)
    nan_mask = np.isnan(X_train)
    X_train[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    logger.info(
        "Training on %d samples | Classes: %s",
        len(X_train),
        list(label_encoder.classes_),
    )

    classifier = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        class_weight=RF_CLASS_WEIGHT,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )
    classifier.fit(X_scaled, y_enc)
    logger.info("Training complete.")

    return classifier, scaler, label_encoder, feature_cols, col_medians


# ---------------------------------------------------------------------------
# Step 5: Evaluate on JSON signals
# ---------------------------------------------------------------------------


def evaluate(
    json_df: pd.DataFrame,
    classifier,
    scaler,
    label_encoder,
    feature_cols: list,
    col_medians: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run inference on all JSON signals and evaluate accuracy per noise condition.

    Parameters
    ----------
    json_df : pd.DataFrame
        Feature rows from JSON signals (all noise types).
    classifier, scaler, label_encoder : fitted sklearn objects
    feature_cols : list of str
        Ordered list of feature column names.
    col_medians : np.ndarray
        Per-column medians from training (used to impute NaN).

    Returns
    -------
    tuple
        (predictions_df, summary_df)
    """
    X = json_df[feature_cols].values.astype(np.float64)

    # Impute NaN with training medians
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    X_scaled = scaler.transform(X)
    y_pred_encoded = classifier.predict(X_scaled)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

    results = (
        json_df[["filename", "class_label", "noise_type"]].copy().reset_index(drop=True)
    )
    results["predicted_label"] = y_pred_labels

    # Per-class probabilities
    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(X_scaled)
        for i, cls_name in enumerate(label_encoder.classes_):
            results[f"prob_{cls_name}"] = proba[:, i]

    # Evaluation summary
    summary_rows = []

    def _report(subset: pd.DataFrame, group: str):
        acc = accuracy_score(subset["class_label"], subset["predicted_label"])
        logger.info("===== Group: '%s' | Accuracy: %.4f =====", group, acc)
        logger.info(
            "\n%s",
            classification_report(
                subset["class_label"], subset["predicted_label"], zero_division=0
            ),
        )
        logger.info(
            "Confusion matrix:\n%s",
            confusion_matrix(subset["class_label"], subset["predicted_label"]),
        )
        summary_rows.append(
            {
                "noise_type": group,
                "accuracy": round(acc, 4),
                "n_samples": len(subset),
            }
        )

    _report(results, "ALL")
    for noise_type in ["clean"] + NOISE_SUBFOLDERS:
        subset = results[results["noise_type"] == noise_type]
        if not subset.empty:
            _report(subset, noise_type)

    return results, pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Step 6: Save artifacts and results
# ---------------------------------------------------------------------------


def save_artifacts(
    classifier,
    scaler,
    label_encoder,
    feature_cols: list,
    col_medians: np.ndarray,
) -> None:
    """Save retrained model artifacts to OUTPUT_DIR."""
    joblib.dump(classifier, os.path.join(OUTPUT_DIR, "afdb_rhythm_classifier.joblib"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "afdb_scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "afdb_label_encoder.joblib"))
    joblib.dump(feature_cols, os.path.join(OUTPUT_DIR, "afdb_feature_columns.joblib"))
    joblib.dump(col_medians, os.path.join(OUTPUT_DIR, "afdb_col_medians.joblib"))
    logger.info("Model artifacts saved to '%s/'.", OUTPUT_DIR)


def save_results(predictions: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Save prediction CSV and evaluation summary CSV to OUTPUT_DIR."""
    pred_path = os.path.join(OUTPUT_DIR, "afdb_predictions.csv")
    summary_path = os.path.join(OUTPUT_DIR, "afdb_evaluation_report.csv")

    predictions.to_csv(pred_path, index=False)
    summary.to_csv(summary_path, index=False)

    logger.info("Predictions saved : '%s'", pred_path)
    logger.info("Report saved      : '%s'", summary_path)
    logger.info("===== Final Summary =====\n%s", summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # 1. Download JSON data
    download_drive_folder(GDRIVE_FOLDER_ID, DATA_DIR)

    # 2. Stream AFDB records and extract features
    logger.info("Loading AFDB training data from PhysioNet...")
    afdb_df = load_afdb_features(AFDB_RECORDS, AFDB_PN_DIR, BLOCK_SEC, CHANNEL_IDX)

    # 3. Load and extract features from new JSON signals
    logger.info("Loading new JSON signals...")
    json_df = load_json_features(DATA_DIR)

    # 4. Retrain classifier on AFDB + clean JSON signals
    logger.info("Retraining classifier...")
    classifier, scaler, label_encoder, feature_cols, col_medians = retrain(
        afdb_df, json_df
    )

    # 5. Evaluate on all JSON signals (clean + noisy)
    logger.info("Evaluating on JSON test signals...")
    predictions, summary = evaluate(
        json_df, classifier, scaler, label_encoder, feature_cols, col_medians
    )

    # 6. Save everything
    save_artifacts(classifier, scaler, label_encoder, feature_cols, col_medians)
    save_results(predictions, summary)


if __name__ == "__main__":
    main()
