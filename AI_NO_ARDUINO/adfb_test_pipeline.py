"""
afdb_test_pipeline.py
---------------------
Test pipeline for the AFDB (Atrial Fibrillation Database) rhythm classifier.

This script replicates the exact feature extraction pipeline used during
training so that the pre-trained scaler and classifier receive the same
feature vector format they were fitted on.

Feature vector (per signal block):
  Temporal (7):   mean, std, median, min, max, RMS, zero-crossing rate
  HRV     (5):   RR mean, RR std, SDNN, RMSSD, pNN50
  QRS     (3):   QRS width, amplitude, area
  Wavelet (variable): energy + stats per decomposition level

Pipeline:
  1. Download test data from Google Drive (gdown).
  2. Load scaler, label encoder, and classifier from .joblib files.
  3. Parse each .json file and extract 'v_raw' samples.
  4. Extract features using the same algorithm used during training.
  5. Scale features and run classifier inference.
  6. Evaluate accuracy broken down by noise condition.
  7. Save predictions (CSV) and evaluation report (CSV).

Folder-to-class mapping:
  Atrial 1   -> AFIB  (Atrial Fibrillation)
  Atrial 2   -> AFL   (Atrial Flutter)
  ECG Normal -> N     (Normal sinus rhythm)
  J (AV Junctional)  -> no test data available

Subfolder structure per class folder:
  *.json          <- clean signals  (noise_type = 'clean')
  Muscular/*.json <- muscular noise (noise_type = 'Muscular')
  Respiracion/*.json <- respiratory noise (noise_type = 'Respiracion')

Signal properties:
  Sampling frequency : 200 Hz
  Voltage field      : v_raw (from each ECG sample object)

Requirements:
  pip install gdown joblib scikit-learn numpy pandas scipy pywt
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
import gdown
import pywt
from scipy.signal import find_peaks
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Google Drive folder ID from the shared link
GDRIVE_FOLDER_ID = "1pwdN3yN-Y79zNcEMWO09i3C9V3D5g-2q"

# Local directory where downloaded data will be stored
DATA_DIR = "data"

# Pre-trained model artifact paths
SCALER_PATH = "results_retrained/afdb_scaler.joblib"
ENCODER_PATH = "results_retrained/afdb_label_encoder.joblib"
CLASSIFIER_PATH = "results_retrained/afdb_rhythm_classifier.joblib"

# Output file paths
OUTPUT_PREDICTIONS_CSV = "afdb_predictions.csv"
OUTPUT_REPORT_CSV = "afdb_evaluation_report.csv"

# Sampling frequency of the new ECG signals (Hz)
SAMPLING_FREQ = 200

# JSON structure keys
JSON_ECG_KEY = "ecg"
JSON_VOLTAGE_KEY = "v_raw"

# Noise subfolders to scan inside each class folder
NOISE_SUBFOLDERS = ["Muscular", "Respiración"]

# Mapping from folder name to the model's class label
# Folder names must match exactly (case-sensitive) what is on disk
FOLDER_TO_CLASS = {
    "ATRIAL 1": "AFIB",
    "ATRIAL 2": "AFL",
    "ECG NORMAL": "N",
}

# Wavelet settings (must match training configuration)
WAVELET_NAME = "db4"
WAVELET_LEVELS = 5

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Download data from Google Drive
# ---------------------------------------------------------------------------


def download_drive_folder(folder_id: str, output_dir: str) -> None:
    """
    Download an entire Google Drive folder using gdown.

    Skips the download if the output directory already contains files.

    Parameters
    ----------
    folder_id : str
        Google Drive folder ID extracted from the share URL.
    output_dir : str
        Local path where the folder contents will be saved.
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
    logger.info("Download complete. Data saved to '%s'.", output_dir)


# ---------------------------------------------------------------------------
# Step 2: Load model artifacts
# ---------------------------------------------------------------------------


def load_artifacts(scaler_path: str, encoder_path: str, classifier_path: str):
    """
    Load the scaler, label encoder, and classifier from .joblib files.

    Parameters
    ----------
    scaler_path : str
        Path to the fitted StandardScaler .joblib file.
    encoder_path : str
        Path to the fitted LabelEncoder .joblib file.
    classifier_path : str
        Path to the fitted RandomForest classifier .joblib file.

    Returns
    -------
    tuple
        (scaler, label_encoder, classifier)
    """
    logger.info("Loading scaler       : '%s'", scaler_path)
    scaler = joblib.load(scaler_path)

    logger.info("Loading label encoder: '%s'", encoder_path)
    label_encoder = joblib.load(encoder_path)

    logger.info("Loading classifier   : '%s'", classifier_path)
    classifier = joblib.load(classifier_path)

    if hasattr(scaler, "n_features_in_"):
        logger.info("Scaler expects %d features per sample.", scaler.n_features_in_)

    logger.info("Known classes: %s", list(label_encoder.classes_))

    return scaler, label_encoder, classifier


# ---------------------------------------------------------------------------
# Step 3: Feature extraction (must match training pipeline exactly)
# ---------------------------------------------------------------------------


def extract_temporal_features(signal: np.ndarray) -> np.ndarray:
    """
    Compute 7 time-domain statistical features from a signal.

    Features: mean, std, median, min, max, RMS, zero-crossing rate.

    Parameters
    ----------
    signal : np.ndarray
        1-D array of signal samples.

    Returns
    -------
    np.ndarray
        Array of 7 float values.
    """
    mean = np.mean(signal)
    std = np.std(signal)
    median = np.median(signal)
    minimum = np.min(signal)
    maximum = np.max(signal)
    rms = np.sqrt(np.mean(signal**2))

    # Zero-crossing rate: fraction of consecutive pairs with opposite signs
    zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
    zcr = zero_crossings / max(len(signal) - 1, 1)

    return np.array([mean, std, median, minimum, maximum, rms, zcr])


def detect_r_peaks(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Detect R-peaks using a simplified Pan-Tompkins approach.

    Steps:
      1. Differentiate and square the signal to emphasize QRS complexes.
      2. Apply a moving-average integration window.
      3. Find peaks with a minimum distance of 0.3 s and a dynamic threshold.

    Parameters
    ----------
    signal : np.ndarray
        1-D ECG signal array.
    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Array of sample indices where R-peaks were detected.
    """
    # Derivative to highlight steep slopes in QRS
    diff_signal = np.diff(signal)

    # Squaring amplifies large slopes and makes all values positive
    squared = diff_signal**2

    # Moving-average integration window (~150 ms)
    window_size = int(0.15 * fs)
    if window_size < 1:
        window_size = 1
    kernel = np.ones(window_size) / window_size
    integrated = np.convolve(squared, kernel, mode="same")

    # Dynamic threshold: 30% of the signal maximum
    threshold = 0.3 * np.max(integrated) if np.max(integrated) > 0 else 0.0
    min_distance = int(0.3 * fs)  # minimum 300 ms between beats

    peaks, _ = find_peaks(integrated, height=threshold, distance=min_distance)
    return peaks


def extract_hrv_features(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Compute 5 heart-rate variability (HRV) features from RR intervals.

    Features: RR mean, RR std, SDNN, RMSSD, pNN50.

    Parameters
    ----------
    signal : np.ndarray
        1-D ECG signal array.
    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Array of 5 float values. Returns zeros if fewer than 2 R-peaks
        are detected.
    """
    peaks = detect_r_peaks(signal, fs)

    if len(peaks) < 2:
        return np.zeros(5)

    # RR intervals in milliseconds
    rr_intervals = np.diff(peaks) / fs * 1000.0

    rr_mean = np.mean(rr_intervals)
    rr_std = np.std(rr_intervals)
    sdnn = np.std(rr_intervals)  # standard deviation of NN intervals

    # RMSSD: root mean square of successive differences
    successive_diffs = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(successive_diffs**2)) if len(successive_diffs) > 0 else 0.0

    # pNN50: percentage of successive RR differences > 50 ms
    pnn50 = (
        np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100.0
        if len(successive_diffs) > 0
        else 0.0
    )

    return np.array([rr_mean, rr_std, sdnn, rmssd, pnn50])


def extract_qrs_features(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Compute 3 QRS morphology features averaged across all detected beats.

    Features: mean QRS width (samples), mean QRS amplitude, mean QRS area.

    QRS window: a 100 ms window centred on each detected R-peak.

    Parameters
    ----------
    signal : np.ndarray
        1-D ECG signal array.
    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Array of 3 float values. Returns zeros if no R-peaks are detected.
    """
    peaks = detect_r_peaks(signal, fs)

    if len(peaks) == 0:
        return np.zeros(3)

    half_window = int(0.05 * fs)  # 50 ms on each side -> 100 ms total QRS window
    widths, amplitudes, areas = [], [], []

    for peak in peaks:
        start = max(0, peak - half_window)
        end = min(len(signal), peak + half_window)
        qrs = signal[start:end]

        if len(qrs) == 0:
            continue

        widths.append(len(qrs))
        amplitudes.append(np.max(qrs) - np.min(qrs))
        areas.append(np.trapz(np.abs(qrs)))

    if not widths:
        return np.zeros(3)

    return np.array([np.mean(widths), np.mean(amplitudes), np.mean(areas)])


def extract_wavelet_features(signal: np.ndarray) -> np.ndarray:
    """
    Compute energy and statistical features from a multi-level wavelet
    decomposition.

    For each decomposition level the following are computed:
      energy, mean, std, max absolute value  (4 features per level)

    Total features = 4 * (WAVELET_LEVELS + 1)  -- levels + approximation.

    Parameters
    ----------
    signal : np.ndarray
        1-D ECG signal array.

    Returns
    -------
    np.ndarray
        1-D float array of wavelet features.
    """
    coeffs = pywt.wavedec(signal, WAVELET_NAME, level=WAVELET_LEVELS)
    features = []

    for coeff in coeffs:
        energy = np.sum(coeff**2)
        mean = np.mean(coeff)
        std = np.std(coeff)
        max_abs = np.max(np.abs(coeff)) if len(coeff) > 0 else 0.0
        features.extend([energy, mean, std, max_abs])

    return np.array(features)


def extract_features(signal: np.ndarray, fs: int = SAMPLING_FREQ) -> np.ndarray:
    """
    Build the complete feature vector for one ECG signal block.

    Concatenates: temporal (7) + HRV (5) + QRS (3) + wavelet (variable).
    This must exactly match the feature extraction performed during training.

    Parameters
    ----------
    signal : np.ndarray
        1-D float32 array of ECG samples (v_raw).
    fs : int
        Sampling frequency in Hz (default: SAMPLING_FREQ).

    Returns
    -------
    np.ndarray
        1-D float64 feature vector.
    """
    temporal = extract_temporal_features(signal)
    hrv = extract_hrv_features(signal, fs)
    qrs = extract_qrs_features(signal, fs)
    wavelet = extract_wavelet_features(signal)

    return np.concatenate([temporal, hrv, qrs, wavelet]).astype(np.float64)


# ---------------------------------------------------------------------------
# Step 4: Load signals from .json files
# ---------------------------------------------------------------------------


def load_json_signal(filepath: str) -> np.ndarray | None:
    """
    Load a single ECG signal from a .json file.

    Extracts the 'v_raw' field from each object inside the 'ecg' array.

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
            logger.warning("Empty or invalid 'ecg' array in '%s'. Skipping.", filepath)
            return None

        signal = np.array(
            [sample[JSON_VOLTAGE_KEY] for sample in ecg_array],
            dtype=np.float32,
        )
        return signal

    except KeyError as exc:
        logger.error("Missing key %s in a sample inside '%s'. Skipping.", exc, filepath)
        return None
    except Exception as exc:
        logger.error("Failed to load '%s': %s", filepath, exc)
        return None


def collect_samples(data_dir: str) -> list[dict]:
    """
    Walk the data directory and collect all ECG samples with their metadata.

    For each valid .json file the function loads the signal, extracts the
    feature vector, and stores it together with the class label and noise type.

    Each returned dict contains:
      - 'filename'    : name of the .json file
      - 'folder'      : original folder name (e.g., 'Atrial 1')
      - 'class_label' : mapped model class (e.g., 'AFIB')
      - 'noise_type'  : 'clean', 'Muscular', or 'Respiracion'
      - 'features'    : 1-D np.ndarray (the complete feature vector)

    Parameters
    ----------
    data_dir : str
        Root directory containing one subfolder per class.

    Returns
    -------
    list of dict
        One entry per valid ECG signal found.
    """
    samples = []

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: '{data_dir}'")

    class_folders = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )

    if not class_folders:
        raise ValueError(
            f"No subfolders found in '{data_dir}'. "
            "Expected one folder per class (e.g., 'Atrial 1', 'ECG Normal')."
        )

    logger.info("Found %d class folder(s): %s", len(class_folders), class_folders)

    for folder_name in class_folders:
        # Resolve the model class label from the folder name
        class_label = FOLDER_TO_CLASS.get(folder_name)
        if class_label is None:
            logger.warning(
                "Folder '%s' has no mapping in FOLDER_TO_CLASS. Skipping.", folder_name
            )
            continue

        class_path = os.path.join(data_dir, folder_name)

        # Locations to scan: root (clean) + each noise subfolder
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
                "  Folder='%s' | Class='%s' | Noise='%s' | Files: %d",
                folder_name,
                class_label,
                noise_type,
                len(json_files),
            )

            for filename in json_files:
                filepath = os.path.join(folder_path, filename)
                signal = load_json_signal(filepath)

                if signal is None:
                    continue

                features = extract_features(signal)

                samples.append(
                    {
                        "filename": filename,
                        "folder": folder_name,
                        "class_label": class_label,
                        "noise_type": noise_type,
                        "features": features,
                    }
                )

    logger.info("Total samples collected: %d", len(samples))
    return samples


# ---------------------------------------------------------------------------
# Step 5: Run inference
# ---------------------------------------------------------------------------


def run_inference(
    samples: list[dict],
    scaler,
    label_encoder,
    classifier,
) -> pd.DataFrame:
    """
    Scale feature vectors and run classifier inference on all samples.

    Parameters
    ----------
    samples : list of dict
        Output of collect_samples().
    scaler : sklearn transformer
        Fitted StandardScaler.
    label_encoder : sklearn LabelEncoder
        Fitted encoder to map numeric predictions back to class names.
    classifier : sklearn estimator
        Fitted RandomForest classifier.

    Returns
    -------
    pd.DataFrame
        One row per sample with columns: filename, folder, class_label,
        noise_type, predicted_label, and optionally prob_<class> columns.
    """
    if not samples:
        raise ValueError("No samples available for inference.")

    X = np.array([s["features"] for s in samples])
    filenames = [s["filename"] for s in samples]
    folders = [s["folder"] for s in samples]
    labels = [s["class_label"] for s in samples]
    noise_types = [s["noise_type"] for s in samples]

    logger.info("Feature matrix shape: %s", X.shape)

    # Validate feature count against scaler expectation
    if hasattr(scaler, "n_features_in_") and X.shape[1] != scaler.n_features_in_:
        raise ValueError(
            f"Feature count mismatch: extracted {X.shape[1]} features but "
            f"scaler expects {scaler.n_features_in_}. "
            "Ensure feature extraction matches the training pipeline exactly."
        )

    logger.info("Applying scaler...")
    X_scaled = scaler.transform(X)

    logger.info("Running classifier predictions...")
    y_pred_encoded = classifier.predict(X_scaled)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

    results = pd.DataFrame(
        {
            "filename": filenames,
            "folder": folders,
            "class_label": labels,
            "noise_type": noise_types,
            "predicted_label": y_pred_labels,
        }
    )

    if hasattr(classifier, "predict_proba"):
        logger.info("Extracting prediction probabilities...")
        proba = classifier.predict_proba(X_scaled)
        for i, cls_name in enumerate(label_encoder.classes_):
            results[f"prob_{cls_name}"] = proba[:, i]

    return results


# ---------------------------------------------------------------------------
# Step 6: Evaluate predictions
# ---------------------------------------------------------------------------


def evaluate(results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accuracy and a detailed classification report per noise condition.

    Groups: ALL, clean, Muscular, Respiracion.

    Parameters
    ----------
    results : pd.DataFrame
        Output of run_inference().

    Returns
    -------
    pd.DataFrame
        Summary table with noise_type, accuracy, and n_samples columns.
    """
    summary_rows = []

    def _report_group(subset: pd.DataFrame, group_name: str) -> float:
        """Log classification report for a subset and return its accuracy."""
        acc = accuracy_score(subset["class_label"], subset["predicted_label"])
        logger.info("===== Group: '%s' | Accuracy: %.4f =====", group_name, acc)
        logger.info(
            "\n%s",
            classification_report(
                subset["class_label"],
                subset["predicted_label"],
                zero_division=0,
            ),
        )
        logger.info(
            "Confusion matrix:\n%s",
            confusion_matrix(subset["class_label"], subset["predicted_label"]),
        )
        return acc

    # Overall
    acc_all = _report_group(results, "ALL")
    summary_rows.append(
        {"noise_type": "ALL", "accuracy": round(acc_all, 4), "n_samples": len(results)}
    )

    # Per noise condition
    for noise_type in ["clean"] + NOISE_SUBFOLDERS:
        subset = results[results["noise_type"] == noise_type]
        if subset.empty:
            logger.warning("No samples for noise_type='%s'. Skipping.", noise_type)
            continue
        acc = _report_group(subset, noise_type)
        summary_rows.append(
            {
                "noise_type": noise_type,
                "accuracy": round(acc, 4),
                "n_samples": len(subset),
            }
        )

    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Step 7: Save results
# ---------------------------------------------------------------------------


def save_results(results: pd.DataFrame, report: pd.DataFrame) -> None:
    """
    Write prediction results and evaluation summary to CSV files.

    Parameters
    ----------
    results : pd.DataFrame
        Full per-sample predictions from run_inference().
    report : pd.DataFrame
        Summary accuracy table from evaluate().
    """
    results.to_csv(OUTPUT_PREDICTIONS_CSV, index=False)
    logger.info("Predictions saved   : '%s'", OUTPUT_PREDICTIONS_CSV)

    report.to_csv(OUTPUT_REPORT_CSV, index=False)
    logger.info("Evaluation report   : '%s'", OUTPUT_REPORT_CSV)

    logger.info("===== Final Summary =====")
    logger.info("\n%s", report.to_string(index=False))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    # 1. Download data from Google Drive
    download_drive_folder(GDRIVE_FOLDER_ID, DATA_DIR)

    # 2. Load model artifacts
    scaler, label_encoder, classifier = load_artifacts(
        SCALER_PATH, ENCODER_PATH, CLASSIFIER_PATH
    )

    # 3. Collect signals, extract features, and map class labels
    samples = collect_samples(DATA_DIR)

    # 4. Run inference
    results = run_inference(samples, scaler, label_encoder, classifier)

    # 5. Evaluate accuracy broken down by noise condition
    report = evaluate(results)

    # 6. Save predictions and report
    save_results(results, report)


if __name__ == "__main__":
    main()
