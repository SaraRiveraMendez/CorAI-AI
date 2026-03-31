"""
afdb_augment.py
---------------
Data augmentation pipeline for JSON ECG signals.

This script performs two operations on every clean JSON signal found in the
data directory:

  1. RESAMPLING (200 Hz -> 250 Hz)
     Resamples v_raw to match the AFDB training frequency so that features
     extracted from JSON signals and AFDB signals live in the same space.
     Resampled signals are saved alongside the originals with the suffix
     '_250hz' in a parallel directory tree.

  2. SYNTHETIC NOISE AUGMENTATION (on the resampled 250 Hz signals)
     Generates two noisy variants per clean signal:
       - Muscular  : broadband EMG-like noise (20-500 Hz bandpass, Gaussian)
       - Respiracion: low-frequency baseline wander (0.1-0.5 Hz sinusoid)
     Noisy variants are saved in Muscular/ and Respiracion/ subfolders
     mirroring the original structure, so the training pipeline can load
     them transparently.

Output directory structure (mirrors input):
  data_augmented/
    ATRIAL 1/
      *_250hz.json          <- resampled clean signals
      Muscular/
        *_250hz_muscular.json
      Respiracion/
        *_250hz_respiracion.json
    ATRIAL 2/
      ...
    ECG NORMAL/
      ...

Usage:
    python afdb_augment.py --data-dir data --output-dir data_augmented
    python afdb_augment.py --data-dir data --output-dir data_augmented --src-fs 200 --dst-fs 250

Requirements:
    pip install numpy scipy
"""

import os
import json
import argparse
import logging
import numpy as np
from scipy.signal import resample_poly, butter, filtfilt
from math import gcd

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

SRC_FS = 200  # Sampling frequency of the original JSON signals (Hz)
DST_FS = 250  # Target sampling frequency matching AFDB (Hz)

# Muscular noise: broadband Gaussian noise bandpass-filtered to EMG range
MUSCULAR_NOISE_LOWCUT = 20.0  # Hz
MUSCULAR_NOISE_HIGHCUT = 150.0  # Hz  (capped well below Nyquist at 125 Hz for 250 Hz)
MUSCULAR_NOISE_SNR_DB = 7.0  # Signal-to-noise ratio in dB (lower = more noise)
# Set to 7 dB for aggressive augmentation that forces
# the model to be robust against strong EMG artifacts.

# Respiratory noise: sinusoidal baseline wander
RESP_FREQ_HZ = 0.25  # Respiratory frequency (Hz), typical at rest
RESP_AMPLITUDE_FRAC = 0.15  # Amplitude as fraction of signal peak-to-peak

# Noise subfolders expected/created inside each class folder
NOISE_SUBFOLDERS = ["Muscular", "Respiracion"]

# Folder names that correspond to known classes (case-insensitive match)
KNOWN_CLASS_FOLDERS = {"ATRIAL 1", "ATRIAL 2", "ECG NORMAL"}

# JSON keys
JSON_ECG_KEY = "ecg"
JSON_VOLTAGE_KEY = "v_raw"
JSON_TS_KEY = "t_s"
JSON_TH_KEY = "t_h"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def resample_signal(signal: np.ndarray, src_fs: int, dst_fs: int) -> np.ndarray:
    """
    Resample a 1-D signal from src_fs to dst_fs using polyphase filtering.

    Uses scipy.signal.resample_poly which applies an anti-aliasing FIR filter
    before decimation, avoiding aliasing artifacts.

    Parameters
    ----------
    signal : np.ndarray
        1-D array of signal samples at src_fs Hz.
    src_fs : int
        Original sampling frequency in Hz.
    dst_fs : int
        Target sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Resampled signal at dst_fs Hz.
    """
    if src_fs == dst_fs:
        return signal.copy()

    common = gcd(dst_fs, src_fs)
    up = dst_fs // common
    down = src_fs // common
    return resample_poly(signal, up, down).astype(np.float32)


def resample_timestamps(n_samples: int, fs: float) -> list[float]:
    """
    Generate evenly spaced time stamps for a resampled signal.

    Parameters
    ----------
    n_samples : int
        Number of samples in the resampled signal.
    fs : float
        Sampling frequency of the resampled signal in Hz.

    Returns
    -------
    list of float
        Time values in seconds, starting at 0 and spaced by 1/fs.
    """
    return [round(i / fs, 6) for i in range(n_samples)]


# ---------------------------------------------------------------------------
# Noise generation
# ---------------------------------------------------------------------------


def add_muscular_noise(
    signal: np.ndarray,
    fs: float,
    lowcut: float = MUSCULAR_NOISE_LOWCUT,
    highcut: float = MUSCULAR_NOISE_HIGHCUT,
    snr_db: float = MUSCULAR_NOISE_SNR_DB,
    seed: int = 0,
) -> np.ndarray:
    """
    Add synthetic EMG-like muscular noise to a signal.

    The noise is generated as Gaussian white noise, bandpass-filtered to the
    EMG frequency range (20-150 Hz), and scaled to achieve the specified
    signal-to-noise ratio.

    Parameters
    ----------
    signal : np.ndarray
        1-D clean ECG signal.
    fs : float
        Sampling frequency in Hz.
    lowcut : float
        Lower cutoff frequency of the EMG band in Hz.
    highcut : float
        Upper cutoff frequency of the EMG band in Hz (must be < fs/2).
    snr_db : float
        Desired signal-to-noise ratio in decibels.
        Lower values produce more noise. Typical range: 10-25 dB.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy signal of the same shape as input.
    """
    rng = np.random.default_rng(seed)
    nyq = 0.5 * fs

    # Clamp highcut safely below Nyquist
    highcut = min(highcut, nyq * 0.95)

    # Generate and bandpass-filter white Gaussian noise
    raw_noise = rng.standard_normal(len(signal)).astype(np.float32)
    b, a = butter(4, [lowcut / nyq, highcut / nyq], btype="band")
    noise = filtfilt(b, a, raw_noise).astype(np.float32)

    # Scale noise to achieve the target SNR
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)

    if noise_power > 0 and signal_power > 0:
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        noise *= np.sqrt(target_noise_power / noise_power)

    return (signal + noise).astype(np.float32)


def add_respiratory_noise(
    signal: np.ndarray,
    fs: float,
    freq_hz: float = RESP_FREQ_HZ,
    amplitude_frac: float = RESP_AMPLITUDE_FRAC,
    seed: int = 0,
) -> np.ndarray:
    """
    Add synthetic respiratory baseline wander to a signal.

    The wander is modeled as a low-frequency sinusoid with a small random
    phase offset, representing the mechanical effect of breathing on ECG
    electrode contact.

    Parameters
    ----------
    signal : np.ndarray
        1-D clean ECG signal.
    fs : float
        Sampling frequency in Hz.
    freq_hz : float
        Respiratory frequency in Hz (typically 0.15-0.4 Hz at rest).
    amplitude_frac : float
        Wander amplitude as a fraction of the signal peak-to-peak range.
    seed : int
        Random seed for the phase offset.

    Returns
    -------
    np.ndarray
        Signal with baseline wander added.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(len(signal)) / fs
    phase = rng.uniform(0, 2 * np.pi)

    # Amplitude relative to the peak-to-peak range of the clean signal
    peak_to_peak = float(np.max(signal) - np.min(signal))
    amplitude = amplitude_frac * peak_to_peak if peak_to_peak > 0 else amplitude_frac

    wander = (amplitude * np.sin(2 * np.pi * freq_hz * t + phase)).astype(np.float32)
    return (signal + wander).astype(np.float32)


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------


def load_json_signal(filepath: str) -> tuple[np.ndarray | None, dict | None]:
    """
    Load a clean JSON ECG signal.

    Parameters
    ----------
    filepath : str
        Path to the .json file.

    Returns
    -------
    tuple
        (signal_array, raw_data_dict) or (None, None) on failure.
        signal_array is a 1-D float32 array of v_raw values.
        raw_data_dict is the full parsed JSON object (for metadata reuse).
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        ecg_array = data.get(JSON_ECG_KEY, [])
        if not ecg_array:
            logger.warning("Empty ecg array in '%s'. Skipping.", filepath)
            return None, None

        signal = np.array([s[JSON_VOLTAGE_KEY] for s in ecg_array], dtype=np.float32)
        return signal, data

    except Exception as exc:
        logger.error("Failed to load '%s': %s", filepath, exc)
        return None, None


def build_json_payload(
    signal: np.ndarray,
    fs: float,
    original_data: dict,
    extra_meta: dict | None = None,
) -> dict:
    """
    Build a JSON-serialisable dict for a processed signal.

    Preserves the original timestamp_inicio and any other top-level metadata
    from the source file. Rebuilds the ecg array with recomputed t_s values.

    Parameters
    ----------
    signal : np.ndarray
        1-D float32 array of processed v_raw samples.
    fs : float
        Sampling frequency of the processed signal in Hz.
    original_data : dict
        Original JSON data dict (used for metadata like timestamp_inicio).
    extra_meta : dict or None
        Additional top-level metadata keys to include (e.g. augmentation info).

    Returns
    -------
    dict
        JSON-serialisable payload.
    """
    timestamps = resample_timestamps(len(signal), fs)

    ecg_array = [
        {
            JSON_VOLTAGE_KEY: float(v),
            "v_": 0.0,  # filtered voltage not recomputed; placeholder
            JSON_TS_KEY: t,
            JSON_TH_KEY: "",  # wall-clock time not meaningful after resampling
        }
        for v, t in zip(signal, timestamps)
    ]

    payload = {
        "timestamp_inicio": original_data.get("timestamp_inicio", ""),
        "fs_hz": float(fs),
        "n_samples": int(len(signal)),
    }

    if extra_meta:
        payload.update(extra_meta)

    payload[JSON_ECG_KEY] = ecg_array
    return payload


def save_json(payload: dict, filepath: str) -> None:
    """
    Write a JSON payload to disk, creating parent directories as needed.

    Parameters
    ----------
    payload : dict
        Data to serialise.
    filepath : str
        Destination file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Main augmentation logic
# ---------------------------------------------------------------------------


def augment_folder(
    data_dir: str,
    output_dir: str,
    src_fs: int,
    dst_fs: int,
) -> dict:
    """
    Walk data_dir, resample all clean signals, and generate noisy variants.

    Only processes files in the root of each class folder (clean signals).
    Files already inside Muscular/ or Respiracion/ subfolders are ignored
    because they are real noisy recordings, not clean signals to augment.

    Parameters
    ----------
    data_dir : str
        Input directory containing one subfolder per class.
    output_dir : str
        Output directory. Mirrors the structure of data_dir.
    src_fs : int
        Source sampling frequency in Hz.
    dst_fs : int
        Target sampling frequency in Hz.

    Returns
    -------
    dict
        Summary counts: {class_folder: {clean, muscular, respiracion}}.
    """
    summary = {}

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: '{data_dir}'")

    class_folders = sorted(
        d
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and d.strip().upper() in KNOWN_CLASS_FOLDERS
    )

    if not class_folders:
        raise ValueError(
            f"No known class folders found in '{data_dir}'. "
            f"Expected one of: {KNOWN_CLASS_FOLDERS}"
        )

    logger.info("Found %d class folder(s): %s", len(class_folders), class_folders)
    logger.info(
        "Resampling: %d Hz -> %d Hz | Noise types: Muscular, Respiracion",
        src_fs,
        dst_fs,
    )

    for folder_name in class_folders:
        class_input_path = os.path.join(data_dir, folder_name)
        class_output_path = os.path.join(output_dir, folder_name)

        # Only process JSON files at the root (clean signals)
        json_files = sorted(
            f for f in os.listdir(class_input_path) if f.endswith(".json")
        )

        logger.info(
            "Class '%s': %d clean signal(s) found.", folder_name, len(json_files)
        )

        counts = {"clean": 0, "Muscular": 0, "Respiracion": 0}

        for filename in json_files:
            src_path = os.path.join(class_input_path, filename)
            signal, original_data = load_json_signal(src_path)

            if signal is None:
                continue

            stem = os.path.splitext(filename)[0]

            # ----------------------------------------------------------
            # 1. Resample to dst_fs
            # ----------------------------------------------------------
            resampled = resample_signal(signal, src_fs, dst_fs)

            resampled_filename = f"{stem}_{dst_fs}hz.json"
            resampled_path = os.path.join(class_output_path, resampled_filename)

            save_json(
                build_json_payload(
                    resampled,
                    dst_fs,
                    original_data,
                    extra_meta={"augmentation": "resampled", "src_fs": src_fs},
                ),
                resampled_path,
            )
            counts["clean"] += 1
            logger.debug("Resampled -> '%s'", resampled_path)

            # ----------------------------------------------------------
            # 2. Muscular noise variant (on resampled signal)
            # ----------------------------------------------------------
            muscular_signal = add_muscular_noise(
                resampled, dst_fs, seed=hash(filename) % 2**31
            )
            muscular_filename = f"{stem}_{dst_fs}hz_muscular.json"
            muscular_path = os.path.join(
                class_output_path, "Muscular", muscular_filename
            )

            save_json(
                build_json_payload(
                    muscular_signal,
                    dst_fs,
                    original_data,
                    extra_meta={
                        "augmentation": "muscular_noise",
                        "src_fs": src_fs,
                        "noise_snr_db": MUSCULAR_NOISE_SNR_DB,
                        "noise_band_hz": [
                            MUSCULAR_NOISE_LOWCUT,
                            MUSCULAR_NOISE_HIGHCUT,
                        ],
                    },
                ),
                muscular_path,
            )
            counts["Muscular"] += 1
            logger.debug("Muscular noise -> '%s'", muscular_path)

            # ----------------------------------------------------------
            # 3. Respiratory noise variant (on resampled signal)
            # ----------------------------------------------------------
            resp_signal = add_respiratory_noise(
                resampled, dst_fs, seed=hash(filename) % 2**31
            )
            resp_filename = f"{stem}_{dst_fs}hz_respiracion.json"
            resp_path = os.path.join(class_output_path, "Respiracion", resp_filename)

            save_json(
                build_json_payload(
                    resp_signal,
                    dst_fs,
                    original_data,
                    extra_meta={
                        "augmentation": "respiratory_noise",
                        "src_fs": src_fs,
                        "resp_freq_hz": RESP_FREQ_HZ,
                        "resp_amplitude_frac": RESP_AMPLITUDE_FRAC,
                    },
                ),
                resp_path,
            )
            counts["Respiracion"] += 1
            logger.debug("Respiratory noise -> '%s'", resp_path)

        summary[folder_name] = counts
        logger.info(
            "  '%s' done: %d resampled, %d muscular, %d respiracion.",
            folder_name,
            counts["clean"],
            counts["Muscular"],
            counts["Respiracion"],
        )

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "ECG signal augmentation: resampling (src_fs -> dst_fs) "
            "and synthetic noise generation (muscular + respiratory)."
        )
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root directory containing clean JSON signals (one subfolder per class).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for augmented signals (mirrors input structure).",
    )
    parser.add_argument(
        "--src-fs",
        type=int,
        default=SRC_FS,
        help=f"Source sampling frequency in Hz (default: {SRC_FS}).",
    )
    parser.add_argument(
        "--dst-fs",
        type=int,
        default=DST_FS,
        help=f"Target sampling frequency in Hz (default: {DST_FS}).",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("AFDB DATA AUGMENTATION")
    logger.info("Input  : %s", os.path.abspath(args.data_dir))
    logger.info("Output : %s", os.path.abspath(args.output_dir))
    logger.info("Resample: %d Hz -> %d Hz", args.src_fs, args.dst_fs)
    logger.info("=" * 60)

    summary = augment_folder(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        src_fs=args.src_fs,
        dst_fs=args.dst_fs,
    )

    logger.info("=" * 60)
    logger.info("AUGMENTATION COMPLETE")
    total_clean = sum(v["clean"] for v in summary.values())
    total_muscular = sum(v["Muscular"] for v in summary.values())
    total_resp = sum(v["Respiracion"] for v in summary.values())
    logger.info(
        "Total: %d resampled | %d muscular | %d respiracion",
        total_clean,
        total_muscular,
        total_resp,
    )
    logger.info("Output saved to: '%s'", os.path.abspath(args.output_dir))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
