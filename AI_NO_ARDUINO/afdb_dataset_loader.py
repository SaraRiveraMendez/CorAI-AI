import wfdb
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm


def load_afdb_dataset(records, pn_dir, block_sec=30.0, fs_target=250):
    """
    Load and preprocess signals from the AFDB dataset directly from PhysioNet.

    Parameters
    ----------
    records : list[str]
        List of AFDB record IDs (e.g., ['04015', '04043']).
    pn_dir : str
        PhysioNet base URL or local path (e.g., "https://physionet.org/content/afdb/1.0.0/").
    block_sec : float, optional
        Segment length (in seconds) for feature extraction.
    fs_target : int, optional
        Target sampling frequency (default 250 Hz).

    Returns
    -------
    df_all : pd.DataFrame
        Processed dataset with extracted features for each signal block.
    """

    logging.info(" Starting AFDB dataset loading...")
    processed_records = []
    all_dfs = []

    # Prefer commonly used ECG leads (MLII, ECG2, etc.)
    preferred_channels = ["MLII", "ML2", "ECG2", "II", "V1"]

    for rec in tqdm(records, desc="Processing AFDB records", ncols=100):
        try:
            # wfdb handles PhysioNet URLs and local paths automatically
            record = wfdb.rdrecord(rec, pn_dir=pn_dir)

            logging.info(f"Record {rec} available leads: {record.sig_name}")

            # --- Channel detection ---
            channel_idx = [
                i
                for i, name in enumerate(record.sig_name)
                if name.upper() in [c.upper() for c in preferred_channels]
            ]

            if not channel_idx:
                logging.warning(f"Skipping {rec}: no preferred ECG lead found.")
                continue

            ch_idx = channel_idx[0]
            ch_name = record.sig_name[ch_idx]
            logging.info(f"Using channel {ch_name} for record {rec}")

            # --- Get signal and sampling rate ---
            fs = record.fs
            signal = record.p_signal[:, ch_idx]

            # Normalize signal
            signal = (signal - np.mean(signal)) / np.std(signal)

            # --- Split into blocks ---
            block_size = int(block_sec * fs)
            n_blocks = len(signal) // block_size
            if n_blocks == 0:
                logging.warning(f"Skipping {rec}: too short for {block_sec}s blocks.")
                continue

            # --- Extract simple statistical features ---
            features = []
            for i in range(n_blocks):
                block = signal[i * block_size : (i + 1) * block_size]
                feat = {
                    "record": rec,
                    "channel": ch_name,
                    "block_idx": i,
                    "t_start": i * block_sec,
                    "t_end": (i + 1) * block_sec,
                    "n_samples": len(block),
                    "fs": fs,
                    "mean": np.mean(block),
                    "std": np.std(block),
                    "median": np.median(block),
                    "min": np.min(block),
                    "max": np.max(block),
                    "rms": np.sqrt(np.mean(block**2)),
                    "zcr": ((block[:-1] * block[1:]) < 0).sum() / len(block),
                }
                features.append(feat)

            df_rec = pd.DataFrame(features)
            all_dfs.append(df_rec)
            processed_records.append(rec)

        except Exception as e:
            logging.error(f"Error processing record {rec}: {str(e)}")
            continue

    # --- Final dataset ---
    if len(all_dfs) == 0:
        raise RuntimeError("No valid records were processed.")

    df_all = pd.concat(all_dfs, ignore_index=True)
    logging.info(f" Loaded and processed {len(processed_records)} valid records.")
    logging.info(f" Final dataset shape: {df_all.shape}")

    return df_all
