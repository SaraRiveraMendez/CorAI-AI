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


def load_json_signals(data_dir: str, json_fs: float) -> pd.DataFrame:
    """
    Load all JSON ECG signals from the downloaded Google Drive folder,
    extract features, and tag each sample with its class label and noise type.

    Directory structure expected:
      data_dir/
        <FOLDER_NAME>/          <- class root (clean signals)
          *.json
          Muscular/             <- muscular noise signals
            *.json
          Respiración/          <- respiratory noise signals
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

                except Exception as exc:
                    logging.error("Failed to load '%s': %s", filepath, exc)
                    continue

                feats = extract_all_features(signal, json_fs)
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
# Metrics report (unchanged)
# ---------------------------------------------------------------------------


def save_metrics_report(metrics: dict, results_dir: str):
    """Save metrics to a timestamped JSON and TXT file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(results_dir, f"metrics_{timestamp}.json")
    try:
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logging.info(
            "Metrics saved -> %s (%d bytes)", json_path, os.path.getsize(json_path)
        )
    except Exception as e:
        logging.error("Failed to save metrics JSON: %s", e)

    txt_path = os.path.join(results_dir, f"report_{timestamp}.txt")
    try:
        with open(txt_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("AFDB SUPERVISED CLASSIFICATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write("DATASET INFO:\n")
            f.write(f"  - Total samples: {metrics.get('n_samples', 'N/A')}\n")
            f.write(f"  - Features: {metrics.get('n_features', 'N/A')}\n")
            f.write(f"  - Classes: {metrics.get('n_classes', 'N/A')}\n")
            f.write(f"  - Records: {metrics.get('records', 'N/A')}\n")
            f.write(f"  - Channel: {metrics.get('channel_idx', 'N/A')}\n\n")
            f.write("CLASS DISTRIBUTION:\n")
            for label, count in metrics.get("class_distribution", {}).items():
                f.write(f"  - {label}: {count} samples\n")
            f.write("\n")
            f.write("MODEL PERFORMANCE:\n")
            f.write(f"  - Accuracy: {metrics.get('accuracy', 'N/A'):.4f}\n")
            f.write(
                f"  - F1-score (weighted): {metrics.get('f1_weighted', 'N/A'):.4f}\n"
            )
            f.write(f"  - F1-score (macro): {metrics.get('f1_macro', 'N/A'):.4f}\n")
            f.write(
                f"  - Precision (weighted): {metrics.get('precision', 'N/A'):.4f}\n"
            )
            f.write(f"  - Recall (weighted): {metrics.get('recall', 'N/A'):.4f}\n\n")
            if "classification_report" in metrics:
                f.write("DETAILED CLASSIFICATION REPORT:\n")
                f.write(metrics["classification_report"])

            # New section: per noise-type results
            if "json_evaluation" in metrics:
                f.write("\n\nJSON SIGNAL EVALUATION BY NOISE TYPE:\n")
                f.write("-" * 40 + "\n")
                for entry in metrics["json_evaluation"]:
                    f.write(
                        f"  {entry['noise_type']:<15} | "
                        f"Accuracy: {entry['accuracy']:.4f} | "
                        f"n={entry['n_samples']}\n"
                    )

        logging.info(
            "Report saved -> %s (%d bytes)", txt_path, os.path.getsize(txt_path)
        )
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
):
    """
    Supervised classification pipeline using real AFDB rhythm labels,
    optionally extended with new JSON ECG signals for retraining.

    When data_dir is provided:
      - Training set : all AFDB blocks + clean JSON signals
      - Evaluation   : all JSON signals broken down by noise_type

    When data_dir is None:
      - Original behaviour: train and evaluate on AFDB data only.

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
        Root directory of downloaded JSON signals. If None, JSON data
        is not used.
    json_fs : float
        Sampling frequency of the JSON signals in Hz.
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
    # Use 'rhythm_label' as the unified label column name
    afdb_df["class_label"] = afdb_df["rhythm_label"]

    # ------------------------------------------------------------------
    # 2. Load JSON data (optional)
    # ------------------------------------------------------------------
    json_df = None
    if data_dir is not None:
        logging.info("Loading JSON signals from '%s'...", data_dir)
        json_df = load_json_signals(data_dir, json_fs)
        json_df["class_label"] = json_df["rhythm_label"]

    # ------------------------------------------------------------------
    # 3. Build training set
    # ------------------------------------------------------------------
    if json_df is not None:
        # Include only clean JSON signals in training to avoid contaminating
        # the model with noise it will later be evaluated on
        json_clean = json_df[json_df["noise_type"] == "clean"].copy()
        logging.info("Adding %d clean JSON samples to training set.", len(json_clean))
        train_df = pd.concat([afdb_df, json_clean], ignore_index=True)
    else:
        train_df = afdb_df.copy()

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

        # Save full predictions CSV
        pred_df = (
            json_df[["filename", "class_label", "noise_type"]]
            .copy()
            .reset_index(drop=True)
        )
        pred_df["predicted_label"] = label_encoder.inverse_transform(y_json_pred)
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_json_scaled)
            for i, cls in enumerate(label_encoder.classes_):
                pred_df[f"prob_{cls}"] = proba[:, i]

        pred_path = os.path.join(results_dir, "json_predictions.csv")
        pred_df.to_csv(pred_path, index=False)
        logging.info("JSON predictions saved: '%s'", pred_path)

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
            logging.info(
                "\n%s",
                classification_report(
                    label_encoder.inverse_transform(y_json_enc[mask]),
                    label_encoder.inverse_transform(y_json_pred[mask]),
                    zero_division=0,
                ),
            )
            json_eval_rows.append(
                {
                    "noise_type": noise_type,
                    "accuracy": round(float(acc), 4),
                    "n_samples": int(mask.sum()),
                }
            )

        # Save summary CSV
        summary_path = os.path.join(results_dir, "json_evaluation_report.csv")
        pd.DataFrame(json_eval_rows).to_csv(summary_path, index=False)
        logging.info("JSON evaluation report saved: '%s'", summary_path)

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
    # 10. Metrics report
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
        "n_samples": int(len(y)),
        "n_features": int(len(feature_cols)),
        "n_classes": int(len(label_encoder.classes_)),
        "classes": label_encoder.classes_.tolist(),
        "class_distribution": class_dist,
        "train_accuracy": float(train_acc),
        "classification_report": class_report,
        "json_evaluation": json_eval_rows,
        **test_metrics,
    }
    save_metrics_report(final_metrics, results_dir)

    # ------------------------------------------------------------------
    # 11. Save model artifacts
    # ------------------------------------------------------------------
    if save_model:
        logging.info("Saving model artifacts...")

        artifacts = {
            "afdb_rhythm_classifier.joblib": clf,
            "afdb_scaler.joblib": scaler,
            "afdb_label_encoder.joblib": label_encoder,
            # Extra artifacts for robust future inference
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
            "Root directory of JSON signals (optional). "
            "If provided, the model is retrained with AFDB + clean JSON signals "
            "and evaluated on all JSON signals by noise type."
        ),
    )
    parser.add_argument(
        "--json-fs",
        type=float,
        default=200.0,
        help="Sampling frequency of the JSON signals in Hz (default: 200)",
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
        )
        logging.info("=" * 60)
        logging.info("TRAINING COMPLETED SUCCESSFULLY")
        logging.info("Results saved in: %s", results_dir)
        logging.info("=" * 60)

    except Exception as e:
        logging.error("TRAINING FAILED: %s", e, exc_info=True)
        raise
 