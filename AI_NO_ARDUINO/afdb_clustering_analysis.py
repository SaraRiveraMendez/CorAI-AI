"""
afdb_supervised_classification.py

CLASIFICACIÓN SUPERVISADA DIRECTA usando las etiquetas reales de AFDB.
- AFIB: Fibrilación Atrial
- AFL: Flutter Atrial  
- J: Ritmo de Unión AV
- N: Otros ritmos (principalmente ritmo sinusal normal)

Usage example:
    python afdb_supervised_classification.py \
        --records 04015 04043 04048 \
        --pn-dir afdb \
        --block-sec 60 \
        --channel-idx 1 \
        --save-model
"""

import os
import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Backend sin GUI
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
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


def load_afdb_annotations(record_name, pn_dir):
    """
    Carga las anotaciones de ritmo (.atr) de AFDB.

    Returns:
        - sample_indices: array de índices de muestra donde cambia el ritmo
        - symbols: array de símbolos de anotación
        - aux_note: notas auxiliares (contiene el tipo de ritmo)
    """
    try:
        if pn_dir == "afdb":
            annotation = wfdb.rdann(record_name, "atr", pn_dir="afdb")
        else:
            ann_path = os.path.join(pn_dir, record_name)
            annotation = wfdb.rdann(ann_path, "atr")

        return annotation.sample, annotation.symbol, annotation.aux_note
    except Exception as e:
        logging.error(f"Could not load annotations for {record_name}: {e}")
        return None, None, None


def get_rhythm_label_for_block(start_sample, end_sample, ann_samples, aux_notes):
    """
    Determina la etiqueta de ritmo predominante en un bloque.

    AFDB usa anotaciones de ritmo en aux_note:
    - (AFIB : Fibrilación Atrial
    - (AFL  : Flutter Atrial
    - (J    : Ritmo de Unión AV
    - (N    : Otros ritmos (normal)
    """
    if ann_samples is None or len(ann_samples) == 0:
        return None

    # Encontrar todas las anotaciones dentro del bloque
    block_annotations = []
    for i, sample in enumerate(ann_samples):
        if start_sample <= sample < end_sample:
            aux = aux_notes[i] if i < len(aux_notes) else ""
            # Extraer tipo de ritmo de aux_note
            if "(AFIB" in aux:
                block_annotations.append("AFIB")
            elif "(AFL" in aux:
                block_annotations.append("AFL")
            elif "(J" in aux:
                block_annotations.append("J")
            elif "(N" in aux:
                block_annotations.append("N")

    # Si no hay anotaciones en el bloque, buscar la anotación previa más cercana
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

    # Retornar la etiqueta más frecuente en el bloque
    counter = Counter(block_annotations)
    return counter.most_common(1)[0][0]


def extract_features_with_labels(records, pn_dir, block_sec=60, channel_idx=1):
    """
    Extrae features Y etiquetas reales de ritmo de AFDB.

    Returns:
        DataFrame con features + columna 'rhythm_label'
    """
    features_list = []

    for rec in records:
        logging.info(f"Loading record {rec}")

        # Cargar señal (modo streaming por defecto)
        try:
            record = wfdb.rdrecord(rec, pn_dir=pn_dir)
        except Exception as e:
            logging.error(f"Could not load record {rec}: {e}")
            continue

        # Cargar anotaciones
        ann_samples, ann_symbols, aux_notes = load_afdb_annotations(rec, pn_dir)
        if ann_samples is None:
            logging.warning(f"No annotations for {rec}, skipping")
            continue

        fs = record.fs
        n_samples = record.sig_len
        signals = record.p_signal

        if channel_idx >= signals.shape[1]:
            logging.warning(f"Channel {channel_idx} not available in {rec}")
            continue

        block_size = int(block_sec * fs)
        n_blocks = n_samples // block_size

        logging.info(f"{rec}: {n_blocks} blocks, fs={fs}")

        signal = signals[:, channel_idx]

        for blk in range(n_blocks):
            start = blk * block_size
            end = start + block_size
            block = signal[start:end]

            # Extraer features
            feats = extract_all_features(block, fs)

            # Obtener etiqueta de ritmo
            rhythm = get_rhythm_label_for_block(start, end, ann_samples, aux_notes)

            if rhythm is None:
                continue  # Saltar bloques sin etiqueta

            # Metadata
            feats["record"] = rec
            feats["channel"] = record.sig_name[channel_idx]
            feats["channel_idx"] = channel_idx
            feats["block_idx"] = blk
            feats["t_start_sec"] = blk * block_sec
            feats["t_end_sec"] = (blk + 1) * block_sec
            feats["rhythm_label"] = rhythm

            features_list.append(feats)

    if len(features_list) == 0:
        raise RuntimeError("No valid labeled blocks extracted")

    df = pd.DataFrame(features_list)
    logging.info(f"\nTotal labeled blocks: {len(df)}")
    logging.info(f"Label distribution:\n{df['rhythm_label'].value_counts()}")

    return df


def save_metrics_report(metrics: dict, results_dir: str):
    """Guarda métricas en JSON y TXT."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = os.path.join(results_dir, f"metrics_{timestamp}.json")
    try:
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logging.info(
            f"✓ Metrics saved -> {json_path} ({os.path.getsize(json_path)} bytes)"
        )
    except Exception as e:
        logging.error(f"✗ Failed to save metrics JSON: {e}")

    # TXT
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

        logging.info(
            f"✓ Report saved -> {txt_path} ({os.path.getsize(txt_path)} bytes)"
        )
    except Exception as e:
        logging.error(f"✗ Failed to save report TXT: {e}")

    return json_path, txt_path


def train_supervised_model(
    records,
    pn_dir,
    block_sec=60,
    channel_idx=1,
    save_model=True,
    results_dir="results_supervised",
):
    """
    Pipeline de clasificación supervisada usando etiquetas reales de AFDB.
    """
    # Crear directorio con ruta absoluta
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    logging.info(f"Results directory: {results_dir}")
    logging.info(f"Processing channel: {channel_idx}")

    # Verificar que el directorio existe y es escribible
    test_file = os.path.join(results_dir, "_test_write.tmp")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        file_size = os.path.getsize(test_file)
        os.remove(test_file)
        logging.info(f"Directory is writable (test file: {file_size} bytes)")
    except Exception as e:
        logging.error(f"Directory is NOT writable: {e}")
        raise RuntimeError(f"Cannot write to {results_dir}: {e}")

    # Extraer features CON ETIQUETAS REALES
    logging.info("Extracting features with rhythm labels...")
    df = extract_features_with_labels(records, pn_dir, block_sec, channel_idx)

    # Preparar datos
    metadata_cols = [
        "record",
        "channel",
        "channel_idx",
        "block_idx",
        "t_start_sec",
        "t_end_sec",
        "rhythm_label",
    ]

    X = df.drop(columns=metadata_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number])

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["rhythm_label"].values)

    logging.info(f"Features shape: {X.shape}")
    logging.info(f"Classes: {label_encoder.classes_}")
    logging.info(f"Class distribution in y: {np.bincount(y)}")

    # Imputar NaNs
    X_filled = X.fillna(X.median())

    # Estandarizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    logging.info(f"Train set: {len(y_train)} samples")
    logging.info(f"Test set: {len(y_test)} samples")

    # Entrenar RandomForest
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

    # Evaluación
    y_pred = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)

    # Métricas con zero_division para evitar warnings
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

    train_acc = accuracy_score(y_train, y_pred_train)

    # Classification report
    class_report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, digits=4, zero_division=0
    )

    logging.info(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")
    logging.info(f"Train Accuracy: {train_acc:.4f}")
    logging.info(f"\n{class_report}")

    # Identificar clases no predichas
    unique_pred = np.unique(y_pred)
    missing_classes = set(range(len(label_encoder.classes_))) - set(unique_pred)
    if missing_classes:
        logging.warning(
            f"Classes NOT predicted: {[label_encoder.classes_[i] for i in missing_classes]}"
        )

    # === VISUALIZACIONES ===

    logging.info("\n" + "=" * 60)
    logging.info("GENERATING PLOTS")
    logging.info("=" * 60)

    # 1) Confusion Matrix
    logging.info("Creating confusion matrix...")
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

    if os.path.exists(cm_path):
        logging.info(
            f"✓ Confusion matrix: {cm_path} ({os.path.getsize(cm_path)} bytes)"
        )
    else:
        logging.error(f"✗ Confusion matrix NOT saved!")

    # 2) Feature Importance
    logging.info("Creating feature importance plot...")
    feat_imp = pd.Series(clf.feature_importances_, index=X.columns)
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

    if os.path.exists(fi_path):
        logging.info(
            f"✓ Feature importance: {fi_path} ({os.path.getsize(fi_path)} bytes)"
        )
    else:
        logging.error(f"✗ Feature importance NOT saved!")

    # 3) PCA Visualization
    logging.info("Creating PCA visualization...")
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

    if os.path.exists(pca_path):
        logging.info(
            f"✓ PCA visualization: {pca_path} ({os.path.getsize(pca_path)} bytes)"
        )
    else:
        logging.error(f"✗ PCA visualization NOT saved!")

    # 4) Class Distribution
    logging.info("Creating class distribution plot...")
    class_counts = pd.Series(y).value_counts()
    class_names = [label_encoder.classes_[i] for i in class_counts.index]

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.bar(class_names, class_counts.values, color="coral", edgecolor="black")
    ax4.set_title("Class Distribution in Dataset", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Rhythm Type", fontsize=12)
    ax4.set_ylabel("Number of Blocks", fontsize=12)
    ax4.tick_params(axis="x", rotation=45)
    ax4.grid(axis="y", alpha=0.3)

    cd_path = os.path.join(results_dir, "class_distribution.png")
    fig4.tight_layout()
    fig4.savefig(cd_path, dpi=300, bbox_inches="tight")
    plt.close(fig4)

    if os.path.exists(cd_path):
        logging.info(
            f"✓ Class distribution: {cd_path} ({os.path.getsize(cd_path)} bytes)"
        )
    else:
        logging.error(f"✗ Class distribution NOT saved!")

    logging.info("=" * 60 + "\n")

    # Métricas completas
    class_dist = dict(
        zip(
            label_encoder.classes_,
            [int(sum(y == i)) for i in range(len(label_encoder.classes_))],
        )
    )

    final_metrics = {
        "timestamp": datetime.now().isoformat(),
        "records": records,
        "channel_idx": channel_idx,
        "block_sec": float(block_sec),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(label_encoder.classes_)),
        "classes": label_encoder.classes_.tolist(),
        "class_distribution": class_dist,
        "train_accuracy": float(train_acc),
        "classification_report": class_report,
        **test_metrics,
    }

    # Guardar métricas
    save_metrics_report(final_metrics, results_dir)

    # === GUARDAR MODELOS ===
    if save_model:
        logging.info("\n" + "=" * 60)
        logging.info("SAVING MODELS")
        logging.info("=" * 60)

        model_path = os.path.join(results_dir, "afdb_rhythm_classifier.joblib")
        scaler_path = os.path.join(results_dir, "afdb_scaler.joblib")
        encoder_path = os.path.join(results_dir, "afdb_label_encoder.joblib")

        try:
            joblib.dump(clf, model_path)
            if os.path.exists(model_path):
                logging.info(
                    f"✓ Classifier: {model_path} ({os.path.getsize(model_path)} bytes)"
                )
            else:
                logging.error(f"✗ Classifier NOT saved!")
        except Exception as e:
            logging.error(f"✗ Failed to save classifier: {e}")

        try:
            joblib.dump(scaler, scaler_path)
            if os.path.exists(scaler_path):
                logging.info(
                    f"✓ Scaler: {scaler_path} ({os.path.getsize(scaler_path)} bytes)"
                )
            else:
                logging.error(f"✗ Scaler NOT saved!")
        except Exception as e:
            logging.error(f"✗ Failed to save scaler: {e}")

        try:
            joblib.dump(label_encoder, encoder_path)
            if os.path.exists(encoder_path):
                logging.info(
                    f"✓ Label Encoder: {encoder_path} ({os.path.getsize(encoder_path)} bytes)"
                )
            else:
                logging.error(f"✗ Label Encoder NOT saved!")
        except Exception as e:
            logging.error(f"✗ Failed to save label encoder: {e}")

        logging.info("=" * 60 + "\n")

    return clf, final_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AFDB Supervised Classification (usando etiquetas reales)"
    )
    parser.add_argument("--records", nargs="+", required=True, help="AFDB record IDs")
    parser.add_argument(
        "--pn-dir", default="afdb", help="AFDB directory or 'afdb' for streaming"
    )
    parser.add_argument("--block-sec", type=float, default=60, help="Block duration")
    parser.add_argument(
        "--channel-idx", type=int, default=1, help="Channel index (0 or 1)"
    )
    parser.add_argument("--save-model", action="store_true", help="Save trained model")
    parser.add_argument(
        "--results-dir", default="results_supervised", help="Output dir"
    )
    args = parser.parse_args()

    # Crear directorio de resultados
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
    logging.info("Using REAL rhythm labels from .atr annotations")
    logging.info("=" * 60)
    logging.info(f"Python: {sys.version}")
    logging.info(f"Working directory: {os.getcwd()}")
    logging.info(f"Results directory: {results_dir}")
    logging.info(f"Directory writable: {os.access(results_dir, os.W_OK)}")

    try:
        model, metrics = train_supervised_model(
            records=args.records,
            pn_dir=args.pn_dir,
            block_sec=args.block_sec,
            channel_idx=args.channel_idx,
            save_model=args.save_model,
            results_dir=results_dir,
        )

        logging.info("=" * 60)
        logging.info("✓ TRAINING COMPLETED SUCCESSFULLY!")
        logging.info(f"Results saved in: {results_dir}")
        logging.info("=" * 60)

    except Exception as e:
        logging.error(f"✗ TRAINING FAILED: {e}", exc_info=True)
        raise
