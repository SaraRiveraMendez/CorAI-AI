import argparse
import logging
import os
import joblib
import numpy as np
import pandas as pd
import wfdb

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from afdb_dataset_loader import extract_features_for_records


def train_clustered_model(
    records,
    pn_dir,
    block_sec=60,
    method="kmeans",
    n_clusters=2,
    eps=0.5,
    min_samples=5,
    save_model=False,
):

    logging.info("Extracting features...")

    df = extract_features_for_records(
        records=records, pn_dir=pn_dir, block_sec=block_sec
    )

    if df.empty:
        raise RuntimeError("No features were extracted. Check data paths.")

    # Select numerical feature columns
    feature_cols = [
        c
        for c in df.columns
        if c
        not in ["record", "channel", "cluster", "block_idx", "t_start_sec", "t_end_sec"]
    ]

    X = df[feature_cols].values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering
    if method.lower() == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X_scaled)
    elif method.lower() == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples)  # corrected
        labels = model.fit_predict(X_scaled)
    else:
        raise ValueError("Unknown clustering method.")

    df["cluster"] = labels

    # Check that we have at least 2 classes
    unique_clusters = set(labels)
    if len(unique_clusters) < 2:
        raise RuntimeError(
            f"Only one cluster found ({unique_clusters})."
            " Cannot train a supervised model."
        )

    # Evaluate clustering with Silhouette
    if len(unique_clusters) > 1 and -1 in unique_clusters:
        valid_idx = labels != -1
        sil = silhouette_score(X_scaled[valid_idx], labels[valid_idx])
    else:
        sil = silhouette_score(X_scaled, labels)

    logging.info(f"Clustering Silhouette score: {sil:.4f}")

    # Train a classifier from clusters
    y = df["cluster"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    logging.info(f"Accuracy: {acc:.4f}")
    logging.info(f"F1-score: {f1:.4f}")

    if save_model:
        os.makedirs("models", exist_ok=True)
        joblib.dump(clf, "models/afdb_clustered_model.joblib")
        joblib.dump(scaler, "models/afdb_supervised_scaler.joblib")
        logging.info("Model and scaler saved.")

    return clf, {"accuracy": acc, "f1": f1, "silhouette": sil}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="AFDB clustering and modeling.")

    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--pn-dir", required=True)
    parser.add_argument("--block-sec", type=int, default=60)

    parser.add_argument("--method", choices=["kmeans", "dbscan"], default="kmeans")
    parser.add_argument("--n-clusters", type=int, default=2)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=5)

    parser.add_argument("--save-model", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    logging.info("Starting AFDB clustering analysis pipeline...")

    train_clustered_model(
        records=args.records,
        pn_dir=args.pn_dir,
        block_sec=args.block_sec,
        method=args.method,
        n_clusters=args.n_clusters,
        eps=args.eps,
        min_samples=args.min_samples,
        save_model=args.save_model,
    )
