"""
AFDB Cluster-Based Classification Model
---------------------------------------
This script trains a supervised machine learning model (Random Forest)
using the previously clustered AFDB dataset as pseudo-labeled data.
It also generates a feature importance chart for interpretability.

Author: Sara Rivera Méndez
Date: 2025-11-08

Requirements:
    - Python 3.10+
    - pandas, numpy, scikit-learn, joblib, matplotlib, seaborn
    - Local modules: afdb_dataset_loader, afdb_clustering_analysis

Usage:
    python afdb_clustered_model.py --records 04015 04043 04048 --n-clusters 3 -> Example
"""

import os
import logging
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
)
from sklearn.decomposition import PCA

from afdb_dataset_loader import extract_all_features


def perform_clustering(df, n_clusters=3, method="kmeans", eps=0.5, min_samples=10):
    """
    Perform clustering (KMeans or DBSCAN) on AFDB dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with extracted ECG features.
    n_clusters : int
        Number of clusters for KMeans.
    method : str
        "kmeans" or "dbscan"
    eps : float
        DBSCAN epsilon parameter.
    min_samples : int
        DBSCAN minimum samples per cluster.

    Returns
    -------
    df_clustered : pd.DataFrame
        Dataset with 'cluster' column added.
    model : object
        The clustering model used (KMeans or DBSCAN).
    X_scaled : np.ndarray
        Scaled feature matrix.
    """
    X = df.select_dtypes("number").values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method.lower() == "dbscan":
        logging.info(
            f"Running DBSCAN clustering (eps={eps}, min_samples={min_samples})..."
        )
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = model.fit_predict(X_scaled)
    else:
        logging.info(f"Running KMeans clustering with {n_clusters} clusters...")
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)

    df_clustered = df.copy()
    df_clustered["cluster"] = labels

    # Try silhouette score (only valid if >1 cluster)
    if len(set(labels)) > 1 and -1 not in labels:
        score = silhouette_score(X_scaled, labels)
        logging.info(f"Silhouette Score: {score:.3f}")
    else:
        logging.warning(
            "Silhouette score not computed (single cluster or noise present)."
        )

    return df_clustered, model, X_scaled


def train_clustered_model(
    records,
    pn_dir,
    method="kmeans",
    n_clusters=3,
    eps=0.5,
    min_samples=10,
    save_model=True,
):
    """
    Loads, clusters, and trains a supervised classifier using pseudo-labels.

    Parameters
    ----------
    records : list[str]
        List of AFDB record IDs to process.
    pn_dir : str
        PhysioNet directory or local path.
    method : str
        "kmeans" or "dbscan"
    n_clusters : int
        Number of clusters for KMeans.
    eps : float
        DBSCAN epsilon parameter.
    min_samples : int
        DBSCAN minimum samples per cluster.
    save_model : bool
        Whether to save the trained RandomForest model.

    Returns
    -------
    model : RandomForestClassifier
        Trained classifier.
    metrics : dict
        Evaluation metrics.
    """

    # --- Create results directory ---
    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)
    logging.info(f"Results will be saved in: {os.path.abspath(results_dir)}")

    # --- Load and preprocess data ---
    logging.info("Loading and preprocessing AFDB dataset...")
    df = extract_all_features(records=records, pn_dir=pn_dir)
    logging.info(f"Dataset shape: {df.shape}")

    # --- Clustering step ---
    df_clustered, cluster_model, X_scaled = perform_clustering(
        df, n_clusters=n_clusters, method=method, eps=eps, min_samples=min_samples
    )

    # Skip -1 labels if using DBSCAN
    df_clustered = df_clustered[df_clustered["cluster"] != -1]
    logging.info(f"Filtered dataset shape (excluding noise): {df_clustered.shape}")

    drop_cols = [
        c
        for c in ["record", "timestamp", "label", "lead", "channel", "cluster"]
        if c in df_clustered.columns
    ]

    X = df_clustered.drop(columns=drop_cols)

    y = df_clustered["cluster"]

    scaler = StandardScaler()
    print("\n[DEBUG] Columnas de X:", X.columns.tolist())
    print("[DEBUG] Primeras filas de X:\n", X.head())

    X_scaled = scaler.fit_transform(X)

    # --- Train/Test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    # --- Train model ---
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    logging.info(f"Model accuracy: {acc:.3f}")
    logging.info(f"Model F1-score: {f1:.3f}")

    # --- PCA visualization ---
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
    pca_df["cluster"] = y.values

    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette="tab10",
        s=40,
        alpha=0.8,
        edgecolor="none",
    )
    plt.title(f"PCA Scatter Plot of Clusters ({method.upper()})")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "pca_clusters.png"), dpi=300)
    plt.close()
    logging.info("Saved PCA scatter plot as pca_clusters.png")

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False)
    plt.title("Confusion Matrix (Cluster-based Model)")
    plt.xlabel("Predicted Cluster")
    plt.ylabel("True Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300)
    plt.close()
    logging.info("Saved confusion matrix as confusion_matrix.png")

    # --- Feature Importance ---
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = feature_importances.sort_values(ascending=False).head(15)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "feature_importance.png"), dpi=300)
    plt.close()
    logging.info("Saved feature importance plot as feature_importance.png")

    # --- Save model and scaler ---
    if save_model:
        joblib.dump(model, os.path.join(results_dir, "afdb_clustered_model.joblib"))
        joblib.dump(scaler, os.path.join(results_dir, "afdb_scaler.joblib"))
        logging.info("Model and scaler saved in 'Results/'")

    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "report": classification_report(y_test, y_pred, output_dict=True),
    }

    return model, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AFDB Clustering and Model Training")

    parser.add_argument(
        "--records",
        nargs="+",
        required=True,
        help="List of AFDB record IDs to process (e.g., 04015 04043 04048)",
    )
    parser.add_argument(
        "--pn-dir",
        default="afdb/1.0.0",
        help="PhysioNet directory or local path",
    )
    parser.add_argument(
        "--method",
        choices=["kmeans", "dbscan"],
        default="kmeans",
        help="Clustering method to use",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=3, help="Number of clusters for KMeans"
    )
    parser.add_argument(
        "--eps", type=float, default=0.5, help="DBSCAN epsilon parameter"
    )
    parser.add_argument(
        "--min-samples", type=int, default=10, help="DBSCAN minimum samples per cluster"
    )
    parser.add_argument(
        "--block-sec",
        type=float,
        default=30.0,
        help="Signal block size in seconds (passed to preprocessing)",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save trained Random Forest model and scaler",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logging.info("tarting AFDB clustering analysis pipeline...")
    model, metrics = train_clustered_model(
        records=args.records,
        pn_dir=args.pn_dir,
        method=args.method,
        n_clusters=args.n_clusters,
        eps=args.eps,
        min_samples=args.min_samples,
        save_model=args.save_model,
    )

    logging.info("Pipeline completed successfully.")
    logging.info(f"Accuracy: {metrics['accuracy']:.3f}")
    logging.info(f"F1-score: {metrics['f1_score']:.3f}")
