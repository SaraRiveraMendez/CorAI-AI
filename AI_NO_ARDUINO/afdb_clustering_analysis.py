"""
afdb_clustering_analysis.py

Performs unsupervised clustering analysis on AFDB ECG features.

Pipeline:
1. Load preprocessed dataset using `load_afdb_dataset()`.
2. Apply PCA for dimensionality reduction.
3. Perform clustering (KMeans and/or DBSCAN).
4. Visualize clusters and report silhouette scores.

Usage:
    python afdb_clustering_analysis.py --records 04015 04043 --method kmeans
"""

import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from afdb_dataset_loader import load_afdb_dataset


# ---------------------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="AFDB clustering analysis.")
    parser.add_argument(
        "--records", nargs="+", required=True, help="List of record IDs to load."
    )
    parser.add_argument(
        "--block-sec", type=float, default=60.0, help="Block length in seconds."
    )
    parser.add_argument(
        "--method",
        choices=["kmeans", "dbscan"],
        default="kmeans",
        help="Clustering method.",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=3, help="Number of clusters (KMeans)."
    )
    parser.add_argument("--log", default="INFO", help="Logging level.")
    return parser.parse_args()


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO))

    # Load dataset (already preprocessed)
    df = load_afdb_dataset(
        records=args.records, block_sec=args.block_sec, normalize=True
    )
    logging.info(f"Loaded dataset shape: {df.shape}")

    # Select numeric feature columns for clustering
    feature_cols = [
        "mean",
        "std",
        "median",
        "min",
        "max",
        "rms",
        "zcr",
        "hr_bpm",
        "n_peaks",
    ]
    X = df[feature_cols].values

    # Reduce dimensionality for visualization (PCA → 2D)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    logging.info(
        f"Explained variance by PCA: {pca.explained_variance_ratio_.sum():.2f}"
    )

    # Clustering
    if args.method == "kmeans":
        model = KMeans(n_clusters=args.n_clusters, random_state=42)
        labels = model.fit_predict(X)
    else:
        model = DBSCAN(eps=0.7, min_samples=10)
        labels = model.fit_predict(X)

    # Evaluate silhouette score (if valid)
    if len(set(labels)) > 1 and -1 not in set(labels):
        sil_score = silhouette_score(X, labels)
        logging.info(f"Silhouette Score: {sil_score:.3f}")
    else:
        sil_score = None
        logging.info("Silhouette score not computed (single cluster or noise).")

    # Visualization
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=30, alpha=0.8
    )
    plt.title(f"AFDB Clustering ({args.method.upper()})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter, label="Cluster ID")
    plt.tight_layout()
    plt.show()

    # Summary table (optional)
    df["cluster"] = labels
    summary = df.groupby("cluster")[feature_cols].mean().round(2)
    print("\nCluster feature means:")
    print(summary)


if __name__ == "__main__":
    main()
