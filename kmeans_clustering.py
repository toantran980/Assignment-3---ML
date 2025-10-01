"""
K-Means Clustering on Iris Dataset

This script:
- Performs unsupervised clustering on the Iris dataset using K-Means (K=3)
- Computes RMSE per cluster and global RMSE
- Compares clustering results to the true class labels using ARI and a confusion matrix
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, confusion_matrix

def setup_kmeans(X: np.ndarray) -> KMeans:
    """Fit KMeans model to the data with K=3 clusters."""
    # TODO: Implement KMeans fitting with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    return kmeans

def calc_kmeans_rmse(model: KMeans, X_size: int) -> float:
    """Compute global RMSE based on model inertia."""
    # TODO: Implement global RMSE using model.inertia_
    return np.sqrt(model.inertia_ / X_size)

def rmse_per_cluster(X, labels, centroids):
    """Compute RMSE for each cluster based on assigned points and centroids."""
    # TODO: Compute RMSE for each cluster
    rmses = []
    for i in range(centroids.shape[0]):
        cluster_points = X[labels == i]
        if len(cluster_points) == 0:
            rmses.append(0.0)
            continue
        mse = np.mean(np.linalg.norm(cluster_points - centroids[i], axis=1) ** 2)
        rmses.append(np.sqrt(mse))
    return rmses

def main():
    # Load data (X: features, y_true: true labels)
    X, y_true = load_iris(return_X_y=True)

    # Run KMeans clustering
    kmeans = setup_kmeans(X)

    # Per-cluster RMSE
    rmses = rmse_per_cluster(X, kmeans.labels_, kmeans.cluster_centers_)
    print("Per-cluster RMSE:")
    for i, rmse in enumerate(rmses):
        print(f"  Cluster {i}: {rmse:.4f}")

    # Global RMSE (inertia-based)
    rmse_global = calc_kmeans_rmse(kmeans, len(X))
    print(f"Global RMSE: {rmse_global:.4f}")

    # Cluster quality validation (Adjusted Rand Index)
    ari = adjusted_rand_score(y_true, kmeans.labels_)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")

    # Optional: Confusion matrix
    print("\nConfusion Matrix (Clusters vs True Labels):")
    print(confusion_matrix(y_true, kmeans.labels_))

if __name__ == "__main__":
    main()