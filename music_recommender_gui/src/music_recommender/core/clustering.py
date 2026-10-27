"""Cluster-based song retrieval and a shared held-out recommendation comparison."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

# shortcut: cap the shared comparison sample, revisit for exact hierarchies above 5,000 songs.
MAX_TRAIN_SONGS = 5000
MAX_TEST_SONGS = 1000


@dataclass
class ClusterRecommender:
    centers: np.ndarray
    labels: np.ndarray

    def predict(self, X):
        """Ward clustering has no predict; assign unseen songs to the nearest centroid."""
        return pairwise_distances_argmin(X, self.centers, metric="euclidean")

    def recommend(self, X, query, n, metric, *, cluster=None, exclude=None):
        if metric != "euclidean":
            raise ValueError("Only Euclidean distance is supported.")
        if cluster is None:
            cluster = int(self.predict(query)[0])
        indices = np.flatnonzero(self.labels == cluster)
        if exclude is not None:
            indices = indices[indices != exclude]
        if not len(indices):
            return np.array([], dtype=float), indices
        distances = pairwise_distances(query, X[indices], metric=metric)[0]
        order = np.argsort(distances, kind="stable")[:n]
        return distances[order], indices[order]


def _sample_indices(y, limit, random_state):
    indices = np.arange(len(y))
    if len(y) <= limit:
        return indices
    _, counts = np.unique(y, return_counts=True)
    stratify = y if counts.min() >= 2 and min(limit, len(y) - limit) >= len(counts) else None
    return np.sort(train_test_split(indices, train_size=limit, random_state=random_state,
                                    stratify=stratify)[0])


def detect_elbow(counts, inertia):
    """Find the largest normalized bend away from the line joining curve endpoints."""
    counts, inertia = np.asarray(counts), np.asarray(inertia)
    if len(counts) < 3 or np.ptp(inertia) <= 0:
        return None
    x = (counts - counts[0]) / (counts[-1] - counts[0])
    y = (inertia - inertia[-1]) / (inertia[0] - inertia[-1])
    bends = 1 - x - y
    index = int(np.argmax(bends[1:-1])) + 1
    return int(counts[index]) if bends[index] > 1e-6 else None


def compare_recommenders(X_train, y_train, X_test, y_test, *, metric, k, n_clusters,
                         random_state=42):
    """Fit on the same training sample; score the same unseen songs for all methods.

    The y arrays are for sampling and scoring; X may include one-hot genre inputs.
    Missing recommendations count as misses in Precision@K and lower Coverage@K.
    """
    if metric != "euclidean":
        raise ValueError("Only Euclidean distance is supported.")
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    train_indices = _sample_indices(y_train, MAX_TRAIN_SONGS, random_state)
    test_indices = _sample_indices(y_test, MAX_TEST_SONGS, random_state)
    X_train, y_train = np.asarray(X_train)[train_indices], y_train[train_indices]
    X_test, y_test = np.asarray(X_test)[test_indices], y_test[test_indices]
    if n_clusters is not None and not 2 <= n_clusters <= len(X_train):
        raise ValueError(f"Cluster count must be between 2 and {len(X_train)} training songs.")

    max_count = min(12, len(np.unique(X_train, axis=0)))
    if max_count < 2:
        raise ValueError("Clustering requires at least two distinct training songs.")
    candidate_counts = set(range(1, max_count + 1))
    if n_clusters is not None:
        candidate_counts.add(n_clusters)
    kmeans_fits, kmeans_times, elbow = {}, {}, []
    for count in sorted(candidate_counts):
        start = perf_counter()
        estimator = KMeans(n_clusters=count, n_init=10, random_state=random_state).fit(X_train)
        kmeans_times[count] = perf_counter() - start
        kmeans_fits[count] = estimator
        elbow.append((count, estimator.inertia_))
    detected = detect_elbow(*zip(*elbow))
    selection = "manual"
    if n_clusters is None:
        n_clusters = detected if detected is not None else 2
        selection = "elbow" if detected is not None else "fallback"

    models, fit_times = {}, {}
    start = perf_counter()
    knn = NearestNeighbors(metric=metric).fit(X_train)
    fit_times["KNN"] = perf_counter() - start
    for name, estimator in (
        ("K-Means", kmeans_fits[n_clusters]),
        ("Hierarchical", AgglomerativeClustering(n_clusters=n_clusters, linkage="ward",
                                                compute_distances=True)),
    ):
        if name == "K-Means":
            labels = estimator.labels_
            fit_times[name] = kmeans_times[n_clusters]
        else:
            start = perf_counter()
            labels = estimator.fit_predict(X_train)
            fit_times[name] = perf_counter() - start
        groups, labels = np.unique(labels, return_inverse=True)
        centers = np.array([X_train[labels == group].mean(axis=0) for group in range(len(groups))])
        models[name] = ClusterRecommender(centers, labels)
        if name == "Hierarchical":
            hierarchy = estimator

    rows = []
    for name in ("KNN", *models):
        start = perf_counter()
        if name == "KNN":
            _, indices = knn.kneighbors(X_test, n_neighbors=min(k, len(X_train)))
        else:
            model = models[name]
            groups = model.predict(X_test)
            indices = [model.recommend(X_train, query[None, :], k, metric, cluster=group)[1]
                       for query, group in zip(X_test, groups)]
        query_time = perf_counter() - start
        hits = [np.count_nonzero(y_train[neighbors] == genre)
                for neighbors, genre in zip(indices, y_test)]
        rows.append({"Model": name, f"Precision@{k}": float(np.mean(hits) / k),
                     f"Coverage@{k}": float(np.mean([len(i) for i in indices]) / k),
                     "Fit seconds": fit_times[name],
                     "Query ms/song": query_time * 1000 / len(X_test)})

    counts = np.zeros(len(hierarchy.children_))
    for i, children in enumerate(hierarchy.children_):
        counts[i] = sum(1 if child < len(X_train) else counts[child - len(X_train)]
                        for child in children)
    linkage = np.column_stack([hierarchy.children_, hierarchy.distances_, counts]).astype(float)
    comparison = {
        "table": pd.DataFrame(rows), "k": k, "metric": metric, "n_clusters": n_clusters,
        "cluster_selection": selection, "detected_clusters": detected,
        "train_indices": train_indices, "test_indices": test_indices,
        "train_songs": len(X_train), "test_songs": len(X_test),
    }
    diagnostics = {
        "elbow": pd.DataFrame(elbow, columns=["Clusters", "Inertia"]), "linkage": linkage,
        "cluster_sizes": pd.DataFrame({name: np.bincount(model.labels, minlength=n_clusters)
                                       for name, model in models.items()}),
    }
    return models, comparison, diagnostics
