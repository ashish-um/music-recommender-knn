"""Evaluation helpers (notebook section 5.5)."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


def classification_metrics(y_true, y_pred) -> dict:
    """Accuracy and weighted Precision, Recall and F1-Score."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def per_class_report(y_true, y_pred, class_names) -> pd.DataFrame:
    """Precision, recall, F1 and support for every genre."""
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    rows = {name: report[name] for name in class_names}
    table = pd.DataFrame(rows).T[["precision", "recall", "f1-score", "support"]]
    table["support"] = table["support"].astype(int)
    return table


def make_confusion_matrix(y_true, y_pred, n_classes) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def precision_at_k(metric, X_train, y_train, X_test, y_test, k=10) -> float:
    """Average fraction of the k recommended songs that share the input song's genre.

    Training songs act as the catalogue and test songs as queries, so the songs
    being scored were never seen by the model.
    """
    if metric != "euclidean":
        raise ValueError("Only Euclidean distance is supported.")
    model = NearestNeighbors(metric="euclidean")
    model.fit(X_train)
    _, indices = model.kneighbors(X_test, n_neighbors=min(k, len(X_train)))

    train_genres = np.asarray(y_train)
    test_genres = np.asarray(y_test)

    hits = (train_genres[indices] == test_genres[:, None]).sum(axis=1)
    return float((hits / k).mean())


def k_sweep(X_train, y_train, X_test, y_test, k_values) -> list[tuple[int, float]]:
    """Test accuracy of KNeighborsClassifier for every K."""
    results = []
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        model.fit(X_train, y_train)
        results.append((k, accuracy_score(y_test, model.predict(X_test))))
    return results
