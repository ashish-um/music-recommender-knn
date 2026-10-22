"""The full training pipeline (notebook sections 5.2 - 5.6) as one function."""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.model_selection import (
    ParameterGrid,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import data, evaluation
from .config import ALL_FEATURES, INFO_COLUMNS, TrainingConfig
from .model_store import TrainedModel


class TrainingCancelled(Exception):
    """Raised when the user cancels training."""


class Reporter:
    """Receives progress from run_training. The GUI subclasses this to emit Qt signals."""

    def stage(self, name: str, percent: int):
        pass

    def progress(self, percent: int, detail: str = ""):
        pass

    def log(self, text: str):
        pass

    def is_cancelled(self) -> bool:
        return False


class PrintReporter(Reporter):
    """Prints the log, for use from the command line."""

    def log(self, text: str):
        print(text)


def _check_cancel(reporter):
    if reporter.is_cancelled():
        raise TrainingCancelled()


def _fit_and_score(estimator, params, X, y, train_idx, test_idx):
    model = clone(estimator).set_params(**params)
    model.fit(X[train_idx], y[train_idx])
    return model.score(X[test_idx], y[test_idx])


def grid_search(X, y, param_grid, cv_folds, n_jobs, reporter, start, end) -> tuple[dict, pd.DataFrame, float]:
    """Same result as GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv_folds),
    but reports progress after every fit and can be cancelled."""
    candidates = list(ParameterGrid(param_grid))
    folds = list(StratifiedKFold(n_splits=cv_folds).split(X, y))
    tasks = [(params, train_idx, test_idx) for params in candidates for train_idx, test_idx in folds]
    total = len(tasks)
    reporter.log(f"Fitting {cv_folds} folds for each of {len(candidates)} candidates, "
                 f"totalling {total} fits")

    estimator = KNeighborsClassifier()
    scores = []
    jobs = (delayed(_fit_and_score)(estimator, p, X, y, tr, te) for p, tr, te in tasks)
    for score in Parallel(n_jobs=n_jobs, return_as="generator")(jobs):
        scores.append(score)
        done = len(scores)
        reporter.progress(start + (end - start) * done // total, f"{done}/{total} fits")
        _check_cancel(reporter)

    scores = np.array(scores).reshape(len(candidates), cv_folds)
    means = scores.mean(axis=1)
    stds = scores.std(axis=1)
    ranks = rankdata(-means, method="min").astype(int)
    best_index = int(np.argmin(ranks))

    table = pd.DataFrame({
        "K": [c["n_neighbors"] for c in candidates],
        "weights": [c["weights"] for c in candidates],
        "metric": [c["metric"] for c in candidates],
        "mean CV accuracy": means,
        "std": stds,
        "rank": ranks,
    }).sort_values(["rank", "K"]).reset_index(drop=True)
    return candidates[best_index], table, float(means[best_index])


def _evaluate(clf, X_test, y_test, n_classes, class_names) -> dict:
    y_pred = clf.predict(X_test)
    return {
        "metrics": evaluation.classification_metrics(y_test, y_pred),
        "confusion_matrix": evaluation.make_confusion_matrix(y_test, y_pred, n_classes),
        "per_class": evaluation.per_class_report(y_test, y_pred, class_names),
    }


def _log_metrics(reporter, metrics):
    for name, value in metrics.items():
        reporter.log(f"{name:<10}: {value:.4f}")


def run_training(config: TrainingConfig, reporter: Reporter | None = None, raw_df=None) -> TrainedModel:
    """Run every step of the notebook with the given settings and return the trained model.

    raw_df can be passed to skip reading the CSV again (the GUI keeps it in memory).
    """
    reporter = reporter or Reporter()
    problems = config.validate()
    if problems:
        raise ValueError("\n".join(problems))
    results = {}

    # 1. Dataset collection
    reporter.stage("Loading dataset", 0)
    if raw_df is None:
        raw_df = data.load_dataset(config.dataset_path)
    results["raw_summary"] = data.dataset_summary(raw_df)
    reporter.log("#### LOADING DATASET")
    reporter.log(f"Number of samples:  {raw_df.shape[0]}")
    reporter.log(f"Number of features:  {raw_df.shape[1]}")
    reporter.log(f"Number of genres: {raw_df['track_genre'].nunique()}")
    _check_cancel(reporter)

    # 2. Cleaning
    reporter.stage("Cleaning data", 5)
    df, cleaning = data.clean_dataset(raw_df, config.genres)
    results["cleaning"] = cleaning
    reporter.log("\n#### DATA PREPROCESSING")
    for col, n in cleaning.missing_values.items():
        reporter.log(f"Missing values in {col}: {n}")
    reporter.log(f"Number of samples after removing missing values: {cleaning.steps[1][1]}")
    reporter.log(f"Number of samples in {len(config.genres)} selected genres: {cleaning.steps[2][1]}")
    reporter.log(f"Duplicate track_id: {cleaning.duplicate_track_ids}")
    reporter.log(f"Duplicate songs (same name and artist): {cleaning.duplicate_songs}")
    reporter.log(f"Number of samples after removing duplicates: {len(df)}")
    for genre, n in cleaning.songs_per_genre.items():
        reporter.log(f"  {genre:<15}{n}")
    _check_cancel(reporter)

    # 3. Encoding
    reporter.stage("Encoding", 10)
    df, genre_encoder = data.encode(df)
    class_names = [str(c) for c in genre_encoder.classes_]
    n_classes = len(class_names)
    reporter.log("Genre classes: " + ", ".join(f"{c} = {i}" for i, c in enumerate(class_names)))

    X = df[config.features]
    y = df["genre_label"]
    reporter.log(f"Number of input features: {X.shape[1]}")

    # 4. Train-test split
    reporter.stage("Train-test split", 12)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state,
        stratify=y if config.stratify else None,
    )
    results["split"] = {"train": len(X_train), "test": len(X_test)}
    reporter.log("\n#### TRAIN TEST SPLIT")
    reporter.log(f"Number of samples in training dataset:  {len(X_train)}")
    reporter.log(f"Number of samples in testing dataset:  {len(X_test)}")

    # 5. Scaling (fitted on training data only)
    reporter.stage("Scaling features", 14)
    scaler = MinMaxScaler() if config.scaler == "minmax" else StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train_arr = y_train.to_numpy()
    y_test_arr = y_test.to_numpy()
    reporter.log(f"Scaler: {type(scaler).__name__} (fitted on training data only)")
    _check_cancel(reporter)

    # 6. Default KNN classifier
    reporter.stage("Training default KNN", 16)
    default_clf = KNeighborsClassifier(n_neighbors=config.default_k)
    default_clf.fit(X_train_scaled, y_train_arr)
    results["default"] = _evaluate(default_clf, X_test_scaled, y_test_arr, n_classes, class_names)
    reporter.log(f"\n#### DEFAULT KNN (K = {config.default_k}, euclidean, uniform)")
    _log_metrics(reporter, results["default"]["metrics"])
    _check_cancel(reporter)

    # 7. Default recommender quality
    reporter.stage(f"Precision@{config.precision_k} (default)", 22)
    results["default"]["precision_at_k"] = evaluation.precision_at_k(
        "euclidean", X_train_scaled, y_train_arr, X_test_scaled, y_test_arr, config.precision_k)
    reporter.log(f"Precision@{config.precision_k} (default recommender): "
                 f"{results['default']['precision_at_k']:.4f}")
    _check_cancel(reporter)

    # 8. Accuracy vs K
    if config.run_k_sweep:
        reporter.stage("Accuracy vs K", 25)
        max_k = min(config.k_sweep_max, len(X_train))
        results["k_sweep"] = evaluation.k_sweep(
            X_train_scaled, y_train_arr, X_test_scaled, y_test_arr, range(1, max_k + 1))
        best_k, best_acc = max(results["k_sweep"], key=lambda item: item[1])
        reporter.log(f"\nBest K from accuracy-vs-K: {best_k} with accuracy {best_acc:.4f}")
        _check_cancel(reporter)

    # 9. Cross validation of the default model
    if config.run_cv:
        reporter.stage("Cross validation", 40)
        scores = cross_val_score(KNeighborsClassifier(n_neighbors=config.default_k),
                                 X_train_scaled, y_train_arr, cv=config.cv_folds, scoring="accuracy")
        results["cv"] = {"scores": scores, "mean": float(scores.mean()), "std": float(scores.std())}
        reporter.log(f"\n#### CROSS VALIDATION ({config.cv_folds}-fold)")
        reporter.log(f"Fold accuracies: {np.round(scores, 4)}")
        reporter.log(f"Mean CV Accuracy: {scores.mean() * 100:.2f} %")
        reporter.log(f"CV Standard Deviation: {scores.std():.4f}")
        _check_cancel(reporter)

    # 10. Grid search
    if config.run_grid_search:
        reporter.stage("Grid search", 45)
        reporter.log("\n#### HYPERPARAMETER TUNING (GridSearchCV)")
        best_params, grid_table, best_score = grid_search(
            X_train_scaled, y_train_arr, config.param_grid, config.cv_folds,
            config.n_jobs, reporter, 45, 90)
        tuned_clf = KNeighborsClassifier(**best_params).fit(X_train_scaled, y_train_arr)
        results["grid"] = {"table": grid_table, "best_score": best_score}
        reporter.log(f"Best Parameters: {best_params}")
        reporter.log(f"Best CV Accuracy: {best_score * 100:.2f} %")
    else:
        best_params = {"n_neighbors": config.default_k, "weights": "uniform", "metric": "euclidean"}
        tuned_clf = default_clf

    # 11. Tuned classifier evaluation
    reporter.stage("Evaluating optimized KNN", 90)
    results["tuned"] = _evaluate(tuned_clf, X_test_scaled, y_test_arr, n_classes, class_names)
    reporter.log("\n#### OPTIMIZED KNN")
    _log_metrics(reporter, results["tuned"]["metrics"])
    _check_cancel(reporter)

    # Compare genre classifiers on exactly the same scaled training and test rows.
    results["model_comparison"] = {"KNN": results["tuned"]}
    for name, classifier in (
        ("Decision Tree", DecisionTreeClassifier(random_state=config.random_state)),
        ("Naïve Bayes", GaussianNB()),
    ):
        reporter.stage(f"Comparing {name}", 92)
        classifier.fit(X_train_scaled, y_train_arr)
        results["model_comparison"][name] = _evaluate(
            classifier, X_test_scaled, y_test_arr, n_classes, class_names)
        reporter.log(f"\n#### {name.upper()} (genre classification)")
        _log_metrics(reporter, results["model_comparison"][name]["metrics"])
        _check_cancel(reporter)
    best_accuracy = max(r["metrics"]["Accuracy"] for r in results["model_comparison"].values())
    winners = [name for name, r in results["model_comparison"].items()
               if r["metrics"]["Accuracy"] == best_accuracy]
    reporter.log(f"\nHighest test accuracy: {', '.join(winners)} ({best_accuracy:.2%})")

    # 12. Tuned recommender quality: only the metric carries over to NearestNeighbors.
    best_metric = best_params["metric"]
    reporter.stage(f"Precision@{config.precision_k} (optimized)", 94)
    results["tuned"]["precision_at_k"] = evaluation.precision_at_k(
        best_metric, X_train_scaled, y_train_arr, X_test_scaled, y_test_arr, config.precision_k)
    reporter.log(f"Precision@{config.precision_k} (optimized recommender, {best_metric}): "
                 f"{results['tuned']['precision_at_k']:.4f}")

    # 13. Final recommender on the whole catalogue
    reporter.stage("Building recommender", 97)
    catalogue_scaled = scaler.transform(X)
    recommender = NearestNeighbors(metric=best_metric).fit(catalogue_scaled)
    catalogue = df[INFO_COLUMNS + ALL_FEATURES].reset_index(drop=True)
    reporter.log(f"\nRecommender is ready. Songs in catalogue: {len(catalogue)}")

    reporter.stage("Done", 100)
    return TrainedModel(
        config=config,
        feature_cols=list(config.features),
        genre_classes=class_names,
        scaler=scaler,
        default_clf=default_clf,
        tuned_clf=tuned_clf,
        best_params=dict(best_params),
        recommender_metric=best_metric,
        recommender=recommender,
        catalogue=catalogue,
        catalogue_scaled=catalogue_scaled,
        results=results,
    )
