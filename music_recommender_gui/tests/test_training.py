import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from music_recommender.core import data, evaluation
from music_recommender.core.config import TrainingConfig
from music_recommender.core.training import Reporter, TrainingCancelled, grid_search, run_training


def test_reproduces_notebook_results(trained_model):
    r = trained_model.results
    assert r["split"] == {"train": 3940, "test": 985}
    assert r["default"]["metrics"]["Accuracy"] == pytest.approx(0.5756, abs=1e-4)
    assert r["default"]["metrics"]["F1-Score"] == pytest.approx(0.5684, abs=1e-4)
    assert r["default"]["precision_at_k"] == pytest.approx(1.0)
    assert r["cv"]["mean"] == pytest.approx(0.5665, abs=1e-4)
    assert trained_model.best_params == {"metric": "euclidean", "n_neighbors": 15, "weights": "distance"}
    assert r["grid"]["best_score"] == pytest.approx(0.6028, abs=1e-4)
    assert r["tuned"]["metrics"]["Accuracy"] == pytest.approx(0.6122, abs=1e-4)
    assert r["tuned"]["metrics"]["F1-Score"] == pytest.approx(0.6048, abs=1e-4)
    assert r["tuned"]["precision_at_k"] == pytest.approx(1.0)
    assert max(r["k_sweep"], key=lambda item: item[1])[0] == 17
    assert np.trace(r["tuned"]["confusion_matrix"]) == 603
    assert trained_model.n_songs == 4925


def test_grid_search_matches_gridsearchcv():
    rng = np.random.default_rng(0)
    X = rng.random((300, 4))
    y = (X[:, 0] + 0.3 * rng.random(300) > 0.6).astype(int) + (X[:, 1] > 0.5)
    grid = {"n_neighbors": [3, 7, 15], "weights": ["uniform", "distance"], "metric": ["euclidean"]}

    best, table, score = grid_search(X, y, grid, 5, 1, Reporter(), 0, 100)
    reference = GridSearchCV(KNeighborsClassifier(), grid, cv=5, scoring="accuracy").fit(X, y)

    assert best == reference.best_params_
    assert score == pytest.approx(reference.best_score_)
    assert sorted(table["mean CV accuracy"]) == pytest.approx(sorted(reference.cv_results_["mean_test_score"]))


def test_recommendation_comparison_uses_same_split(raw_df, trained_model):
    config = trained_model.config
    df, _ = data.clean_dataset(raw_df, config.genres)
    df, _ = data.encode(df)
    X_train, X_test, y_train, y_test = train_test_split(
        df[config.features], df["genre_label"], test_size=config.test_size,
        random_state=config.random_state, stratify=df["genre_label"],
    )
    comparison = trained_model.results["recommendation_comparison"]
    table = comparison["table"].set_index("Model")
    assert set(table.index) == {"KNN", "K-Means", "Hierarchical"}
    assert "model_comparison" not in trained_model.results
    assert comparison["train_songs"] == len(X_train)
    assert comparison["test_songs"] == len(X_test)
    encoder = trained_model.genre_feature_encoder
    train_inputs = np.hstack([trained_model.scaler.transform(X_train),
                              encoder.transform(df.loc[X_train.index, ["track_genre"]])])
    test_inputs = np.hstack([trained_model.scaler.transform(X_test),
                             encoder.transform(df.loc[X_test.index, ["track_genre"]])])
    reference = evaluation.precision_at_k(
        trained_model.recommender_metric, train_inputs, y_train,
        test_inputs, y_test, config.precision_k)
    assert table.loc["KNN", f"Precision@{config.precision_k}"] == pytest.approx(reference)
    assert table[f"Precision@{config.precision_k}"].between(0, 1).all()
    for name, cluster in trained_model.cluster_models.items():
        assert len(cluster.labels) == trained_model.n_songs
        assert len(cluster.centers) == config.n_clusters
        assert trained_model.results["clustering"]["cluster_sizes"][name].sum() == len(X_train)


def test_without_grid_search_uses_default_model(raw_df, small_config):
    model = run_training(small_config, raw_df=raw_df)
    assert model.tuned_clf is model.default_clf
    assert model.recommender_metric == "euclidean"
    assert model.genre_classes == ["classical", "heavy-metal"]


def test_cancel(raw_df):
    class CancelNow(Reporter):
        def is_cancelled(self):
            return True

    with pytest.raises(TrainingCancelled):
        run_training(TrainingConfig(), CancelNow(), raw_df=raw_df)


def test_invalid_config_rejected(raw_df):
    with pytest.raises(ValueError, match="at least 2 genres"):
        run_training(TrainingConfig(genres=["jazz"]), raw_df=raw_df)


def test_only_euclidean_is_accepted():
    config = TrainingConfig()
    assert config.grid_size() == 18
    config.param_grid["metric"] = ["unsupported"]
    assert "Only Euclidean distance is supported." in config.validate()
    with pytest.raises(ValueError, match="Only Euclidean"):
        grid_search(np.zeros((10, 2)), np.zeros(10), config.param_grid, 2, 1, Reporter(), 0, 100)
