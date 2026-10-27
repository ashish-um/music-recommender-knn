from dataclasses import replace

import numpy as np
import pytest

from music_recommender.core import clustering, recommender as rec
from music_recommender.core.config import TrainingConfig


@pytest.mark.parametrize("inertia, expected", [
    ([100, 20, 17, 14, 11], 2),
    ([100, 90, 20, 15, 10], 3),
    ([10, 8, 6, 4, 2], None),
    ([5, 5, 5, 5, 5], None),
])
def test_detects_bends_but_does_not_invent_an_elbow(inertia, expected):
    assert clustering.detect_elbow(range(1, 6), inertia) == expected


def test_automatic_count_and_manual_override_use_training_curve_only():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=60, centers=[[-10, -10], [0, 10], [10, -10]],
                      cluster_std=0.1, random_state=42)
    assert TrainingConfig(n_clusters=None).validate() == []
    for requested, expected, selection in ((None, 3, "elbow"), (4, 4, "manual")):
        models, comparison, _ = clustering.compare_recommenders(
            X, y, X[:3] + 1000, y[:3], metric="euclidean", k=4, n_clusters=requested)
        assert comparison["detected_clusters"] == 3
        assert comparison["n_clusters"] == expected
        assert comparison["cluster_selection"] == selection
        assert all(len(model.centers) == expected for model in models.values())


def test_automatic_count_falls_back_when_curve_has_only_two_points():
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    models, comparison, _ = clustering.compare_recommenders(
        X, [0, 0, 1, 1], [[0.1]], [0], metric="euclidean", k=2, n_clusters=None)
    assert comparison["n_clusters"] == 2
    assert comparison["cluster_selection"] == "fallback"
    assert comparison["detected_clusters"] is None


def test_small_clusters_do_not_backfill_and_missing_slots_count_as_misses():
    X = np.array([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    models, comparison, diagnostics = clustering.compare_recommenders(
        X, y, np.array([[0.05], [0.95]]), [0, 1], metric="euclidean", k=4, n_clusters=2)
    table = comparison["table"].set_index("Model")
    assert table["Precision@4"].to_numpy() == pytest.approx([0.75, 0.75, 0.75])
    assert table["Coverage@4"].to_numpy() == pytest.approx([1, 0.75, 0.75])
    for model in models.values():
        _, indices = model.recommend(X, X[[0]], 4, "euclidean", cluster=model.labels[0], exclude=0)
        assert set(indices) == {1, 2}
        _, indices = model.recommend(X, np.array([[0.95]]), 4, "euclidean")
        assert set(indices) == {3, 4, 5}
    assert diagnostics["linkage"].shape == (len(X) - 1, 4)
    assert diagnostics["linkage"][-1, 3] == len(X)


def test_sampling_is_shared_and_test_features_do_not_change_clusters(monkeypatch):
    monkeypatch.setattr(clustering, "MAX_TRAIN_SONGS", 12)
    monkeypatch.setattr(clustering, "MAX_TEST_SONGS", 6)
    X = np.random.default_rng(42).random((40, 3))
    y = np.repeat([0, 1], 20)
    models, comparison, _ = clustering.compare_recommenders(
        X, y, X + 10, y, metric="euclidean", k=4, n_clusters=2)
    other, repeated, _ = clustering.compare_recommenders(
        X, y, X - 10, 1 - y, metric="euclidean", k=4, n_clusters=2)
    assert comparison["train_songs"] == 12
    assert comparison["test_songs"] == 6
    np.testing.assert_array_equal(comparison["train_indices"], repeated["train_indices"])
    for name, model in models.items():
        np.testing.assert_allclose(model.centers, other[name].centers)
        assert len(model.labels) == 12


def test_cluster_recommendations_preserve_membership_and_exclude_input(trained_model):
    for method, model in trained_model.cluster_models.items():
        for index in range(0, trained_model.n_songs, 500):
            result = rec.recommend(trained_model, index, n=15, method=method)
            assert index not in set(result["song_index"])
            assert len(result) == min(15, np.count_nonzero(model.labels == model.labels[index]) - 1)
            assert result["distance"].is_monotonic_increasing
            assert (model.labels[result["song_index"].to_numpy()] == model.labels[index]).all()


def test_legacy_model_keeps_knn_but_requests_retraining_for_clusters(trained_model):
    old = replace(trained_model, cluster_models={})
    assert len(rec.recommend(old, 0)) == 10
    with pytest.raises(ValueError, match="Retrain"):
        rec.recommend(old, 0, method="K-Means")
