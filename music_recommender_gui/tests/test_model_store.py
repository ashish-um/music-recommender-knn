import pytest
import pandas as pd

from music_recommender.core import recommender as rec
from music_recommender.core.model_store import ModelFileError, load_model, save_model


def test_round_trip(trained_model, tmp_path):
    path = save_model(trained_model, tmp_path / "model.joblib")
    loaded, notes = load_model(path)
    assert notes == []
    assert loaded.best_params == trained_model.best_params
    assert loaded.recommendation_feature_cols == trained_model.recommendation_feature_cols
    assert loaded.genre_feature_encoder is not None
    pd.testing.assert_frame_equal(loaded.results["recommendation_comparison"]["table"],
                                  trained_model.results["recommendation_comparison"]["table"])
    for method in ("KNN", "K-Means", "Hierarchical"):
        before = rec.recommend(trained_model, 10, method=method)
        after = rec.recommend(loaded, 10, method=method)
        assert before.equals(after)


def test_rejects_other_files(tmp_path):
    import joblib
    path = tmp_path / "other.joblib"
    joblib.dump({"not": "a model"}, path)
    with pytest.raises(ModelFileError):
        load_model(path)
    with pytest.raises(ModelFileError):
        load_model(tmp_path / "missing.joblib")


def test_old_distance_models_require_retraining(trained_model, tmp_path):
    from dataclasses import replace
    old = replace(trained_model, recommender_metric="unsupported")
    path = save_model(old, tmp_path / "old_distance.joblib")
    with pytest.raises(ModelFileError, match="Retrain using Euclidean"):
        load_model(path)
