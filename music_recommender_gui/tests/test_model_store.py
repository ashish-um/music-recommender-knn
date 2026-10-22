import pytest

from music_recommender.core import recommender as rec
from music_recommender.core.model_store import ModelFileError, load_model, save_model


def test_round_trip(trained_model, tmp_path):
    path = save_model(trained_model, tmp_path / "model.joblib")
    loaded, notes = load_model(path)
    assert notes == []
    assert loaded.best_params == trained_model.best_params
    for name, result in trained_model.results["model_comparison"].items():
        assert loaded.results["model_comparison"][name]["metrics"] == result["metrics"]
    before = rec.recommend(trained_model, 10)
    after = rec.recommend(loaded, 10)
    assert before.equals(after)


def test_rejects_other_files(tmp_path):
    import joblib
    path = tmp_path / "other.joblib"
    joblib.dump({"not": "a model"}, path)
    with pytest.raises(ModelFileError):
        load_model(path)
    with pytest.raises(ModelFileError):
        load_model(tmp_path / "missing.joblib")
