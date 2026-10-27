from dataclasses import replace
from pathlib import Path
import pytest

from streamlit.testing.v1 import AppTest
from yt_dlp import YoutubeDL

from music_recommender.core import recommender as rec

APP = Path(__file__).parents[1] / "src/music_recommender/streamlit_app.py"


def test_streamlit_selects_each_recommender_and_renders_comparison(trained_model, monkeypatch):
    monkeypatch.setattr(YoutubeDL, "extract_info", lambda *args, **kwargs: {"entries": []})
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.session_state["model"] = trained_model
    app.run()
    assert not app.exception
    selector = next(widget for widget in app.selectbox if widget.label == "Recommendation model")
    assert selector.options == ["KNN", "K-Means", "Hierarchical"]
    for method in ("K-Means", "Hierarchical"):
        next(widget for widget in app.selectbox if widget.label == "Recommendation model").select(method).run()
        assert not app.exception
        expected = rec.recommend(trained_model, 0, method=method)
        assert app.dataframe[0].value["track_name"].tolist() == expected["track_name"].tolist()
    assert any("K-Means vs hierarchical" in widget.value for widget in app.subheader)
    assert not any("feature profile" in widget.value.lower() for widget in app.subheader)
    assert not any("feature profile" in widget.label.lower() for widget in app.checkbox)
    assert any(widget.label == "Show recommendation distance chart" for widget in app.checkbox)


def test_streamlit_opens_legacy_model_with_retraining_message(trained_model, monkeypatch):
    monkeypatch.setattr(YoutubeDL, "extract_info", lambda *args, **kwargs: {"entries": []})
    old = replace(trained_model, cluster_models={}, results={
        key: value for key, value in trained_model.results.items()
        if key not in ("recommendation_comparison", "clustering")})
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.session_state["model"] = old
    app.run()
    assert not app.exception
    selector = next(widget for widget in app.selectbox if widget.label == "Recommendation model")
    assert selector.options == ["KNN"]
    assert any("Retrain this saved model" in widget.value for widget in app.info)


@pytest.mark.parametrize("clusters", [None, 4])
def test_blank_or_entered_cluster_count_reaches_training(clusters, trained_model, raw_df, monkeypatch):
    from music_recommender.core import training
    from music_recommender.core.config import DEFAULT_GENRES
    calls = []

    def capture_training(config, *args, **kwargs):
        calls.append(config)
        return trained_model

    monkeypatch.setattr(training, "run_training", capture_training)
    monkeypatch.setattr(YoutubeDL, "extract_info", lambda *args, **kwargs: {"entries": []})
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.session_state["raw_df"] = raw_df
    app.run()
    field = next(widget for widget in app.number_input if widget.label == "Number of clusters")
    assert field.value is None
    genres = next(widget for widget in app.multiselect if widget.label == "Genres")
    assert set(genres.value) == set(DEFAULT_GENRES)
    if clusters is not None:
        field.set_value(clusters).run()
    next(widget for widget in app.button if widget.label == "Train model").click().run()
    assert not app.exception
    assert len(calls) == 1
    assert calls[0].n_clusters == clusters
