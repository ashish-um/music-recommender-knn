import pytest
import numpy as np

from music_recommender.core import recommender as rec


def test_nuvole_bianche_gets_classical(trained_model):
    index = rec.find_song(trained_model, "nuvole bianche")
    result = rec.recommend(trained_model, index, n=10)
    assert len(result) == 10
    assert (result["track_genre"] == "classical").all()
    assert result["same_genre"].all()
    assert "Una Mattina" in set(result["track_name"])


def test_input_song_never_recommended(trained_model):
    for index in range(0, trained_model.n_songs, 500):
        for metric in (None, "euclidean"):
            result = rec.recommend(trained_model, index, n=15, metric=metric)
            assert index not in set(result["song_index"])
            assert len(result) == 15
            assert result["distance"].is_monotonic_increasing


def test_unknown_song(trained_model):
    assert rec.find_song(trained_model, "Some Unknown Song") is None
    assert rec.search_songs(trained_model, "zzzzqqqq").empty


def test_search_matches_title_and_artist(trained_model):
    by_title = rec.search_songs(trained_model, "nuvole")
    by_artist = rec.search_songs(trained_model, "einaudi")
    assert "Nuvole Bianche" in set(by_title["track_name"])
    assert set(by_artist["artists"].str.contains("Einaudi"))  == {True}


def test_custom_features_of_a_song_find_that_song(trained_model):
    index = rec.find_song(trained_model, "Balls to the Wall")
    result = rec.recommend_from_features(trained_model, rec.song_features(trained_model, index), n=5)
    assert result.iloc[0]["song_index"] == index
    assert result.iloc[0]["distance"] == 0


def test_predict_genre(trained_model):
    index = rec.find_song(trained_model, "Nuvole Bianche")
    genre, probabilities = rec.predict_genre(trained_model, rec.song_features(trained_model, index))
    assert genre == "classical"
    assert set(probabilities) == set(trained_model.genre_classes)
    assert abs(sum(probabilities.values()) - 1) < 1e-9


def test_rejects_other_distances_for_every_method(trained_model):
    for method in ("KNN", "K-Means", "Hierarchical"):
        with pytest.raises(ValueError, match="Only Euclidean"):
            rec.recommend(trained_model, 0, metric="unsupported", method=method)


def test_genre_is_used_for_retrieval_but_not_genre_classification(trained_model):
    model = trained_model
    audio_count = len(model.feature_cols)
    assert model.default_clf.n_features_in_ == audio_count
    assert model.tuned_clf.n_features_in_ == audio_count
    assert model.recommender.n_features_in_ == audio_count + len(model.genre_classes)
    expected = model.genre_feature_encoder.transform(model.catalogue[["track_genre"]])
    np.testing.assert_array_equal(model.catalogue_scaled[:, audio_count:], expected)
    assert len(model.recommendation_feature_cols) == model.catalogue_scaled.shape[1]


def test_genre_changes_ranking_when_audio_alone_prefers_another_genre(trained_model):
    from dataclasses import replace
    from sklearn.neighbors import NearestNeighbors

    audio_count = len(trained_model.feature_cols)
    catalogue = trained_model.catalogue.iloc[:3].copy().reset_index(drop=True)
    catalogue["track_genre"] = [trained_model.genre_classes[0], trained_model.genre_classes[1],
                                 trained_model.genre_classes[0]]
    audio = np.zeros((3, audio_count))
    audio[:, 0] = [0, 0.01, 0.5]
    inputs = np.hstack([audio, trained_model.genre_feature_encoder.transform(catalogue[["track_genre"]])])
    model = replace(trained_model, catalogue=catalogue, catalogue_scaled=inputs,
                    recommender=NearestNeighbors(metric="euclidean").fit(inputs))
    assert NearestNeighbors(metric="euclidean").fit(audio).kneighbors(audio[[0]], n_neighbors=2)[1][0, 1] == 1
    assert rec.recommend(model, 0, n=1).iloc[0]["song_index"] == 2


def test_custom_recommendations_require_a_known_genre(trained_model):
    features = rec.song_features(trained_model, 0)
    features.pop("track_genre")
    with pytest.raises(ValueError, match="Provide track_genre"):
        rec.recommend_from_features(trained_model, features)
    rec.predict_genre(trained_model, features)
