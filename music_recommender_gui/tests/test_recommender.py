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
        for metric in (None, "euclidean", "cosine"):
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
