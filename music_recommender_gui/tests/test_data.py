import pytest

from music_recommender.core import data


def test_cleaning_counts_for_explicit_subset(raw_df, example_genres):
    df, report = data.clean_dataset(raw_df, example_genres)
    assert [rows for _, rows in report.steps] == [114000, 113999, 7000, 7000 - 255, 4925]
    assert report.duplicate_track_ids == 255
    assert report.duplicate_songs == 1820
    assert report.songs_per_genre["heavy-metal"] == 986
    assert report.songs_per_genre["jazz"] == 472
    assert len(df) == 4925


def test_encoding(raw_df, example_genres):
    df, _ = data.clean_dataset(raw_df, example_genres)
    df, encoder = data.encode(df)
    assert list(encoder.classes_) == ["classical", "country", "edm", "heavy-metal", "hip-hop", "jazz", "pop"]
    assert set(df["explicit"].unique()) <= {0, 1}


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(data.DatasetError, match="not found"):
        data.load_dataset(tmp_path / "nope.csv")


def test_wrong_columns_give_clear_error(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(data.DatasetError, match="missing these columns"):
        data.load_dataset(path)


def test_unknown_genres(raw_df):
    with pytest.raises(data.DatasetError):
        data.clean_dataset(raw_df, ["not-a-genre", "neither"])


def test_explicit_all_genres_before_deduplication(raw_df):
    from music_recommender.core.config import TrainingConfig
    assert len(TrainingConfig().genres) == 7
    df, report = data.clean_dataset(raw_df, None)
    assert report.steps[2][1] == len(raw_df.dropna())
    assert len(df) == 81343
    assert df["track_genre"].nunique() == 113
