"""Using a trained model: search songs, recommend similar songs and predict genre."""

import numpy as np
import pandas as pd

from .config import ALL_FEATURES
from .model_store import TrainedModel


def song_label(row) -> str:
    return f"{row['track_name']} — {row['artists']} ({row['track_genre']})"


def song_labels(model: TrainedModel) -> list[str]:
    """One "Title — Artist (genre)" label per catalogue row, used for autocomplete."""
    cat = model.catalogue
    return (cat["track_name"] + " — " + cat["artists"] + " (" + cat["track_genre"] + ")").tolist()


def search_songs(model: TrainedModel, text: str, limit: int = 50) -> pd.DataFrame:
    """Songs whose title or artist contains the text (not case sensitive)."""
    text = text.strip().lower()
    cat = model.catalogue
    if not text:
        return cat.iloc[0:0]
    mask = (cat["track_name"].str.lower().str.contains(text, regex=False)
            | cat["artists"].str.lower().str.contains(text, regex=False))
    matches = cat[mask]
    # Exact title matches first, then the rest in catalogue order.
    exact = matches["track_name"].str.lower() == text
    return pd.concat([matches[exact], matches[~exact]]).head(limit)


def find_song(model: TrainedModel, name: str, artist: str | None = None) -> int | None:
    """Index of a song by exact title (and artist, if given), like the notebook's search."""
    cat = model.catalogue
    mask = cat["track_name"].str.lower() == name.strip().lower()
    if artist:
        mask &= cat["artists"].str.lower() == artist.strip().lower()
    matches = cat.index[mask]
    return int(matches[0]) if len(matches) else None


def _neighbours_model(model: TrainedModel, metric: str | None):
    if metric not in (None, "euclidean"):
        raise ValueError("Only Euclidean distance is supported.")
    if model.recommender_metric != "euclidean":
        raise ValueError("Retrain this model using Euclidean distance.")
    return model.recommender


def _result_table(model: TrainedModel, indices, distances, input_genre=None) -> pd.DataFrame:
    result = model.catalogue.loc[indices, ["track_name", "artists", "track_genre", "track_id"]].copy()
    result.insert(0, "rank", range(1, len(result) + 1))
    result["distance"] = np.round(distances, 4)
    if input_genre is not None:
        result["same_genre"] = result["track_genre"] == input_genre
    return result.reset_index().rename(columns={"index": "song_index"})


def recommend(model: TrainedModel, song_index: int, n: int = 10, metric: str | None = None,
              method: str = "KNN") -> pd.DataFrame:
    """Top-n songs closest to the catalogue song at song_index, excluding the song itself."""
    if method == "KNN":
        nn = _neighbours_model(model, metric)
        n_query = min(n + 1, model.n_songs)
        distances, indices = nn.kneighbors(model.catalogue_scaled[[song_index]], n_neighbors=n_query)
        keep = indices[0] != song_index
        indices, distances = indices[0][keep][:n], distances[0][keep][:n]
    else:
        cluster = getattr(model, "cluster_models", {}).get(method)
        if cluster is None:
            raise ValueError(f"{method} is unavailable in this model. Retrain to add clustering models.")
        distances, indices = cluster.recommend(
            model.catalogue_scaled, model.catalogue_scaled[[song_index]], n,
            metric or "euclidean", cluster=cluster.labels[song_index], exclude=song_index)
    genre = model.catalogue.at[song_index, "track_genre"]
    return _result_table(model, indices, distances, genre)


def _scale_features(model: TrainedModel, features: dict) -> np.ndarray:
    row = pd.DataFrame([[float(features[col]) for col in model.feature_cols]], columns=model.feature_cols)
    return model.scaler.transform(row)


def recommend_from_features(model: TrainedModel, features: dict, n: int = 10,
                            metric: str | None = None, method: str = "KNN") -> pd.DataFrame:
    """Top-n songs closest to a custom set of audio features."""
    query = _scale_features(model, features)
    encoder = getattr(model, "genre_feature_encoder", None)
    if encoder is not None:
        genre = features.get("track_genre")
        if genre not in model.genre_classes:
            raise ValueError("Provide track_genre from the trained catalogue for recommendations.")
        query = np.hstack([query, encoder.transform(pd.DataFrame({"track_genre": [genre]}))])
    if method == "KNN":
        nn = _neighbours_model(model, metric)
        distances, indices = nn.kneighbors(query, n_neighbors=min(n, model.n_songs))
        return _result_table(model, indices[0], distances[0])
    cluster = getattr(model, "cluster_models", {}).get(method)
    if cluster is None:
        raise ValueError(f"{method} is unavailable in this model. Retrain to add clustering models.")
    distances, indices = cluster.recommend(model.catalogue_scaled, query, n, metric or "euclidean")
    return _result_table(model, indices, distances)


def predict_genre(model: TrainedModel, features: dict, which: str = "tuned") -> tuple[str, dict]:
    """Predicted genre and the probability of every genre for a custom set of features."""
    clf = model.tuned_clf if which == "tuned" else model.default_clf
    probabilities = clf.predict_proba(_scale_features(model, features))[0]
    by_genre = {model.genre_classes[label]: float(p) for label, p in zip(clf.classes_, probabilities)}
    predicted = max(by_genre, key=by_genre.get)
    return predicted, by_genre


def song_features(model: TrainedModel, song_index: int) -> dict:
    """Raw feature values of a catalogue song."""
    row = model.catalogue.loc[song_index]
    return {**{col: float(row[col]) for col in ALL_FEATURES}, "track_genre": row["track_genre"]}
