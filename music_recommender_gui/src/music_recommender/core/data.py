"""Loading, cleaning and encoding the Spotify Tracks dataset (notebook section 5.2 - 5.3)."""

from dataclasses import dataclass, field

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import REQUIRED_COLUMNS


class DatasetError(Exception):
    """Raised when the dataset file cannot be used."""


@dataclass
class CleaningReport:
    """Number of songs after each cleaning step, plus details printed in the notebook."""

    steps: list[tuple[str, int]] = field(default_factory=list)
    missing_values: dict[str, int] = field(default_factory=dict)
    duplicate_track_ids: int = 0
    duplicate_songs: int = 0
    songs_per_genre: dict[str, int] = field(default_factory=dict)

    def add(self, step: str, rows: int):
        self.steps.append((step, rows))


def load_dataset(path) -> pd.DataFrame:
    """Read dataset.csv and check that the columns the app needs are present."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise DatasetError(f"File not found: {path}") from None
    except Exception as exc:
        raise DatasetError(f"Could not read {path}: {exc}") from exc

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DatasetError("The dataset is missing these columns: " + ", ".join(missing))
    return df


def clean_dataset(df: pd.DataFrame, genres) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply the notebook's cleaning steps and record how many songs remain after each."""
    report = CleaningReport()
    report.add("Loaded", len(df))

    # "Unnamed: 0" is only a saved row number.
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    report.missing_values = {col: int(n) for col, n in df.isnull().sum().items() if n > 0}
    df = df.dropna()
    report.add("Removed missing values", len(df))

    df = df[df["track_genre"].isin(genres)].reset_index(drop=True)
    report.add(f"Kept {len(genres)} genres", len(df))
    if df.empty:
        raise DatasetError("None of the selected genres are present in the dataset.")

    # The same track can be listed under two genres, and the same song can be released
    # more than once. Keep only the first copy so a song is never recommended to itself.
    report.duplicate_track_ids = int(df.duplicated(subset="track_id").sum())
    df = df.drop_duplicates(subset="track_id")
    report.add("Removed duplicate track_id", len(df))

    report.duplicate_songs = int(df.duplicated(subset=["track_name", "artists"]).sum())
    df = df.drop_duplicates(subset=["track_name", "artists"]).reset_index(drop=True)
    report.add("Removed duplicate songs", len(df))

    report.songs_per_genre = {g: int(n) for g, n in df["track_genre"].value_counts().items()}
    return df, report


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """Convert explicit to 0/1 and genre names to numbers."""
    df = df.copy()
    df["explicit"] = LabelEncoder().fit_transform(df["explicit"])

    genre_encoder = LabelEncoder()
    df["genre_label"] = genre_encoder.fit_transform(df["track_genre"])
    return df, genre_encoder


def dataset_summary(df: pd.DataFrame) -> dict:
    """Basic numbers about a raw or cleaned dataset, shown on the Dataset tab."""
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "genres": int(df["track_genre"].nunique()),
        "missing": int(df.isnull().sum().sum()),
    }
