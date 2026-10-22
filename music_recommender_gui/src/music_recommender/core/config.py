"""Constants and training settings, taken from the notebook."""

from dataclasses import dataclass, field
from pathlib import Path

# The 12 audio features used in the notebook (implementation.md, Step 9).
DEFAULT_FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms", "mode", "explicit",
]

# Extra numeric columns the user may switch on. The notebook left them out
# because they lowered accuracy (key, time_signature) or do not describe sound (popularity).
OPTIONAL_FEATURES = ["popularity", "key", "time_signature"]

ALL_FEATURES = DEFAULT_FEATURES + OPTIONAL_FEATURES

# Audio features shown in the histograms and correlation heatmap.
AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
]

DEFAULT_GENRES = ["classical", "hip-hop", "heavy-metal", "edm", "jazz", "country", "pop"]

# Columns kept in the catalogue to display recommendations.
INFO_COLUMNS = ["track_id", "track_name", "artists", "album_name", "track_genre"]

REQUIRED_COLUMNS = INFO_COLUMNS + ALL_FEATURES

DEFAULT_PARAM_GRID = {
    "n_neighbors": [3, 5, 7, 9, 11, 15, 21, 25, 31],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "cosine"],
}

METRICS = ["euclidean", "manhattan", "cosine"]
WEIGHTS = ["uniform", "distance"]
SCALERS = ["minmax", "standard"]

# Range and display info for every feature, used by the feature editor.
# (minimum, maximum, step, unit)
FEATURE_RANGES = {
    "danceability": (0.0, 1.0, 0.01, ""),
    "energy": (0.0, 1.0, 0.01, ""),
    "loudness": (-60.0, 5.0, 0.1, "dB"),
    "speechiness": (0.0, 1.0, 0.01, ""),
    "acousticness": (0.0, 1.0, 0.01, ""),
    "instrumentalness": (0.0, 1.0, 0.01, ""),
    "liveness": (0.0, 1.0, 0.01, ""),
    "valence": (0.0, 1.0, 0.01, ""),
    "tempo": (0.0, 250.0, 1.0, "BPM"),
    "duration_ms": (10_000.0, 1_200_000.0, 1000.0, "ms"),
    "mode": (0.0, 1.0, 1.0, ""),
    "explicit": (0.0, 1.0, 1.0, ""),
    "popularity": (0.0, 100.0, 1.0, ""),
    "key": (0.0, 11.0, 1.0, ""),
    "time_signature": (0.0, 5.0, 1.0, ""),
}

BINARY_FEATURES = {"mode", "explicit"}

# Default location of dataset.csv: the project folder one level above this app.
APP_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_PATH = APP_ROOT.parent / "dataset.csv"
MODELS_DIR = APP_ROOT / "models"


@dataclass
class TrainingConfig:
    """Every setting of the training pipeline. Defaults reproduce the notebook."""

    dataset_path: str = str(DEFAULT_DATASET_PATH)
    genres: list[str] = field(default_factory=lambda: list(DEFAULT_GENRES))
    features: list[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    scaler: str = "minmax"
    default_k: int = 5
    precision_k: int = 10
    run_k_sweep: bool = True
    k_sweep_max: int = 30
    run_cv: bool = True
    cv_folds: int = 5
    run_grid_search: bool = True
    param_grid: dict = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_PARAM_GRID.items()})
    n_jobs: int = -1

    def grid_size(self) -> int:
        size = 1
        for values in self.param_grid.values():
            size *= len(values)
        return size

    def validate(self) -> list[str]:
        """Return a list of problems with the settings (empty if all is fine)."""
        problems = []
        if len(self.genres) < 2:
            problems.append("Select at least 2 genres.")
        if not self.features:
            problems.append("Select at least 1 feature.")
        if not 0.05 <= self.test_size <= 0.5:
            problems.append("Test size must be between 0.05 and 0.5.")
        if self.default_k < 1:
            problems.append("Default K must be at least 1.")
        if self.precision_k < 1:
            problems.append("Precision@K must be at least 1.")
        if self.run_cv and self.cv_folds < 2:
            problems.append("Cross validation needs at least 2 folds.")
        if self.run_grid_search:
            if self.cv_folds < 2:
                problems.append("Grid search needs at least 2 CV folds.")
            for key in ("n_neighbors", "weights", "metric"):
                if not self.param_grid.get(key):
                    problems.append(f"Grid search: choose at least one value for '{key}'.")
        if self.scaler not in SCALERS:
            problems.append(f"Unknown scaler '{self.scaler}'.")
        return problems
