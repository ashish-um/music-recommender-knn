"""The trained model bundle and saving / loading it with joblib."""

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .config import TrainingConfig

FORMAT_VERSION = 1


@dataclass
class TrainedModel:
    """Everything produced by one training run: fitted objects, song catalogue and results."""

    config: TrainingConfig
    feature_cols: list[str]
    genre_classes: list[str]
    scaler: object
    default_clf: object
    tuned_clf: object
    best_params: dict
    recommender_metric: str
    recommender: object
    catalogue: pd.DataFrame          # info columns + raw features of every song
    catalogue_scaled: np.ndarray     # scaled features, same row order as catalogue
    results: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    sklearn_version: str = sklearn.__version__
    format_version: int = FORMAT_VERSION

    @property
    def n_songs(self) -> int:
        return len(self.catalogue)

    @property
    def tuned_accuracy(self) -> float:
        return self.results["tuned"]["metrics"]["Accuracy"]

    def summary(self) -> str:
        return (f"{len(self.genre_classes)} genres · {self.n_songs:,} songs · "
                f"Accuracy {self.tuned_accuracy:.2%}")


class ModelFileError(Exception):
    """Raised when a model file cannot be loaded."""


def default_model_filename() -> str:
    return datetime.now().strftime("knn_model_%Y-%m-%d_%H%M%S.joblib")


def save_model(model: TrainedModel, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    return path


def load_model(path) -> tuple[TrainedModel, list[str]]:
    """Load a model saved by this app. Returns the model and a list of warnings.

    joblib files are pickles: only load files you trust (ones saved by this app).
    """
    path = Path(path)
    if not path.exists():
        raise ModelFileError(f"File not found: {path}")
    notes = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = joblib.load(path)
    except Exception as exc:
        raise ModelFileError(f"Could not load {path.name}: {exc}") from exc

    if not isinstance(model, TrainedModel):
        raise ModelFileError(f"{path.name} is not a model saved by Music Recommender.")
    if model.sklearn_version != sklearn.__version__ or caught:
        notes.append(f"The model was trained with scikit-learn {model.sklearn_version}, "
                     f"but {sklearn.__version__} is installed. Retrain if results look wrong.")
    return model, notes
