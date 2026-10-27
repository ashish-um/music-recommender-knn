import pytest

from music_recommender.core import data
from music_recommender.core.config import DEFAULT_DATASET_PATH, TrainingConfig
from music_recommender.core.training import run_training

if not DEFAULT_DATASET_PATH.exists():
    pytest.skip(f"dataset.csv not found at {DEFAULT_DATASET_PATH}", allow_module_level=True)


@pytest.fixture(scope="session")
def raw_df():
    return data.load_dataset(DEFAULT_DATASET_PATH)


@pytest.fixture(scope="session")
def example_genres():
    return ["classical", "hip-hop", "heavy-metal", "edm", "jazz", "country", "pop"]


@pytest.fixture(scope="session")
def trained_model(raw_df, example_genres):
    """Use an explicit small subset for fast regression checks."""
    return run_training(TrainingConfig(genres=example_genres), raw_df=raw_df)


@pytest.fixture(scope="session")
def small_config():
    """A fast configuration for tests that only need some model."""
    return TrainingConfig(genres=["classical", "heavy-metal"], run_grid_search=False,
                          run_k_sweep=False, run_cv=False)
