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
def trained_model(raw_df):
    """The model trained with notebook defaults (takes ~20 s, shared by all tests)."""
    return run_training(TrainingConfig(), raw_df=raw_df)


@pytest.fixture(scope="session")
def small_config():
    """A fast configuration for tests that only need some model."""
    return TrainingConfig(genres=["classical", "heavy-metal"], run_grid_search=False,
                          run_k_sweep=False, run_cv=False)
