# Music Recommender

A PyQt6 desktop application for the mini project *Music Recommendation System using K-Nearest Neighbors* (see `../implementation.md`). It can:

- **train** the model with every step of the notebook (cleaning, scaling, KNN, cross validation, grid search), with the settings exposed as controls, and save it;
- **use** a trained model to recommend similar songs, predict the genre of a song you describe, and browse every evaluation result and plot.
- **compare** KNN, Decision Tree and Gaussian Naïve Bayes genre classifiers on the same features and train/test split.

With the default settings the app reproduces the notebook exactly: accuracy 57.56% → 61.83%, Precision@10 0.4936 → 0.5048, best parameters K = 25, Manhattan distance, distance weighting.

## Install and run

You need [uv](https://docs.astral.sh/uv/) and the dataset file `dataset.csv` in the project folder (one level above this folder).

```bash
cd music_recommender_gui
uv sync                    # creates .venv and installs PyQt6, scikit-learn, pandas, ...
uv run music-recommender   # or: uv run python -m music_recommender
```

A model trained with the notebook settings is included at `models/notebook_default.joblib`. Open it with **Model → Open Model…** to use the app without training first. The app remembers the last model and reopens it on the next start.

## The workspace

The app uses a light workspace with a gray navigation sidebar, white panels,
rounded controls and blue accents. Select a page in the sidebar; the header shows
your current page and catalogue size. **Open model…** is available in the header.
Menus, keyboard shortcuts and plot export tools remain available. At smaller widths,
recommendation, prediction, dataset and training panels stack vertically. Pages and
training settings scroll to keep controls and charts readable. Dropdowns and numeric
steppers have explicit arrow icons, and scrollbars remain visible when content overflows.

| Page | What it does |
|---|---|
| **Recommend** | Type part of a title or artist and pick a song from the list. The list shows "Title — Artist (genre)", so songs with the same name can be told apart. You get the top-N most similar songs, with same-genre rows highlighted, a "same genre: x / N" score and a chart comparing the input song's features with the recommendations. You can switch the distance metric live. Double-click a result to get recommendations for it. Right-click to copy it or open it in Spotify. **Export CSV…** saves the list. |
| **Train** | Choose the dataset, genres (any of the 114), input features, test size, random state, scaler, default K, Precision@K, accuracy-vs-K range, CV folds and the grid search values. **Train model** runs in the background with a progress bar and a log in the notebook's output format. **Cancel** stops it, and **Restore notebook defaults** resets every setting. When training finishes, the app offers to save the model. |
| **Results** | Before/after KNN metrics and chart; **Model comparison** table and accuracy chart for KNN, Decision Tree and Naïve Bayes, naming the highest test accuracy (including ties); confusion matrices and genre-wise reports for each classifier; accuracy vs K, cross-validation folds, and grid search results. Plot toolbars include a **save** button for report figures. |
| **Dataset** | Size of the original dataset, the number of songs after each cleaning step, and the notebook's plots: songs per genre, histograms, boxplots by genre (any feature), duration boxplot, correlation heatmap, plus average features by genre. Includes a searchable table of every song the model uses. |
| **About** | Project details and developer credits: Ashish Umrao, Abhijeet Kumar and Aryan Singh. Available without loading a model; also shown in **Help → About**. |

**Model menu:** Open, Save As, Open Recent, Model Info (all training settings and the best parameters), and *Use This Model's Settings for Training* (copies a saved model's settings to the Train page).

> Model files are Python pickles (joblib). Only open model files you trust, such as ones saved by this app.

## Project structure

```
music_recommender_gui/
├── pyproject.toml / uv.lock
├── PLAN.md                      implementation plan
├── models/                      saved models (*.joblib)
├── src/music_recommender/
│   ├── core/                    machine learning only, no Qt
│   │   ├── config.py            constants, TrainingConfig (defaults = notebook)
│   │   ├── data.py              load / clean / encode the dataset
│   │   ├── evaluation.py        metrics, Precision@K, accuracy vs K
│   │   ├── training.py          run_training(): the whole notebook pipeline
│   │   ├── model_store.py       TrainedModel, save / load
│   │   └── recommender.py       search, recommend, predict genre
│   └── gui/
│       ├── app.py               entry point
│       ├── main_window.py       sidebar, pages, menus, status bar, settings
│       ├── theme.py             shared light palette and widget styling
│       ├── workers.py           background threads (dataset loading, training)
│       ├── widgets/             plot canvas, DataFrame table, song search, feature sliders
│       └── tabs/                one file per workspace page
└── tests/                       pytest: notebook parity, recommender, persistence, GUI smoke tests
```

The core can also be used without the GUI:

```python
from music_recommender.core.config import TrainingConfig
from music_recommender.core.training import run_training, PrintReporter
from music_recommender.core import recommender as rec

model = run_training(TrainingConfig(), PrintReporter())
print(rec.recommend(model, rec.find_song(model, "Nuvole Bianche")))
```

## Comparing the three classifiers

Every new training run evaluates `KNeighborsClassifier`, `DecisionTreeClassifier` and
`GaussianNB` using the same cleaned catalogue, selected features, training/test rows
and training-fitted scaler. KNN uses its grid-search parameters when grid search is
enabled, otherwise its configured baseline. Decision Tree uses sklearn defaults with
the configured random seed; Gaussian Naïve Bayes uses defaults. No extra dependencies
are needed. This follows PR7 (KNN) and PR8 (Decision Tree and Naïve Bayes).

Open **Results → Model comparison** for accuracy, weighted precision, recall and F1,
sorted by accuracy, plus an accuracy chart and the highest-scoring classifier(s).
Results opens on this comparison when the loaded model contains comparison data.
The confusion-matrix and genre-wise report views also have selectors for the new
classifiers. Scores are saved in the model bundle. Older saved models still open;
retrain and save a new model to populate the comparison.

Measured with the default dataset and settings: **KNN 61.83%**, Decision Tree
**56.04%**, Gaussian Naïve Bayes **55.13%** test accuracy. Optimized KNN has the
highest accuracy among these tested configurations.

The comparison reports the best **tested configuration on this test set**. Only KNN
is tuned, so it is not an equally tuned comparison of model families. Test accuracy
is descriptive; use training-set cross-validation for further model selection rather
than repeatedly tuning on the test set. These scores measure genre prediction,
not listener satisfaction. KNN provides song recommendations; recommendation quality
is measured separately by Precision@K.

## Tests

```bash
uv run pytest
```

The tests need `../dataset.csv` and are skipped if it is missing. They check that:
- the default training reproduces every number in implementation.md (accuracy, F1, CV, best parameters, Precision@10, 609 correct predictions);
- the progress-reporting grid search gives the same result as scikit-learn's `GridSearchCV`;
- the three-classifier comparison uses the same split, persists through saving/loading, and renders alongside older saved models;
- recommendations never include the input song;
- a saved model gives identical recommendations after loading;
- the GUI can train, cancel and recommend in an offscreen window.

## Notes

- **Grid search** runs in a loop that gives the same result as `GridSearchCV`, but reports progress after each of the 270 fits so it can show a progress bar and be cancelled. A full training run takes about 15–20 seconds on all CPU cores.
- **Wayland:** if the window does not appear, try `QT_QPA_PLATFORM=xcb uv run music-recommender`.
