# Music Recommender

A PyQt6 desktop application for the mini project *Music Recommendation System using K-Nearest Neighbors* (see `../implementation.md`). It can:

- **train** the model with every step of the notebook (cleaning, scaling, KNN, cross validation, grid search), with the settings exposed as controls, and save it;
- **use** a trained model to recommend similar songs, predict the genre of a song you describe, and browse every evaluation result and plot.
- **compare** KNN, K-Means and Ward hierarchical recommendations on the same training candidates and held-out test songs.

All models use **Euclidean distance only**, with **seven genres selected by default**: classical, hip-hop, heavy-metal, EDM, jazz, country and pop. Recommendations combine scaled audio features and one-hot genre inputs.
The separate KNN genre classifier uses audio only to avoid including its target as
an input. Precision@K now measures genre consistency; because genre is an input,
it is not an independent recommendation-quality score.

## Install and run

You need [uv](https://docs.astral.sh/uv/) and the dataset file `dataset.csv` in the project folder (one level above this folder).

```bash
cd music_recommender_gui
uv sync                    # creates .venv and installs PyQt6, scikit-learn, pandas, ...
uv run music-recommender   # or: uv run python -m music_recommender
uv run streamlit run src/music_recommender/streamlit_app.py  # browser interface
```

The Streamlit interface reuses the same model files and training pipeline. It includes
song recommendations with distance charts, classifier evaluation
and confusion-matrix plots, catalogue exploration, and model training. It reads
`dataset.csv` from the usual project location or accepts a CSV upload. Choose **KNN**,
**K-Means** or **Hierarchical** under **Recommend → Recommendation model**. The Train
tab accepts an optional cluster count: leave it blank for automatic elbow detection,
or enter a count to override detection. Results includes an elbow plot and dendrogram.
After training, a save dialog lets you choose the model file name and local path.
The Recommend page can also search YouTube for the selected song and embed the first
match. This uses yt-dlp without an API key; results can be inaccurate or unavailable
for embedding, and YouTube may change the page format used for search.
Settings are saved locally in `~/.music-recommender/settings.json` and retained when
the app or a model is reopened.

Train and save a new model to use the seven default genres and genre input features. The desktop app remembers the last model and reopens it on the next start. Older audio-only models show a notice to retrain.

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
| **Recommend** | Type part of a title or artist and pick a song from the list. The list shows "Title — Artist (genre)", so songs with the same name can be told apart. You get the top-N most similar songs, with same-genre rows highlighted, a "same genre: x / N" score and a chart comparing the input song's features with the recommendations. Euclidean distance is fixed. Double-click a result to get recommendations for it. Right-click to copy it or open it in Spotify. **Export CSV…** saves the list. |
| **Train** | Choose the dataset, genres (any of the 114), input features, test size, random state, scaler, default K, Precision@K, accuracy-vs-K range, CV folds and the grid search values. **Train model** runs in the background with a progress bar and a log in the notebook's output format. **Cancel** stops it, and **Restore notebook defaults** resets every setting. When training finishes, the app offers to save the model. |
| **Results** | Recommendation Precision@K, coverage and runtime for KNN, K-Means and hierarchical clustering; elbow plot, dendrogram and cluster sizes in Streamlit; KNN confusion matrices and per-genre reports; accuracy vs K, cross-validation and grid search. |
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
│   │   ├── clustering.py        cluster retrieval and shared recommendation comparison
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

## Comparing the three recommendation methods

This follows PR7 (KNN), PR9 (K-Means and hierarchical clustering) and PR11
(cross-validation and grid search). The notebook and Streamlit app reuse the same
sklearn clustering and evaluation code. No new application dependencies are needed.

| Method | Recommendation rule |
|---|---|
| KNN (`NearestNeighbors`) | Rank all candidates by Euclidean distance. |
| K-Means (`KMeans`, 10 initializations) | Assign a query to its nearest centroid, then rank songs within that cluster. |
| Hierarchical (`AgglomerativeClustering`, Ward linkage) | Build training clusters; assign unseen queries to the nearest training-cluster centroid, then rank songs within that cluster. |

Hierarchical clustering has no native prediction operation. Nearest-centroid
assignment is the stated extension for new songs; fitted songs keep their original
cluster memberships. Both clustering algorithms form groups using Euclidean geometry.
All three methods rank candidates with Euclidean distance.
Genre is one-hot encoded and appended to scaled audio inputs for all three recommendation methods, including cluster fitting. It is excluded from the separate genre classifier inputs.

All methods use the same reproducible sample of at most **5,000 training candidates**
and **1,000 held-out test queries**. This bounds hierarchical memory use. The seven default genres leave 4,925 cleaned songs. Their full train/test split
fits within these limits, so every training/test song is included in the comparison.
The scaler is fitted on training data only. The table reports:

- **Precision@K:** average same-genre matches divided by the requested K.
- **Coverage@K:** average returned songs divided by K. Small clusters return fewer
  songs; missing slots count as misses in Precision@K. Results are not backfilled.
- **Fit seconds:** estimator fitting, excluding elbow diagnostics.
- **Query ms/song:** recommendation time including assignment for unseen queries.

In Streamlit, leave **Number of clusters** blank to select the largest normalized
bend in the K-Means training inertia curve (1–12 clusters, limited by distinct
training songs). The detected count is used for both K-Means and hierarchical
clustering. Entering a count overrides detection. If the curve has no detectable
bend or too few points, training uses 2 clusters and identifies the fallback.
The Results plot marks the count actually used and, for a manual override, the
detected elbow when different. This geometric heuristic does not guarantee an
optimal cluster count. The notebook keeps an explicit example of 7; set
`n_clusters = None` there or in `TrainingConfig` to enable automatic selection. Cluster numbers are arbitrary, and numbers across methods
have no shared meaning. Classification accuracy, precision, recall, F1 and confusion
matrices remain available for KNN genre prediction. They are separate from the
recommendation comparison. KNN grid search tunes classification K and voting weights; Euclidean retrieval remains fixed.

Interactive KNN searches the full catalogue. Cluster models learn from the training
sample; other catalogue songs are assigned to learned centroids. The sample-based
held-out comparison and interactive full-catalogue recommendations serve different
purposes. Because genre is an input feature, Precision@K measures genre consistency rather than independent recommendation quality or listener satisfaction.

Models save cluster centers, catalogue memberships, comparison scores and plot data.
Train a fresh model to include all three methods and genre inputs. Older Euclidean models
still open with KNN; retrain them to add clustering. Models trained with a different
distance must be retrained so that saved scores match Euclidean behavior. The desktop app retains KNN
retrieval and displays the new recommendation comparison.

## Tests

```bash
uv run pytest
```

The tests need `../dataset.csv` and are skipped if it is missing. They check that:
- a fixed seven-genre regression run reproduces its expected classification scores; recommendation queries include one-hot genre inputs;
- the progress-reporting grid search gives the same result as scikit-learn's `GridSearchCV`;
- clustering uses training data only, excludes the input song, handles small clusters without backfilling, and persists through saving/loading;
- the recommendation comparison uses the same candidates and queries and renders alongside older saved models;
- recommendations never include the input song;
- a saved model gives identical recommendations after loading;
- the GUI can train, cancel and recommend in an offscreen window.

## Notes

- **Grid search** runs in a loop that gives the same result as `GridSearchCV`, but reports progress after each of the 90 fits so it can show a progress bar and be cancelled. The seven-genre default keeps training quick; selecting more genres increases classification CV, grid-search and accuracy-by-K time.
- **Wayland:** if the window does not appear, try `QT_QPA_PLATFORM=xcb uv run music-recommender`.
