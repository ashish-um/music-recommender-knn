# Implementation Plan: PyQt GUI for the KNN Music Recommender

> **Status:** implemented (milestones M1–M9). See README.md for usage.

This plan turns the notebook `Music_Recommendation_KNN.ipynb` (described in `../implementation.md`) into a desktop application with two jobs:

1. **Train**: run the full pipeline (load → clean → encode → split → scale → KNN → evaluate → Cross Validation → GridSearchCV → recommender) from the GUI, with the settings exposed as controls, and save the result as a model file.
2. **Use**: load a trained model and get song recommendations, predict the genre of a song, and browse the evaluation results.

---

## 1. Goals and non-goals

### Goals
- **Same results as the notebook.** With default settings, the app must reproduce the notebook's numbers exactly: Accuracy 0.5756 → 0.6183, Precision@10 0.4936 → 0.5048, best params `{metric: manhattan, n_neighbors: 25, weights: distance}`. This is the main correctness check.
- **Responsive UI.** Loading the CSV and running Grid Search (270 fits) happen in a background thread, with a progress bar and a live log. The window never freezes.
- **Train once, use many times.** A trained model is saved to a `.joblib` file and reloaded at startup, so you don't have to retrain every time you open the app.
- **Fix the limitations listed in implementation.md §7** where the GUI makes this easy:
  - *Duplicate names:* search by "Title — Artist" with autocomplete instead of the first exact name match.
  - *Search by artist:* the search box matches both title and artist.
  - *User interface:* this app is the "future work" item 7.
- **Testable core.** All ML code lives in a Qt-free package, so it can be unit-tested and reused.

### Non-goals
- No web version, no user accounts, no collaborative filtering.
- No audio playback. The dataset has no audio files. A "Open in Spotify" link using `track_id` is the closest we get (optional, see §9).
- No changes to the existing notebook or dataset.

---

## 2. Technology choices

| Concern | Choice | Why |
|---|---|---|
| GUI toolkit | **PyQt6** | Current PyQt version. Ships as an abi3 wheel, so it works on Python 3.14. (PyQt5 is already installed system-wide and would also work; see §11.) |
| ML | scikit-learn, pandas, NumPy | Same libraries as the notebook. |
| Plots | Matplotlib (`FigureCanvasQTAgg`) + Seaborn | Same plots as the notebook, embedded in Qt widgets. |
| Model persistence | `joblib` | Standard for scikit-learn objects. |
| Environment | **uv** (`pyproject.toml` + `.venv`) | Already installed. Keeps the dependencies out of the system Python, which is missing scikit-learn. |
| Tests | pytest (+ `QT_QPA_PLATFORM=offscreen` for GUI smoke tests) | Runs without a display. |

---

## 3. Project layout

```
Mini Project/
├── dataset.csv                       (existing, NOT copied: the app points to it)
├── implementation.md                 (existing)
├── Music_Recommendation_KNN.ipynb    (existing)
└── music_recommender_gui/
    ├── PLAN.md                       this file
    ├── README.md                     install / run / usage guide with screenshots
    ├── pyproject.toml                uv project, deps, `music-recommender` script entry point
    ├── models/                       saved models (*.joblib), created on first save
    ├── src/music_recommender/
    │   ├── __init__.py
    │   ├── __main__.py               `python -m music_recommender`
    │   ├── core/                     ── pure Python, no Qt imports ──
    │   │   ├── config.py             constants + TrainingConfig dataclass
    │   │   ├── data.py               load / clean / encode the dataset
    │   │   ├── evaluation.py         metrics, precision_at_k, K sweep
    │   │   ├── training.py           run_training(config, reporter) -> TrainedModel
    │   │   ├── model_store.py        TrainedModel dataclass, save/load .joblib
    │   │   └── recommender.py        search, recommend, recommend_from_features, predict_genre
    │   └── gui/
    │       ├── app.py                QApplication setup, style, main()
    │       ├── main_window.py        QMainWindow: tabs, menus, status bar, model state
    │       ├── workers.py            QThread workers (dataset loading, training)
    │       ├── widgets/
    │       │   ├── mpl_canvas.py     reusable Matplotlib canvas widget
    │       │   ├── dataframe_model.py QAbstractTableModel for pandas DataFrames
    │       │   └── feature_editor.py  sliders + spin boxes for the 12 audio features
    │       └── tabs/
    │           ├── data_tab.py       dataset overview and EDA plots
    │           ├── train_tab.py      training settings, Train button, progress, log
    │           ├── results_tab.py    metrics, confusion matrix, K curve, grid results
    │           ├── recommend_tab.py  pick a song → top-N similar songs
    │           └── predict_tab.py    custom audio features → predicted genre + similar songs
    └── tests/
        ├── conftest.py               dataset path fixture (skip if dataset.csv missing)
        ├── test_data.py
        ├── test_training.py          reproduces the notebook numbers
        ├── test_recommender.py
        ├── test_model_store.py
        └── test_gui_smoke.py         offscreen: build window, load model, recommend
```

**Why a `core/` and `gui/` split:** the ML logic can then be tested without a display, the GUI stays thin, and the same core could later back a CLI or web app.

---

## 4. Core layer (port of the notebook)

### 4.1 `config.py`
- `ALL_FEATURE_COLS`: the 12 notebook features plus the optional `popularity`, `key`, `time_signature` (off by default, since the notebook found they lower accuracy).
- `DEFAULT_FEATURES`: the 12 features from §5.3 Step 9.
- `DEFAULT_GENRES = ["classical", "hip-hop", "heavy-metal", "edm", "jazz", "country", "pop"]`
- `DEFAULT_PARAM_GRID`: the grid from §5.6.2.
- `@dataclass TrainingConfig`: `dataset_path`, `genres`, `features`, `test_size=0.2`, `random_state=42`, `stratify=True`, `scaler="minmax"`, `default_k=5`, `run_cv=True`, `cv_folds=5`, `run_grid_search=True`, `param_grid`, `run_k_sweep=True`, `k_sweep_range=(1, 30)`, `precision_k=10`, `n_jobs=-1`.

### 4.2 `data.py`
Each function is one notebook step, and each returns a small report dict that the GUI shows in the log:
- `load_dataset(path) -> DataFrame` checks that the required columns are present and raises a clear error if not.
- `clean_dataset(df, genres) -> (df, CleaningReport)`: drop `Unnamed: 0`, `dropna`, filter genres, drop duplicate `track_id`, then drop duplicate (`track_name`, `artists`). The report records the count after each step (114000 → 113999 → 7000 → 4925 for defaults).
- `encode(df) -> (df, genre_encoder)`: `explicit` → 0/1 and `track_genre` → `genre_label`.
- `dataset_summary(df)`: rows, columns, missing values, genres, songs per genre (for the Data tab).

### 4.3 `evaluation.py`
- `classification_metrics(y_true, y_pred)` returns accuracy and weighted precision, recall and F1 (as in §5.5.1).
- `per_class_report(...)`: `classification_report(output_dict=True)` as a DataFrame.
- `precision_at_k(metric, X_train, y_train, X_test, y_test, k)`: identical logic to §5.5.2.
- `k_sweep(X_train, y_train, X_test, y_test, k_range)` returns a list of (k, accuracy).

### 4.4 `training.py`
`run_training(config, reporter) -> TrainedModel`, where `reporter` is a small protocol with `stage(name, percent)`, `log(text)` and `is_cancelled()`. The core therefore knows nothing about Qt, and the worker turns these calls into signals.

The stages and their approximate progress weights:

| # | Stage | Output kept in the model |
|---|---|---|
| 1 | Load CSV | — |
| 2 | Clean (drop column, missing values, genres, duplicates) | cleaning report |
| 3 | Encode | genre classes |
| 4 | Train-test split (stratified, `random_state`) | split sizes |
| 5 | Fit scaler on train only | scaler |
| 6 | Default KNN (`n_neighbors=default_k`) + test metrics + confusion matrix | default classifier, metrics |
| 7 | Default Precision@K (Euclidean) | value |
| 8 | K sweep 1–30 (optional) | (k, acc) list |
| 9 | 5-fold CV of the default model (optional) | fold scores, mean, std |
| 10 | GridSearchCV (optional, the longest step) | `cv_results_` table, best params, best CV score |
| 11 | Tuned classifier metrics + confusion matrix | tuned classifier, metrics |
| 12 | Tuned Precision@K (best metric) | value |
| 13 | Fit `NearestNeighbors(metric=best_metric)` on **all** scaled songs | recommender, scaled catalogue |

If Grid Search is turned off, the "tuned" model is the default model and the recommender uses Euclidean distance.

**Cancellation:** `is_cancelled()` is checked between stages. A running `GridSearchCV.fit` cannot be stopped part-way, so for the Grid Search stage the worker runs it with a custom loop over `ParameterGrid` × folds, checking for cancel and reporting progress after each fit. The results are the same as `GridSearchCV`; `tests/test_training.py` asserts that the best parameters and score match. A simpler fallback, if the custom loop proves fiddly: call `GridSearchCV` directly and show an indeterminate progress bar for that stage only.

### 4.5 `model_store.py`
`@dataclass TrainedModel` holds:
- `config`, `created_at`, `sklearn_version`, `n_songs`
- `scaler`, `genre_classes`, `feature_cols`
- `default_clf`, `tuned_clf`, `best_params`
- `recommender` (fitted `NearestNeighbors`), `catalogue` (DataFrame: `track_id`, `track_name`, `artists`, `album_name`, `track_genre`, raw feature columns), `catalogue_scaled` (ndarray)
- `results`: metrics for both models, confusion matrices, per-class reports, CV scores, grid table, K sweep, Precision@K values, cleaning report

The scaled catalogue is about 4,925 × 12 floats, so the file stays small (around 1–2 MB).

- `save_model(model, path)` / `load_model(path)` using joblib. The file includes a format version, and loading warns when the scikit-learn version differs from the one used for training.
- **Security note:** joblib files are pickles. The app only loads files the user chooses or the app saved itself, and the Open dialog says this.

### 4.6 `recommender.py`
All functions take a `TrainedModel`:
- `search_songs(model, text, limit=50)`: case-insensitive substring match on title **or** artist, returning rows to fill the autocomplete list.
- `recommend(model, song_index, n=10, metric=None)` excludes the input song **by index**, not by `head()`. The optional `metric` re-fits a `NearestNeighbors` with another distance; this takes milliseconds on 4,925 songs and lets the user compare metrics live. Returns rank, title, artist, genre, distance and a `same_genre` flag.
- `recommend_from_features(model, raw_features: dict, n=10)` scales a custom feature vector and finds the nearest songs (there is no self to exclude).
- `predict_genre(model, raw_features, which="tuned")` returns the predicted genre and per-genre probabilities (`predict_proba`).

---

## 5. GUI design

### 5.1 Main window

```
┌ Music Recommender (KNN) ────────────────────────────────────────────────┐
│ File  Model  Help                                                       │
├─────────────────────────────────────────────────────────────────────────┤
│ [ Recommend ] [ Predict Genre ] [ Train ] [ Results ] [ Dataset ]       │
│                                                                         │
│                         (active tab content)                            │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ Model: knn_2026-10-22.joblib · 7 genres · 4,925 songs · Acc 61.83%      │
└─────────────────────────────────────────────────────────────────────────┘
```

- **Menus:**
  - *File*: Open Dataset…, Quit
  - *Model*: Open Model…, Save Model As…, Recent Models, Model Info
  - *Help*: About (project, syllabus, authors from the notebook's "Developed by" cell)
- **Model state:** `MainWindow.current_model` (a `TrainedModel` or None). Setting it emits a `modelChanged` signal, and every tab updates itself from that signal. Recommend, Predict and Results are disabled, with a hint to "Train or open a model", until a model is present.
- **Startup:** `QSettings` remembers the last model path, the dataset path and the window geometry. The last model is reloaded automatically if the file still exists.
- **Style:** Fusion style, which follows the system light/dark palette. The Matplotlib figure background matches the palette.

### 5.2 Recommend tab (main "use" screen)

```
Song: [ nuvole bia_______________________ ▾ ]  Top N: [10]  Distance: [Model default ▾]  [Recommend] [🎲 Random]
      ┌ Nuvole Bianche — Ludovico Einaudi (classical) ┐  ← autocomplete popup
Input: Nuvole Bianche | Ludovico Einaudi | classical
┌────┬───────────────────────┬──────────────────┬───────────┬──────────┐
│ #  │ Track                 │ Artist           │ Genre     │ Distance │   rows with the same genre are highlighted
│ 1  │ Una Mattina           │ Ludovico Einaudi │ classical │ 0.4123   │
│ …  │                       │                  │           │          │
└────┴───────────────────────┴──────────────────┴───────────┴──────────┘
Same genre: 10 / 10                                           [Export CSV]
┌ Feature profile: input song vs average of the recommendations (bar chart) ┐
```

- `QCompleter` on a model of "Title — Artist (genre)" strings, so duplicate titles can be told apart.
- Double-clicking a recommendation makes it the new input song, so you can "walk" through similar songs.
- Right-click → *Copy* / *Open in Spotify* (`https://open.spotify.com/track/<track_id>`).

### 5.3 Predict Genre tab (classifier + custom song)

- A `FeatureEditor` with one slider and spin box per feature, in real units:
  - danceability, energy and similar features: 0–1
  - loudness: −60–0 dB
  - tempo: 0–250 BPM
  - duration: mm:ss
  - mode, explicit: checkboxes
- "Fill from song…" copies an existing song's features, so you can start from a real song and adjust it.
- Output:
  - Predicted genre (large label), from the tuned model, with a switch to the default model.
  - A horizontal bar chart of the per-genre probabilities.
  - The top-10 nearest songs to this custom feature vector.
- It updates live, with a short debounce while sliders move.

### 5.4 Train tab

```
Dataset: [../dataset.csv                     ] [Browse]   ✓ 114,000 rows loaded
┌ Data ──────────────────┐ ┌ Model ──────────────────────┐ ┌ Optimization ───────────────────┐
│ Genres: [7 selected ▾] │ │ Default K: [5]              │ │ ☑ Cross validation  folds [5]   │
│  (searchable checklist │ │ Scaler: (•) MinMax ( ) Std  │ │ ☑ Grid search                   │
│   of all 114 genres)   │ │ Precision@K, K: [10]        │ │   K values: [3,5,7,9,11,15,...] │
│ Features: [12 sel. ▾]  │ │ ☑ K sweep 1 – [30]          │ │   weights: ☑uniform ☑distance   │
│ Test size: [0.20]      │ │                             │ │   metric: ☑eucl ☑manh ☑cosine   │
│ Random state: [42]     │ │                             │ │   → 54 combinations, 270 fits   │
│ ☑ Stratify             │ │                             │ │ CPU cores: [all ▾]              │
└────────────────────────┘ └─────────────────────────────┘ └─────────────────────────────────┘
[Restore notebook defaults]                          [▶ Train]  [■ Cancel]
Stage: Grid search (132/270 fits)   [██████████░░░░░░░░░░] 48%
┌ Log ───────────────────────────────────────────────────────────────────────────────────────┐
│ Number of samples: 114000 … Duplicate track_id: 255 … Best Parameters: {...}               │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

- Inputs are checked before training starts: at least 2 genres and 1 feature, and K must be smaller than the smallest training fold.
- When training finishes:
  - The new model becomes the current model.
  - The app switches to the Results tab.
  - It asks whether to save the model, with a timestamped default file name in `models/`.
- The log text matches the notebook's printed output, so the GUI also works as a teaching aid for the report.

### 5.5 Results tab (evaluation of the current model)

A left-hand list selects a view, and the right side shows it:
1. **Summary:** the before/after table from §5.7.1, with an Improvement column.
2. **Comparison chart:** grouped bar chart of Default vs Optimized (PR11 style).
3. **Confusion matrix:** heatmap, with a Default/Optimized toggle.
4. **Per-genre report:** precision, recall, F1 and support table.
5. **Accuracy vs K:** line plot from the K sweep, with the best K marked.
6. **Cross validation:** fold scores, mean and standard deviation.
7. **Grid search:** sortable table of all combinations (K, weights, metric, mean, std, rank).

Every plot has a "Save image…" button, which is useful for the project report.

### 5.6 Dataset tab (EDA)
- Summary cards: rows, columns, genres, missing values. The cleaning report shows the counts after each step.
- A plot selector with the notebook's plots: songs per genre, feature histograms, boxplots by genre (with a feature picker), duration boxplot and correlation heatmap.
- A paged table preview of the cleaned data.
- Works from the current model's catalogue: the cleaned data plus the original dataset's summary saved in the model. *(Implementation note: before any model exists, the Train tab shows the raw CSV summary and the genre counts instead.)*

### 5.7 Threading model
- `workers.py`: `TrainingWorker(QObject)` runs inside a `QThread`. It has signals `stageChanged(str, int)`, `logLine(str)`, `finished(object)`, `failed(str)` and `cancelled()`. A `QtReporter` adapts the core `reporter` protocol to these signals.
- `DatasetLoadWorker` does the same for the roughly 1 s CSV load.
- The GUI never touches model objects from the worker thread. The finished `TrainedModel` is handed over through the `finished` signal.
- Errors such as a missing column or a bad path appear in a `QMessageBox` with the details, and do not crash the app.

---

## 6. Build order (milestones)

Each milestone ends in something runnable and checked before the next one starts.

| # | Milestone | Done when |
|---|---|---|
| M1 | **Scaffold:** `uv init`, `pyproject.toml`, add dependencies, package skeleton, `music-recommender` entry point | `uv run music-recommender` opens an empty window |
| M2 | **Core pipeline:** `config`, `data`, `evaluation`, `training` (no GUI) | `pytest tests/test_training.py` reproduces the notebook numbers exactly |
| M3 | **Persistence + recommender:** `model_store`, `recommender` | Save → load round-trip test passes; "Nuvole Bianche" returns 10/10 classical |
| M4 | **GUI shell:** main window, menus, status bar, model state, QSettings, workers | Can open a saved model; status bar updates; tabs enable and disable correctly |
| M5 | **Train tab** | Training runs from the GUI with live progress and log; cancel works; model can be saved |
| M6 | **Recommend tab** | Search, recommend, metric switch, double-click to walk, export CSV |
| M7 | **Predict tab** | Sliders → genre probabilities + nearest songs; "Fill from song" works |
| M8 | **Results + Dataset tabs** | All notebook plots and tables shown and exportable |
| M9 | **Polish:** README with screenshots, error handling, GUI smoke test, final pass | All tests pass; offscreen screenshots of every tab look right |

---

## 7. Verification

1. **Numerical parity (automated):** with the default `TrainingConfig`:
   - default accuracy is 0.5756 and tuned accuracy 0.6183 (±1e-4)
   - Precision@10 is 0.4936 → 0.5048
   - best params are `manhattan / 25 / distance`
   - cleaned catalogue has 4,925 songs and the split is 3,940 / 985
2. **Behaviour tests:**
   - unknown song search returns empty, with no crash
   - recommendations never contain the input song
   - `recommend_from_features` on a song's own features returns that song first at distance 0
   - a model saved and loaded again gives identical recommendations
3. **GUI smoke test (offscreen):** build `MainWindow`, inject a trained model, trigger a recommendation and check the table row count. Also run a tiny training job (2 genres, no grid search) through the worker and wait for `finished`.
4. **Manual check:** launch the app, train with defaults, and compare the Results tab against implementation.md §5.7. Then try the sample songs from §5.7.4.

---

## 8. Dependencies (`pyproject.toml`)

```toml
[project]
name = "music-recommender"
requires-python = ">=3.10"
dependencies = ["PyQt6", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "joblib"]

[project.scripts]
music-recommender = "music_recommender.gui.app:main"

[dependency-groups]
dev = ["pytest"]
```

Exact versions are pinned by `uv.lock`.

**Running the app:**
```bash
cd music_recommender_gui
uv sync
uv run music-recommender
```

---

## 9. Optional extras (only if time permits; not in the core milestones)
- "Open in Spotify" link from `track_id` (cheap, planned in M6).
- PCA 2-D map of the catalogue, highlighting the input song and its recommendations. This ties to Unit IV / PR10 and §7 future work item 4.
- Feature weighting sliders for the recommender (§7 future work item 3).

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Grid Search freezes the UI | Runs in a QThread; the custom loop gives fine-grained progress and cancel. |
| Custom grid loop gives different results from `GridSearchCV` | Uses the same `StratifiedKFold` (which `GridSearchCV` uses for classifiers with `cv=5`) and the same tie-breaking (lowest rank, first in grid order); a test compares it against the real `GridSearchCV`. If it differs, fall back to `GridSearchCV` with an indeterminate progress bar. |
| Model file from an older scikit-learn | Store the version and warn on mismatch; retraining takes under a minute. |
| Wayland/Qt platform plugin issues on Arch | PyQt6 wheels bundle Qt; if needed, set `QT_QPA_PLATFORM=xcb`. Documented in the README. |
| Dataset not next to the app | Default path `../dataset.csv`; otherwise the Browse dialog sets it, and the path is remembered in QSettings. |

---

## 11. Open decisions (defaults in bold)

1. **Qt binding:** **PyQt6 in a uv-managed venv**, or the system PyQt5 that is already installed.
2. **Subdirectory name:** **`music_recommender_gui/`**.
3. **Scope:** **all five tabs** (Recommend, Predict Genre, Train, Results, Dataset). The minimum version is Train + Recommend + Results.
