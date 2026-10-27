"""Streamlit interface for the music recommender."""

from io import BytesIO
import json
from pathlib import Path
import tempfile
import re

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
import seaborn as sns
import streamlit as st
from yt_dlp import YoutubeDL

from music_recommender.core import data, recommender as rec
from music_recommender.core.config import (
    ALL_FEATURES, AUDIO_FEATURES, DEFAULT_FEATURES, DEFAULT_GENRES,
    DEFAULT_DATASET_PATH, MODELS_DIR, TrainingConfig,
)
from music_recommender.core.model_store import default_model_filename, load_model, save_model
from music_recommender.core.training import Reporter, run_training

SETTING_DEFAULTS = {
    "show_distance": True,
    "show_track_id": True,
    "show_same_genre": True,
    "show_recommendation_distance_chart": True,
}
SETTINGS_FILE = Path.home() / ".music-recommender" / "settings.json"


def _load_settings():
    try:
        saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(saved, dict):
        return {}
    return {key: value for key, value in saved.items()
            if key in SETTING_DEFAULTS and isinstance(value, bool)}


def _save_settings():
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps({key: st.session_state[key] for key in SETTING_DEFAULTS}, indent=2),
                                 encoding="utf-8")
        st.session_state.settings_save_error = ""
    except OSError as exc:
        st.session_state.settings_save_error = str(exc)


@st.cache_data(ttl=3600, show_spinner=False)
def _youtube_first_match(query):
    with YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True}) as ydl:
        results = ydl.extract_info(f"ytsearch1:{query}", download=False)
    entry = next(iter(results.get("entries", [])), None)
    if not entry or not re.fullmatch(r"[A-Za-z0-9_-]{11}", entry.get("id", "")):
        return None
    return {"id": entry["id"], "title": entry.get("title", "YouTube result")}


def _select_recommended_song():
    rows = st.session_state[st.session_state.recommendation_table_key].selection.rows
    indices = st.session_state.recommendation_song_indices
    if rows and 0 <= rows[0] < len(indices):
        song_index = indices[rows[0]]
        st.session_state.recommend_song_picker = st.session_state.recommendation_song_labels[song_index]


def _show_results(model):
    results = model.results
    tuned = results.get("tuned", {})
    metrics = tuned.get("metrics", {})
    st.caption(f"Best KNN: K={model.best_params['n_neighbors']}, {model.best_params['weights']} weights, "
               f"{model.best_params['metric']} distance · Test set: {results.get('split', {}).get('test', 0):,} songs")
    cards = st.columns(5)
    for card, name in zip(cards, ("Accuracy", "Precision", "Recall", "F1-Score")):
        card.metric(name, f"{metrics[name]:.1%}" if name in metrics else "—")
    precision_k = model.config.precision_k
    if "precision_at_k" in tuned:
        cards[4].metric(f"Precision@{precision_k}", f"{tuned['precision_at_k']:.1%}")

    summary_tab, classifier_tab, clustering_tab, tuning_tab = st.tabs(
        ["Recommendation comparison", "KNN genre evaluation", "Clustering graphs", "KNN tuning"])
    with summary_tab:
        default_metrics = results.get("default", {}).get("metrics", {})
        knn_rows = [{"Metric": name, "Default KNN": default_metrics[name], "Optimized KNN": metrics[name]}
                    for name in ("Accuracy", "Precision", "Recall", "F1-Score")
                    if name in default_metrics and name in metrics]
        if "precision_at_k" in results.get("default", {}) and "precision_at_k" in tuned:
            knn_rows.append({"Metric": f"Precision@{precision_k}",
                             "Default KNN": results["default"]["precision_at_k"],
                             "Optimized KNN": tuned["precision_at_k"]})
        if knn_rows:
            st.subheader("Default vs optimized KNN")
            st.bar_chart(pd.DataFrame(knn_rows).set_index("Metric"), y_label="Score")

        comparison = results.get("recommendation_comparison")
        if comparison:
            st.subheader("KNN vs K-Means vs hierarchical recommendations")
            rows = comparison["table"].set_index("Model")
            score = f"Precision@{comparison['k']}"
            coverage = f"Coverage@{comparison['k']}"
            st.caption(f"Same {comparison['train_songs']:,} training songs and "
                       f"{comparison['test_songs']:,} held-out test songs for all three methods. "
                       f"{comparison['n_clusters']} clusters; {comparison['metric']} distance for ranking.")
            left, right = st.columns(2)
            left.bar_chart(rows[[score, coverage]], y_label="Fraction")
            right.bar_chart(rows["Query ms/song"], y_label="Milliseconds per test song")
            st.dataframe(rows.style.format("{:.4f}").highlight_max(subset=[score, coverage])
                         .highlight_min(subset=["Fit seconds", "Query ms/song"]), use_container_width=True)
            winners = ", ".join(rows.index[rows[score] == rows[score].max()])
            st.caption(f"Highest {score}: {winners} ({rows[score].max():.1%}). "
                       "Genre is also an input feature, so Precision@K measures genre consistency rather than independent recommendation quality or listening preference. "
                       "Coverage is the fraction of requested recommendations returned; missing slots count as misses. "
                       "Fit time excludes elbow diagnostics; query time includes cluster assignment.")
        else:
            st.info("Retrain this saved model to compare KNN, K-Means and hierarchical recommendations.")

    with classifier_tab:
        st.caption("Accuracy, precision, recall and F1 below describe KNN genre prediction. "
                   "Cluster numbers are not genre labels; recommendation scores are compared in the first tab.")
        classifiers = {"Default KNN": results.get("default"), "Optimized KNN": tuned}
        classifiers = {name: item for name, item in classifiers.items() if item and "confusion_matrix" in item}
        selected = st.selectbox("Classifier", list(classifiers))
        report = classifiers[selected]
        left, right = st.columns([1.2, 1])
        with left:
            fig, ax = plt.subplots(figsize=(max(7, len(model.genre_classes) * .55), 5))
            sns.heatmap(report["confusion_matrix"], annot=len(model.genre_classes) <= 15, fmt="d",
                        cmap="Blues", xticklabels=model.genre_classes,
                        yticklabels=model.genre_classes, ax=ax)
            ax.set(title=f"{selected} confusion matrix", xlabel="Predicted genre", ylabel="Actual genre")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with right:
            st.subheader("Per-genre scores")
            st.dataframe(report["per_class"].sort_values("f1-score"),
                         use_container_width=True, height=420)

    with clustering_tab:
        diagnostics = results.get("clustering")
        if diagnostics:
            st.caption("Clusters use scaled audio features and one-hot genre inputs. K-Means uses Euclidean "
                       "distance; hierarchical clustering uses Ward linkage. New songs are assigned to the "
                       "nearest cluster centroid. Both methods rank songs within the selected cluster.")
            left, right = st.columns(2)
            with left:
                fig, ax = plt.subplots(figsize=(7, 4))
                elbow = diagnostics["elbow"]
                ax.plot(elbow["Clusters"], elbow["Inertia"], marker="o")
                selection = comparison.get("cluster_selection", "manual")
                detected = comparison.get("detected_clusters")
                ax.axvline(comparison["n_clusters"], linestyle="--", color="#ff4b4b",
                           label=f"Used: {comparison['n_clusters']} ({selection})")
                if detected is not None and detected != comparison["n_clusters"]:
                    ax.axvline(detected, linestyle=":", color="#2a9d6f", label=f"Detected elbow: {detected}")
                ax.set(title="K-Means elbow plot", xlabel="Number of clusters", ylabel="Within-cluster squared distance")
                ax.legend()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                if selection == "elbow":
                    st.caption(f"Automatically selected {comparison['n_clusters']} clusters from the largest "
                               "normalized bend in the training inertia curve. Both clustering methods use this count.")
                elif selection == "fallback":
                    st.caption("No elbow could be detected; training used 2 clusters as a fallback.")
                else:
                    st.caption("The entered cluster count overrides elbow detection. "
                               "Leave Number of clusters blank in Train to select it automatically.")
            with right:
                fig, ax = plt.subplots(figsize=(8, 4))
                dendrogram(diagnostics["linkage"], truncate_mode="lastp", p=25,
                           show_leaf_counts=True, ax=ax)
                ax.set(title="Hierarchical dendrogram (last 25 groups)",
                       xlabel="Training songs or merged groups (counts in parentheses)", ylabel="Ward merge distance")
                ax.tick_params(axis="x", rotation=90)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.caption("Branches show groups joining. A larger merge height means more dissimilar groups.")
            st.subheader("Training songs per cluster")
            st.bar_chart(diagnostics["cluster_sizes"].rename_axis("Cluster"), y_label="Songs")
            st.caption("Cluster numbers are arbitrary and do not correspond across the two methods.")
        else:
            st.info("Retrain to generate an elbow plot, dendrogram and cluster sizes.")

    with tuning_tab:
        if "k_sweep" in results:
            sweep = pd.DataFrame(results["k_sweep"], columns=["K neighbours", "Test accuracy"])
            st.subheader("Accuracy by number of neighbours")
            st.line_chart(sweep.set_index("K neighbours"), y_label="Test accuracy")
        if "cv" in results:
            cv = pd.DataFrame({"Fold": range(1, len(results["cv"]["scores"]) + 1),
                               "Accuracy": results["cv"]["scores"]}).set_index("Fold")
            st.subheader(f"{len(cv)}-fold cross-validation")
            st.bar_chart(cv, y_label="Accuracy")
            st.caption(f"Mean accuracy {results['cv']['mean']:.1%} ± {results['cv']['std']:.1%}")
        if "grid" in results:
            st.subheader("Grid search")
            grid = results["grid"]["table"]
            weights = st.selectbox("Voting weights", sorted(grid["weights"].unique()))
            selected_grid = grid[grid["weights"] == weights]
            st.line_chart(selected_grid, x="K", y="mean CV accuracy", color="metric")
            st.dataframe(grid, use_container_width=True)
        if not any(key in results for key in ("k_sweep", "cv", "grid")):
            st.info("No tuning diagnostics were saved with this model.")


def _show_dataset(model):
    catalogue = model.catalogue
    overview_tab, features_tab, catalogue_tab = st.tabs(["Overview", "Audio features", "Catalogue"])
    with overview_tab:
        summary = model.results.get("raw_summary", {})
        cards = st.columns(4)
        cards[0].metric("Songs used", f"{len(catalogue):,}")
        cards[1].metric("Genres", catalogue["track_genre"].nunique())
        cards[2].metric("Recommendation features", len(model.recommendation_feature_cols))
        cards[3].metric("Original rows", f"{summary.get('rows', len(catalogue)):,}")
        left, right = st.columns([1.2, 1])
        with left:
            st.subheader("Songs by genre")
            st.bar_chart(catalogue["track_genre"].value_counts().sort_values(), y_label="Songs")
        with right:
            st.subheader("Cleaning steps")
            cleaning = model.results.get("cleaning")
            if cleaning:
                steps = pd.DataFrame(cleaning.steps, columns=["Step", "Songs"])
                steps["Removed"] = -steps["Songs"].diff().fillna(0).astype(int)
                st.dataframe(steps, hide_index=True, use_container_width=True)
            else:
                st.info("Cleaning details are unavailable in this saved model.")

    with features_tab:
        feature, genre = st.columns(2)
        selected_feature = feature.selectbox("Feature", model.feature_cols)
        genres = sorted(catalogue["track_genre"].unique())
        selected_genre = genre.selectbox("Genre", ["All genres", *genres])
        feature_data = catalogue if selected_genre == "All genres" else catalogue[catalogue["track_genre"] == selected_genre]
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(data=feature_data, x=selected_feature, bins=30, color="#5570e8", ax=ax)
        ax.set(title=f"{selected_feature} distribution · {selected_genre}",
               xlabel=selected_feature, ylabel="Songs")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        if all(name in catalogue for name in AUDIO_FEATURES):
            fig, ax = plt.subplots(figsize=(9, 7))
            sns.heatmap(catalogue[AUDIO_FEATURES].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title("Audio feature correlation")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with catalogue_tab:
        query = st.text_input("Search title or artist")
        genres = sorted(catalogue["track_genre"].unique())
        selected_genres = st.multiselect("Genres", genres, default=genres)
        filtered = catalogue[catalogue["track_genre"].isin(selected_genres)]
        if query:
            mask = filtered["track_name"].str.contains(query, case=False, regex=False) | \
                filtered["artists"].str.contains(query, case=False, regex=False)
            filtered = filtered[mask]
        st.caption(f"Showing {len(filtered):,} of {len(catalogue):,} songs")
        st.dataframe(filtered, use_container_width=True, height=420)
        st.download_button("Download filtered catalogue", filtered.to_csv(index=False),
                           "music_catalogue.csv", "text/csv")


def _show_recommendations(model):
    cat = model.catalogue
    options = rec.song_labels(model)
    if st.session_state.get("recommend_song_picker") not in options:
        st.session_state.recommend_song_picker = options[0]
    left, right = st.columns([1.5, 1])
    with left:
        choice = st.selectbox("Choose a song", options, key="recommend_song_picker")
        index = options.index(choice)
        methods = ["KNN", *getattr(model, "cluster_models", {})]
        method = st.selectbox("Recommendation model", methods)
        if getattr(model, "genre_feature_encoder", None) is not None:
            st.caption("Inputs: scaled audio features and one-hot genre. Different genres add to Euclidean distance.")
        else:
            st.info("This saved model uses audio only. Retrain to include genre in recommendations.")
        if len(methods) == 1:
            st.caption("Retrain to enable K-Means and hierarchical recommendations.")
        col1, col2 = st.columns(2)
        count = col1.slider("Recommendations", 1, 50, 10)
        col2.caption("Distance: Euclidean")
        metric = "euclidean"
        result = rec.recommend(model, index, count, metric, method=method)
        if len(result) < count:
            st.info(f"This cluster has only {len(result)} other songs. Try KNN or fewer clusters for more results.")
        same = int(result["same_genre"].sum())
        st.caption(f"{same} of {len(result)} share the input genre; lower distance means more similar. "
                   "Select a row to use that song as the new input.")
        hidden = [column for column, setting in (("distance", "show_distance"),
                                                  ("track_id", "show_track_id"),
                                                  ("same_genre", "show_same_genre"))
                  if not st.session_state[setting]]
        st.session_state.recommendation_song_indices = result["song_index"].tolist()
        st.session_state.recommendation_song_labels = options
        st.session_state.recommendation_table_key = f"recommended_songs_{method}_{index}"
        st.dataframe(result.drop(columns=["song_index", *hidden]), use_container_width=True,
                     key=st.session_state.recommendation_table_key, on_select=_select_recommended_song,
                     selection_mode="single-row")
        if st.session_state["show_recommendation_distance_chart"]:
            fig, ax = plt.subplots(figsize=(9, 4))
            display = result.sort_values("distance", ascending=True).copy()
            labels = display["track_name"].str.slice(0, 24) + " — " + display["artists"].str.slice(0, 22)
            ax.barh(labels.iloc[::-1], display["distance"].iloc[::-1], color="#5570e8")
            ax.set(title="Recommended songs by distance", xlabel="Distance (lower is more similar)", ylabel="Song")
            ax.grid(axis="x", linestyle="--", alpha=.35)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        st.download_button("Download recommendations as CSV",
                           result.drop(columns="song_index").to_csv(index=False),
                           "recommendations.csv", "text/csv")
    song = cat.iloc[index]
    query = f"{song['track_name'][:100].strip()} {song['artists'][:80].strip()} official audio"
    with right:
        st.subheader("YouTube player")
        try:
            with st.spinner("Searching YouTube…"):
                video = _youtube_first_match(query)
        except Exception as exc:
            st.error(f"YouTube search failed: {exc}")
            return
        if video:
            st.caption(f"First result: {video['title']}")
            st.iframe(f"https://www.youtube.com/embed/{video['id']}", height=360)
        else:
            st.info("No YouTube results found for this song.")


class StreamlitReporter(Reporter):
    def __init__(self, progress, stage, log):
        self.progress_bar, self.stage_text, self.log_box = progress, stage, log
        self.lines = []

    def stage(self, name, percent):
        self.stage_text.write(name)
        self.progress_bar.progress(percent)

    def progress(self, percent, detail=""):
        self.progress_bar.progress(percent, text=detail or None)

    def log(self, text):
        self.lines.append(text)
        self.log_box.code("\n".join(self.lines[-30:]))


@st.dialog("Save trained model")
def _save_trained_model_dialog(model):
    file_path = Path(st.text_input("File name and location", key="model_save_path")).expanduser()
    st.caption("This path is on the computer running the local Streamlit app.")
    overwrite = st.checkbox("Overwrite existing file") if file_path.exists() else True
    save_col, skip_col = st.columns(2)
    if save_col.button("Save model", type="primary"):
        if file_path.suffix.lower() != ".joblib":
            st.error("Choose a file name ending in .joblib.")
        elif file_path.exists() and not overwrite:
            st.warning("Check the overwrite box or choose a different file name.")
        else:
            try:
                saved = save_model(model, file_path)
                st.session_state.saved_model_path = str(saved)
                st.session_state.show_model_save_dialog = False
                st.rerun(scope="app")
            except Exception as exc:
                st.error(f"Could not save model: {exc}")
    if skip_col.button("Not now"):
        st.session_state.show_model_save_dialog = False
        st.rerun(scope="app")


def _show_training():
    st.subheader("Train a model")
    uploaded = st.file_uploader("Dataset CSV", type="csv", key="dataset_upload")
    if uploaded:
        try:
            raw = data.load_dataset(uploaded)
            st.session_state.raw_df = raw
            st.success(f"Loaded {len(raw):,} rows and {raw['track_genre'].nunique()} genres.")
        except Exception as exc:
            st.error(str(exc))
    elif "raw_df" not in st.session_state and DEFAULT_DATASET_PATH.exists():
        try:
            st.session_state.raw_df = data.load_dataset(DEFAULT_DATASET_PATH)
        except Exception as exc:
            st.error(str(exc))
    if "raw_df" not in st.session_state:
        st.info(f"Upload dataset.csv to train. Default location: {DEFAULT_DATASET_PATH}")
        return
    raw = st.session_state.raw_df
    genres = sorted(raw["track_genre"].dropna().unique().tolist())
    chosen_genres = st.multiselect("Genres", genres, default=[g for g in DEFAULT_GENRES if g in genres])
    features = st.multiselect("Audio input features", ALL_FEATURES,
                              default=[f for f in DEFAULT_FEATURES if f in raw.columns])
    st.caption("Genre is automatically added as one-hot input columns for all recommendation methods. "
               "The separate genre classifier uses audio features only.")
    c1, c2, c3 = st.columns(3)
    test_size = c1.slider("Test set size", .05, .5, .2, .05)
    default_k = c2.number_input("Default K", 1, 200, 5)
    precision_k = c3.number_input("Precision@K", 1, 50, 10)
    c4, c5, c6 = st.columns(3)
    run_cv = c4.checkbox("Cross-validation", True)
    run_grid = c5.checkbox("Grid search (90 fits)", True)
    run_sweep = c6.checkbox("Accuracy by K", True)
    n_clusters = st.number_input("Number of clusters", min_value=2, max_value=100,
                                 value=None, placeholder="Automatic (elbow detection)",
                                 key="training_cluster_count",
                                 help="Leave blank to detect the elbow from training data. Enter a value to override it. "
                                      "Both K-Means and Ward hierarchical clustering use the selected count.")
    st.caption("The recommendation comparison uses the same sample of at most 5,000 training and 1,000 test songs "
               "for all methods. Clustering learns from that training sample; other catalogue songs use "
               "nearest-centroid assignment. Genre is included in cluster formation and retrieval; "
               "Precision@K therefore reports genre consistency.")
    if st.button("Train model", type="primary"):
        config = TrainingConfig(dataset_path=str(DEFAULT_DATASET_PATH), genres=chosen_genres,
                                features=features, test_size=test_size, default_k=int(default_k),
                                precision_k=int(precision_k), run_cv=run_cv, run_grid_search=run_grid,
                                run_k_sweep=run_sweep, n_clusters=int(n_clusters) if n_clusters is not None else None)
        problems = config.validate()
        if problems:
            st.error("\n".join(problems))
            return
        progress, stage, log = st.progress(0), st.empty(), st.empty()
        try:
            st.session_state.model = run_training(config, StreamlitReporter(progress, stage, log), raw_df=raw)
            comparison = st.session_state.model.results["recommendation_comparison"]
            st.success(f"Training complete. Using {comparison['n_clusters']} clusters "
                       f"({comparison['cluster_selection']}).")
            st.session_state.model_save_path = str(MODELS_DIR / default_model_filename())
            st.session_state.pop("saved_model_path", None)
            st.session_state.show_model_save_dialog = True
        except Exception as exc:
            st.exception(exc)
    if st.session_state.get("saved_model_path"):
        st.success(f"Model saved to {st.session_state.saved_model_path}")


def main():
    st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="wide")
    if not st.session_state.get("settings_loaded"):
        st.session_state.update(SETTING_DEFAULTS | _load_settings())
        st.session_state.settings_loaded = True
    st.title("🎵 Music Recommender")
    st.caption("Compare KNN, K-Means and hierarchical song recommendations, and explore the catalogue.")
    with st.sidebar:
        st.header("Model")
        models_dir = Path(__file__).resolve().parents[2] / "models"
        model_files = sorted(models_dir.glob("*.joblib"), reverse=True) if models_dir.exists() else []
        selected = st.selectbox("Saved model", ["Choose a file…"] + [str(p) for p in model_files])
        if selected != "Choose a file…" and st.button("Load saved model"):
            try:
                st.session_state.model, notes = load_model(selected)
                for note in notes:
                    st.warning(note)
            except Exception as exc:
                st.error(str(exc))
        uploaded_model = st.file_uploader("Or upload a model file", type="joblib")
        if uploaded_model and st.button("Load uploaded model"):
            st.warning("Model files use pickle and can execute code. Load only files you trust.")
            try:
                with tempfile.NamedTemporaryFile(suffix=".joblib") as temp:
                    temp.write(uploaded_model.getvalue())
                    temp.flush()
                    st.session_state.model, notes = load_model(temp.name)
                for note in notes:
                    st.warning(note)
            except Exception as exc:
                st.error(str(exc))
        model = st.session_state.get("model")
        if model:
            st.success(model.summary())
            stream = BytesIO()
            import joblib
            joblib.dump(model, stream, compress=3)
            st.download_button("Download model", stream.getvalue(), "music_recommender.joblib")

    pages = st.tabs(["About", "Recommend", "Results", "Dataset", "Train", "Settings"])
    with pages[0]:
        st.markdown("**Developed by**")
        team = st.columns(3)
        for column, name in zip(team, ("Ashish Umrao (4315)", "Abhijeet Kumar (4301)", "Aryan Singh (4313)")):
            with column.container(border=True):
                st.markdown(f"### {name}")
        st.header("Music Recommendation System using KNN and Clustering")
        st.write("A mini project comparing KNN, K-Means and hierarchical recommendations using audio features and genre.")
        st.markdown("**Features**\n\n"
                    "- Train KNN, K-Means and hierarchical models and compare recommendation Precision@K.\n"
                    "- Explore recommendations, model metrics, and the cleaned song catalogue.\n"
                    "- Play the first YouTube search match for a selected song.")
        with st.container(border=True):
            st.info("### Checked by: **Prof. Vaishali Ingle**")
    with pages[1]:
        if model:
            _show_recommendations(model)
        else:
            st.info("Load a saved model or train one to start recommending songs.")
    with pages[2]:
        if model:
            _show_results(model)
        else:
            st.info("Load a model to view evaluation graphs.")
    with pages[3]:
        if model:
            _show_dataset(model)
        else:
            st.info("Load a model to explore its cleaned catalogue.")
    with pages[4]:
        _show_training()
    with pages[5]:
        st.subheader("Recommended song columns")
        st.checkbox("Show distance", key="show_distance", help="Lower distances indicate more similar songs.", on_change=_save_settings)
        st.checkbox("Show track ID", key="show_track_id", on_change=_save_settings)
        st.checkbox("Show same genre", key="show_same_genre", on_change=_save_settings)
        st.subheader("Recommendation graphs")
        st.checkbox("Show recommendation distance chart", key="show_recommendation_distance_chart", on_change=_save_settings)
        if st.session_state.get("settings_save_error"):
            st.error(f"Could not save settings: {st.session_state.settings_save_error}")
    if st.session_state.get("show_model_save_dialog") and st.session_state.get("model"):
        _save_trained_model_dialog(st.session_state.model)


if __name__ == "__main__":
    main()
