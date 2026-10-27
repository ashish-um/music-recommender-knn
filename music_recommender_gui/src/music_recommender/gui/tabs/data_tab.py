"""Dataset tab: cleaning report, exploratory plots and the songs used by the current model."""

import pandas as pd
import seaborn as sns
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...core.config import ALL_FEATURES, AUDIO_FEATURES, INFO_COLUMNS
from ..widgets.dataframe_model import DataFrameModel, make_table_view
from ..widgets.mpl_canvas import PlotWidget

PLOTS = ["Songs per genre", "Histograms of audio features", "Boxplot by genre",
         "Boxplot of song duration", "Correlation heatmap", "Average features by genre"]


class DataTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None

        self.summary = QLabel()
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.summary.setWordWrap(True)
        self.cleaning_model = DataFrameModel()
        cleaning_box = QGroupBox("Cleaning steps")
        cleaning_layout = QVBoxLayout(cleaning_box)
        cleaning_table = make_table_view(self.cleaning_model, stretch_columns=(0,))
        cleaning_table.setMinimumHeight(220)
        cleaning_layout.addWidget(cleaning_table)

        left = QWidget()
        left.setMinimumWidth(390)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.summary)
        left_layout.addWidget(cleaning_box, 1)

        # Plots
        self.plot_choice = QComboBox()
        self.plot_choice.addItems(PLOTS)
        self.box_feature = QComboBox()
        self.box_feature.addItems(AUDIO_FEATURES + ["duration_ms"])
        self.box_feature.setCurrentText("energy")
        self.plot_choice.currentIndexChanged.connect(self.draw_plot)
        self.box_feature.currentIndexChanged.connect(self.draw_plot)
        plot_row = QHBoxLayout()
        plot_row.addWidget(QLabel("Plot:"))
        plot_row.addWidget(self.plot_choice)
        plot_row.addWidget(QLabel("Feature:"))
        plot_row.addWidget(self.box_feature)
        plot_row.addStretch()
        self.plot = PlotWidget()
        self.plot.setMinimumHeight(360)
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addLayout(plot_row)
        plot_layout.addWidget(self.plot, 1)

        # Song table
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter songs by title, artist or genre…")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self.filter_songs)
        self.songs_model = DataFrameModel(float_format="{:.3f}")
        self.songs_count = QLabel()
        songs_box = QGroupBox("Songs in the model's catalogue (after cleaning)")
        songs_layout = QVBoxLayout(songs_box)
        filter_row = QHBoxLayout()
        filter_row.addWidget(self.filter, 1)
        filter_row.addWidget(self.songs_count)
        songs_layout.addLayout(filter_row)
        songs_table = make_table_view(self.songs_model, sortable=True)
        songs_table.setMinimumHeight(180)
        songs_layout.addWidget(songs_table)

        self.top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.top_splitter.setChildrenCollapsible(False)
        self.top_splitter.addWidget(left)
        self.top_splitter.addWidget(plot_panel)
        self.top_splitter.setSizes([390, 710])
        main = QSplitter(Qt.Orientation.Vertical)
        main.setChildrenCollapsible(False)
        main.addWidget(self.top_splitter)
        main.addWidget(songs_box)
        main.setSizes([480, 260])

        layout = QVBoxLayout(self)
        layout.addWidget(main)

    def resizeEvent(self, event):
        self.top_splitter.setOrientation(Qt.Orientation.Vertical if self.width() < 1100
                                         else Qt.Orientation.Horizontal)
        super().resizeEvent(event)

    def set_model(self, model):
        self.model = model
        if model is None:
            return
        raw = model.results["raw_summary"]
        cleaning = model.results["cleaning"]
        split = model.results["split"]
        self.summary.setText(
            f"<b>Original dataset:</b> {raw['rows']:,} songs, {raw['columns']} columns, "
            f"{raw['genres']} genres, {raw['missing']} missing values.<br>"
            f"<b>Used by this model:</b> {model.n_songs:,} songs in {len(model.genre_classes)} genres, "
            f"{len(model.recommendation_feature_cols)} input features.<br>"
            f"<b>Split:</b> {split['train']:,} training / {split['test']:,} testing songs."
        )
        steps = pd.DataFrame(cleaning.steps, columns=["Step", "Songs"])
        steps["Removed"] = (-steps["Songs"].diff().fillna(0)).astype(int)
        self.cleaning_model.set_dataframe(steps)
        self.filter_songs(self.filter.text())
        self.draw_plot()

    def filter_songs(self, text):
        if self.model is None:
            return
        cat = self.model.catalogue[INFO_COLUMNS[1:] + ALL_FEATURES]
        text = text.strip().lower()
        if text:
            mask = (cat["track_name"].str.lower().str.contains(text, regex=False)
                    | cat["artists"].str.lower().str.contains(text, regex=False)
                    | cat["track_genre"].str.lower().str.contains(text, regex=False))
            cat = cat[mask]
        self.songs_model.set_dataframe(cat)
        self.songs_count.setText(f"{len(cat):,} songs")

    def draw_plot(self):
        if self.model is None:
            return
        df = self.model.catalogue
        choice = self.plot_choice.currentText()
        self.box_feature.setEnabled(choice == "Boxplot by genre")
        fig = self.plot.figure
        self.plot.clear()
        order = df["track_genre"].value_counts().index

        if choice == "Songs per genre":
            ax = fig.add_subplot()
            counts = df["track_genre"].value_counts()
            bars = ax.bar(counts.index, counts.values, color="#5570e8")
            ax.bar_label(bars, fontsize=8)
            ax.set_title("Number of Songs per Genre")
            ax.set_xlabel("Genre")
            ax.set_ylabel("Number of Songs")
            ax.tick_params(axis="x", rotation=45)
        elif choice == "Histograms of audio features":
            axes = fig.subplots(3, 3)
            for ax, feature in zip(axes.flat, AUDIO_FEATURES):
                ax.hist(df[feature], bins=30, color="#5570e8")
                ax.set_title(feature)
            fig.suptitle("Histograms of Audio Features")
        elif choice == "Boxplot by genre":
            feature = self.box_feature.currentText()
            ax = fig.add_subplot()
            sns.boxplot(data=df, x="track_genre", y=feature, order=order, ax=ax)
            ax.set_title(f"{feature.capitalize()} by Genre")
            ax.set_xlabel("Genre")
            ax.tick_params(axis="x", rotation=45)
        elif choice == "Boxplot of song duration":
            ax = fig.add_subplot()
            sns.boxplot(x=df["duration_ms"], ax=ax)
            ax.set_title("Boxplot of Song Duration (ms)")
        elif choice == "Correlation heatmap":
            ax = fig.add_subplot()
            sns.heatmap(df[AUDIO_FEATURES].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                        annot_kws={"fontsize": 8})
            ax.set_title("Correlation between Audio Features")
        elif choice == "Average features by genre":
            ax = fig.add_subplot()
            features = [f for f in AUDIO_FEATURES if f not in ("loudness", "tempo")]
            means = df.groupby("track_genre")[features].mean()
            sns.heatmap(means, annot=True, fmt=".2f", cmap="viridis", ax=ax, annot_kws={"fontsize": 8})
            ax.set_title("Average audio features by genre (0 to 1 features)")
            ax.set_ylabel("Genre")
        self.plot.draw()
