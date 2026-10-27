"""Recommend tab: pick a song and see the songs that sound most similar."""

import numpy as np
from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtGui import QColor, QDesktopServices, QGuiApplication
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...core import recommender as rec
from ..widgets.dataframe_model import DataFrameModel, make_table_view
from ..widgets.mpl_canvas import PlotWidget
from ..widgets.song_picker import SongPicker

SAME_GENRE_COLOR = QColor(84, 112, 230, 20)
TABLE_COLUMNS = ["rank", "track_name", "artists", "track_genre", "distance", "same_genre"]
HEADERS = {"rank": "#", "track_name": "Track", "artists": "Artist", "track_genre": "Genre",
           "distance": "Distance", "same_genre": "Same genre"}


class RecommendTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.song_index = None
        self.result = None

        self.picker = SongPicker()
        self.picker.songChosen.connect(self.set_input_song)
        self.picker.notFound.connect(lambda text: self.info.setText(
            f"<b>'{text}'</b> is not present in the dataset."))

        self.top_n = QSpinBox(minimum=1, maximum=50, value=10)
        self.top_n.setPrefix("Top ")
        self.metric = QComboBox()
        self.metric.setToolTip("Distance used to measure how similar two songs are.")
        recommend_button = QPushButton("Recommend")
        recommend_button.setProperty("primary", True)
        recommend_button.setDefault(True)
        recommend_button.clicked.connect(self.picker.choose_typed)
        random_button = QPushButton("Random song")
        random_button.clicked.connect(self.random_song)
        self.top_n.valueChanged.connect(self.refresh)
        self.metric.currentIndexChanged.connect(self.refresh)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Song:"))
        search_row.addWidget(self.picker, 1)
        search_row.addWidget(recommend_button)
        search_row.addWidget(random_button)
        options_row = QHBoxLayout()
        options_row.addWidget(self.top_n)
        options_row.addWidget(QLabel("Distance:"))
        options_row.addWidget(self.metric)
        options_row.addStretch()
        top = QVBoxLayout()
        top.addLayout(search_row)
        top.addLayout(options_row)

        self.info = QLabel("Choose a song to get recommendations. "
                           "You can search by title or artist.")
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.info.setWordWrap(True)

        self.table_model = DataFrameModel(headers=HEADERS)
        self.table_model.row_brush = lambda row: SAME_GENRE_COLOR if row.get("same_genre") else None
        self.table = make_table_view(self.table_model, stretch_columns=(1, 2))
        self.table.setMinimumHeight(220)
        self.table.doubleClicked.connect(lambda index: self.use_row_as_input(index.row()))
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.table_menu)
        self.table.setToolTip("Double-click a song to get recommendations for it.")

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        export_button = QPushButton("Export CSV…")
        export_button.clicked.connect(self.export_csv)
        below_table = QHBoxLayout()
        below_table.addWidget(self.summary, 1)
        below_table.addWidget(export_button)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.table)
        left_layout.addLayout(below_table)

        self.plot = PlotWidget()
        self.plot.setMinimumHeight(360)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(left)
        self.splitter.addWidget(self.plot)
        self.splitter.setSizes([620, 480])

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.info)
        layout.addWidget(self.splitter, 1)

    def resizeEvent(self, event):
        self.splitter.setOrientation(Qt.Orientation.Vertical if self.width() < 1100
                                     else Qt.Orientation.Horizontal)
        super().resizeEvent(event)

    def set_model(self, model):
        self.model = model
        self.picker.set_model(model)
        self.metric.blockSignals(True)
        self.metric.clear()
        if model:
            self.metric.addItem("Euclidean distance", "euclidean")
        self.metric.setEnabled(False)
        self.metric.blockSignals(False)
        self.song_index = None
        self.result = None
        self.table_model.set_dataframe(self.table_model.dataframe().iloc[0:0])
        self.summary.clear()
        self.plot.clear()
        self.plot.draw()

    def set_input_song(self, index: int):
        self.song_index = index
        self.picker.set_song(index)
        self.refresh()

    def random_song(self):
        if self.model:
            self.set_input_song(int(np.random.default_rng().integers(self.model.n_songs)))

    def refresh(self):
        if self.model is None or self.song_index is None:
            return
        song = self.model.catalogue.loc[self.song_index]
        self.info.setText(f"Input song: <b>{song['track_name']}</b> &nbsp;|&nbsp; Artist: {song['artists']}"
                          f" &nbsp;|&nbsp; Album: {song['album_name']} &nbsp;|&nbsp; Genre: <b>{song['track_genre']}</b>")
        self.result = rec.recommend(self.model, self.song_index, self.top_n.value(), self.metric.currentData())
        self.table_model.set_dataframe(self.result[TABLE_COLUMNS])
        same = int(self.result["same_genre"].sum())
        self.summary.setText(f"Same genre as the input song: <b>{same} / {len(self.result)}</b>"
                             "  (smaller distance = more similar)")
        self.draw_profile()

    def draw_profile(self):
        """Input song's scaled features next to the average of the recommendations."""
        cols = self.model.recommendation_feature_cols
        song = self.model.catalogue_scaled[self.song_index]
        recs = self.model.catalogue_scaled[self.result["song_index"].to_numpy()].mean(axis=0)
        keep = [i for i in range(len(cols)) if i < len(self.model.feature_cols) or song[i] or recs[i]]
        cols, song, recs = [cols[i] for i in keep], song[keep], recs[keep]
        y = np.arange(len(cols))

        self.plot.clear()
        ax = self.plot.figure.add_subplot()
        ax.barh(y - 0.2, song, height=0.4, label="Input song", color="#5570e8")
        ax.barh(y + 0.2, recs, height=0.4, label=f"Average of top {len(self.result)}", color="#a8b5cf")
        ax.set_yticks(y, cols)
        ax.invert_yaxis()
        ax.set_xlabel("Scaled value")
        ax.set_title("Audio and genre feature profile")
        ax.legend(loc="lower right")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        self.plot.draw()

    def _row(self, view_row: int):
        """Result row for a table row (the table may be sorted differently)."""
        rank = self.table_model.dataframe().iloc[view_row]["rank"]
        return self.result[self.result["rank"] == rank].iloc[0]

    def use_row_as_input(self, view_row: int):
        self.set_input_song(int(self._row(view_row)["song_index"]))

    def table_menu(self, pos):
        index = self.table.indexAt(pos)
        if not index.isValid():
            return
        row = self._row(index.row())
        menu = QMenu(self)
        menu.addAction("Recommend songs like this", lambda: self.use_row_as_input(index.row()))
        menu.addAction("Copy title and artist", lambda: QGuiApplication.clipboard().setText(
            f"{row['track_name']} - {row['artists']}"))
        menu.addAction("Open in Spotify", lambda: QDesktopServices.openUrl(
            QUrl(f"https://open.spotify.com/track/{row['track_id']}")))
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def export_csv(self):
        if self.result is None:
            return
        name = self.model.catalogue.at[self.song_index, "track_name"]
        path, _ = QFileDialog.getSaveFileName(self, "Export recommendations",
                                              f"recommendations_{name}.csv", "CSV files (*.csv)")
        if path:
            self.result.drop(columns=["song_index"]).to_csv(path, index=False)
