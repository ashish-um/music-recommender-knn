"""Predict Genre tab: describe a song with audio features, see its genre and similar songs."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...core import recommender as rec
from ..widgets.dataframe_model import DataFrameModel, make_table_view
from ..widgets.feature_editor import FeatureEditor
from ..widgets.mpl_canvas import PlotWidget
from ..widgets.song_picker import SongPicker

HEADERS = {"rank": "#", "track_name": "Track", "artists": "Artist", "track_genre": "Genre",
           "distance": "Distance"}


class PredictTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None

        # Left: feature controls
        self.picker = SongPicker(placeholder="Start from an existing song…")
        self.picker.songChosen.connect(self.fill_from_song)
        average_button = QPushButton("Reset to average")
        average_button.setToolTip("Set every feature to the average of all songs.")
        average_button.clicked.connect(self.reset_to_average)
        fill_row = QHBoxLayout()
        fill_row.addWidget(self.picker, 1)
        fill_row.addWidget(average_button)

        self.editor = FeatureEditor()
        self.editor.valuesChanged.connect(self.update_prediction)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.editor)
        scroll.setMinimumHeight(340)

        features_box = QGroupBox("Audio features of your song")
        features_layout = QVBoxLayout(features_box)
        features_layout.addLayout(fill_row)
        features_layout.addWidget(scroll)

        # Right: prediction and similar songs
        self.classifier = QComboBox()
        self.classifier.addItem("Optimized KNN", "tuned")
        self.classifier.addItem("Default KNN", "default")
        self.classifier.currentIndexChanged.connect(lambda: self.update_prediction(self.editor.values()))
        self.predicted = QLabel("–")
        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.predicted.setFont(font)
        head = QHBoxLayout()
        head.addWidget(QLabel("Predicted genre:"))
        head.addWidget(self.predicted, 1)
        head.addWidget(QLabel("Classifier:"))
        head.addWidget(self.classifier)

        self.plot = PlotWidget(toolbar=False)

        self.table_model = DataFrameModel(headers=HEADERS)
        table = make_table_view(self.table_model, stretch_columns=(1, 2))
        table.setMinimumHeight(220)
        similar_box = QGroupBox("Songs that sound most like this")
        similar_layout = QVBoxLayout(similar_box)
        similar_layout.addWidget(table)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addLayout(head)
        right_layout.addWidget(self.plot, 2)
        right_layout.addWidget(similar_box, 3)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(features_box)
        self.splitter.addWidget(right)
        self.splitter.setSizes([480, 620])

        # Toggle between exploring "what if" values and only looking up existing songs.
        self.edit_toggle = QPushButton()
        self.edit_toggle.setCheckable(True)
        self.edit_toggle.setChecked(True)
        self.edit_toggle.toggled.connect(self.set_editing)
        self.edit_hint = QLabel()
        self.edit_hint.setWordWrap(True)
        toggle_row = QHBoxLayout()
        toggle_row.addWidget(self.edit_toggle)
        toggle_row.addWidget(self.edit_hint, 1)
        self.set_editing(True)

        layout = QVBoxLayout(self)
        layout.addLayout(toggle_row)
        layout.addWidget(self.splitter)

    def resizeEvent(self, event):
        self.splitter.setOrientation(Qt.Orientation.Vertical if self.width() < 1100
                                     else Qt.Orientation.Horizontal)
        super().resizeEvent(event)

    def set_model(self, model):
        self.model = model
        self.picker.set_model(model)
        if model:
            self.editor.set_features(model.feature_cols)
            self.reset_to_average()

    def set_editing(self, enabled: bool):
        """Enable or lock the feature sliders. Picking a song still fills in its values."""
        self.edit_toggle.setChecked(enabled)
        self.editor.setEnabled(enabled)
        if enabled:
            self.edit_toggle.setText("🔓  Editing: ON")
            self.edit_hint.setText("Move the sliders to describe any song (\"what if?\"). "
                                   "This does not change the dataset or the model.")
        else:
            self.edit_toggle.setText("🔒  Editing: OFF")
            self.edit_hint.setText("Sliders are locked: pick a song to see its real features, "
                                   "predicted genre and similar songs.")

    def reset_to_average(self):
        if self.model:
            self.editor.set_values(self.model.catalogue[self.model.feature_cols].mean().to_dict())

    def fill_from_song(self, index: int):
        self.editor.set_values(rec.song_features(self.model, index))

    def update_prediction(self, values: dict):
        if self.model is None or set(values) != set(self.model.feature_cols):
            return
        genre, probabilities = rec.predict_genre(self.model, values, self.classifier.currentData())
        self.predicted.setText(genre)

        self.plot.clear()
        ax = self.plot.figure.add_subplot()
        genres = sorted(probabilities, key=probabilities.get)
        colors = ["#5570e8" if g == genre else "#a8b5cf" for g in genres]
        ax.barh(genres, [probabilities[g] for g in genres], color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability (share of neighbour votes)")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        self.plot.draw()

        similar = rec.recommend_from_features(self.model, values, n=10)
        self.table_model.set_dataframe(similar[["rank", "track_name", "artists", "track_genre", "distance"]])
