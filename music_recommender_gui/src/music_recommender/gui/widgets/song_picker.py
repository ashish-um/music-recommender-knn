"""Search box with autocomplete over "Title — Artist (genre)"."""

from PyQt6.QtCore import QStringListModel, Qt, pyqtSignal
from PyQt6.QtWidgets import QCompleter, QHBoxLayout, QLineEdit, QWidget

from ...core import recommender as rec


class SongPicker(QWidget):
    songChosen = pyqtSignal(int)        # catalogue index
    notFound = pyqtSignal(str)

    def __init__(self, parent=None, placeholder="Type a song title or artist…"):
        super().__init__(parent)
        self.model = None
        self._label_to_index = {}

        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.setClearButtonEnabled(True)
        self._list = QStringListModel(self)
        self.completer = QCompleter(self._list, self)
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.completer.setMaxVisibleItems(15)
        self.edit.setCompleter(self.completer)

        self.completer.activated[str].connect(self._choose_label)
        self.edit.returnPressed.connect(self.choose_typed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit)

    def set_model(self, model):
        self.model = model
        labels = rec.song_labels(model) if model else []
        self._label_to_index = {label: i for i, label in enumerate(labels)}
        self._list.setStringList(labels)

    def _choose_label(self, label):
        index = self._label_to_index.get(label)
        if index is not None:
            self.songChosen.emit(index)

    def choose_typed(self):
        """Use the typed text: an exact label, otherwise the best search match."""
        text = self.edit.text().strip()
        if not text or self.model is None:
            return
        if text in self._label_to_index:
            self._choose_label(text)
            return
        matches = rec.search_songs(self.model, text, limit=1)
        if matches.empty:
            self.notFound.emit(text)
            return
        index = int(matches.index[0])
        self.set_song(index)
        self.songChosen.emit(index)

    def set_song(self, index: int):
        """Show a song in the box without emitting songChosen."""
        label = rec.song_label(self.model.catalogue.loc[index])
        self.edit.blockSignals(True)
        self.edit.setText(label)
        self.edit.blockSignals(False)
