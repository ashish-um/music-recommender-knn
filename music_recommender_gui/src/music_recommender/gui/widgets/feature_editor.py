"""Sliders and spin boxes for editing the audio features of a custom song."""

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QDoubleSpinBox, QGridLayout, QLabel, QSlider, QWidget

from ...core.config import BINARY_FEATURES, FEATURE_RANGES

# Shown in seconds instead of milliseconds.
DISPLAY_SCALE = {"duration_ms": (1000.0, "s", "duration")}

FEATURE_HINTS = {
    "danceability": "How suitable for dancing (steady beat).",
    "energy": "Intensity and activity. Metal is high, soft piano low.",
    "loudness": "Overall loudness in decibels.",
    "speechiness": "Presence of spoken words.",
    "acousticness": "Confidence that the track is acoustic.",
    "instrumentalness": "Likelihood that the track has no vocals.",
    "liveness": "Probability of a live recording.",
    "valence": "Musical positiveness (happy vs sad).",
    "tempo": "Speed in beats per minute.",
    "duration_ms": "Length of the song.",
    "mode": "Checked = major key, unchecked = minor key.",
    "explicit": "Whether the lyrics are explicit.",
    "popularity": "Spotify popularity (0-100).",
    "key": "Pitch class (0 = C, 1 = C#, ...).",
    "time_signature": "Beats per bar.",
}

SLIDER_STEPS = 1000


class _NumberFeature:
    """A slider and spin box kept in sync, working in display units."""

    def __init__(self, name, on_change):
        low, high, step, unit = FEATURE_RANGES[name]
        self.factor, display_unit, _ = DISPLAY_SCALE.get(name, (1.0, unit, name))
        self.low, self.high = low / self.factor, high / self.factor

        self.spin = QDoubleSpinBox()
        self.spin.setRange(self.low, self.high)
        self.spin.setSingleStep(max(step / self.factor, 0.01))
        self.spin.setDecimals(2 if step < 1 else 0)
        if display_unit:
            self.spin.setSuffix(" " + display_unit)
        self.spin.setMinimumWidth(95)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)

        # The spin box rounds (e.g. duration to whole seconds), so keep the exact value that was
        # set from a song until the user changes it. Otherwise a song would not match itself exactly.
        self._exact = None
        self._shown = None

        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self.spin.valueChanged.connect(on_change)

    def _from_slider(self, position):
        value = self.low + (self.high - self.low) * position / SLIDER_STEPS
        if abs(value - self.spin.value()) > (self.high - self.low) / SLIDER_STEPS:
            self.spin.setValue(value)

    def _from_spin(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(round((value - self.low) / (self.high - self.low) * SLIDER_STEPS))
        self.slider.blockSignals(False)

    def value(self) -> float:
        if self._exact is not None and self.spin.value() == self._shown:
            return self._exact
        return self.spin.value() * self.factor

    def set_value(self, value: float):
        self._exact = value
        self.spin.setValue(value / self.factor)
        self._shown = self.spin.value()


class _BinaryFeature:
    def __init__(self, name, on_change):
        self.check = QCheckBox({"mode": "Major key", "explicit": "Explicit lyrics"}.get(name, name))
        self.check.toggled.connect(on_change)

    def value(self) -> float:
        return 1.0 if self.check.isChecked() else 0.0

    def set_value(self, value: float):
        self.check.setChecked(value >= 0.5)


class FeatureEditor(QWidget):
    """Emits valuesChanged(dict) shortly after the user stops moving a control."""

    valuesChanged = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._controls = {}
        self._layout = QGridLayout(self)
        self._layout.setColumnStretch(1, 1)
        self._timer = QTimer(self, singleShot=True, interval=150)
        self._timer.timeout.connect(lambda: self.valuesChanged.emit(self.values()))

    def set_features(self, names):
        """Rebuild the controls for the given feature columns."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._controls = {}

        for row, name in enumerate(names):
            label = QLabel(DISPLAY_SCALE.get(name, (1, "", name))[2])
            label.setToolTip(FEATURE_HINTS.get(name, ""))
            self._layout.addWidget(label, row, 0)
            if name in BINARY_FEATURES:
                control = _BinaryFeature(name, self._changed)
                self._layout.addWidget(control.check, row, 1, 1, 2)
            else:
                control = _NumberFeature(name, self._changed)
                control.slider.setToolTip(FEATURE_HINTS.get(name, ""))
                self._layout.addWidget(control.slider, row, 1)
                self._layout.addWidget(control.spin, row, 2)
            self._controls[name] = control
        self._layout.setRowStretch(len(names), 1)

    def _changed(self, *_):
        self._timer.start()

    def values(self) -> dict:
        return {name: control.value() for name, control in self._controls.items()}

    def set_values(self, values: dict):
        for name, control in self._controls.items():
            if name in values:
                control.set_value(values[name])
        self._timer.start()
