"""Train tab: every training setting as a control, a Train button, progress and the log."""

import os
import time

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (
    QCheckBox,
    QBoxLayout,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...core.config import (
    ALL_FEATURES,
    DEFAULT_FEATURES,
    DEFAULT_GENRES,
    METRICS,
    WEIGHTS,
    TrainingConfig,
)
from ..workers import TrainingWorker, start_worker


def _checked_items(widget: QListWidget) -> list[str]:
    return [widget.item(i).data(Qt.ItemDataRole.UserRole) for i in range(widget.count())
            if widget.item(i).checkState() == Qt.CheckState.Checked]


def _set_checked(widget: QListWidget, names):
    names = set(names)
    for i in range(widget.count()):
        item = widget.item(i)
        item.setCheckState(Qt.CheckState.Checked if item.data(Qt.ItemDataRole.UserRole) in names
                           else Qt.CheckState.Unchecked)


def _parse_int_list(text: str) -> list[int]:
    values = sorted({int(part) for part in text.replace(";", ",").split(",") if part.strip()})
    if any(v < 1 for v in values):
        raise ValueError("K values must be positive")
    return values


class TrainTab(QWidget):
    trainingFinished = pyqtSignal(object)            # TrainedModel
    loadDatasetRequested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("trainPage")
        self.raw_df = None
        self.raw_path = None
        self.worker = None
        self.started_at = 0.0

        self.settings_panel = QWidget()
        self.settings_panel.setObjectName("trainSettingsPanel")
        self.settings_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        settings_layout = self.settings_layout = QBoxLayout(QBoxLayout.Direction.LeftToRight, self.settings_panel)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(24)
        settings_layout.addWidget(self._build_data_box(), 3)
        right = QVBoxLayout()
        right.setSpacing(18)
        right.addWidget(self._build_model_box())
        right.addWidget(self._build_optimization_box())
        right.addStretch()
        settings_layout.addLayout(right, 2)

        # Buttons and progress
        defaults_button = QPushButton("Restore notebook defaults")
        defaults_button.clicked.connect(lambda: self.set_config(TrainingConfig(dataset_path=self.path_edit.text())))
        self.train_button = QPushButton("▶  Train model")
        self.train_button.setProperty("primary", True)
        self.train_button.setDefault(True)
        self.train_button.clicked.connect(self.start_training)
        self.cancel_button = QPushButton("■  Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_training)
        buttons = QHBoxLayout()
        buttons.addWidget(defaults_button)
        buttons.addStretch()
        buttons.addWidget(self.train_button)
        buttons.addWidget(self.cancel_button)

        self.stage_label = QLabel("Ready.")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.log = QPlainTextEdit()
        self.log.setObjectName("trainingLog")
        self.log.setReadOnly(True)
        self.log.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.log.setPlaceholderText("Training progress and evaluation results will appear here.")

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(12)
        settings_scroll = QScrollArea()
        settings_scroll.setObjectName("trainSettings")
        settings_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setWidget(self.settings_panel)
        top_layout.addWidget(settings_scroll, 1)
        top_layout.addLayout(buttons)
        top_layout.addWidget(self.stage_label)
        top_layout.addWidget(self.progress)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top)
        log_panel = QWidget()
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(QLabel("<b>Training log</b>"))
        log_layout.addWidget(self.log)
        splitter.addWidget(log_panel)
        splitter.setSizes([620, 120])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        self._fill_genres({g: None for g in DEFAULT_GENRES})
        self.set_config(TrainingConfig())

    def resizeEvent(self, event):
        self.settings_layout.setDirection(QBoxLayout.Direction.TopToBottom if self.width() < 1100
                                          else QBoxLayout.Direction.LeftToRight)
        super().resizeEvent(event)

    # ---------- building the settings panels ----------

    def _build_data_box(self):
        box = QGroupBox("Dataset")

        self.path_edit = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.browse_dataset)
        load = QPushButton("Load")
        load.clicked.connect(lambda: self.loadDatasetRequested.emit(self.path_edit.text()))
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Dataset:"))
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse)
        path_row.addWidget(load)
        self.dataset_status = QLabel("Dataset not loaded yet.")

        # Genres
        self.genre_filter = QLineEdit()
        self.genre_filter.setPlaceholderText("Filter genres…")
        self.genre_filter.setClearButtonEnabled(True)
        self.genre_filter.textChanged.connect(self._filter_genres)
        self.genre_list = QListWidget()
        self.genre_list.setMinimumHeight(156)
        self.genre_list.itemChanged.connect(self._update_genre_count)
        self.genre_count = QLabel()
        genre_buttons = QHBoxLayout()
        for text, action in (("Notebook 7", lambda: _set_checked(self.genre_list, DEFAULT_GENRES)),
                             ("None", lambda: _set_checked(self.genre_list, []))):
            button = QPushButton(text)
            button.clicked.connect(action)
            genre_buttons.addWidget(button)
        genre_buttons.addStretch()
        genre_buttons.addWidget(self.genre_count)
        genres = QVBoxLayout()
        genres.addWidget(QLabel("<b>Genres</b> (more genres = harder problem)"))
        genres.addWidget(self.genre_filter)
        genres.addWidget(self.genre_list, 1)
        genres.addLayout(genre_buttons)

        # Features
        self.feature_list = QListWidget()
        self.feature_list.setMinimumHeight(156)
        for name in ALL_FEATURES:
            item = QListWidgetItem(name if name in DEFAULT_FEATURES else f"{name}  (not in notebook)")
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            self.feature_list.addItem(item)
        features = QVBoxLayout()
        features.addWidget(QLabel("<b>Input features</b>"))
        features.addWidget(self.feature_list, 1)

        lists = QHBoxLayout()
        lists.addLayout(genres, 3)
        lists.addLayout(features, 2)

        # Split
        self.test_size = QDoubleSpinBox(minimum=0.05, maximum=0.5, singleStep=0.05, decimals=2)
        self.random_state = QSpinBox(minimum=0, maximum=1_000_000)
        self.stratify = QCheckBox("Stratify by genre")
        split_row = QHBoxLayout()
        split_row.addWidget(QLabel("Test size:"))
        split_row.addWidget(self.test_size)
        split_row.addWidget(QLabel("Random state:"))
        split_row.addWidget(self.random_state)
        split_row.addWidget(self.stratify)
        split_row.addStretch()

        layout = QVBoxLayout(box)
        layout.addLayout(path_row)
        layout.addWidget(self.dataset_status)
        layout.addLayout(lists, 1)
        layout.addLayout(split_row)
        return box

    def _build_model_box(self):
        box = QGroupBox("Model settings")
        self.scaler = QComboBox()
        self.scaler.addItem("Min-Max (0 to 1)", "minmax")
        self.scaler.addItem("Standard (mean 0, std 1)", "standard")
        self.default_k = QSpinBox(minimum=1, maximum=200)
        self.precision_k = QSpinBox(minimum=1, maximum=50)
        self.k_sweep = QCheckBox("Accuracy vs K, for K = 1 to")
        self.k_sweep_max = QSpinBox(minimum=2, maximum=100)
        self.k_sweep.toggled.connect(self.k_sweep_max.setEnabled)
        sweep_row = QHBoxLayout()
        sweep_row.addWidget(self.k_sweep)
        sweep_row.addWidget(self.k_sweep_max)
        sweep_row.addStretch()

        form = QFormLayout(box)
        form.addRow("Feature scaling:", self.scaler)
        form.addRow("Default K (neighbours):", self.default_k)
        form.addRow("Precision@K, K =", self.precision_k)
        form.addRow(sweep_row)
        return box

    def _build_optimization_box(self):
        box = QGroupBox("Optimization")
        self.run_cv = QCheckBox("Cross validation of the default model")
        self.cv_folds = QSpinBox(minimum=2, maximum=20)
        self.cv_folds.setSuffix(" folds")
        self.run_grid = QCheckBox("Grid search (GridSearchCV)")
        self.k_values = QLineEdit()
        self.k_values.setToolTip("Comma-separated values of K to try.")
        self.weight_checks = {w: QCheckBox(w) for w in WEIGHTS}
        self.metric_checks = {m: QCheckBox(m) for m in METRICS}
        self.grid_size = QLabel()
        self.n_jobs = QComboBox()
        self.n_jobs.addItem("All cores", -1)
        for n in (1, 2, 4, 8):
            if n <= (os.cpu_count() or 1):
                self.n_jobs.addItem(str(n), n)

        layout = QGridLayout(box)
        layout.addWidget(self.run_cv, 0, 0, 1, 3)
        layout.addWidget(QLabel("CV folds:"), 0, 3)
        layout.addWidget(self.cv_folds, 0, 4)
        layout.addWidget(self.run_grid, 1, 0, 1, 5)
        layout.addWidget(QLabel("K values:"), 2, 0)
        layout.addWidget(self.k_values, 2, 1, 1, 4)
        layout.addWidget(QLabel("Weights:"), 3, 0)
        for col, check in enumerate(self.weight_checks.values(), start=1):
            layout.addWidget(check, 3, col)
        layout.addWidget(QLabel("Metric:"), 4, 0)
        for col, check in enumerate(self.metric_checks.values(), start=1):
            layout.addWidget(check, 4, col)
        layout.addWidget(self.grid_size, 5, 0, 1, 5)
        layout.addWidget(QLabel("CPU cores:"), 6, 0)
        layout.addWidget(self.n_jobs, 6, 1, 1, 2)

        grid_controls = [self.k_values, *self.weight_checks.values(), *self.metric_checks.values()]
        self.run_grid.toggled.connect(lambda on: [w.setEnabled(on) for w in grid_controls])
        for widget in (self.run_grid, *self.weight_checks.values(), *self.metric_checks.values()):
            widget.toggled.connect(self._update_grid_size)
        self.k_values.textChanged.connect(self._update_grid_size)
        self.cv_folds.valueChanged.connect(self._update_grid_size)
        return box

    # ---------- genres ----------

    def _fill_genres(self, counts: dict):
        """counts: genre -> number of songs (None if unknown)."""
        checked = _checked_items(self.genre_list) or DEFAULT_GENRES
        self.genre_list.blockSignals(True)
        self.genre_list.clear()
        for genre in sorted(counts):
            text = genre if counts[genre] is None else f"{genre}  ({counts[genre]})"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, genre)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            self.genre_list.addItem(item)
        self.genre_list.blockSignals(False)
        _set_checked(self.genre_list, checked)
        self._filter_genres(self.genre_filter.text())

    def _filter_genres(self, text):
        text = text.lower()
        for i in range(self.genre_list.count()):
            item = self.genre_list.item(i)
            item.setHidden(text not in item.data(Qt.ItemDataRole.UserRole))

    def _update_genre_count(self, *_):
        self.genre_count.setText(f"{len(_checked_items(self.genre_list))} selected")

    def _update_grid_size(self, *_):
        if not self.run_grid.isChecked():
            self.grid_size.setText("Grid search off: the default model is used as the final model.")
            return
        try:
            k_count = len(_parse_int_list(self.k_values.text()))
        except ValueError:
            self.grid_size.setText("<span style='color:#c0392b'>K values must be whole numbers, "
                                   "separated by commas.</span>")
            return
        weights = sum(c.isChecked() for c in self.weight_checks.values())
        metrics = sum(c.isChecked() for c in self.metric_checks.values())
        combos = k_count * weights * metrics
        self.grid_size.setText(f"→ {k_count} × {weights} × {metrics} = <b>{combos} combinations</b>, "
                               f"{combos * self.cv_folds.value()} fits")

    # ---------- dataset ----------

    def browse_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open dataset", self.path_edit.text(), "CSV files (*.csv)")
        if path:
            self.path_edit.setText(path)
            self.loadDatasetRequested.emit(path)

    def set_dataset_loading(self, path):
        self.path_edit.setText(path)
        self.dataset_status.setText(f"Loading {os.path.basename(path)}…")

    def set_dataset(self, df, path):
        self.raw_df, self.raw_path = df, path
        self.path_edit.setText(path)
        self.dataset_status.setText(f"✓ Loaded: {len(df):,} songs, {df.shape[1]} columns, "
                                    f"{df['track_genre'].nunique()} genres.")
        self._fill_genres(df["track_genre"].value_counts().to_dict())

    def set_dataset_failed(self, message):
        self.raw_df = self.raw_path = None
        self.dataset_status.setText(f"<span style='color:#c0392b'>✗ {message}</span>")

    # ---------- config <-> controls ----------

    def set_config(self, config: TrainingConfig):
        self.path_edit.setText(config.dataset_path)
        _set_checked(self.genre_list, config.genres)
        _set_checked(self.feature_list, config.features)
        self.test_size.setValue(config.test_size)
        self.random_state.setValue(config.random_state)
        self.stratify.setChecked(config.stratify)
        self.scaler.setCurrentIndex(self.scaler.findData(config.scaler))
        self.default_k.setValue(config.default_k)
        self.precision_k.setValue(config.precision_k)
        self.k_sweep.setChecked(config.run_k_sweep)
        self.k_sweep_max.setValue(config.k_sweep_max)
        self.run_cv.setChecked(config.run_cv)
        self.cv_folds.setValue(config.cv_folds)
        self.run_grid.setChecked(config.run_grid_search)
        self.k_values.setText(", ".join(str(k) for k in config.param_grid["n_neighbors"]))
        for name, check in self.weight_checks.items():
            check.setChecked(name in config.param_grid["weights"])
        for name, check in self.metric_checks.items():
            check.setChecked(name in config.param_grid["metric"])
        self.n_jobs.setCurrentIndex(max(0, self.n_jobs.findData(config.n_jobs)))
        self._update_genre_count()
        self._update_grid_size()

    def config(self) -> TrainingConfig:
        return TrainingConfig(
            dataset_path=self.path_edit.text().strip(),
            genres=_checked_items(self.genre_list),
            features=_checked_items(self.feature_list),
            test_size=self.test_size.value(),
            random_state=self.random_state.value(),
            stratify=self.stratify.isChecked(),
            scaler=self.scaler.currentData(),
            default_k=self.default_k.value(),
            precision_k=self.precision_k.value(),
            run_k_sweep=self.k_sweep.isChecked(),
            k_sweep_max=self.k_sweep_max.value(),
            run_cv=self.run_cv.isChecked(),
            cv_folds=self.cv_folds.value(),
            run_grid_search=self.run_grid.isChecked(),
            param_grid={
                "n_neighbors": _parse_int_list(self.k_values.text()),
                "weights": [w for w, c in self.weight_checks.items() if c.isChecked()],
                "metric": [m for m, c in self.metric_checks.items() if c.isChecked()],
            },
            n_jobs=self.n_jobs.currentData(),
        )

    # ---------- training ----------

    def start_training(self):
        try:
            config = self.config()
        except ValueError:
            QMessageBox.warning(self, "Check settings", "K values must be whole numbers separated by commas.")
            return
        problems = config.validate()
        if problems:
            QMessageBox.warning(self, "Check settings", "\n".join(problems))
            return

        # Reuse the dataset already in memory if it is the same file.
        raw_df = self.raw_df if self.raw_path and os.path.abspath(self.raw_path) == os.path.abspath(
            config.dataset_path) else None

        self.log.clear()
        self.progress.setValue(0)
        self._set_running(True)
        self.started_at = time.monotonic()

        self.worker = TrainingWorker(config, raw_df)
        self.worker.stageChanged.connect(self._on_stage)
        self.worker.progressChanged.connect(self._on_progress)
        self.worker.logLine.connect(self.log.appendPlainText)
        self.worker.trained.connect(self._on_trained)
        self.worker.failed.connect(self._on_failed)
        self.worker.cancelled.connect(self._on_cancelled)
        thread = start_worker(self.worker, self)
        thread.finished.connect(self._on_thread_finished)

    def cancel_training(self):
        if self.worker:
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.stage_label.setText(self.stage_label.text() + "  — cancelling…")

    def is_running(self) -> bool:
        return self.worker is not None

    def _set_running(self, running):
        self.settings_panel.setEnabled(not running)
        self.train_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def _on_stage(self, name, percent):
        self.stage_label.setText(f"Stage: {name}")
        self.progress.setValue(percent)

    def _on_progress(self, percent, detail):
        self.progress.setValue(percent)
        stage = self.stage_label.text().split(" (")[0]
        self.stage_label.setText(f"{stage} ({detail})")

    def _on_trained(self, model):
        seconds = time.monotonic() - self.started_at
        self.stage_label.setText(f"✓ Training finished in {seconds:.1f} s.")
        self.log.appendPlainText(f"\nTraining finished in {seconds:.1f} s.")
        self.trainingFinished.emit(model)

    def _on_failed(self, message):
        self.stage_label.setText("✗ Training failed.")
        self.log.appendPlainText(f"\nERROR: {message}")
        QMessageBox.critical(self, "Training failed", message)

    def _on_cancelled(self):
        self.stage_label.setText("Training cancelled.")
        self.progress.setValue(0)
        self.log.appendPlainText("\nTraining cancelled.")

    def _on_thread_finished(self):
        self.worker = None
        self._set_running(False)
