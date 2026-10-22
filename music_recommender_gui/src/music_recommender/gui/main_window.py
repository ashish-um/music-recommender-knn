"""Main window: tabs, menus, status bar, and the currently loaded model."""

import json
from dataclasses import asdict
from pathlib import Path

from PyQt6.QtCore import QSettings, QSize, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.config import DEFAULT_DATASET_PATH, MODELS_DIR
from ..core.model_store import ModelFileError, default_model_filename, load_model, save_model
from .tabs.data_tab import DataTab
from .tabs.predict_tab import PredictTab
from .tabs.recommend_tab import RecommendTab
from .tabs.results_tab import ResultsTab
from .tabs.train_tab import TrainTab
from .workers import DatasetLoadWorker, start_worker

MAX_RECENT = 6
MODEL_FILTER = "Model files (*.joblib)"
ABOUT_TEXT = (
    "<h2>Music Recommendation System</h2>"
    "<p>Mini Project: Artificial Intelligence &amp; Machine Learning (PCC-302-ITT)<br>"
    "Savitribai Phule Pune University, TE Information Technology (2024 Course)</p>"
    "<p>Content-based music recommendations using KNN on the Spotify Tracks Dataset,<br>"
    "with KNN, Decision Tree and Naïve Bayes genre classification.</p>"
    "<h3>Developed by</h3>"
    "<p>Ashish Umrao<br>Abhijeet Kumar<br>Aryan Singh</p>"
    "<p>Checked by: वैशाली इंगळे</p>"
)
PAGE_DESCRIPTIONS = {
    "Recommend": "Find songs with a similar sound. Start with a title or an artist.",
    "Predict Genre": "Explore audio features and discover the genre behind a sound.",
    "Train": "Choose your data, configure the models and follow their progress.",
    "Results": "Compare classifiers and explore how your models perform.",
    "Dataset": "Get to know your catalogue, from cleaning steps to audio features.",
    "About": "The project and the people behind it.",
}


class NeedsModel(QStackedWidget):
    """Shows a hint instead of the page until a model has been trained or opened."""

    def __init__(self, page, window: "MainWindow"):
        super().__init__()
        self.page = page
        hint = QWidget()
        layout = QVBoxLayout(hint)
        layout.addStretch()
        text = QLabel("<h3>No model loaded</h3>Train a model on the <b>Train</b> tab, "
                      "or open a model you saved earlier.")
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(text)
        buttons = QHBoxLayout()
        buttons.addStretch()
        train = QPushButton("Go to Train tab")
        train.clicked.connect(lambda: window.tabs.setCurrentWidget(window.train_tab))
        open_button = QPushButton("Open model…")
        open_button.clicked.connect(window.open_model_dialog)
        buttons.addWidget(train)
        buttons.addWidget(open_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        layout.addStretch()
        self.addWidget(hint)
        scroll = QScrollArea()
        scroll.setObjectName("pageScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(page)
        self.addWidget(scroll)

    def set_model(self, model):
        self.page.set_model(model)
        self.setCurrentIndex(1 if model else 0)


class MainWindow(QMainWindow):
    modelChanged = pyqtSignal(object)

    def __init__(self, autoload=True):
        super().__init__()
        self.setWindowTitle("Music Recommender")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 720)
        self.settings = QSettings("AIML-MiniProject", "MusicRecommender")
        self.model = None
        self.model_path = None
        self.dataset_worker = None

        self.recommend_tab = RecommendTab()
        self.predict_tab = PredictTab()
        self.train_tab = TrainTab()
        self.results_tab = ResultsTab()
        self.data_tab = DataTab()

        self.tabs = QTabWidget()
        self.tabs.setObjectName("pages")
        self.model_pages = []
        for page, title in ((self.recommend_tab, "Recommend"), (self.predict_tab, "Predict Genre")):
            self._add_model_page(page, title)
        self.tabs.addTab(self.train_tab, "Train")
        for page, title in ((self.results_tab, "Results"), (self.data_tab, "Dataset")):
            self._add_model_page(page, title)
        self.about_tab = QLabel(ABOUT_TEXT)
        self.about_tab.setObjectName("aboutCard")
        self.about_tab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.about_tab.setWordWrap(True)
        self.about_tab.setMargin(24)
        self.about_tab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.tabs.addTab(self.about_tab, "About")
        self.tabs.tabBar().hide()
        for page in (self.recommend_tab, self.predict_tab, self.train_tab, self.results_tab, self.data_tab):
            page.layout().setContentsMargins(20, 8, 20, 20)
            page.layout().setSpacing(14)
        self._build_workspace()

        self.train_tab.trainingFinished.connect(self.on_training_finished)
        self.train_tab.loadDatasetRequested.connect(self.load_dataset)

        self.model_label = QLabel()
        self.statusBar().addPermanentWidget(self.model_label)
        self._build_menus()
        self.set_model(None)

        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

        if autoload:
            self._autoload()

    def _add_model_page(self, page, title):
        wrapper = NeedsModel(page, self)
        self.model_pages.append(wrapper)
        self.tabs.addTab(wrapper, title)

    def _build_workspace(self):
        shell = QWidget()
        layout = QHBoxLayout(shell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(224)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(12, 16, 12, 12)
        sidebar_layout.setSpacing(8)
        brand = QFrame()
        brand.setObjectName("brandCard")
        brand_layout = QHBoxLayout(brand)
        brand_layout.setContentsMargins(10, 10, 10, 10)
        icon = QLabel("♫")
        icon.setObjectName("brandIcon")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setFixedSize(32, 32)
        brand_name = QLabel("Music Recommender")
        brand_name.setWordWrap(True)
        brand_name.setObjectName("brandName")
        brand_layout.addWidget(icon)
        brand_layout.addWidget(brand_name, 1)
        sidebar_layout.addWidget(brand)
        section = QLabel("WORKSPACE")
        section.setObjectName("sectionLabel")
        sidebar_layout.addWidget(section)
        self.navigation = QListWidget()
        self.navigation.setObjectName("navigation")
        self.navigation.setAccessibleName("Main navigation")
        self.navigation.addItems([f"{symbol}   {self.tabs.tabText(i)}"
                                  for i, symbol in enumerate(("♫", "≋", "⚙", "▥", "▤", "ⓘ"))])
        for index in range(self.navigation.count()):
            self.navigation.item(index).setSizeHint(QSize(180, 42))
        sidebar_layout.addWidget(self.navigation, 1)
        footnote = QLabel("AIML MINI PROJECT\nSpotify Tracks Dataset")
        footnote.setObjectName("sidebarFootnote")
        sidebar_layout.addWidget(footnote)
        layout.addWidget(sidebar)

        content = QVBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)
        header = QFrame()
        header.setObjectName("workspaceHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 12, 24, 12)
        self.breadcrumb = QLabel()
        self.breadcrumb.setObjectName("breadcrumb")
        header_layout.addWidget(self.breadcrumb, 1)
        open_button = QPushButton("Open model…")
        open_button.setProperty("primary", True)
        open_button.clicked.connect(self.open_model_dialog)
        header_layout.addWidget(open_button)
        content.addWidget(header)
        heading = QHBoxLayout()
        heading.setContentsMargins(24, 22, 24, 18)
        titles = QVBoxLayout()
        self.page_title = QLabel()
        self.page_title.setObjectName("pageTitle")
        self.page_subtitle = QLabel()
        self.page_subtitle.setObjectName("pageSubtitle")
        self.page_subtitle.setWordWrap(True)
        titles.addWidget(self.page_title)
        titles.addWidget(self.page_subtitle)
        heading.addLayout(titles, 1)
        self.model_badge = QLabel("No model loaded")
        self.model_badge.setObjectName("modelBadge")
        heading.addWidget(self.model_badge)
        content.addLayout(heading)
        content.addWidget(self.tabs, 1)
        layout.addLayout(content, 1)
        self.setCentralWidget(shell)
        self.navigation.currentRowChanged.connect(self.tabs.setCurrentIndex)
        self.tabs.currentChanged.connect(self._sync_navigation)
        self._sync_navigation(self.tabs.currentIndex())

    def _sync_navigation(self, index):
        self.navigation.setCurrentRow(index)
        title = self.tabs.tabText(index)
        self.breadcrumb.setText(f"Workspace  /  {title}")
        self.page_title.setText(title)
        self.page_subtitle.setText(PAGE_DESCRIPTIONS[title])

    # ---------- menus ----------

    def _build_menus(self):
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction("Open &Dataset…", self.train_tab.browse_dataset)
        file_menu.addSeparator()
        quit_action = file_menu.addAction("&Quit", self.close)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)

        model_menu = self.menuBar().addMenu("&Model")
        open_action = model_menu.addAction("&Open Model…", self.open_model_dialog)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.save_action = model_menu.addAction("&Save Model As…", self.save_model_dialog)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.recent_menu = model_menu.addMenu("Open &Recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        model_menu.addSeparator()
        self.info_action = model_menu.addAction("Model &Info", self.show_model_info)
        self.reuse_action = model_menu.addAction("Use This Model's Settings for Training",
                                                 self.reuse_model_settings)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction("&About", self.show_about)

    def _fill_recent_menu(self):
        self.recent_menu.clear()
        recent = [p for p in self._recent_models() if Path(p).exists()]
        if not recent:
            self.recent_menu.addAction("(none)").setEnabled(False)
        for path in recent:
            action = QAction(Path(path).name, self.recent_menu)
            action.setToolTip(path)
            action.triggered.connect(lambda _=False, p=path: self.open_model(p))
            self.recent_menu.addAction(action)

    def _recent_models(self) -> list[str]:
        value = self.settings.value("recent_models", [])
        return [value] if isinstance(value, str) else list(value or [])

    def _remember_model_path(self, path):
        path = str(Path(path).resolve())
        recent = [path] + [p for p in self._recent_models() if p != path]
        self.settings.setValue("recent_models", recent[:MAX_RECENT])
        self.settings.setValue("last_model", path)

    # ---------- model state ----------

    def set_model(self, model, path=None):
        self.model = model
        self.model_path = str(path) if path else None
        for page in self.model_pages:
            page.set_model(model)
        for action in (self.save_action, self.info_action, self.reuse_action):
            action.setEnabled(model is not None)
        if model is None:
            self.model_label.setText("No model loaded")
            self.model_badge.setText("No model loaded")
        else:
            name = Path(path).name if path else "unsaved model"
            self.model_label.setText(f"Model: {name} · {model.summary()}")
            self.model_badge.setText(f"{model.n_songs:,} songs  ·  {len(model.genre_classes)} genres")
        self.modelChanged.emit(model)

    def open_model_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open model (only open model files you trust)", str(MODELS_DIR), MODEL_FILTER)
        if path:
            self.open_model(path)

    def open_model(self, path, quiet=False) -> bool:
        try:
            model, notes = load_model(path)
        except ModelFileError as exc:
            if not quiet:
                QMessageBox.critical(self, "Could not open model", str(exc))
            return False
        self.set_model(model, path)
        self._remember_model_path(path)
        for note in notes:
            self.statusBar().showMessage(note, 10000)
        if not quiet:
            self.tabs.setCurrentWidget(self.model_pages[0])
        return True

    def save_model_dialog(self) -> bool:
        if self.model is None:
            return False
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        suggested = self.model_path or str(MODELS_DIR / default_model_filename())
        path, _ = QFileDialog.getSaveFileName(self, "Save model", suggested, MODEL_FILTER)
        if not path:
            return False
        if not path.endswith(".joblib"):
            path += ".joblib"
        try:
            save_model(self.model, path)
        except OSError as exc:
            QMessageBox.critical(self, "Could not save model", str(exc))
            return False
        self.model_path = path
        self._remember_model_path(path)
        self.model_label.setText(f"Model: {Path(path).name} · {self.model.summary()}")
        self.statusBar().showMessage(f"Model saved to {path}", 6000)
        return True

    def on_training_finished(self, model):
        self.set_model(model)
        self.tabs.setCurrentWidget(self.model_pages[2])     # Results
        answer = QMessageBox.question(
            self, "Training finished",
            f"Training finished: {model.summary()}.\n\nSave this model so it can be reused later?")
        if answer == QMessageBox.StandardButton.Yes:
            self.save_model_dialog()

    def show_model_info(self):
        if self.model is None:
            return
        config = asdict(self.model.config)
        text = (f"<b>File:</b> {self.model_path or '(not saved)'}<br>"
                f"<b>Trained:</b> {self.model.created_at.replace('T', ' ')} with scikit-learn "
                f"{self.model.sklearn_version}<br>"
                f"<b>Songs:</b> {self.model.n_songs:,} &nbsp; <b>Genres:</b> {', '.join(self.model.genre_classes)}<br>"
                f"<b>Features:</b> {', '.join(self.model.feature_cols)}<br>"
                f"<b>Best parameters:</b> {self.model.best_params}<br>"
                f"<b>Recommender metric:</b> {self.model.recommender_metric}")
        box = QMessageBox(QMessageBox.Icon.Information, "Model info", text, parent=self)
        box.setDetailedText("Training settings:\n" + json.dumps(config, indent=2))
        box.exec()

    def reuse_model_settings(self):
        if self.model:
            self.train_tab.set_config(self.model.config)
            self.tabs.setCurrentWidget(self.train_tab)

    def show_about(self):
        QMessageBox.about(self, "About Music Recommender", ABOUT_TEXT)

    # ---------- dataset ----------

    def load_dataset(self, path):
        if self.dataset_worker is not None:
            return
        self.train_tab.set_dataset_loading(path)
        self.dataset_worker = DatasetLoadWorker(path)
        self.dataset_worker.loaded.connect(self._on_dataset_loaded)
        self.dataset_worker.failed.connect(self.train_tab.set_dataset_failed)
        thread = start_worker(self.dataset_worker, self)
        thread.finished.connect(self._on_dataset_thread_finished)

    def _on_dataset_loaded(self, df, path):
        self.train_tab.set_dataset(df, path)
        self.settings.setValue("dataset_path", path)

    def _on_dataset_thread_finished(self):
        self.dataset_worker = None

    # ---------- startup and shutdown ----------

    def _autoload(self):
        last_model = self.settings.value("last_model")
        if last_model and Path(last_model).exists() and self.open_model(last_model, quiet=True):
            self.statusBar().showMessage(f"Opened last model {Path(last_model).name}", 5000)
        else:
            self.tabs.setCurrentWidget(self.train_tab)

        dataset = self.settings.value("dataset_path") or str(DEFAULT_DATASET_PATH)
        if Path(dataset).exists():
            self.load_dataset(dataset)
        else:
            self.train_tab.path_edit.setText(dataset)

    def closeEvent(self, event):
        if self.train_tab.is_running():
            answer = QMessageBox.question(self, "Training in progress",
                                          "Training is still running. Cancel it and quit?")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.train_tab.cancel_training()
            # Wait for the worker thread to stop before the window is destroyed.
            for thread in self.train_tab.findChildren(QThread):
                thread.wait(10000)
        self.settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)
