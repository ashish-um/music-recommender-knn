"""Builds the real window offscreen and drives each tab."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    # Keep QSettings away from the user's real settings.
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path_factory.mktemp("config"))
    application = QApplication.instance() or QApplication([])
    from music_recommender.gui.theme import apply_theme
    apply_theme(application)
    return application


def wait_until(app, condition, timeout=60):
    import time
    end = time.monotonic() + timeout
    while not condition() and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)
    return condition()


def test_sidebar_navigation(app):
    from music_recommender.gui.main_window import MainWindow

    window = MainWindow(autoload=False)
    window.show()
    app.processEvents()
    assert window.navigation.count() == window.tabs.count() == 5
    for index in range(window.navigation.count()):
        window.navigation.setCurrentRow(index)
        assert window.tabs.currentIndex() == index
        assert window.page_title.text() == window.tabs.tabText(index)
        assert window.page_subtitle.text()
        assert window.navigation.visualItemRect(window.navigation.item(index)).height() >= 40
    window.tabs.setCurrentWidget(window.train_tab)
    assert window.navigation.currentRow() == window.tabs.currentIndex()
    assert "Train" in window.breadcrumb.text()
    window.tabs.setCurrentWidget(window.about_tab)
    assert "Ashish Umrao" in window.about_tab.text()
    assert window.model_badge.text() == "No model loaded"
    window.close()


def test_window_with_model(app, trained_model):
    from music_recommender.gui.main_window import MainWindow

    window = MainWindow(autoload=False)
    assert window.model_pages[0].currentIndex() == 0          # "No model loaded" hint
    window.set_model(trained_model)
    assert all(page.currentIndex() == 1 for page in window.model_pages)
    assert "7 genres" in window.model_label.text()

    # Recommend tab
    tab = window.recommend_tab
    tab.picker.edit.setText("nuvole bianche")
    tab.picker.choose_typed()
    assert tab.table_model.rowCount() == 10
    assert "10 / 10" in tab.summary.text()
    tab.top_n.setValue(20)
    assert tab.table_model.rowCount() == 20
    tab.metric.setCurrentIndex(tab.metric.findData("euclidean"))
    assert tab.table_model.rowCount() == 20
    first = int(tab.result["song_index"].iloc[0])
    tab.use_row_as_input(0)
    assert tab.song_index == first
    tab.picker.edit.setText("Some Unknown Song zzz")
    tab.picker.choose_typed()
    assert "not present" in tab.info.text()

    # Results and Dataset tabs draw every view
    assert window.results_tab.summary_model.rowCount() == 6
    comparison = window.results_tab.classifier_model.dataframe()
    assert window.results_tab.views.currentItem().text() == "Model comparison"
    assert len(comparison) == 3
    assert comparison["Precision@10"].is_monotonic_decreasing
    winners = comparison.loc[comparison["Precision@10"] == comparison["Precision@10"].max(), "Model"]
    assert all(name in window.results_tab.classifier_note.text() for name in winners)
    from PyQt6.QtWidgets import QRadioButton
    for button in window.results_tab.findChildren(QRadioButton):
        button.setChecked(True)
    assert window.results_tab.report_model.rowCount() == 7
    for row in range(window.results_tab.views.count()):
        window.results_tab.views.setCurrentRow(row)
        app.processEvents()
    for choice in range(window.data_tab.plot_choice.count()):
        window.data_tab.plot_choice.setCurrentIndex(choice)
    assert window.data_tab.songs_model.rowCount() == 4925
    window.close()


def test_compact_layouts_keep_controls_readable(app, trained_model):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QBoxLayout, QScrollArea, QStyle, QStyleOptionSpinBox
    from music_recommender.gui.main_window import MainWindow

    window = MainWindow(autoload=False)
    window.resize(1100, 720)
    window.set_model(trained_model)
    window.show()
    page = window.recommend_tab
    wrapper = window.model_pages[0]
    window.tabs.setCurrentWidget(wrapper)
    app.processEvents()
    assert page.splitter.orientation() == Qt.Orientation.Vertical
    scroll = wrapper.currentWidget()
    assert isinstance(scroll, QScrollArea)
    assert scroll.horizontalScrollBar().maximum() == 0
    assert page.plot.height() >= 280
    assert page.picker.edit.width() >= 250

    window.tabs.setCurrentWidget(window.model_pages[2])
    app.processEvents()
    assert window.data_tab.top_splitter.orientation() == Qt.Orientation.Vertical
    assert window.model_pages[2].currentWidget().horizontalScrollBar().maximum() == 0

    window.tabs.setCurrentWidget(window.train_tab)
    app.processEvents()
    assert window.train_tab.settings_layout.direction() == QBoxLayout.Direction.TopToBottom
    settings_scroll = window.train_tab.settings_panel.parentWidget().parentWidget()
    assert settings_scroll.horizontalScrollBar().maximum() == 0
    spin = window.train_tab.default_k
    settings_scroll.ensureWidgetVisible(spin)
    app.processEvents()
    option = QStyleOptionSpinBox()
    spin.initStyleOption(option)
    up = spin.style().subControlRect(QStyle.ComplexControl.CC_SpinBox, option,
                                    QStyle.SubControl.SC_SpinBoxUp, spin)
    before = spin.value()
    QTest.mouseClick(spin, Qt.MouseButton.LeftButton, pos=up.center())
    assert spin.value() == before + 1

    window.resize(1440, 900)
    window.tabs.setCurrentWidget(window.model_pages[0])
    app.processEvents()
    assert window.recommend_tab.splitter.orientation() == Qt.Orientation.Horizontal
    window.close()


def test_older_model_has_no_comparison(app, trained_model):
    from dataclasses import replace
    from music_recommender.gui.tabs.results_tab import ResultsTab
    from PyQt6.QtWidgets import QRadioButton

    old_results = {k: v for k, v in trained_model.results.items() if k != "recommendation_comparison"}
    tab = ResultsTab()
    tab.set_model(trained_model)
    tab.set_model(replace(trained_model, results=old_results))
    assert tab.classifier_model.rowCount() == 0
    assert tab.views.currentItem().text() == "Summary"
    assert "Retrain" in tab.classifier_note.text()
    assert tab.report_model.rowCount() == 7
    tab.close()


def test_train_tab_runs_training(app, raw_df):
    from music_recommender.core.config import DEFAULT_DATASET_PATH, TrainingConfig
    from music_recommender.gui.tabs.train_tab import TrainTab

    tab = TrainTab()
    tab.set_dataset(raw_df, str(DEFAULT_DATASET_PATH))
    tab.set_config(TrainingConfig(genres=["classical", "heavy-metal", "jazz"], run_k_sweep=False,
                                  param_grid={"n_neighbors": [5, 15], "weights": ["distance"],
                                              "metric": ["euclidean"]}))
    assert "2 combinations" in tab.grid_size.text()
    results = []
    tab.trainingFinished.connect(results.append)
    tab.start_training()
    assert wait_until(app, lambda: not tab.is_running())
    assert len(results) == 1
    assert results[0].genre_classes == ["classical", "heavy-metal", "jazz"]
    assert "Best Parameters" in tab.log.toPlainText()
    assert tab.progress.value() == 100


def test_train_tab_cancel(app, raw_df):
    from music_recommender.core.config import DEFAULT_DATASET_PATH
    from music_recommender.gui.tabs.train_tab import TrainTab

    tab = TrainTab()
    tab.set_dataset(raw_df, str(DEFAULT_DATASET_PATH))
    finished = []
    tab.trainingFinished.connect(finished.append)
    tab.start_training()
    tab.cancel_training()
    assert wait_until(app, lambda: not tab.is_running())
    assert finished == []
    assert "cancelled" in tab.stage_label.text()
