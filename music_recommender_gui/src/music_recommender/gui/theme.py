"""Light workspace styling shared by the desktop app and its plots."""

from pathlib import Path

from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import QApplication

from .widgets.mpl_canvas import apply_palette_to_matplotlib


def apply_theme(app: QApplication):
    app.setStyle("Fusion")
    app.setFont(QFont("DejaVu Sans", 9))
    palette = QPalette()
    for role, color in (
        (QPalette.ColorRole.Window, "#fafbfc"),
        (QPalette.ColorRole.WindowText, "#252830"),
        (QPalette.ColorRole.Base, "#ffffff"),
        (QPalette.ColorRole.AlternateBase, "#f8f9fb"),
        (QPalette.ColorRole.Text, "#252830"),
        (QPalette.ColorRole.Button, "#ffffff"),
        (QPalette.ColorRole.ButtonText, "#252830"),
        (QPalette.ColorRole.Highlight, "#e7edff"),
        (QPalette.ColorRole.HighlightedText, "#304fba"),
        (QPalette.ColorRole.Mid, "#e2e5eb"),
        (QPalette.ColorRole.Light, "#ffffff"),
        (QPalette.ColorRole.Midlight, "#f2f4f7"),
        (QPalette.ColorRole.Dark, "#bcc4d1"),
        (QPalette.ColorRole.Shadow, "#8b95a5"),
        (QPalette.ColorRole.ToolTipBase, "#252830"),
        (QPalette.ColorRole.ToolTipText, "#ffffff"),
        (QPalette.ColorRole.PlaceholderText, "#787e89"),
    ):
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#9298a3"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#9298a3"))
    app.setPalette(palette)
    icons = (Path(__file__).parent / "icons").as_posix()
    app.setStyleSheet("""
        QMainWindow, QDialog { background: #fafbfc; }
        QLabel { background: transparent; }
        QFrame#sidebar { background: #f1f2f4; border-right: 1px solid #e2e5eb; }
        QFrame#brandCard { background: white; border: 1px solid #e2e5eb; border-radius: 10px; }
        QLabel#brandIcon { background: #5265dd; color: white; border-radius: 8px; font-size: 23px; }
        QLabel#brandName { color: #20232b; font-size: 13px; font-weight: 600; }
        QLabel#sectionLabel { color: #69717e; font-size: 11px; padding: 12px 10px 4px; }
        QLabel#sidebarFootnote { color: #69717e; font-size: 11px; padding: 12px 8px; }
        QListWidget#navigation { background: transparent; border: none; outline: none; }
        QListWidget#navigation::item { padding: 11px 12px; margin: 2px 0; border-radius: 7px; color: #505661; }
        QListWidget#navigation::item:hover { background: #e7e9ee; }
        QListWidget#navigation::item:selected { background: #e0e4ee; color: #202738; font-weight: 600; }
        QListWidget#navigation::item:focus { border: 1px solid #5265dd; }
        QFrame#workspaceHeader { background: white; border-bottom: 1px solid #e5e7ec; }
        QLabel#breadcrumb, QLabel#pageSubtitle { color: #68707d; }
        QLabel#pageTitle { color: #20232b; font-size: 24px; font-weight: 600; }
        QLabel#modelBadge { background: #eef2fc; color: #3b55a6; border: 1px solid #dee5f7;
                            border-radius: 7px; padding: 7px 10px; }
        QLabel#aboutCard { background: white; border: 1px solid #e3e6ec; border-radius: 10px;
                           margin: 8px 20px 20px; padding: 24px; }
        QTabWidget#pages::pane { border: none; background: #fafbfc; }
        QGroupBox { background: white; border: 1px solid #e3e6ec; border-radius: 10px;
                    margin-top: 12px; padding: 16px 12px 10px; font-weight: 600; }
        QGroupBox::title { subcontrol-origin: margin; left: 13px; padding: 0 5px; color: #444c59; }
        QWidget#trainSettingsPanel { background: white; }
        QScrollArea#trainSettings { background: white; border: 1px solid #e3e6ec; border-radius: 10px; }
        QWidget#trainPage QGroupBox {
            background: white; border: 1px solid #d5dbe5; margin-top: 16px; padding: 20px 14px 14px;
        }
        QWidget#trainPage QGroupBox::title {
            left: 14px; padding: 0 5px; color: #303a4c; font-size: 14px; font-weight: 600;
        }
        QPlainTextEdit#trainingLog { background: #f1f3f7; border: 1px solid #e3e6ec; border-radius: 8px; }
        QWidget#trainPage QSplitter::handle:vertical { background: #e2e5eb; height: 1px; margin: 3px 0; }
        QWidget#trainPage QSplitter::handle:vertical:hover { background: #dce3f0; }
        QPushButton, QToolButton { background: white; color: #343a46; border: 1px solid #dde1e8;
                                  border-radius: 7px; padding: 7px 12px; }
        QPushButton:hover, QToolButton:hover { background: #f1f4f9; border-color: #c7cdd8; }
        QPushButton:pressed, QToolButton:pressed { background: #e7ecf6; }
        QPushButton:focus, QToolButton:focus { border: 1px solid #5265dd; }
        QPushButton[primary="true"] { background: #252830; color: white; border-color: #252830; font-weight: 600; }
        QPushButton[primary="true"]:hover { background: #3d4350; }
        QPushButton:checked { background: #edf1ff; color: #3653b0; border-color: #b8c6f3; }
        QPushButton:disabled, QToolButton:disabled { background: #f5f6f8; color: #9298a3; border-color: #e6e8ee; }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background: white; border: 1px solid #dde1e8; border-radius: 7px;
            padding: 6px 9px; min-height: 18px; selection-background-color: #e0e8ff; selection-color: #263f91;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #5265dd; }
        QComboBox { padding-right: 22px; }
        QComboBox::drop-down { border: none; width: 22px; }
        QComboBox::down-arrow { image: url("ICONS/chevron-down.svg"); width: 12px; height: 12px; }
        QSpinBox, QDoubleSpinBox { padding-right: 26px; }
        QSpinBox::up-button, QDoubleSpinBox::up-button {
            subcontrol-origin: border; subcontrol-position: top right;
            width: 22px; border-left: 1px solid #dde1e8; border-top-right-radius: 7px;
        }
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            subcontrol-origin: border; subcontrol-position: bottom right;
            width: 22px; border-left: 1px solid #dde1e8; border-bottom-right-radius: 7px;
        }
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
            image: url("ICONS/chevron-up.svg"); width: 12px; height: 12px;
        }
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
            image: url("ICONS/chevron-down.svg"); width: 12px; height: 12px;
        }
        QComboBox QAbstractItemView { background: white; selection-background-color: #e7edff; selection-color: #304fba; }
        QTableView, QListWidget, QPlainTextEdit, QScrollArea {
            background: white; border: 1px solid #e3e6ec; border-radius: 8px;
            selection-background-color: #e7edff; selection-color: #304fba;
        }
        QListWidget::item { padding: 5px 8px; }
        QScrollArea#pageScroll { background: transparent; border: none; border-radius: 0; }
        QListWidget::item:selected { background: #e7edff; color: #304fba; }
        QListWidget#resultNavigation { background: #f4f5f8; border: none; padding: 6px; }
        QListWidget#resultNavigation::item { padding: 9px 6px; border-radius: 5px; }
        QListWidget#resultNavigation::item:selected { background: white; color: #304fba; }
        QTableView { gridline-color: #edf0f5; alternate-background-color: #fafbfd; }
        QTableView::item { padding: 6px; border-bottom: 1px solid #edf0f5; }
        QHeaderView::section { background: #f8f9fb; color: #626b7b; padding: 9px 10px;
                               border: none; border-bottom: 1px solid #e3e6ec; font-weight: 600; }
        QTableCornerButton::section { background: #f8f9fb; border: none; }
        QPlainTextEdit { padding: 10px; color: #596374; }
        QSplitter::handle { background: #eef0f4; }
        QSplitter::handle:horizontal { width: 1px; margin: 6px; }
        QSplitter::handle:vertical { height: 1px; margin: 6px; }
        QScrollBar:vertical { background: #f4f5f8; width: 12px; margin: 2px; }
        QScrollBar:horizontal { background: #f4f5f8; height: 12px; margin: 2px; }
        QScrollBar::handle { background: #aab4c4; border-radius: 3px; min-height: 24px; min-width: 24px; }
        QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
        QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
        QProgressBar { background: #edf0f6; border: none; border-radius: 5px; min-height: 10px; text-align: center; }
        QProgressBar::chunk { background: #5874e7; border-radius: 5px; }
        QSlider::groove:horizontal { background: #e5e9f1; height: 4px; border-radius: 2px; }
        QSlider::sub-page:horizontal { background: #6682eb; border-radius: 2px; }
        QSlider::handle:horizontal { background: white; border: 2px solid #6682eb; width: 12px;
                                      height: 12px; margin: -6px 0; border-radius: 7px; }
        QCheckBox, QRadioButton { spacing: 7px; padding: 3px 0; }
        QCheckBox::indicator, QAbstractItemView::indicator {
            width: 16px; height: 16px; border: 2px solid #64748b;
            border-radius: 3px; background: white;
        }
        QCheckBox::indicator:hover, QAbstractItemView::indicator:hover { border-color: #3653b0; }
        QCheckBox::indicator:checked, QAbstractItemView::indicator:checked {
            background: #4263d5; border-color: #4263d5; image: url("ICONS/check.svg");
        }
        QCheckBox::indicator:disabled, QAbstractItemView::indicator:disabled {
            background: #f0f2f5; border-color: #a8b2c2;
        }
        QCheckBox::indicator:checked:disabled, QAbstractItemView::indicator:checked:disabled {
            background: #8b9bc4; border-color: #8b9bc4;
        }
        QMenuBar { background: white; border-bottom: 1px solid #e5e7ec; padding: 3px 10px; }
        QMenuBar::item { padding: 4px 9px; border-radius: 4px; }
        QMenuBar::item:selected, QMenu::item:selected { background: #e7edff; color: #304fba; }
        QMenu { background: white; border: 1px solid #e0e4eb; padding: 5px; }
        QMenu::item { padding: 7px 24px 7px 12px; }
        QStatusBar { background: white; border-top: 1px solid #e5e7ec; color: #68707d; font-size: 11px; padding: 3px; }
        QStatusBar::item { border: none; }
        QToolTip { background: #252830; color: white; border: none; padding: 6px; }
        QWidget#plotCard { background: white; border: 1px solid #e3e6ec; border-radius: 10px; }
        NavigationToolbar2QT { background: white; border: none; }
        NavigationToolbar2QT QToolButton { border: none; padding: 4px; }
    """.replace("ICONS", icons))
    apply_palette_to_matplotlib(app)
