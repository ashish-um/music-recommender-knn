"""A Matplotlib figure inside a Qt widget, with the standard toolbar (zoom, save image)."""

import matplotlib
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


def apply_palette_to_matplotlib(app: QApplication):
    """Make plots follow the application's light or dark palette."""
    palette = app.palette()
    base = palette.color(QPalette.ColorRole.Base).name()
    text = palette.color(QPalette.ColorRole.WindowText).name()
    mid = palette.color(QPalette.ColorRole.Mid).name()
    matplotlib.rcParams.update({
        "figure.facecolor": base,
        "axes.facecolor": base,
        "axes.edgecolor": mid,
        "axes.labelcolor": text,
        "axes.titlecolor": text,
        "text.color": text,
        "xtick.color": text,
        "ytick.color": text,
        "legend.facecolor": base,
        "legend.edgecolor": mid,
        "grid.color": mid,
        "savefig.facecolor": "white",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


class PlotWidget(QWidget):
    def __init__(self, parent=None, toolbar=True):
        super().__init__(parent)
        self.setObjectName("plotCard")
        self.setMinimumSize(320, 280)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        if toolbar:
            navigation = NavigationToolbar2QT(self.canvas, self)
            navigation.setIconSize(QSize(16, 16))
            layout.addWidget(navigation)
        layout.addWidget(self.canvas)

    def clear(self):
        self.figure.clear()

    def draw(self):
        self.canvas.draw_idle()

    def message(self, text: str):
        """Show a text message instead of a plot."""
        self.figure.clear()
        self.figure.text(0.5, 0.5, text, ha="center", va="center", wrap=True)
        self.draw()
