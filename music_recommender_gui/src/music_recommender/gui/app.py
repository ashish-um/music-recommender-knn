"""Application entry point."""

import sys

from PyQt6.QtWidgets import QApplication

from .theme import apply_theme


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Music Recommender")
    app.setOrganizationName("AIML-MiniProject")
    apply_theme(app)

    # Imported after QApplication exists so Matplotlib picks the Qt backend.
    from .main_window import MainWindow

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
