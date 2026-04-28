import sys

from PyQt5.QtWidgets import QApplication

from app.gui.main_window import MainWindow


def run():
    # Fusion style gives a consistent look across Windows/Linux/macOS.
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
