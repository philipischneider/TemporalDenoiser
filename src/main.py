import sys
from pathlib import Path

# Make sure src/ is on the path when running directly
sys.path.insert(0, str(Path(__file__).parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt
from ui.main_window import MainWindow


def apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    dark = QColor(43, 43, 43)
    mid_dark = QColor(60, 63, 65)
    mid = QColor(75, 78, 80)
    light = QColor(100, 104, 107)
    highlight = QColor(0, 120, 212)
    text = QColor(220, 220, 220)
    disabled_text = QColor(120, 120, 120)
    palette.setColor(QPalette.Window, dark)
    palette.setColor(QPalette.WindowText, text)
    palette.setColor(QPalette.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.AlternateBase, mid_dark)
    palette.setColor(QPalette.ToolTipBase, mid_dark)
    palette.setColor(QPalette.ToolTipText, text)
    palette.setColor(QPalette.Text, text)
    palette.setColor(QPalette.Button, mid_dark)
    palette.setColor(QPalette.ButtonText, text)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, highlight)
    palette.setColor(QPalette.Highlight, highlight)
    palette.setColor(QPalette.HighlightedText, Qt.white)
    palette.setColor(QPalette.Disabled, QPalette.Text, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, disabled_text)
    palette.setColor(QPalette.Mid, mid)
    palette.setColor(QPalette.Dark, QColor(25, 25, 25))
    palette.setColor(QPalette.Shadow, QColor(15, 15, 15))
    app.setPalette(palette)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("TemporalDenoiser")
    app.setApplicationDisplayName("Temporal Denoiser")
    apply_dark_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
