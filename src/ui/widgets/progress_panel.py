from __future__ import annotations

import time

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QPlainTextEdit, QGroupBox,
)
from PySide6.QtGui import QTextCharFormat, QColor, QFont


_LOG_COLORS = {
    "info":    "#c8c8c8",
    "success": "#6ec26e",
    "warning": "#e6c46e",
    "error":   "#e66e6e",
    "debug":   "#888888",
}


class ProgressPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # --- Progress bar area ---
        bar_box = QGroupBox("Progress")
        bar_layout = QVBoxLayout(bar_box)

        stats_row = QHBoxLayout()
        self._frame_label = QLabel("Frame: 0 / 0")
        self._eta_label = QLabel("ETA: —")
        self._eta_label.setStyleSheet("color: #888;")
        self._elapsed_label = QLabel("Elapsed: 0s")
        self._elapsed_label.setStyleSheet("color: #888;")
        stats_row.addWidget(self._frame_label)
        stats_row.addStretch()
        stats_row.addWidget(self._elapsed_label)
        stats_row.addWidget(self._eta_label)
        bar_layout.addLayout(stats_row)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setStyleSheet(
            "QProgressBar { border: 1px solid #555; border-radius: 3px; text-align: center; height: 18px; }"
            "QProgressBar::chunk { background-color: #0078d4; border-radius: 2px; }"
        )
        bar_layout.addWidget(self._bar)
        layout.addWidget(bar_box)

        # --- Log area ---
        log_box = QGroupBox("Log")
        log_layout = QVBoxLayout(log_box)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(2000)
        self._log.setFont(QFont("Consolas", 9))
        self._log.setStyleSheet(
            "QPlainTextEdit { background: #1a1a1a; color: #c8c8c8; border: none; }"
        )
        log_layout.addWidget(self._log)
        layout.addWidget(log_box)

        self._start_time: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._bar.setValue(0)
        self._frame_label.setText("Frame: 0 / 0")
        self._eta_label.setText("ETA: —")
        self._elapsed_label.setText("Elapsed: 0s")
        self._log.clear()
        self._start_time = time.monotonic()

    @Slot(int, int, float)
    def update_progress(self, current: int, total: int, eta_seconds: float) -> None:
        pct = int(current / max(total, 1) * 100)
        self._bar.setValue(pct)
        self._frame_label.setText(f"Frame: {current} / {total}")

        if eta_seconds >= 0:
            m, s = divmod(int(eta_seconds), 60)
            self._eta_label.setText(f"ETA: {m}m {s:02d}s")
        else:
            self._eta_label.setText("ETA: —")

        if self._start_time is not None:
            elapsed = time.monotonic() - self._start_time
            m, s = divmod(int(elapsed), 60)
            self._elapsed_label.setText(f"Elapsed: {m}m {s:02d}s")

    @Slot(str, str)
    def log(self, message: str, level: str = "info") -> None:
        color = _LOG_COLORS.get(level, _LOG_COLORS["info"])
        timestamp = time.strftime("%H:%M:%S")
        html = f'<span style="color:{color};">[{timestamp}] {message}</span>'
        self._log.appendHtml(html)
        self._log.ensureCursorVisible()
