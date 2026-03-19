from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QPixmap, QImage, QIcon, QAction
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QVBoxLayout,
    QHBoxLayout, QLabel, QStatusBar, QToolBar,
    QFileDialog, QMessageBox, QSizePolicy,
)
import numpy as np

from ui.widgets.settings_panel import SettingsPanel
from ui.widgets.progress_panel import ProgressPanel
from core.pipeline import DenoisePipeline, PipelineConfig, PipelineWorker
from core.exr_handler import list_exr_layers
from utils.frame_sequence import detect_frame_sequence


class PreviewLabel(QLabel):
    """Displays a single denoised/noisy frame scaled to fit."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #444;")
        self.setText("No preview available")

    def set_image_array(self, array: np.ndarray) -> None:
        """Accept float32 (H, W, 3) in linear space and display as sRGB."""
        if array is None:
            self.setText("No preview available")
            return
        # Tonemap: simple gamma 2.2 for display
        display = np.clip(array ** (1.0 / 2.2), 0.0, 1.0)
        display = (display * 255).astype(np.uint8)
        h, w, _ = display.shape
        img = QImage(display.data, w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.setPixmap(
            pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Temporal Denoiser")
        self.resize(1400, 840)

        self._worker: PipelineWorker | None = None
        self._thread: QThread | None = None

        self._build_ui()
        self._build_toolbar()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter, stretch=1)

        # Left: settings
        self.settings_panel = SettingsPanel()
        self.settings_panel.setMinimumWidth(300)
        self.settings_panel.setMaximumWidth(380)
        splitter.addWidget(self.settings_panel)

        # Center + bottom: preview above, log below
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # Preview header with mode toggle
        preview_header = QHBoxLayout()
        preview_label_title = QLabel("Preview")
        preview_label_title.setStyleSheet("font-weight: bold; color: #aaa;")
        preview_header.addWidget(preview_label_title)
        preview_header.addStretch()

        self.frame_info_label = QLabel("Frame: —")
        self.frame_info_label.setStyleSheet("color: #888;")
        preview_header.addWidget(self.frame_info_label)
        right_layout.addLayout(preview_header)

        self.preview = PreviewLabel()
        right_layout.addWidget(self.preview, stretch=3)

        self.progress_panel = ProgressPanel()
        right_layout.addWidget(self.progress_panel, stretch=1)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready.")

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setStyleSheet("QToolBar { spacing: 6px; padding: 4px; }")
        self.addToolBar(toolbar)

        self.action_start = QAction("▶  Start", self)
        self.action_start.setToolTip("Start denoising sequence")
        toolbar.addAction(self.action_start)

        self.action_stop = QAction("■  Stop", self)
        self.action_stop.setToolTip("Stop denoising")
        self.action_stop.setEnabled(False)
        toolbar.addAction(self.action_stop)

        toolbar.addSeparator()

        action_open_input = QAction("📂 Input Folder", self)
        action_open_input.setToolTip("Select input folder with EXR frames")
        action_open_input.triggered.connect(self._browse_input)
        toolbar.addAction(action_open_input)

        action_open_output = QAction("💾 Output Folder", self)
        action_open_output.setToolTip("Select output folder")
        action_open_output.triggered.connect(self._browse_output)
        toolbar.addAction(action_open_output)

    def _connect_signals(self) -> None:
        self.action_start.triggered.connect(self._on_start)
        self.action_stop.triggered.connect(self._on_stop)
        self.settings_panel.input_changed.connect(self._on_input_changed)

    # ------------------------------------------------------------------
    # Slots / helpers
    # ------------------------------------------------------------------

    @Slot()
    def _browse_input(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if path:
            self.settings_panel.set_input_folder(path)

    @Slot()
    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.settings_panel.set_output_folder(path)

    @Slot(str)
    def _on_input_changed(self, path: str) -> None:
        folder = Path(path)
        if not folder.is_dir():
            return

        frames = detect_frame_sequence(folder)
        if not frames:
            self.status_bar.showMessage(f"Input: {path} — no EXR files found.")
            return

        self.status_bar.showMessage(f"Input: {path} — {len(frames)} frames. Scanning passes…")
        try:
            layers = list_exr_layers(frames[0])
            self.settings_panel.populate_passes(layers)
            self.status_bar.showMessage(
                f"Input: {path} — {len(frames)} frames, {len(layers)} passes detected."
            )
        except Exception as e:
            self.status_bar.showMessage(f"Could not read EXR passes: {e}")

    @Slot()
    def _on_start(self) -> None:
        config = self.settings_panel.build_config()
        if config is None:
            return

        errors = config.validate()
        if errors:
            QMessageBox.warning(self, "Configuration Error", "\n".join(errors))
            return

        self._start_pipeline(config)

    @Slot()
    def _on_stop(self) -> None:
        if self._worker:
            self._worker.request_stop()
        self.action_stop.setEnabled(False)

    def _start_pipeline(self, config: PipelineConfig) -> None:
        self.action_start.setEnabled(False)
        self.action_stop.setEnabled(True)
        self.progress_panel.reset()
        self.progress_panel.log("Starting denoising pipeline…", "info")

        self._thread = QThread(self)
        self._worker = PipelineWorker(config)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_pipeline_finished)
        self._worker.error.connect(self._on_pipeline_error)
        self._worker.progress.connect(self.progress_panel.update_progress)
        self._worker.log_message.connect(self.progress_panel.log)
        self._worker.frame_ready.connect(self._on_frame_ready)

        self._thread.start()

    @Slot(int, np.ndarray)
    def _on_frame_ready(self, frame_num: int, image: np.ndarray) -> None:
        self.frame_info_label.setText(f"Frame: {frame_num:04d}")
        self.preview.set_image_array(image)

    @Slot()
    def _on_pipeline_finished(self) -> None:
        self._cleanup_worker()
        self.action_start.setEnabled(True)
        self.action_stop.setEnabled(False)
        self.status_bar.showMessage("Denoising complete.")
        self.progress_panel.log("Done.", "success")

    @Slot(str)
    def _on_pipeline_error(self, message: str) -> None:
        self._cleanup_worker()
        self.action_start.setEnabled(True)
        self.action_stop.setEnabled(False)
        self.progress_panel.log(f"ERROR: {message}", "error")
        self.status_bar.showMessage("Error during denoising.")
        QMessageBox.critical(self, "Pipeline Error", message)

    def _cleanup_worker(self) -> None:
        if self._thread:
            self._thread.quit()
            self._thread.wait(5000)
        self._worker = None
        self._thread = None
