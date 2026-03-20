from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QComboBox, QCheckBox, QSlider,
    QLabel, QGroupBox, QHBoxLayout, QScrollArea,
    QFrame, QFileDialog,
)
from PySide6.QtCore import Qt

from core.pipeline import PipelineConfig


class _FolderPicker(QWidget):
    changed = Signal(str)

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._edit = QLineEdit()
        self._edit.setPlaceholderText(f"Select {label}…")
        self._edit.textChanged.connect(self.changed)
        layout.addWidget(self._edit)

        btn = QPushButton("…")
        btn.setFixedWidth(32)
        btn.clicked.connect(self._browse)
        layout.addWidget(btn)

        self._label = label

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, f"Select {self._label}")
        if path:
            self._edit.setText(path)

    def value(self) -> str:
        return self._edit.text().strip()

    def set_value(self, v: str) -> None:
        self._edit.setText(v)


class SettingsPanel(QScrollArea):
    input_changed = Signal(str)
    scan_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scanned = False
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        self.setWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(self._build_io_group())
        layout.addWidget(self._build_denoiser_group())
        layout.addWidget(self._build_passes_group())
        layout.addWidget(self._build_temporal_group())
        layout.addWidget(self._build_video_group())
        layout.addStretch()

    # ------------------------------------------------------------------
    # Group builders
    # ------------------------------------------------------------------

    def _build_io_group(self) -> QGroupBox:
        box = QGroupBox("Input / Output")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._input_picker = _FolderPicker("Input Folder")
        self._input_picker.changed.connect(self.input_changed)
        form.addRow("Input:", self._input_picker)

        self._output_picker = _FolderPicker("Output Folder")
        form.addRow("Output:", self._output_picker)

        return box

    def _build_denoiser_group(self) -> QGroupBox:
        box = QGroupBox("Denoiser")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        engine_label = QLabel("OptiX (NVIDIA)")
        form.addRow("Engine:", engine_label)

        self._hdr_check = QCheckBox("HDR mode (linear float)")
        self._hdr_check.setChecked(True)
        form.addRow("", self._hdr_check)

        return box

    def _build_passes_group(self) -> QGroupBox:
        box = QGroupBox("EXR Pass Mapping")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._passes_tip = QLabel("Select an input folder, then click \"Scan Passes\".")
        self._passes_tip.setStyleSheet("color: #888; font-size: 10px;")
        self._passes_tip.setWordWrap(True)
        form.addRow(self._passes_tip)

        self._scan_btn = QPushButton("Scan Passes")
        self._scan_btn.setToolTip("Read EXR layers from the first file in the input folder")
        self._scan_btn.clicked.connect(self.scan_requested)
        form.addRow("", self._scan_btn)

        self._pass_noisy = QComboBox()
        self._pass_noisy.setToolTip("Pass used as the noisy input (required)")
        form.addRow("Noisy pass:", self._pass_noisy)

        self._pass_albedo = QComboBox()
        self._pass_albedo.setToolTip("Albedo guide pass (optional but recommended)")
        form.addRow("Albedo pass:", self._pass_albedo)

        self._pass_normal = QComboBox()
        self._pass_normal.setToolTip("Normal guide pass (optional but recommended)")
        form.addRow("Normal pass:", self._pass_normal)

        self._pass_vector = QComboBox()
        self._pass_vector.setToolTip("Motion vector pass for temporal reprojection (optional)")
        form.addRow("Vector pass:", self._pass_vector)

        return box

    def _build_temporal_group(self) -> QGroupBox:
        box = QGroupBox("Temporal Settings")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._temporal_check = QCheckBox("Enable temporal denoising")
        self._temporal_check.setChecked(True)
        form.addRow("", self._temporal_check)

        # Blend weight slider
        blend_widget = QWidget()
        blend_layout = QHBoxLayout(blend_widget)
        blend_layout.setContentsMargins(0, 0, 0, 0)
        self._blend_slider = QSlider(Qt.Horizontal)
        self._blend_slider.setRange(0, 100)
        self._blend_slider.setValue(80)
        self._blend_value_label = QLabel("0.80")
        self._blend_value_label.setFixedWidth(36)
        self._blend_slider.valueChanged.connect(
            lambda v: self._blend_value_label.setText(f"{v/100:.2f}")
        )
        blend_layout.addWidget(self._blend_slider)
        blend_layout.addWidget(self._blend_value_label)
        form.addRow("Blend weight:", blend_widget)

        self._vector_scale_edit = QLineEdit("1.0")
        self._vector_scale_edit.setToolTip(
            "Scale factor for motion vectors. Use -1 to invert direction."
        )
        form.addRow("Vector scale:", self._vector_scale_edit)

        return box

    def _build_video_group(self) -> QGroupBox:
        box = QGroupBox("Video Preview")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._video_check = QCheckBox("Create video preview after denoising")
        self._video_check.setChecked(False)
        form.addRow("", self._video_check)

        self._video_fps_edit = QLineEdit("24")
        self._video_fps_edit.setToolTip("Frames per second for the output video")
        form.addRow("FPS:", self._video_fps_edit)

        return box

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_scanned(self) -> bool:
        return self._scanned

    def set_input_folder(self, path: str) -> None:
        self._scanned = False
        self._input_picker.set_value(path)

    def set_output_folder(self, path: str) -> None:
        self._output_picker.set_value(path)

    def get_input_folder(self) -> str:
        return self._input_picker.value()

    def populate_passes(self, layers: list[str], raw_channel_count: int = 0) -> None:
        self._scanned = True
        """Fill pass dropdowns with detected EXR layers and auto-select best matches."""
        none_label = "(none)"

        # Keywords used for auto-selection (checked in order, case-insensitive)
        _AUTO = {
            "noisy":  ["noisy image", "combined", "beauty", "rgba", "image"],
            "albedo": ["albedo", "diffuse color", "denoising albedo"],
            "normal": ["normal", "denoising normal"],
            "vector": ["vector", "speed", "motion"],
        }

        def best_match(keywords: list[str]) -> str:
            for kw in keywords:
                for layer in layers:
                    if kw in layer.lower():
                        return layer
            return ""

        def fill(combo: QComboBox, required: bool, keywords: list[str]) -> None:
            combo.blockSignals(True)
            combo.clear()
            if not required:
                combo.addItem(none_label)
            combo.addItems(layers)
            match = best_match(keywords)
            if match:
                idx = combo.findText(match)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            elif required and layers:
                combo.setCurrentIndex(0 if required else 1)
            combo.blockSignals(False)

        fill(self._pass_noisy,  required=True,  keywords=_AUTO["noisy"])
        fill(self._pass_albedo, required=False, keywords=_AUTO["albedo"])
        fill(self._pass_normal, required=False, keywords=_AUTO["normal"])
        fill(self._pass_vector, required=False, keywords=_AUTO["vector"])

        count = len(layers)
        detail = f" ({raw_channel_count} raw channels)" if raw_channel_count else ""
        self._passes_tip.setText(
            f"{count} layer{'s' if count != 1 else ''} detected{detail}. Adjust selections as needed."
        )
        self._passes_tip.setStyleSheet("color: #6ec26e; font-size: 10px;")

    def build_config(self) -> PipelineConfig | None:
        none_label = "(none)"
        try:
            vector_scale = float(self._vector_scale_edit.text())
        except ValueError:
            vector_scale = 1.0

        def pass_value(combo: QComboBox) -> str:
            text = combo.currentText()
            return "" if text == none_label else text

        try:
            video_fps = max(1, int(self._video_fps_edit.text()))
        except ValueError:
            video_fps = 24

        return PipelineConfig(
            input_folder=Path(self._input_picker.value()),
            output_folder=Path(self._output_picker.value()),
            hdr=self._hdr_check.isChecked(),
            temporal=self._temporal_check.isChecked(),
            temporal_blend=self._blend_slider.value() / 100.0,
            vector_scale=vector_scale,
            pass_noisy=pass_value(self._pass_noisy) or "Combined",
            pass_albedo=pass_value(self._pass_albedo),
            pass_normal=pass_value(self._pass_normal),
            pass_vector=pass_value(self._pass_vector),
            video_preview=self._video_check.isChecked(),
            video_fps=video_fps,
        )
