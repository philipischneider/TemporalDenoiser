from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QComboBox, QCheckBox, QSlider,
    QLabel, QGroupBox, QHBoxLayout, QScrollArea,
    QFrame, QFileDialog,
)

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

    # Compression codecs offered in the UI
    _COMPRESSION_OPTIONS = [
        ("DWAB (lossy, high compression)",  "dwab"),
        ("DWAA (lossy, moderate)",          "dwaa"),
        ("PIZ (lossless, wavelet)",         "piz"),
        ("ZIP (lossless, per scanline)",    "zip"),
        ("ZIPS (lossless, per pixel)",      "zips"),
        ("RLE (lossless)",                  "rle"),
        ("None",                            "none"),
    ]

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

        layout.addWidget(self._build_mode_group())
        layout.addWidget(self._build_io_group())
        layout.addWidget(self._build_denoiser_group())
        layout.addWidget(self._build_passes_group())
        layout.addWidget(self._build_temporal_group())
        layout.addWidget(self._build_interpolation_group())
        layout.addWidget(self._build_video_group())
        layout.addStretch()

        # Apply initial mode visibility
        self._on_mode_changed(self._mode_combo.currentText())

    # ------------------------------------------------------------------
    # Group builders
    # ------------------------------------------------------------------

    def _build_mode_group(self) -> QGroupBox:
        box = QGroupBox("Pipeline Mode")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Denoise", "Interpolate"])
        self._mode_combo.setToolTip(
            "Denoise: apply OptiX temporal denoising to an EXR sequence.\n"
            "Interpolate: generate intermediate frames using motion vectors and depth."
        )
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        form.addRow("Mode:", self._mode_combo)

        return box

    def _build_io_group(self) -> QGroupBox:
        box = QGroupBox("Input / Output")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._input_picker = _FolderPicker("Input Folder")
        self._input_picker.changed.connect(self.input_changed)
        form.addRow("Input:", self._input_picker)

        self._output_picker = _FolderPicker("Output Folder")
        form.addRow("Output:", self._output_picker)

        self._compression_combo = QComboBox()
        for label, _ in self._COMPRESSION_OPTIONS:
            self._compression_combo.addItem(label)
        self._compression_combo.setToolTip(
            "EXR compression codec used when saving output frames.\n"
            "DWAB/DWAA are lossy but produce much smaller files.\n"
            "PIZ/ZIP are lossless."
        )
        form.addRow("Compression:", self._compression_combo)

        return box

    def _build_denoiser_group(self) -> QGroupBox:
        self._denoiser_group = QGroupBox("Denoiser")
        form = QFormLayout(self._denoiser_group)
        form.setLabelAlignment(Qt.AlignRight)

        engine_label = QLabel("OptiX (NVIDIA)")
        form.addRow("Engine:", engine_label)

        self._hdr_check = QCheckBox("HDR mode (linear float)")
        self._hdr_check.setChecked(True)
        form.addRow("", self._hdr_check)

        return self._denoiser_group

    def _build_passes_group(self) -> QGroupBox:
        box = QGroupBox("EXR Pass Mapping")
        self._passes_form = QFormLayout(box)
        self._passes_form.setLabelAlignment(Qt.AlignRight)

        self._passes_tip = QLabel("Select an input folder, then click \"Scan Passes\".")
        self._passes_tip.setStyleSheet("color: #888; font-size: 10px;")
        self._passes_tip.setWordWrap(True)
        self._passes_form.addRow(self._passes_tip)

        self._scan_btn = QPushButton("Scan Passes")
        self._scan_btn.setToolTip("Read EXR layers from the first file in the input folder")
        self._scan_btn.clicked.connect(self.scan_requested)
        self._passes_form.addRow("", self._scan_btn)

        # "Noisy pass" label is stored so we can relabel it in interpolation mode
        self._pass_noisy_label = QLabel("Noisy pass:")
        self._pass_noisy = QComboBox()
        self._pass_noisy.setToolTip("Pass used as the noisy input (required)")
        self._passes_form.addRow(self._pass_noisy_label, self._pass_noisy)

        self._pass_albedo_label = QLabel("Albedo pass:")
        self._pass_albedo = QComboBox()
        self._pass_albedo.setToolTip("Albedo guide pass (optional but recommended)")
        self._passes_form.addRow(self._pass_albedo_label, self._pass_albedo)

        self._pass_normal_label = QLabel("Normal pass:")
        self._pass_normal = QComboBox()
        self._pass_normal.setToolTip("Normal guide pass (optional but recommended)")
        self._passes_form.addRow(self._pass_normal_label, self._pass_normal)

        self._pass_vector_label = QLabel("Vector pass:")
        self._pass_vector = QComboBox()
        self._pass_vector.setToolTip("Motion vector pass for temporal reprojection / interpolation")
        self._passes_form.addRow(self._pass_vector_label, self._pass_vector)

        self._pass_depth_label = QLabel("Depth pass:")
        self._pass_depth = QComboBox()
        self._pass_depth.setToolTip(
            "Depth (Z) pass for interpolation occlusion resolution (optional)"
        )
        self._passes_form.addRow(self._pass_depth_label, self._pass_depth)

        return box

    def _build_temporal_group(self) -> QGroupBox:
        self._temporal_group = QGroupBox("Temporal Settings")
        form = QFormLayout(self._temporal_group)
        form.setLabelAlignment(Qt.AlignRight)

        self._temporal_check = QCheckBox("Enable temporal denoising")
        self._temporal_check.setChecked(True)
        form.addRow("", self._temporal_check)

        blend_widget = QWidget()
        blend_layout = QHBoxLayout(blend_widget)
        blend_layout.setContentsMargins(0, 0, 0, 0)
        self._blend_slider = QSlider(Qt.Horizontal)
        self._blend_slider.setRange(0, 100)
        self._blend_slider.setValue(15)
        self._blend_value_label = QLabel("0.15")
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

        return self._temporal_group

    def _build_interpolation_group(self) -> QGroupBox:
        self._interp_group = QGroupBox("Frame Interpolation")
        form = QFormLayout(self._interp_group)
        form.setLabelAlignment(Qt.AlignRight)

        self._interp_factor_combo = QComboBox()
        self._interp_factor_combo.addItems(["2x", "3x", "4x"])
        self._interp_factor_combo.setToolTip(
            "2x: one new frame between each pair.\n"
            "4x: three new frames between each pair."
        )
        form.addRow("Factor:", self._interp_factor_combo)

        self._interp_src_picker = _FolderPicker("Vector/Depth Source")
        self._interp_src_picker._edit.setPlaceholderText("Same as Input Folder…")
        self._interp_src_picker._edit.setToolTip(
            "Folder containing the original Blender renders with Vector and Depth passes.\n"
            "Leave empty to use the Input Folder (useful when all passes are in the same EXR)."
        )
        form.addRow("Source:", self._interp_src_picker)

        self._interp_sigmoid_edit = QLineEdit("10.0")
        self._interp_sigmoid_edit.setToolTip(
            "Controls sharpness of depth-based blend boundary.\n"
            "Higher = harder occlusion edges. Only used when a Depth pass is selected."
        )
        form.addRow("Depth sharpness:", self._interp_sigmoid_edit)

        self._interp_splat_check = QCheckBox("Forward splatting (experimental)")
        self._interp_splat_check.setChecked(False)
        self._interp_splat_check.setToolTip(
            "Push each source pixel along its own motion vector instead of\n"
            "resampling backward. Sharper motion boundaries, but unvalidated —\n"
            "compare against the default (unchecked) on your footage first."
        )
        form.addRow("", self._interp_splat_check)

        return self._interp_group

    def _build_video_group(self) -> QGroupBox:
        self._video_group = QGroupBox("Video Preview")
        form = QFormLayout(self._video_group)
        form.setLabelAlignment(Qt.AlignRight)

        self._video_check = QCheckBox("Create video preview after denoising")
        self._video_check.setChecked(False)
        form.addRow("", self._video_check)

        self._video_fps_edit = QLineEdit("24")
        self._video_fps_edit.setToolTip("Frames per second for the output video")
        form.addRow("FPS:", self._video_fps_edit)

        return self._video_group

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def _on_mode_changed(self, mode: str) -> None:
        is_interp = (mode == "Interpolate")

        # Show/hide whole groups
        self._denoiser_group.setVisible(not is_interp)
        self._temporal_group.setVisible(not is_interp)
        self._video_group.setVisible(not is_interp)
        self._interp_group.setVisible(is_interp)

        # Relabel the primary pass and show/hide denoising-specific rows
        if is_interp:
            self._pass_noisy_label.setText("Layer to interpolate:")
            self._pass_noisy.setToolTip(
                "The EXR layer whose pixel values will be interpolated (required)."
            )
        else:
            self._pass_noisy_label.setText("Noisy pass:")
            self._pass_noisy.setToolTip("Pass used as the noisy input (required)")

        # Albedo and normal are denoising-only guides — hide in interpolation mode
        self._pass_albedo_label.setVisible(not is_interp)
        self._pass_albedo.setVisible(not is_interp)
        self._pass_normal_label.setVisible(not is_interp)
        self._pass_normal.setVisible(not is_interp)

        # Depth is interpolation-specific — hide in denoise mode
        self._pass_depth_label.setVisible(is_interp)
        self._pass_depth.setVisible(is_interp)

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
        none_label = "(none)"

        _AUTO = {
            "noisy":  ["noisy image", "combined", "beauty", "rgba", "image"],
            "albedo": ["albedo", "diffuse color", "denoising albedo"],
            "normal": ["normal", "denoising normal"],
            "vector": ["vector", "speed", "motion"],
            "depth":  ["depth", "z depth", "mist", "z"],
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
                combo.setCurrentIndex(0)
            combo.blockSignals(False)

        fill(self._pass_noisy,  required=True,  keywords=_AUTO["noisy"])
        fill(self._pass_albedo, required=False, keywords=_AUTO["albedo"])
        fill(self._pass_normal, required=False, keywords=_AUTO["normal"])
        fill(self._pass_vector, required=False, keywords=_AUTO["vector"])
        fill(self._pass_depth,  required=False, keywords=_AUTO["depth"])

        count = len(layers)
        detail = f" ({raw_channel_count} raw channels)" if raw_channel_count else ""
        self._passes_tip.setText(
            f"{count} layer{'s' if count != 1 else ''} detected{detail}. "
            "Adjust selections as needed."
        )
        self._passes_tip.setStyleSheet("color: #6ec26e; font-size: 10px;")

    def build_config(self) -> PipelineConfig | None:
        none_label = "(none)"
        is_interp = (self._mode_combo.currentText() == "Interpolate")

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

        interp_factor = int(self._interp_factor_combo.currentText()[0])  # "2x" → 2
        interp_src = self._interp_src_picker.value()
        try:
            interp_sigmoid = float(self._interp_sigmoid_edit.text())
        except ValueError:
            interp_sigmoid = 10.0

        compression_idx = self._compression_combo.currentIndex()
        compression = self._COMPRESSION_OPTIONS[compression_idx][1]

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
            compression=compression,
            interpolate=is_interp,
            interpolation_factor=interp_factor,
            interp_source_folder=Path(interp_src) if interp_src else Path(""),
            pass_depth=pass_value(self._pass_depth),
            interp_sigmoid_sharpness=interp_sigmoid,
            interp_use_splatting=self._interp_splat_check.isChecked(),
        )
