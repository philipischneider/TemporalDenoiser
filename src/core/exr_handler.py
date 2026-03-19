"""OpenEXR Multilayer read/write utilities for Blender renders."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import OpenEXR
    import Imath
    _HAS_OPENEXR = True
except ImportError:
    _HAS_OPENEXR = False


@dataclass
class ExrLayer:
    """A single named pass extracted from a multilayer EXR."""
    name: str
    channels: Dict[str, np.ndarray]  # channel suffix (R/G/B/X/Y/Z/A/W) -> array

    def to_array(self, order: str = "RGB") -> np.ndarray:
        """Stack channels in the given order to (H, W, C) float32."""
        arrays = [self.channels[c] for c in order if c in self.channels]
        if not arrays:
            raise KeyError(f"None of '{order}' found in layer '{self.name}'")
        return np.stack(arrays, axis=-1)


@dataclass
class ExrFrame:
    """All layers parsed from a single multilayer EXR file."""
    width: int
    height: int
    layers: Dict[str, ExrLayer] = field(default_factory=dict)

    def get_layer_rgb(self, layer_name: str) -> np.ndarray:
        """Return (H, W, 3) float32 for a layer, matching by partial name."""
        layer = self._find_layer(layer_name)
        return layer.to_array("RGB")

    def get_layer_xyz(self, layer_name: str) -> np.ndarray:
        """Return (H, W, 3) float32 for XYZ channels (normals, vectors)."""
        layer = self._find_layer(layer_name)
        # Try XYZ then RGB fallback
        for order in ("XYZ", "RGB"):
            try:
                return layer.to_array(order)
            except KeyError:
                continue
        raise KeyError(f"Cannot extract 3-channel data from layer '{layer_name}'")

    def get_layer_xy(self, layer_name: str) -> np.ndarray:
        """Return (H, W, 2) float32 — first two channels of a layer."""
        layer = self._find_layer(layer_name)
        for order in ("XY", "RG"):
            try:
                return layer.to_array(order)
            except KeyError:
                continue
        # fallback: first two available channels
        keys = list(layer.channels.keys())
        if len(keys) >= 2:
            return np.stack([layer.channels[keys[0]], layer.channels[keys[1]]], axis=-1)
        raise KeyError(f"Cannot extract 2-channel data from layer '{layer_name}'")

    def available_layers(self) -> List[str]:
        return list(self.layers.keys())

    def _find_layer(self, name: str) -> ExrLayer:
        # Exact match first
        if name in self.layers:
            return self.layers[name]
        # Case-insensitive partial match
        name_lower = name.lower()
        for key, layer in self.layers.items():
            if name_lower in key.lower():
                return layer
        raise KeyError(
            f"Layer '{name}' not found. Available: {self.available_layers()}"
        )


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def load_exr(path: Path) -> ExrFrame:
    """Load a multilayer OpenEXR file into an ExrFrame."""
    if not _HAS_OPENEXR:
        raise ImportError("OpenEXR Python package is not installed.")

    exr = OpenEXR.InputFile(str(path))
    header = exr.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    float_type = Imath.PixelType(Imath.PixelType.FLOAT)
    frame = ExrFrame(width=width, height=height)

    for channel_name in header["channels"].keys():
        # Parse: "LayerName.ChannelSuffix" or just "ChannelSuffix"
        parts = channel_name.rsplit(".", 1)
        if len(parts) == 2:
            layer_name, suffix = parts
        else:
            layer_name = "default"
            suffix = parts[0]

        # Normalize layer name: strip render layer prefix if present
        layer_name = _normalize_layer_name(layer_name)

        raw = exr.channel(channel_name, float_type)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(height, width)

        if layer_name not in frame.layers:
            frame.layers[layer_name] = ExrLayer(name=layer_name, channels={})
        frame.layers[layer_name].channels[suffix] = arr

    exr.close()
    return frame


def _normalize_layer_name(name: str) -> str:
    """Strip common render layer prefixes like 'RenderLayer.' or 'View Layer.'."""
    for prefix in ("RenderLayer.", "View Layer.", "ViewLayer.", "Scene."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def save_exr(path: Path, image: np.ndarray, layer_name: str = "Combined") -> None:
    """Save a (H, W, 3) float32 array as an EXR file."""
    if not _HAS_OPENEXR:
        raise ImportError("OpenEXR Python package is not installed.")

    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    h, w, c = image.shape
    path.parent.mkdir(parents=True, exist_ok=True)

    channel_names = {1: ["Y"], 3: ["R", "G", "B"], 4: ["R", "G", "B", "A"]}
    suffixes = channel_names.get(c, [str(i) for i in range(c)])

    header = OpenEXR.Header(w, h)
    header["channels"] = {
        f"{layer_name}.{s}": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        for s in suffixes
    }

    out = OpenEXR.OutputFile(str(path), header)
    channel_data = {
        f"{layer_name}.{s}": image[:, :, i].astype(np.float32).tobytes()
        for i, s in enumerate(suffixes)
    }
    out.writePixels(channel_data)
    out.close()
