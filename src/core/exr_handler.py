"""OpenEXR Multilayer read/write utilities for Blender renders."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import OpenImageIO as oiio
    _HAS_OIIO = True
except ImportError:
    _HAS_OIIO = False


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
        keys = list(layer.channels.keys())
        if len(keys) >= 2:
            return np.stack([layer.channels[keys[0]], layer.channels[keys[1]]], axis=-1)
        raise KeyError(f"Cannot extract 2-channel data from layer '{layer_name}'")

    def get_layer_4ch(self, layer_name: str) -> np.ndarray:
        """Return (H, W, 4) float32 — all 4 channels of a pass (e.g. Vector RGBA or XYZW).

        Tries orderings RGBA then XYZW; falls back to the first 4 available channels.
        """
        layer = self._find_layer(layer_name)
        for order in ("RGBA", "XYZW"):
            try:
                result = layer.to_array(order)
                if result.shape[2] >= 4:
                    return result.astype(np.float32)
            except KeyError:
                continue
        keys = list(layer.channels.keys())
        arrays = [layer.channels[k] for k in keys[:4]]
        return np.stack(arrays, axis=-1).astype(np.float32)

    def get_layer_z(self, layer_name: str) -> np.ndarray:
        """Return (H, W) float32 — single-channel depth pass.

        Tries channel keys Z, V, R in order; falls back to the first available channel.
        Designed for Blender's Depth (Z buffer) and Mist passes.
        """
        layer = self._find_layer(layer_name)
        for key in ("Z", "V", "R"):
            if key in layer.channels:
                return layer.channels[key].astype(np.float32)
        return list(layer.channels.values())[0].astype(np.float32)

    def available_layers(self) -> List[str]:
        return list(self.layers.keys())

    def _find_layer(self, name: str) -> ExrLayer:
        if name in self.layers:
            return self.layers[name]
        name_lower = name.lower()
        for key, layer in self.layers.items():
            if name_lower in key.lower():
                return layer
        raise KeyError(
            f"Layer '{name}' not found. Available: {self.available_layers()}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_layer_name(name: str) -> str:
    """Strip common render layer prefixes like 'RenderLayer.' or 'View Layer.'."""
    for prefix in ("RenderLayer.", "View Layer.", "ViewLayer.", "Scene."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _open_oiio(path: Path):
    """Open an EXR file with OIIO, raising a clear error if it fails."""
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise IOError(
            f"OpenImageIO could not open '{path.name}'. "
            f"Error: {oiio.geterror()}"
        )
    return inp


# ---------------------------------------------------------------------------
# Layer discovery (header only — no pixel data loaded)
# ---------------------------------------------------------------------------

def list_exr_layers(path: Path) -> Tuple[List[str], List[str]]:
    """Return (sorted unique layer names, raw channel descriptors) from an EXR.

    Handles both multi-part EXR (Blender's default format) and single-part
    multilayer EXR (dot-separated channel names). Uses OpenImageIO.
    """
    if not _HAS_OIIO:
        raise ImportError("OpenImageIO Python package is not installed.")

    inp = _open_oiio(path)
    layer_set: set[str] = set()
    raw_channels: List[str] = []

    subimage = 0
    while True:
        spec = inp.spec()
        part_name = spec.getattribute("name")

        if part_name:
            # Multi-part EXR: each subimage is a named pass
            layer_set.add(part_name)
            prefix = part_name + "."
            for ch in spec.channelnames:
                suffix = ch[len(prefix):] if ch.startswith(prefix) else ch
                raw_channels.append(f"{part_name}.{suffix}")
        else:
            # Single-part multilayer: channels named like "Combined.R"
            for ch in spec.channelnames:
                parts = ch.rsplit(".", 1)
                layer_set.add(
                    _normalize_layer_name(parts[0]) if len(parts) == 2 else "default"
                )
                raw_channels.append(ch)

        subimage += 1
        if not inp.seek_subimage(subimage, 0):
            break

    inp.close()
    return sorted(layer_set), sorted(raw_channels)


# ---------------------------------------------------------------------------
# Reading (pixel data)
# ---------------------------------------------------------------------------

def load_exr(path: Path) -> ExrFrame:
    """Load a multilayer OpenEXR file into an ExrFrame using OpenImageIO.

    Supports both multi-part EXR (Blender's default) and single-part
    multilayer EXR (dot-separated channel names).
    """
    if not _HAS_OIIO:
        raise ImportError("OpenImageIO Python package is not installed.")

    inp = _open_oiio(path)
    spec0 = inp.spec()
    frame = ExrFrame(width=spec0.width, height=spec0.height)

    subimage = 0
    while True:
        spec = inp.spec()
        part_name = spec.getattribute("name")

        pixels = inp.read_image(oiio.FLOAT)
        if pixels is None:
            subimage += 1
            if not inp.seek_subimage(subimage, 0):
                break
            continue

        if pixels.ndim == 2:
            pixels = pixels[:, :, np.newaxis]

        if part_name:
            # Multi-part: strip the part-name prefix if OIIO includes it
            # e.g. "Noisy Image.R" -> "R", or plain "R" stays "R"
            prefix = part_name + "."
            channels = {}
            for i, ch_name in enumerate(spec.channelnames):
                if i >= pixels.shape[2]:
                    break
                suffix = ch_name[len(prefix):] if ch_name.startswith(prefix) else ch_name
                channels[suffix] = pixels[:, :, i]
            frame.layers[part_name] = ExrLayer(name=part_name, channels=channels)
        else:
            # Single-part multilayer: channel names include layer prefix
            for i, ch_name in enumerate(spec.channelnames):
                if i >= pixels.shape[2]:
                    break
                parts = ch_name.rsplit(".", 1)
                if len(parts) == 2:
                    layer_name = _normalize_layer_name(parts[0])
                    suffix = parts[1]
                else:
                    layer_name = "default"
                    suffix = ch_name
                if layer_name not in frame.layers:
                    frame.layers[layer_name] = ExrLayer(name=layer_name, channels={})
                frame.layers[layer_name].channels[suffix] = pixels[:, :, i]

        subimage += 1
        if not inp.seek_subimage(subimage, 0):
            break

    inp.close()
    return frame


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def save_exr(
    path: Path,
    image: np.ndarray,
    layer_name: str = "Combined",
    compression: str = "zip",
) -> None:
    """Save a (H, W, 3) float32 array as an EXR file using OpenImageIO.

    Args:
        path:        Output file path.
        image:       (H, W, C) float32 array in linear space.
        layer_name:  Layer name stored in EXR metadata (informational).
        compression: EXR compression codec. Common values:
                     "none", "rle", "zip", "zips", "piz",
                     "pxr24", "b44", "b44a", "dwaa", "dwab".
    """
    if not _HAS_OIIO:
        raise ImportError("OpenImageIO Python package is not installed.")

    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    h, w, c = image.shape
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix_map = {1: ["Y"], 3: ["R", "G", "B"], 4: ["R", "G", "B", "A"]}
    suffixes = suffix_map.get(c, [str(i) for i in range(c)])

    spec = oiio.ImageSpec(w, h, c, oiio.FLOAT)
    spec.channelnames = suffixes          # plain R/G/B — compatible with all EXR viewers
    spec.attribute("compression", compression)

    out = oiio.ImageOutput.create(str(path))
    if out is None:
        raise IOError(f"OpenImageIO cannot create output file '{path.name}'")
    out.open(str(path), spec)
    out.write_image(image.astype(np.float32))
    out.close()
