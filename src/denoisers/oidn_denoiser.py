"""Intel Open Image Denoise 2.x backend — loaded via ctypes from the official SDK DLL.

Setup:
    1. Download OIDN 2.x for Windows from:
       https://github.com/RenderKit/oidn/releases
       (e.g. oidn-2.3.0.x86_64.windows.zip)
    2. Extract to a folder, e.g. C:\\oidn
    3. Set the environment variable:
       OIDN_PATH=C:\\oidn
    No pip install is needed.

The OIDN C API is consumed directly via ctypes.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

import numpy as np

from denoisers.base import BaseDenoiser

# ---------------------------------------------------------------------------
# OIDN C API constants (from oidn.h)
# ---------------------------------------------------------------------------
_OIDN_FORMAT_FLOAT2 = 0x102
_OIDN_FORMAT_FLOAT3 = 0x103

_OIDN_DEVICE_TYPE_DEFAULT = 0
_OIDN_DEVICE_TYPE_CPU     = 1
_OIDN_DEVICE_TYPE_CUDA    = 3

_OIDN_QUALITY_FAST     = 3
_OIDN_QUALITY_BALANCED = 4
_OIDN_QUALITY_HIGH     = 5

_OIDN_ERROR_NONE = 0

_QUALITY_MAP = {
    "high":     _OIDN_QUALITY_HIGH,
    "balanced": _OIDN_QUALITY_BALANCED,
    "fast":     _OIDN_QUALITY_FAST,
}


def _find_oidn_dll() -> Path:
    """Locate OpenImageDenoise.dll from OIDN_PATH or common locations."""
    candidates: list[Path] = []

    oidn_env = os.environ.get("OIDN_PATH", "")
    if oidn_env:
        base = Path(oidn_env)
        candidates += [
            base / "bin" / "OpenImageDenoise.dll",
            base / "OpenImageDenoise.dll",
        ]

    # Common manual install locations
    for base in [Path("C:/oidn"), Path("C:/Program Files/Intel/Open Image Denoise")]:
        candidates += [
            base / "bin" / "OpenImageDenoise.dll",
            base / "OpenImageDenoise.dll",
        ]

    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        "OpenImageDenoise.dll not found.\n\n"
        "To install OIDN:\n"
        "  1. Download from https://github.com/RenderKit/oidn/releases\n"
        "     (choose 'oidn-2.x.x.x86_64.windows.zip')\n"
        "  2. Extract to C:\\oidn  (or any folder)\n"
        "  3. Set env var: OIDN_PATH=C:\\oidn\n"
        "  4. Restart this application."
    )


def _load_oidn_lib() -> ctypes.CDLL:
    dll_path = _find_oidn_dll()
    # Add the DLL's directory so its own dependencies (CUDA, etc.) are found
    os.add_dll_directory(str(dll_path.parent))
    lib = ctypes.CDLL(str(dll_path))
    _setup_signatures(lib)
    return lib


def _setup_signatures(lib: ctypes.CDLL) -> None:
    """Declare argtypes/restype for each OIDN function we use."""
    vp = ctypes.c_void_p
    sz = ctypes.c_size_t

    lib.oidnNewDevice.restype  = vp
    lib.oidnNewDevice.argtypes = [ctypes.c_int]

    lib.oidnCommitDevice.restype  = None
    lib.oidnCommitDevice.argtypes = [vp]

    lib.oidnGetDeviceError.restype  = ctypes.c_int
    lib.oidnGetDeviceError.argtypes = [vp, ctypes.POINTER(ctypes.c_char_p)]

    lib.oidnReleaseDevice.restype  = None
    lib.oidnReleaseDevice.argtypes = [vp]

    lib.oidnNewFilter.restype  = vp
    lib.oidnNewFilter.argtypes = [vp, ctypes.c_char_p]

    # void oidnSetFilterImage(filter, name, ptr, format, w, h, byteOffset, pixelStride, rowStride)
    lib.oidnSetFilterImage.restype  = None
    lib.oidnSetFilterImage.argtypes = [vp, ctypes.c_char_p, vp,
                                       ctypes.c_int, sz, sz, sz, sz, sz]

    lib.oidnRemoveFilterImage.restype  = None
    lib.oidnRemoveFilterImage.argtypes = [vp, ctypes.c_char_p]

    lib.oidnSetFilter1b.restype  = None
    lib.oidnSetFilter1b.argtypes = [vp, ctypes.c_char_p, ctypes.c_bool]

    lib.oidnSetFilter1i.restype  = None
    lib.oidnSetFilter1i.argtypes = [vp, ctypes.c_char_p, ctypes.c_int]

    lib.oidnCommitFilter.restype  = None
    lib.oidnCommitFilter.argtypes = [vp]

    lib.oidnExecuteFilter.restype  = None
    lib.oidnExecuteFilter.argtypes = [vp]

    lib.oidnGetFilterError.restype  = ctypes.c_int
    lib.oidnGetFilterError.argtypes = [vp, ctypes.POINTER(ctypes.c_char_p)]

    lib.oidnReleaseFilter.restype  = None
    lib.oidnReleaseFilter.argtypes = [vp]


# ---------------------------------------------------------------------------
# Denoiser class
# ---------------------------------------------------------------------------

class OIDNDenoiser(BaseDenoiser):
    """OIDN 2.x temporal denoiser using the official SDK via ctypes."""

    def __init__(
        self,
        device: str = "cuda",   # "cuda" or "cpu"
        hdr: bool = True,
        quality: str = "high",  # "high", "balanced", "fast"
        temporal: bool = True,
    ) -> None:
        self._device_type = device
        self._hdr = hdr
        self._quality = quality
        self._temporal = temporal

        self._lib: ctypes.CDLL | None = None
        self._device_handle: int | None = None
        self._filter_handle: int | None = None
        self._shape: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _ensure_lib(self) -> None:
        if self._lib is None:
            self._lib = _load_oidn_lib()

    def _initialize(self, h: int, w: int) -> None:
        self._ensure_lib()
        lib = self._lib

        # Try requested device type, fall back to CPU
        device_order = (
            [_OIDN_DEVICE_TYPE_CUDA, _OIDN_DEVICE_TYPE_CPU]
            if self._device_type == "cuda"
            else [_OIDN_DEVICE_TYPE_CPU]
        )

        dev = None
        for dtype in device_order:
            handle = lib.oidnNewDevice(dtype)
            if not handle:
                continue
            lib.oidnCommitDevice(handle)
            msg = ctypes.c_char_p()
            err = lib.oidnGetDeviceError(handle, ctypes.byref(msg))
            if err == _OIDN_ERROR_NONE:
                dev = handle
                break
            lib.oidnReleaseDevice(handle)

        if dev is None:
            raise RuntimeError(
                "Failed to initialise any OIDN device. "
                "Make sure the OIDN SDK and CUDA drivers are properly installed."
            )

        self._device_handle = dev
        self._filter_handle = lib.oidnNewFilter(dev, b"RT")
        self._shape = (h, w)

    def _ensure_ready(self, h: int, w: int) -> None:
        if self._filter_handle is None or self._shape != (h, w):
            self.cleanup()
            self._initialize(h, w)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_image(self, name: bytes, arr: np.ndarray, fmt: int) -> None:
        h, w = arr.shape[:2]
        channels = arr.shape[2] if arr.ndim == 3 else 1
        pixel_stride = channels * 4          # float32 = 4 bytes
        row_stride   = w * pixel_stride
        ptr = arr.ctypes.data_as(ctypes.c_void_p)
        self._lib.oidnSetFilterImage(
            self._filter_handle, name, ptr,
            fmt, w, h, 0, pixel_stride, row_stride,
        )

    def _remove_image(self, name: bytes) -> None:
        self._lib.oidnRemoveFilterImage(self._filter_handle, name)

    def _check_filter_error(self) -> None:
        msg = ctypes.c_char_p()
        err = self._lib.oidnGetFilterError(self._filter_handle, ctypes.byref(msg))
        if err != _OIDN_ERROR_NONE:
            text = msg.value.decode() if msg.value else f"OIDN error code {err}"
            raise RuntimeError(f"OIDN filter error: {text}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def denoise(
        self,
        noisy: np.ndarray,
        albedo: Optional[np.ndarray] = None,
        normal: Optional[np.ndarray] = None,
        prev_output: Optional[np.ndarray] = None,
        flow: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        h, w = noisy.shape[:2]
        self._ensure_ready(h, w)

        noisy_c  = np.ascontiguousarray(noisy, dtype=np.float32)
        output   = np.zeros((h, w, 3), dtype=np.float32)

        self._set_image(b"color",  noisy_c, _OIDN_FORMAT_FLOAT3)
        self._set_image(b"output", output,  _OIDN_FORMAT_FLOAT3)

        if albedo is not None:
            self._set_image(b"albedo", np.ascontiguousarray(albedo, np.float32), _OIDN_FORMAT_FLOAT3)
        else:
            self._remove_image(b"albedo")

        if normal is not None:
            self._set_image(b"normal", np.ascontiguousarray(normal, np.float32), _OIDN_FORMAT_FLOAT3)
        else:
            self._remove_image(b"normal")

        use_temporal = self._temporal and prev_output is not None
        if use_temporal:
            self._set_image(
                b"previousOutput",
                np.ascontiguousarray(prev_output, np.float32),
                _OIDN_FORMAT_FLOAT3,
            )
            if flow is not None:
                flow_c = np.ascontiguousarray(flow[..., :2], np.float32)
                self._set_image(b"flow", flow_c, _OIDN_FORMAT_FLOAT2)
            else:
                self._remove_image(b"flow")
        else:
            self._remove_image(b"previousOutput")
            self._remove_image(b"flow")

        self._lib.oidnSetFilter1b(self._filter_handle, b"hdr", self._hdr)
        quality_val = _QUALITY_MAP.get(self._quality, _OIDN_QUALITY_HIGH)
        self._lib.oidnSetFilter1i(self._filter_handle, b"quality", quality_val)

        self._lib.oidnCommitFilter(self._filter_handle)
        self._lib.oidnExecuteFilter(self._filter_handle)
        self._check_filter_error()

        return output

    def cleanup(self) -> None:
        if self._lib is None:
            return
        if self._filter_handle:
            self._lib.oidnReleaseFilter(self._filter_handle)
            self._filter_handle = None
        if self._device_handle:
            self._lib.oidnReleaseDevice(self._device_handle)
            self._device_handle = None
        self._shape = None
