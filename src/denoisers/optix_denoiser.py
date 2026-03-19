"""NVIDIA OptiX Denoiser backend — calls a compiled C++ bridge executable."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from denoisers.base import BaseDenoiser

# Path to the compiled optix_bridge executable (adjust after building)
_DEFAULT_BRIDGE = Path(__file__).parents[2] / "optix_bridge" / "build" / "Release" / "optix_bridge.exe"


class OptiXDenoiser(BaseDenoiser):
    """Invokes the OptiX C++ bridge via subprocess for GPU-accelerated denoising.

    Build requirements:
        - NVIDIA Driver 565+
        - CUDA Toolkit 12.6+
        - OptiX SDK 8.1+ (set OPTIX_PATH env var to install root)
        - CMake 3.25+

    Build steps:
        cd optix_bridge
        cmake -B build -DCMAKE_BUILD_TYPE=Release
        cmake --build build --config Release

    The bridge executable reads/writes numpy .npy binary files for zero-copy
    interop with the Python side.
    """

    def __init__(
        self,
        bridge_exe: Path = _DEFAULT_BRIDGE,
        hdr: bool = True,
        temporal: bool = True,
    ) -> None:
        self._bridge_exe = Path(bridge_exe)
        self._hdr = hdr
        self._temporal = temporal
        self._verify_bridge()

    def _verify_bridge(self) -> None:
        if not self._bridge_exe.is_file():
            raise FileNotFoundError(
                f"OptiX bridge not found at: {self._bridge_exe}\n"
                "Please build it first:\n"
                "  cd optix_bridge && cmake -B build && cmake --build build --config Release"
            )

    def denoise(
        self,
        noisy: np.ndarray,
        albedo: Optional[np.ndarray] = None,
        normal: Optional[np.ndarray] = None,
        prev_output: Optional[np.ndarray] = None,
        flow: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        h, w = noisy.shape[:2]

        with tempfile.TemporaryDirectory(prefix="optix_bridge_") as tmp:
            tmp_path = Path(tmp)

            # Write input buffers
            inputs: dict = {
                "width": w,
                "height": h,
                "hdr": self._hdr,
                "temporal": self._temporal,
            }

            def save(name: str, arr: np.ndarray | None) -> None:
                if arr is not None:
                    p = tmp_path / f"{name}.npy"
                    np.save(str(p), arr.astype(np.float32))
                    inputs[name] = str(p)

            save("noisy", noisy)
            save("albedo", albedo)
            save("normal", normal)
            save("prev_output", prev_output)
            save("flow", flow[..., :2] if flow is not None else None)

            output_path = tmp_path / "output.npy"
            inputs["output"] = str(output_path)

            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps(inputs))

            # Call the bridge
            result = subprocess.run(
                [str(self._bridge_exe), str(config_path)],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"OptiX bridge failed (exit {result.returncode}):\n"
                    f"{result.stderr}"
                )

            if not output_path.is_file():
                raise RuntimeError("OptiX bridge did not produce output file.")

            return np.load(str(output_path)).astype(np.float32)

    def cleanup(self) -> None:
        pass  # Stateless — each call spawns a new process
