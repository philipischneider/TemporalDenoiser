"""Abstract base class for all denoiser backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseDenoiser(ABC):
    """Common interface for denoiser backends."""

    @abstractmethod
    def denoise(
        self,
        noisy: np.ndarray,
        albedo: Optional[np.ndarray] = None,
        normal: Optional[np.ndarray] = None,
        prev_output: Optional[np.ndarray] = None,
        flow: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Denoise a single frame.

        Args:
            noisy:       (H, W, 3) float32 — noisy input image.
            albedo:      (H, W, 3) float32 — albedo guide pass (optional).
            normal:      (H, W, 3) float32 — world-space normal guide (optional).
            prev_output: (H, W, 3) float32 — denoised output from previous frame,
                         already warped to the current frame (optional).
            flow:        (H, W, 2) float32 — backward optical flow in pixels (optional).

        Returns:
            (H, W, 3) float32 — denoised image.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Release GPU/CPU resources."""
