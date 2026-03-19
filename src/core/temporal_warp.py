"""Warp a previous frame onto the current frame using optical flow."""
from __future__ import annotations

import numpy as np
import cv2


def warp_frame(
    prev_frame: np.ndarray,
    backward_flow: np.ndarray,
    border_mode: int = cv2.BORDER_REPLICATE,
) -> np.ndarray:
    """Warp prev_frame to the current frame using backward optical flow.

    The backward flow convention used here:
        flow[y, x] = (dx, dy) such that the pixel at (x, y) in the current
        frame corresponds to the pixel at (x - dx, y - dy) in the previous frame.

    Args:
        prev_frame:    (H, W, C) float32 — previous denoised frame.
        backward_flow: (H, W, 2) float32 — per-pixel (dx, dy) in pixels.
        border_mode:   OpenCV border interpolation mode for out-of-bounds.

    Returns:
        (H, W, C) float32 — warped previous frame.
    """
    h, w = prev_frame.shape[:2]
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)

    # Source coordinates in the previous frame
    map_x = grid_x - backward_flow[..., 0]
    map_y = grid_y - backward_flow[..., 1]

    if prev_frame.ndim == 2:
        warped = cv2.remap(
            prev_frame, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=border_mode,
        )
    else:
        channels = []
        for c in range(prev_frame.shape[2]):
            ch = cv2.remap(
                prev_frame[..., c], map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=border_mode,
            )
            channels.append(ch)
        warped = np.stack(channels, axis=-1)

    return warped.astype(np.float32)


def blend_temporal(
    current_denoised: np.ndarray,
    warped_prev: np.ndarray,
    blend_weight: float = 0.8,
    validity_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Blend current denoised frame with the warped previous frame.

    Args:
        current_denoised: (H, W, C) float32 — current frame after denoising.
        warped_prev:      (H, W, C) float32 — temporally warped previous output.
        blend_weight:     Weight for warped_prev. Higher = more temporal stability,
                          but slower response to changes.
        validity_mask:    (H, W) float32 in [0, 1]. 0 = disoccluded / new pixel
                          (uses current only). 1 = valid warp.

    Returns:
        (H, W, C) float32 — blended result.
    """
    if validity_mask is not None:
        w = validity_mask[..., np.newaxis] * blend_weight
    else:
        w = blend_weight

    return (1.0 - w) * current_denoised + w * warped_prev
