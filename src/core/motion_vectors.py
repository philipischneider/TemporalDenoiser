"""Utilities for converting Blender's Vector pass to usable motion vectors."""
from __future__ import annotations

import numpy as np


def blender_vector_to_backward_flow(
    vector_layer: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Convert Blender's 4-channel Vector pass to a backward optical flow.

    Blender's Vector pass layout (in pixel space):
        R, G  = backward motion  (current <- previous, i.e. where did this pixel come from)
        B, A  = forward motion   (current -> next)

    The backward flow (RG channels) is what we need to warp the previous
    denoised frame onto the current frame.

    Args:
        vector_layer: (H, W, 4) or (H, W, 2) float32 array from the EXR.
        scale:        Multiplier applied to the raw pixel-space values.
                      Use -1.0 if the convention appears inverted.

    Returns:
        (H, W, 2) float32 array where [..., 0] = dx and [..., 1] = dy
        in pixel coordinates (positive = right/down).
    """
    if vector_layer.ndim == 3 and vector_layer.shape[2] >= 2:
        flow = vector_layer[..., :2].copy()
    elif vector_layer.ndim == 2:
        raise ValueError("Expected at least 2 channels for motion vectors.")
    else:
        flow = vector_layer[..., :2].copy()

    flow *= scale
    return flow.astype(np.float32)


def detect_static_regions(
    flow: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Return a boolean mask where True = static pixel (negligible motion).

    Args:
        flow:      (H, W, 2) optical flow.
        threshold: Magnitude threshold in pixels.

    Returns:
        (H, W) bool mask.
    """
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return magnitude < threshold


def detect_disocclusions(
    flow: np.ndarray,
    forward_flow: np.ndarray | None = None,
    threshold: float = 2.0,
) -> np.ndarray:
    """Estimate disocclusion mask using forward-backward consistency.

    If forward_flow is None, returns a zero mask (no disocclusions detected).
    """
    if forward_flow is None:
        return np.zeros(flow.shape[:2], dtype=bool)

    # Simple forward-backward check: |backward(forward(p)) - p| > threshold
    h, w = flow.shape[:2]
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)

    # Warp forward flow with backward flow
    sample_x = np.clip(grid_x + flow[..., 0], 0, w - 1).astype(np.int32)
    sample_y = np.clip(grid_y + flow[..., 1], 0, h - 1).astype(np.int32)

    fwd_at_prev = forward_flow[sample_y, sample_x]
    consistency = np.sqrt(
        (flow[..., 0] + fwd_at_prev[..., 0]) ** 2
        + (flow[..., 1] + fwd_at_prev[..., 1]) ** 2
    )
    return consistency > threshold
