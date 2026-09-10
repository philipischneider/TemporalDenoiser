"""Frame interpolation using Blender's Vector and Depth passes."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from core.temporal_warp import warp_frame


def compute_interpolation_t_values(factor: int) -> list[float]:
    """Return sub-frame t values in (0, 1) for the given interpolation factor.

    The original frames at t=0 and t=1 are always included by the pipeline;
    this function returns only the NEW intermediate frames to generate.

    Examples:
        factor=2 → [0.5]
        factor=3 → [0.333..., 0.666...]
        factor=4 → [0.25, 0.5, 0.75]
    """
    return [i / factor for i in range(1, factor)]


def depth_aware_blend(
    warp_fwd: np.ndarray,
    warp_bwd: np.ndarray,
    depth_fwd: np.ndarray | None,
    depth_bwd: np.ndarray | None,
    t: float,
    sigmoid_sharpness: float = 10.0,
) -> np.ndarray:
    """Blend two warped frames using depth to resolve occlusion conflicts.

    When depth is unavailable, falls back to a simple linear blend.

    Algorithm:
        1. diff = depth_fwd - depth_bwd
           Positive diff → fwd pixel is farther from camera → bwd pixel occludes it.
        2. alpha = sigmoid(diff * sigmoid_sharpness)
           alpha→1 means bwd wins, alpha→0 means fwd wins.
        3. certainty = clip(2*|alpha - 0.5|, 0, 1)
           Pixels where the sigmoid is near 0.5 are ambiguous; certainty=0 there.
        4. depth_blend  = (1-alpha)*warp_fwd + alpha*warp_bwd
           linear_blend = (1-t)*warp_fwd + t*warp_bwd
        5. result = certainty*depth_blend + (1-certainty)*linear_blend

    Args:
        warp_fwd:          (H, W, C) float32 — frame N warped forward by t.
        warp_bwd:          (H, W, C) float32 — frame N+1 warped backward by (1-t).
        depth_fwd:         (H, W) float32 — depth of frame N warped forward, or None.
        depth_bwd:         (H, W) float32 — depth of frame N+1 warped backward, or None.
        t:                 Interpolation fraction in (0, 1).
        sigmoid_sharpness: Controls edge hardness. Higher = sharper occlusion boundaries.

    Returns:
        (H, W, C) float32 blended result in linear space.
    """
    linear_blend = (1.0 - t) * warp_fwd + t * warp_bwd

    if depth_fwd is None or depth_bwd is None:
        return linear_blend.astype(np.float32)

    # Depth difference: positive → fwd is behind bwd → bwd should win
    diff = depth_fwd - depth_bwd

    # Sigmoid alpha: 1 = prefer bwd, 0 = prefer fwd. Clipped before exp() —
    # the sigmoid is already fully saturated (0 or 1) well before this range,
    # clipping just avoids a harmless-but-noisy float overflow warning.
    exponent = np.clip(-diff * sigmoid_sharpness, -60.0, 60.0)
    alpha = 1.0 / (1.0 + np.exp(exponent))  # (H, W)

    # Certainty: 0 near the decision boundary, 1 where occlusion is clear
    certainty = np.clip(2.0 * np.abs(alpha - 0.5), 0.0, 1.0)  # (H, W)

    # Expand for broadcasting against (H, W, C)
    alpha_3d = alpha[..., np.newaxis]
    certainty_3d = certainty[..., np.newaxis]

    depth_blend = (1.0 - alpha_3d) * warp_fwd + alpha_3d * warp_bwd
    result = certainty_3d * depth_blend + (1.0 - certainty_3d) * linear_blend

    return result.astype(np.float32)


def splat_forward(
    frame: np.ndarray,
    flow: np.ndarray,
    depth: Optional[np.ndarray] = None,
    sharpness: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Forward (scatter) warp: push each source pixel to its position at time t.

    Unlike warp_frame() (a backward/gather resample — "what value lands here?"),
    this pushes every source pixel to `position + flow` and writes it there,
    which is the physically correct operation for a flow that describes actual
    per-pixel motion. This introduces two things gather doesn't have:

        - Collisions: multiple source pixels can land near the same
          destination (surface moving toward the camera). Resolved with a
          soft depth-weighted average (nearer pixels dominate) when `depth`
          is provided, an approach known as softmax splatting — this avoids
          the aliasing/misalignment a hard nearest-pixel z-test would cause.
          Without `depth`, contributions are averaged with equal weight.
        - Holes: destinations no source pixel reaches (disocclusion revealed
          only at the intermediate frame). Reported via the returned validity
          mask; callers should fall back to something else (e.g. warp_frame)
          for those pixels.

    Destinations are sub-pixel (`position + flow` is rarely integer): each
    source pixel is distributed bilinearly across its 4 neighboring output
    pixels rather than rounded to the nearest one. Rounding to the nearest
    integer destination was tried first and rejected — it introduces a
    positional jitter that shows up as ghosting/misalignment once two
    independently-jittered splats (from frame N and frame N+1) are combined.

    Args:
        frame:     (H, W, C) or (H, W) float32 — source image to splat.
        flow:      (H, W, 2) float32 — per-pixel displacement to reach time t
                   (i.e. already the *total* motion scaled by t/(1-t), not the
                   raw N->N+1 or N+1->N vector).
        depth:     (H, W) float32 — optional per-source-pixel depth.
                   Smaller = nearer to camera. Normalized per-call to [0, 1]
                   (min/max over finite values) before weighting, so absolute
                   scene units don't matter.
        sharpness: Controls how strongly nearer pixels dominate collisions.
                   Higher = harder (closer to a hard z-test); 0 = pure
                   bilinear averaging regardless of depth. Same knob as
                   `interpolate_frame`'s `sigmoid_sharpness`.

    Returns:
        (output, valid_mask, out_depth):
            output:     (H, W, C) float32 — splatted result (garbage/zero at
                        pixels where valid_mask is False).
            valid_mask: (H, W) bool — True where at least one source pixel
                        contributed (accumulated weight above a small floor).
            out_depth:  (H, W) float32 weighted-average depth of whatever
                        landed at each destination, or None if `depth` was
                        not provided. Reusable as a z-buffer for combining
                        two splats.
    """
    h, w = frame.shape[:2]
    squeeze = frame.ndim == 2
    src = frame[..., np.newaxis] if squeeze else frame
    c = src.shape[2]
    n = h * w

    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    dst_x = grid_x + flow[..., 0]
    dst_y = grid_y + flow[..., 1]

    x0 = np.floor(dst_x).astype(np.int64)
    y0 = np.floor(dst_y).astype(np.int64)
    fx = dst_x - x0
    fy = dst_y - y0

    # Per-source importance: nearer (smaller normalized depth) -> more weight.
    # Depth is normalized per-call since Blender's Z pass has no fixed range
    # (background/miss pixels can be an arbitrarily large sentinel value).
    if depth is not None:
        depth32 = depth.astype(np.float32)
        finite = np.isfinite(depth32)
        if finite.any():
            d_min = float(depth32[finite].min())
            d_max = float(depth32[finite].max())
        else:
            d_min, d_max = 0.0, 1.0
        d_range = max(d_max - d_min, 1e-6)
        d_norm = np.where(finite, (depth32 - d_min) / d_range, 1.0)
        d_norm = np.clip(d_norm, 0.0, 1.0)
        importance = np.exp(-d_norm * sharpness).astype(np.float32)
        depth_flat = depth32.ravel().astype(np.float64)
    else:
        importance = np.ones((h, w), dtype=np.float32)
        depth_flat = None

    src_flat = src.reshape(-1, c).astype(np.float64)
    importance_flat = importance.ravel()

    accum = np.zeros((n, c), dtype=np.float64)
    weight = np.zeros(n, dtype=np.float64)
    depth_accum = np.zeros(n, dtype=np.float64) if depth_flat is not None else None

    # Distribute each source pixel across its 4 bilinear neighbor destinations.
    corners = (
        (x0, y0, (1.0 - fx) * (1.0 - fy)),
        (x0 + 1, y0, fx * (1.0 - fy)),
        (x0, y0 + 1, (1.0 - fx) * fy),
        (x0 + 1, y0 + 1, fx * fy),
    )
    for cx, cy, bilinear_w in corners:
        in_bounds = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
        ib = in_bounds.ravel()
        if not ib.any():
            continue
        flat_idx = (cy * w + cx).ravel()[ib]
        w_contrib = (bilinear_w.ravel()[ib] * importance_flat[ib]).astype(np.float64)

        np.add.at(accum, flat_idx, src_flat[ib] * w_contrib[:, np.newaxis])
        np.add.at(weight, flat_idx, w_contrib)
        if depth_accum is not None:
            np.add.at(depth_accum, flat_idx, depth_flat[ib] * w_contrib)

    valid = weight > 1e-8
    output = np.zeros((n, c), dtype=np.float32)
    output[valid] = (accum[valid] / weight[valid, np.newaxis]).astype(np.float32)

    out_depth = None
    if depth_accum is not None:
        # A large-but-finite "farther than everything" sentinel for holes —
        # np.inf here would make depth_aware_blend's later diff*sharpness
        # overflow in exp() for every invalid pixel (harmless since those
        # pixels get masked out downstream, but noisy RuntimeWarnings).
        sentinel = d_max + 1.0 if finite.any() else 1e6
        out_depth_flat = np.full(n, sentinel, dtype=np.float32)
        out_depth_flat[valid] = (depth_accum[valid] / weight[valid]).astype(np.float32)
        out_depth = out_depth_flat.reshape(h, w)

    output = output.reshape(h, w, c)
    valid = valid.reshape(h, w)
    if squeeze:
        output = output[..., 0]

    return output.astype(np.float32), valid, out_depth


def interpolate_frame(
    frame_n: np.ndarray,
    frame_n1: np.ndarray,
    flow_fwd_n: np.ndarray,
    flow_bwd_n1: np.ndarray,
    t: float,
    depth_n: np.ndarray | None = None,
    depth_n1: np.ndarray | None = None,
    sigmoid_sharpness: float = 10.0,
    use_splatting: bool = False,
) -> np.ndarray:
    """Generate a single interpolated frame at time t between frame_n and frame_n1.

    Uses Blender's pre-computed motion vectors (forward from N, backward from N+1)
    to warp both frames toward time t, then blends them with optional depth-based
    occlusion resolution.

    Args:
        frame_n:           (H, W, C) float32 — source frame at time N.
        frame_n1:          (H, W, C) float32 — source frame at time N+1.
        flow_fwd_n:        (H, W, 2) float32 — forward flow from frame N's BA channels
                           (where pixels of N move toward N+1).
        flow_bwd_n1:       (H, W, 2) float32 — backward flow from frame N+1's RG channels
                           (where pixels of N+1 came from in N).
        t:                 Target sub-frame time in (0, 1) exclusive.
        depth_n:           (H, W) float32 — Z depth at frame N, or None.
        depth_n1:          (H, W) float32 — Z depth at frame N+1, or None.
        sigmoid_sharpness: Passed through to depth_aware_blend.
        use_splatting:     If True, forward-splat (scatter) both source frames
                           to time t instead of gather-resampling them, using
                           `depth_n`/`depth_n1` (and `sigmoid_sharpness`) to
                           weight collisions toward the nearer surface. Gives
                           motion boundaries with real per-pixel motion, at
                           the cost of holes where no source pixel lands —
                           those pixels fall back to the gather result (the
                           old, hole-free behavior). Works without depth too
                           (collisions become a plain average) but depth is
                           recommended for sharp occlusion edges.

    Returns:
        (H, W, C) float32 interpolated frame in linear space.
    """
    # Partial warp: move frame N forward by fraction t of its forward flow
    warp_fwd = warp_frame(frame_n, flow_fwd_n * t)

    # Partial warp: move frame N+1 backward by fraction (1-t) of its backward flow
    # Negation because backward flow points back to N; we want to move N+1 toward N
    warp_bwd = warp_frame(frame_n1, -flow_bwd_n1 * (1.0 - t))

    # Warp depth arrays the same way (warp_frame handles 2D arrays natively)
    depth_fwd: np.ndarray | None = None
    depth_bwd: np.ndarray | None = None
    if depth_n is not None and depth_n1 is not None:
        depth_fwd = warp_frame(depth_n, flow_fwd_n * t)
        depth_bwd = warp_frame(depth_n1, -flow_bwd_n1 * (1.0 - t))

    gathered = depth_aware_blend(warp_fwd, warp_bwd, depth_fwd, depth_bwd, t, sigmoid_sharpness)

    if not use_splatting:
        return gathered

    # Forward-splat: push N's pixels along their own forward flow (scaled by
    # t), and N+1's pixels along their own backward flow (scaled by 1-t) —
    # same physical motion, opposite direction of traversal from gather.
    # The negation on the N+1 side mirrors the one in warp_bwd above: Blender's
    # backward-flow convention is pixel(p) in N+1 == pixel(p - bwd_flow(p)) in
    # N, so the pixel's own trajectory from N to N+1 runs along -bwd_flow.
    splat_n, valid_n, z_n = splat_forward(
        frame_n, flow_fwd_n * t, depth=depth_n, sharpness=sigmoid_sharpness
    )
    splat_n1, valid_n1, z_n1 = splat_forward(
        frame_n1, -flow_bwd_n1 * (1.0 - t), depth=depth_n1, sharpness=sigmoid_sharpness
    )

    if z_n is not None and z_n1 is not None:
        combined = depth_aware_blend(splat_n, splat_n1, z_n, z_n1, t, sigmoid_sharpness)
    else:
        combined = (1.0 - t) * splat_n + t * splat_n1

    both_valid = valid_n & valid_n1
    only_n = valid_n & ~valid_n1
    only_n1 = valid_n1 & ~valid_n
    neither = ~(valid_n | valid_n1)

    result = np.empty_like(gathered)
    result[both_valid] = combined[both_valid]
    result[only_n] = splat_n[only_n]
    result[only_n1] = splat_n1[only_n1]
    result[neither] = gathered[neither]

    return result.astype(np.float32)
