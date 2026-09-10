"""Main denoising pipeline orchestrator."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from core.exr_handler import load_exr, save_exr
from core.motion_vectors import (
    blender_vector_to_backward_flow,
    blender_vector_to_forward_flow,
    detect_disocclusions,
)
from core.temporal_warp import warp_frame, blend_temporal
from core.frame_interpolator import interpolate_frame, compute_interpolation_t_values
from utils.frame_sequence import detect_frame_sequence
from denoisers.base import BaseDenoiser


@dataclass
class PipelineConfig:
    input_folder: Path
    output_folder: Path
    hdr: bool = True
    video_preview: bool = False
    video_fps: int = 24
    temporal: bool = True
    temporal_blend: float = 0.15
    vector_scale: float = 1.0
    pass_noisy: str = "Combined"
    pass_albedo: str = "Denoising Albedo"
    pass_normal: str = "Denoising Normal"
    pass_vector: str = "Vector"
    compression: str = "zip"
    # Frame interpolation
    interpolate: bool = False
    interpolation_factor: int = 2
    interp_source_folder: Path = field(default_factory=lambda: Path(""))
    pass_depth: str = ""
    interp_vector_scale: float = 1.0
    interp_sigmoid_sharpness: float = 10.0
    interp_use_splatting: bool = False

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.input_folder.is_dir():
            errors.append(f"Input folder does not exist: {self.input_folder}")
        if not self.output_folder or str(self.output_folder).strip() == "":
            errors.append("Output folder is not set.")
        if self.interpolate and self.interpolation_factor < 2:
            errors.append("Interpolation factor must be at least 2.")
        return errors


class DenoisePipeline:
    """Stateless pipeline executor. Runs synchronously."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(
        self,
        on_progress=None,   # (current, total, eta) -> None
        on_log=None,        # (message, level) -> None
        on_frame_ready=None # (frame_num, array) -> None
    ) -> None:
        cfg = self.config
        log = on_log or (lambda msg, lvl="info": None)
        progress = on_progress or (lambda c, t, e: None)
        frame_ready = on_frame_ready or (lambda n, a: None)

        # Discover frames
        frames = detect_frame_sequence(cfg.input_folder)
        if not frames:
            raise RuntimeError(f"No EXR frames found in {cfg.input_folder}")

        log(f"Found {len(frames)} frames.", "info")
        log("Denoiser: OptiX", "info")

        # Build denoiser
        denoiser = self._build_denoiser()
        log(f"Denoiser initialized.", "success")

        cfg.output_folder.mkdir(parents=True, exist_ok=True)

        prev_denoised: Optional[np.ndarray] = None
        t_start = time.monotonic()

        for idx, frame_path in enumerate(frames):
            if self._stop_requested:
                log("Stop requested — halting.", "warning")
                break

            frame_num = idx + 1
            log(f"Processing {frame_path.name}…", "info")

            try:
                exr = load_exr(frame_path)
            except Exception as e:
                log(f"Failed to load {frame_path.name}: {e}", "error")
                continue

            # Extract passes
            noisy = self._extract_pass(exr, cfg.pass_noisy, "RGB", log)
            if noisy is None:
                log(f"Noisy pass not found in {frame_path.name}, skipping.", "error")
                continue

            albedo = self._extract_pass(exr, cfg.pass_albedo, "RGB", log) if cfg.pass_albedo else None
            normal = self._extract_pass_xyz(exr, cfg.pass_normal, log) if cfg.pass_normal else None

            # Motion vectors for temporal
            flow: Optional[np.ndarray] = None
            forward_flow: Optional[np.ndarray] = None
            if cfg.temporal and cfg.pass_vector and prev_denoised is not None:
                flow, forward_flow = self._extract_flow(exr, cfg, log)

            # Warp previous frame
            warped_prev: Optional[np.ndarray] = None
            validity_mask: Optional[np.ndarray] = None
            if cfg.temporal and prev_denoised is not None and flow is not None:
                warped_prev = warp_frame(prev_denoised, flow)
                disocclusion = detect_disocclusions(flow, forward_flow=forward_flow)
                validity_mask = (~disocclusion).astype(np.float32)

            # Denoise
            denoised = denoiser.denoise(
                noisy=noisy,
                albedo=albedo,
                normal=normal,
                prev_output=warped_prev,
                flow=flow,
            )

            # Temporal blend — OptiX already accumulates temporal history
            # internally, so this is a secondary soft blend on top of that;
            # keep cfg.temporal_blend low (see PipelineConfig) to avoid ghosting.
            if warped_prev is not None and cfg.temporal:
                denoised = blend_temporal(
                    denoised, warped_prev,
                    blend_weight=cfg.temporal_blend,
                    validity_mask=validity_mask,
                )

            prev_denoised = denoised

            # Save output
            out_path = cfg.output_folder / frame_path.name
            try:
                save_exr(out_path, denoised, layer_name=cfg.pass_noisy, compression=cfg.compression)
                log(f"Saved → {out_path.name}", "success")
            except Exception as e:
                log(f"Failed to save {out_path.name}: {e}", "error")

            frame_ready(frame_num, denoised)

            # ETA
            elapsed = time.monotonic() - t_start
            rate = frame_num / elapsed if elapsed > 0 else 0
            remaining = len(frames) - frame_num
            eta = remaining / rate if rate > 0 else -1.0
            progress(frame_num, len(frames), eta)

        denoiser.cleanup()

        if cfg.video_preview and not self._stop_requested:
            self._create_video(frames, cfg, log)

        log("Pipeline finished.", "success")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_video(self, frames, cfg: PipelineConfig, log) -> None:
        import cv2

        output_files = [cfg.output_folder / f.name for f in frames]
        output_files = [p for p in output_files if p.is_file()]
        if not output_files:
            log("No output frames found for video preview.", "warning")
            return

        try:
            first_exr = load_exr(output_files[0])
            first_arr = first_exr.get_layer_rgb("default")
        except Exception as e:
            log(f"Could not read output frames for video preview: {e}", "error")
            return

        h, w = first_arr.shape[:2]
        video_path = cfg.output_folder / "preview.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, float(cfg.video_fps), (w, h))

        log(f"Creating video preview ({len(output_files)} frames @ {cfg.video_fps} fps)…", "info")
        for path in output_files:
            try:
                exr = load_exr(path)
                arr = exr.get_layer_rgb("default")
            except Exception:
                continue
            arr = np.clip(arr ** (1.0 / 2.2), 0.0, 1.0)
            arr = (arr * 255).astype(np.uint8)
            # OpenCV expects BGR
            writer.write(arr[:, :, ::-1])

        writer.release()
        log(f"Video preview saved → {video_path.name}", "success")

    def _build_denoiser(self) -> BaseDenoiser:
        from denoisers.optix_denoiser import OptiXDenoiser
        return OptiXDenoiser(hdr=self.config.hdr, temporal=self.config.temporal)

    def _extract_pass(self, exr, name: str, order: str, log) -> Optional[np.ndarray]:
        if not name:
            return None
        try:
            return exr.get_layer_rgb(name)
        except KeyError as e:
            log(f"Pass '{name}' not found: {e}", "warning")
            return None

    def _extract_pass_xyz(self, exr, name: str, log) -> Optional[np.ndarray]:
        if not name:
            return None
        try:
            return exr.get_layer_xyz(name)
        except KeyError as e:
            log(f"Pass '{name}' not found: {e}", "warning")
            return None

    def _extract_flow(
        self, exr, cfg: PipelineConfig, log
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (backward_flow, forward_flow) for disocclusion detection.

        Backward flow (RG) warps the previous frame onto the current one;
        forward flow (BA) is only needed for the forward-backward consistency
        check in detect_disocclusions(). Falls back to backward-only (via
        get_layer_xy) if the vector pass doesn't carry 4 channels.
        """
        try:
            layer = exr._find_layer(cfg.pass_vector)
            if len(layer.channels) >= 4:
                raw = exr.get_layer_4ch(cfg.pass_vector)
                backward = blender_vector_to_backward_flow(raw, scale=cfg.vector_scale)
                forward = blender_vector_to_forward_flow(raw, scale=cfg.vector_scale)
                return backward, forward
        except KeyError:
            pass
        try:
            raw = exr.get_layer_xy(cfg.pass_vector)
            backward = blender_vector_to_backward_flow(raw, scale=cfg.vector_scale)
            return backward, None
        except KeyError as e:
            log(f"Vector pass '{cfg.pass_vector}' not found: {e}", "warning")
            return None, None


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def _interp_output_name(source_path: Path, t: float) -> str:
    """Return output filename for an interpolated sub-frame.

    frame_0042.exr at t=0.0  → frame_0042_000.exr  (original frame copy)
    frame_0042.exr at t=0.5  → frame_0042_500.exr  (2x interpolated)
    frame_0042.exr at t=0.25 → frame_0042_250.exr  (4x interpolated)
    """
    frac = round(t * 1000)
    return f"{source_path.stem}_{frac:03d}.exr"


class InterpolatePipeline:
    """Generate intermediate frames from an existing EXR sequence using Blender's
    Vector (motion) and optional Depth passes.

    Reads color frames from input_folder, motion vectors and depth from
    interp_source_folder (falls back to input_folder when not set).
    Writes interpolated frames to output_folder.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(
        self,
        on_progress=None,
        on_log=None,
        on_frame_ready=None,
    ) -> None:
        cfg = self.config
        log = on_log or (lambda msg, lvl="info": None)
        progress = on_progress or (lambda c, t, e: None)
        frame_ready = on_frame_ready or (lambda n, a: None)

        # Resolve source folder for vector/depth passes
        src_folder = cfg.interp_source_folder \
            if cfg.interp_source_folder.parts else cfg.input_folder

        input_frames = detect_frame_sequence(cfg.input_folder)
        if not input_frames:
            raise RuntimeError(f"No EXR frames found in {cfg.input_folder}")

        src_frames = detect_frame_sequence(src_folder)
        if not src_frames:
            raise RuntimeError(f"No EXR frames found in source folder {src_folder}")

        if len(input_frames) != len(src_frames):
            log(
                f"Frame count mismatch: input={len(input_frames)}, source={len(src_frames)}. "
                "Using minimum count.",
                "warning",
            )
        n_pairs = min(len(input_frames), len(src_frames))
        input_frames = input_frames[:n_pairs]
        src_frames = src_frames[:n_pairs]

        log(f"Found {n_pairs} frames. Interpolation factor: {cfg.interpolation_factor}x.", "info")

        t_values = compute_interpolation_t_values(cfg.interpolation_factor)
        total_output = (n_pairs - 1) * cfg.interpolation_factor + 1
        log(f"Output frames: {total_output}.", "info")

        cfg.output_folder.mkdir(parents=True, exist_ok=True)

        output_idx = 0
        t_start = time.monotonic()

        for idx in range(n_pairs - 1):
            if self._stop_requested:
                log("Stop requested — halting.", "warning")
                break

            log(f"Interpolating pair {idx + 1}/{n_pairs - 1}: {input_frames[idx].name} …", "info")

            # Load color frames
            frame_n = self._load_color(input_frames[idx], cfg, log)
            frame_n1 = self._load_color(input_frames[idx + 1], cfg, log)
            if frame_n is None or frame_n1 is None:
                output_idx += cfg.interpolation_factor
                continue

            # Load vector passes (4 channels needed for forward+backward)
            vector_n = self._load_vector(src_frames[idx], cfg, log)
            vector_n1 = self._load_vector(src_frames[idx + 1], cfg, log)

            flow_fwd_n = blender_vector_to_forward_flow(vector_n, cfg.interp_vector_scale) \
                if vector_n is not None else None
            flow_bwd_n1 = blender_vector_to_backward_flow(vector_n1, cfg.interp_vector_scale) \
                if vector_n1 is not None else None

            # Fall back to zero flow if vectors are unavailable
            if flow_fwd_n is None:
                flow_fwd_n = np.zeros((*frame_n.shape[:2], 2), dtype=np.float32)
                log("  Vector pass missing — using zero flow for forward.", "warning")
            if flow_bwd_n1 is None:
                flow_bwd_n1 = np.zeros((*frame_n1.shape[:2], 2), dtype=np.float32)
                log("  Vector pass missing — using zero flow for backward.", "warning")

            # Load depth passes (optional)
            depth_n, depth_n1 = self._load_depth_pair(
                src_frames[idx], src_frames[idx + 1], cfg, log
            )

            # Write the original frame N as _000
            out_path = cfg.output_folder / _interp_output_name(input_frames[idx], 0.0)
            try:
                save_exr(out_path, frame_n, layer_name=cfg.pass_noisy, compression=cfg.compression)
            except Exception as e:
                log(f"  Failed to save {out_path.name}: {e}", "error")

            output_idx += 1
            frame_ready(output_idx, frame_n)

            # Generate and write sub-frames
            for t in t_values:
                if self._stop_requested:
                    break

                interp = interpolate_frame(
                    frame_n, frame_n1,
                    flow_fwd_n, flow_bwd_n1,
                    t,
                    depth_n=depth_n,
                    depth_n1=depth_n1,
                    sigmoid_sharpness=cfg.interp_sigmoid_sharpness,
                    use_splatting=cfg.interp_use_splatting,
                )

                out_path = cfg.output_folder / _interp_output_name(input_frames[idx], t)
                try:
                    save_exr(out_path, interp, layer_name=cfg.pass_noisy, compression=cfg.compression)
                    log(f"  Saved {out_path.name}", "success")
                except Exception as e:
                    log(f"  Failed to save {out_path.name}: {e}", "error")

                output_idx += 1
                frame_ready(output_idx, interp)

            # ETA
            elapsed = time.monotonic() - t_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (n_pairs - 1) - (idx + 1)
            eta = remaining / rate if rate > 0 else -1.0
            progress(idx + 1, n_pairs - 1, eta)

        # Write the last frame as _000
        if not self._stop_requested:
            last_color = self._load_color(input_frames[-1], cfg, log)
            if last_color is not None:
                out_path = cfg.output_folder / _interp_output_name(input_frames[-1], 0.0)
                try:
                    save_exr(out_path, last_color, layer_name=cfg.pass_noisy, compression=cfg.compression)
                    log(f"Saved final frame → {out_path.name}", "success")
                except Exception as e:
                    log(f"Failed to save final frame: {e}", "error")

        log("Interpolation pipeline finished.", "success")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_color(self, path: Path, cfg: PipelineConfig, log) -> Optional[np.ndarray]:
        try:
            exr = load_exr(path)
            return exr.get_layer_rgb(cfg.pass_noisy)
        except Exception as e:
            log(f"Could not load color from {path.name}: {e}", "error")
            return None

    def _load_vector(self, path: Path, cfg: PipelineConfig, log) -> Optional[np.ndarray]:
        if not cfg.pass_vector:
            return None
        try:
            exr = load_exr(path)
            return exr.get_layer_4ch(cfg.pass_vector)
        except Exception as e:
            log(f"Could not load vector pass from {path.name}: {e}", "warning")
            return None

    def _load_depth_pair(
        self,
        path_n: Path,
        path_n1: Path,
        cfg: PipelineConfig,
        log,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not cfg.pass_depth:
            return None, None
        try:
            exr_n = load_exr(path_n)
            depth_n = exr_n.get_layer_z(cfg.pass_depth)
        except Exception as e:
            log(f"Could not load depth from {path_n.name}: {e}", "warning")
            return None, None
        try:
            exr_n1 = load_exr(path_n1)
            depth_n1 = exr_n1.get_layer_z(cfg.pass_depth)
        except Exception as e:
            log(f"Could not load depth from {path_n1.name}: {e}", "warning")
            return None, None
        return depth_n, depth_n1


# ---------------------------------------------------------------------------
# Qt Worker wrapper
# ---------------------------------------------------------------------------

class PipelineWorker(QObject):
    progress = Signal(int, int, float)    # current, total, eta_seconds
    log_message = Signal(str, str)        # message, level
    frame_ready = Signal(int, np.ndarray) # frame_num, image array
    finished = Signal()
    error = Signal(str)

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__()
        self._pipeline = DenoisePipeline(config)

    def request_stop(self) -> None:
        self._pipeline.request_stop()

    @Slot()
    def run(self) -> None:
        try:
            self._pipeline.run(
                on_progress=self.progress.emit,
                on_log=self.log_message.emit,
                on_frame_ready=self.frame_ready.emit,
            )
            self.finished.emit()
        except Exception as exc:
            self.error.emit(str(exc))


class InterpolateWorker(QObject):
    progress = Signal(int, int, float)
    log_message = Signal(str, str)
    frame_ready = Signal(int, np.ndarray)
    finished = Signal()
    error = Signal(str)

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__()
        self._pipeline = InterpolatePipeline(config)

    def request_stop(self) -> None:
        self._pipeline.request_stop()

    @Slot()
    def run(self) -> None:
        try:
            self._pipeline.run(
                on_progress=self.progress.emit,
                on_log=self.log_message.emit,
                on_frame_ready=self.frame_ready.emit,
            )
            self.finished.emit()
        except Exception as exc:
            self.error.emit(str(exc))
