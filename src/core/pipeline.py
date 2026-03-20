"""Main denoising pipeline orchestrator."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from core.exr_handler import load_exr, save_exr
from core.motion_vectors import blender_vector_to_backward_flow, detect_disocclusions
from core.temporal_warp import warp_frame, blend_temporal
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
    temporal_blend: float = 0.8
    vector_scale: float = 1.0
    pass_noisy: str = "Combined"
    pass_albedo: str = "Denoising Albedo"
    pass_normal: str = "Denoising Normal"
    pass_vector: str = "Vector"

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.input_folder.is_dir():
            errors.append(f"Input folder does not exist: {self.input_folder}")
        if not self.output_folder or str(self.output_folder).strip() == "":
            errors.append("Output folder is not set.")
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
            if cfg.temporal and cfg.pass_vector and prev_denoised is not None:
                flow = self._extract_flow(exr, cfg, log)

            # Warp previous frame
            warped_prev: Optional[np.ndarray] = None
            validity_mask: Optional[np.ndarray] = None
            if cfg.temporal and prev_denoised is not None and flow is not None:
                warped_prev = warp_frame(prev_denoised, flow)
                disocclusion = detect_disocclusions(flow)
                validity_mask = (~disocclusion).astype(np.float32)

            # --- diagnostic ---
            layer = exr._find_layer(cfg.pass_noisy)
            log(f"  noisy layer channels: {list(layer.channels.keys())}", "info")
            log(f"  noisy  R:[{noisy[:,:,0].min():.4f},{noisy[:,:,0].max():.4f}]"
                f"  G:[{noisy[:,:,1].min():.4f},{noisy[:,:,1].max():.4f}]"
                f"  B:[{noisy[:,:,2].min():.4f},{noisy[:,:,2].max():.4f}]", "info")

            # Denoise
            denoised = denoiser.denoise(
                noisy=noisy,
                albedo=albedo,
                normal=normal,
                prev_output=warped_prev,
                flow=flow,
            )

            log(f"  denoised range: [{denoised.min():.4f}, {denoised.max():.4f}]", "info")
            log(f"  denoised R:[{denoised[:,:,0].min():.4f},{denoised[:,:,0].max():.4f}]"
                f"  G:[{denoised[:,:,1].min():.4f},{denoised[:,:,1].max():.4f}]"
                f"  B:[{denoised[:,:,2].min():.4f},{denoised[:,:,2].max():.4f}]", "info")

            # Temporal blend (for OIDN which handles temporal internally,
            # this is a secondary soft blend; for custom warp, it's primary)
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
                save_exr(out_path, denoised, layer_name=cfg.pass_noisy)
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

    def _extract_flow(self, exr, cfg: PipelineConfig, log) -> Optional[np.ndarray]:
        try:
            raw = exr.get_layer_xy(cfg.pass_vector)
            return blender_vector_to_backward_flow(raw, scale=cfg.vector_scale)
        except KeyError as e:
            log(f"Vector pass '{cfg.pass_vector}' not found: {e}", "warning")
            return None


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
