"""Export an EXR sequence into the .npy format dlssg_bridge.exe's
--visual-test-sequence mode reads, so you can watch real DLSS-G frame
generation running on your own Blender renders (not the synthetic test
pattern baked into the bridge).

This is a viewer, not an exporter — dlssg_bridge has no way to save the
generated frames back to disk (see dlssg_bridge/README.md, "No readback
API"). It opens a window and shows DLSS-G interpolating your footage live.

Usage:
    python tools/export_dlssg_sequence.py <input_folder> <output_folder> \
        [--noisy Combined] [--depth "Depth"] [--vector Vector] \
        [--camera path/to/camera.json] [--vector-scale 1.0]

    <input_folder>  EXR sequence (same passes the main app uses: color,
                     depth, vector).
    <output_folder> Where color_0000.npy / depth_0000.npy / mvec_0000.npy
                     triplets are written, sequentially indexed regardless
                     of the original frame numbers.
    --camera        Optional camera.json from tools/blender_export_camera.py
                     (keyed by original Blender frame number). If given,
                     re-keys/reorders it into a sequential JSON array aligned
                     with the exported frames. If omitted, dlssg_bridge falls
                     back to a static default camera (fine for scenes with
                     little/no camera motion; will look wrong otherwise).

Then run:
    dlssg_bridge.exe --visual-test-sequence <output_folder> [width height]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from core.exr_handler import load_exr
from core.motion_vectors import blender_vector_to_backward_flow
from utils.frame_sequence import detect_frame_sequence


def export_sequence(
    input_folder: Path,
    output_folder: Path,
    pass_noisy: str = "Combined",
    pass_depth: str = "Depth",
    pass_vector: str = "Vector",
    vector_scale: float = 1.0,
    camera_json: Path | None = None,
) -> int:
    frames = detect_frame_sequence(input_folder)
    if not frames:
        raise RuntimeError(f"No EXR frames found in {input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)

    camera_by_frame_num = {}
    if camera_json is not None:
        with open(camera_json, encoding="utf-8") as f:
            camera_by_frame_num = json.load(f)

    camera_sequence = []
    exported = 0

    for path in frames:
        try:
            exr = load_exr(path)
            color = exr.get_layer_rgb(pass_noisy)
            depth = exr.get_layer_z(pass_depth) if pass_depth else None
            vector4 = exr.get_layer_4ch(pass_vector) if pass_vector else None
        except (KeyError, IOError) as e:
            print(f"  Skipping {path.name}: {e}")
            continue

        if depth is None:
            depth = np.full(color.shape[:2], 50.0, dtype=np.float32)
        if vector4 is not None:
            mvec = blender_vector_to_backward_flow(vector4, scale=vector_scale)
        else:
            mvec = np.zeros((*color.shape[:2], 2), dtype=np.float32)

        idx_str = f"{exported:04d}"
        np.save(output_folder / f"color_{idx_str}.npy", color.astype(np.float32))
        np.save(output_folder / f"depth_{idx_str}.npy", depth.astype(np.float32))
        np.save(output_folder / f"mvec_{idx_str}.npy", mvec.astype(np.float32))

        if camera_by_frame_num:
            # frame_sequence.py's frame-number extraction (last integer in
            # the filename) is how camera.json is keyed too.
            import re
            match = re.findall(r"(\d+)", path.stem)
            frame_num = match[-1] if match else None
            cam = camera_by_frame_num.get(str(int(frame_num))) if frame_num else None
            if cam is not None:
                camera_sequence.append({
                    "pos": cam["cameraPos"],
                    "fwd": cam["cameraFwd"],
                    "right": cam["cameraRight"],
                    "fov": cam["cameraFOV"],
                    "aspect": cam["cameraAspectRatio"],
                    "near": cam["cameraNear"],
                    "far": cam["cameraFar"],
                })

        exported += 1
        print(f"  [{exported}] {path.name} -> {idx_str}")

    if camera_sequence:
        if len(camera_sequence) != exported:
            print(
                f"  Warning: camera.json only matched {len(camera_sequence)}/{exported} "
                "frames -- dlssg_bridge will fall back to a static camera for this sequence "
                "(it requires a camera entry for every frame or none at all)."
            )
        else:
            with open(output_folder / "camera_sequence.json", "w", encoding="utf-8") as f:
                json.dump(camera_sequence, f, indent=2)
            print(f"  Wrote camera_sequence.json ({len(camera_sequence)} entries)")

    print(f"Exported {exported} frames to {output_folder}")
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--noisy", default="Combined")
    parser.add_argument("--depth", default="Depth")
    parser.add_argument("--vector", default="Vector")
    parser.add_argument("--vector-scale", type=float, default=1.0)
    parser.add_argument("--camera", type=Path, default=None)
    args = parser.parse_args()

    export_sequence(
        args.input_folder, args.output_folder,
        pass_noisy=args.noisy, pass_depth=args.depth, pass_vector=args.vector,
        vector_scale=args.vector_scale, camera_json=args.camera,
    )


if __name__ == "__main__":
    main()
