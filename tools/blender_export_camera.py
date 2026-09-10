"""Blender-side camera exporter for the DLSS-G (Streamline) interpolation backend.

Run this INSIDE Blender (Scripting tab, or `blender --background scene.blend
--python blender_export_camera.py`), not as part of the TemporalDenoiser app.
It has no dependency on the rest of this repo — it only needs bpy.

Exports one JSON record per rendered frame with everything the C++ DLSS-G
host needs to fill `sl::Constants` for that frame:

    - cameraPos, cameraFwd, cameraRight (world space; cameraUp is derived by
      Streamline's own `recalculateCameraMatrices()` helper via cross product,
      so we don't need to export it — included anyway for sanity-checking).
    - cameraFOV (vertical, radians), cameraAspectRatio, cameraNear, cameraFar.

Deliberately NOT exported: Blender's own projection matrix. The DLSS-G host
builds `cameraViewToClip` itself from FOV/aspect/near/far using a standard
D3D perspective formula — trying to convert Blender's projection matrix
convention (OpenGL-style, right-handed) into D3D clip space directly would be
another sign/handedness puzzle for no benefit, since FOV+aspect+near+far
already fully determine a standard perspective matrix.

Output: a single JSON file, `<output>/camera.json`, keyed by frame number as
a string (matching the frame numbers frame_sequence.py extracts from EXR
filenames) so the two sequences line up regardless of naming gaps.

NOTE: this has not been validated end-to-end against a live Streamline
session (no GPU/D3D12 context available where this was written). If the
generated intermediate frames show swimming/tearing, the two likeliest
culprits are `depthInverted` (set on the C++ side, not here) or a flipped
`cameraFwd` sign — both are one-line fixes, not a sign this file is wrong
in its entirety.
"""
import json
import math
import os

import bpy


def export_camera_track(output_dir: str, frame_start: int = None, frame_end: int = None) -> str:
    """Export one camera record per frame to <output_dir>/camera.json.

    Args:
        output_dir:  Folder to write camera.json into (created if missing).
        frame_start: First frame (defaults to scene.frame_start).
        frame_end:   Last frame, inclusive (defaults to scene.frame_end).

    Returns:
        Path to the written camera.json.
    """
    scene = bpy.context.scene
    cam_obj = scene.camera
    if cam_obj is None or cam_obj.data.type != 'PERSP':
        raise RuntimeError(
            "Active scene camera is missing or not a perspective camera. "
            "DLSS-G requires a perspective projection (orthographicProjection "
            "is supported by Streamline in principle, but this exporter only "
            "handles the common case)."
        )

    frame_start = scene.frame_start if frame_start is None else frame_start
    frame_end = scene.frame_end if frame_end is None else frame_end

    render = scene.render
    res_x = render.resolution_x * render.resolution_percentage / 100.0
    res_y = render.resolution_y * render.resolution_percentage / 100.0
    aspect_ratio = (res_x / res_y) * (render.pixel_aspect_x / render.pixel_aspect_y)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    records = {}

    original_frame = scene.frame_current
    try:
        for frame in range(frame_start, frame_end + 1):
            scene.frame_set(frame)
            depsgraph.update()

            cam_eval = cam_obj.evaluated_get(depsgraph)
            cam_data = cam_eval.data
            mat_world = cam_eval.matrix_world

            # Blender camera local axes -> world space. Blender cameras look
            # down their own local -Z, with +Y as local "up" and +X as local
            # "right" (right-handed, Z-up world). Streamline's own
            # recalculateCameraMatrices() only needs Right + Fwd (it derives
            # Up via cross product) but Up is exported too for sanity checks.
            right = mat_world.col[0].xyz.normalized()
            up = mat_world.col[1].xyz.normalized()
            forward = (-mat_world.col[2].xyz).normalized()
            position = mat_world.translation

            # Vertical FOV in radians, matching the aspect-ratio-corrected
            # sensor fit Blender actually uses for this camera/resolution.
            fov_y = cam_data.angle_y if cam_data.sensor_fit != 'VERTICAL' else cam_data.angle
            if cam_data.sensor_fit == 'AUTO':
                # angle_y already accounts for AUTO fit via Blender's own
                # angle_x/angle_y properties, which factor in the resolution.
                fov_y = cam_data.angle_y

            records[str(frame)] = {
                "cameraPos": [position.x, position.y, position.z],
                "cameraFwd": [forward.x, forward.y, forward.z],
                "cameraRight": [right.x, right.y, right.z],
                "cameraUp": [up.x, up.y, up.z],  # sanity-check only, not required by SL
                "cameraFOV": fov_y,
                "cameraAspectRatio": aspect_ratio,
                "cameraNear": cam_data.clip_start,
                "cameraFar": cam_data.clip_end,
            }
    finally:
        scene.frame_set(original_frame)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "camera.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"[blender_export_camera] Wrote {len(records)} frames to {out_path}")
    return out_path


if __name__ == "__main__":
    # Default: export next to the current .blend file's render output path,
    # or alongside the .blend itself if no output path is configured.
    scene = bpy.context.scene
    out_dir = bpy.path.abspath(scene.render.filepath) or bpy.path.abspath("//")
    if not os.path.isdir(out_dir):
        out_dir = os.path.dirname(out_dir) or bpy.path.abspath("//")
    export_camera_track(out_dir)
