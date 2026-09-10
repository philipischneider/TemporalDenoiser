# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Windows-only PySide6 desktop app that temporally denoises Blender OpenEXR Multilayer render sequences with NVIDIA OptiX, plus a second mode that generates interpolated in-between frames from the Vector/Depth passes. See `README.md` for the user-facing workflow and the required Blender pass setup.

## Commands

Run the app (no console entry point is installed by default; `src/main.py` inserts `src/` on `sys.path` itself):

```bash
python src/main.py
```

Install Python deps:

```bash
pip install -e .
```

Build the OptiX bridge (required before any denoise run — the app raises `FileNotFoundError` if the exe is missing):

```bash
cd optix_bridge
cmake -B build -DOPTIX_PATH="C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0"
cmake --build build --config Release
```

CMake writes the exe to `optix_bridge/build/Release/Release/optix_bridge.exe`, which is exactly the path `_DEFAULT_BRIDGE` in `src/denoisers/optix_denoiser.py` expects (`RUNTIME_OUTPUT_DIRECTORY` plus MSVC's per-config subdir). If you change the CMake output dir, update that constant too.

There is no test suite, linter config, or CI. `tools/inspect_exr.py <folder_or_file>` dumps EXR subimages/channels — handy for inspecting a real render's layer names before wiring up the pass mapping.

## Dependency caveat

All EXR I/O goes through **OpenImageIO** (`import OpenImageIO as oiio` in `src/core/exr_handler.py`, guarded by `_HAS_OIIO`), not the `OpenEXR` PyPI package. OpenImageIO's Python bindings aren't reliably pip-installable on Windows; install via `conda install -c conda-forge openimageio`. If EXR loading raises `ImportError`, that's why.

## Architecture

Import paths are rooted at `src/` (`from core.pipeline import ...`), so anything that imports the package must have `src/` on the path.

**Layers, from the bottom up:**

- `core/exr_handler.py` — the only place that touches OIIO. `load_exr()` returns an `ExrFrame` of named `ExrLayer`s and handles *both* EXR flavors Blender produces: multi-part (one subimage per pass, part name is the layer) and single-part multilayer (channels named `Combined.R`). `_find_layer()` does substring matching, so a config value of `"Combined"` matches `"ViewLayer.Combined"`. `save_exr()` always writes flat `R/G/B` channels — output EXRs are single-layer, not multilayer.
- `core/motion_vectors.py` — Blender Vector pass convention: **RG = backward flow** (where the pixel came from), **BA = forward flow**. Denoising uses backward only; interpolation uses both.
- `core/temporal_warp.py` — `warp_frame()` (per-channel `cv2.remap` with `map = grid - flow`) and `blend_temporal()`.
- `core/frame_interpolator.py` — warps frame N forward and N+1 backward to time *t*, then depth-aware sigmoid blend to resolve occlusion conflicts; falls back to linear blend when no Depth pass is configured.
- `denoisers/` — `BaseDenoiser` is the interface; `OptiXDenoiser` is the only implementation. **It is not an in-process library binding**: each `denoise()` call writes `.npy` buffers plus a `config.json` into a temp dir, spawns `optix_bridge.exe`, and reads back `output.npy`. The C++ side (`optix_bridge/optix_denoiser.cpp`) has a hand-rolled flat-JSON parser, so keep the config strictly flat string/int/bool.
- `core/pipeline.py` — `PipelineConfig` is the single config dataclass shared by both modes, built in `SettingsPanel.build_config()`. Two executors: `DenoisePipeline` (per frame: load → extract passes → warp previous denoised output → OptiX → optional secondary blend → save; optional `preview.mp4` at the end) and `InterpolatePipeline` (color frames from `input_folder`, vector/depth from `interp_source_folder`, output named `<stem>_<t*1000:03d>.exr`). Both are plain synchronous classes with a `request_stop()` flag and three callbacks (`on_progress`, `on_log`, `on_frame_ready`); the `PipelineWorker`/`InterpolateWorker` `QObject`s wrap them and bind those callbacks straight to Qt signals.
- `ui/` — `MainWindow` owns a `QThread` + worker pair per mode and does the moveToThread wiring; `SettingsPanel` owns all widget state and is the only place that constructs a `PipelineConfig`; `ProgressPanel` renders the color-coded log.

**Adding a setting** therefore means touching three places: the `PipelineConfig` field, the widget + `build_config()` in `settings_panel.py`, and the consuming pipeline.

## Behavioral notes worth knowing before changing pipeline code

- OptiX accumulates temporal history internally. `temporal_blend` in the UI is an *additional* Python-side blend on top of that; `PipelineConfig.temporal_blend` defaults to `0.15` (mirrored in the slider default in `settings_panel.py`) to match the README's ghosting-avoidance advice — keep both in sync if you change one.
- `detect_disocclusions()` needs a `forward_flow` argument to detect anything (otherwise it always returns an all-zero mask). `DenoisePipeline._extract_flow()` fetches the vector pass via `get_layer_4ch` and derives both backward and forward flow so the validity mask is actually populated; it falls back to backward-only (`get_layer_xy`, no disocclusion detection) if the pass has fewer than 4 channels.
- The video-preview step reads back saved output EXRs with `get_layer_rgb("default")`, which relies on `save_exr()` writing unprefixed channels; changing the write path breaks preview generation.
- `vector_scale` (and `interp_vector_scale`) exist because Blender's flow sign can appear inverted per render; `-1.0` flips it.
