# Temporal Denoiser

A desktop application for applying **temporal denoising** to Blender's OpenEXR Multilayer render sequences, using **NVIDIA OptiX** as the denoising engine.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Platform Windows](https://img.shields.io/badge/Platform-Windows-lightgrey)
![License MIT](https://img.shields.io/badge/License-MIT-green)

---

## Features

- Reads Blender's **OpenEXR Multilayer** files and extracts:
  - Noisy pass (Combined / Beauty)
  - Albedo guide pass (Denoising Albedo)
  - Normal guide pass (Denoising Normal)
  - Motion vectors pass (Vector) for temporal reprojection
- **NVIDIA OptiX 9.x temporal denoiser**, invoked via a compiled C++ bridge subprocess
- **Temporal denoising**: warps the previous denoised frame using motion vectors and feeds it as a guide to the denoiser, reducing flickering across frames
- **Video preview**: optionally generates a `preview.mp4` from the denoised output frames at the end of the pipeline (configurable FPS)
- **Dark-themed PySide6 GUI** with:
  - Folder pickers for input/output
  - Auto scan of EXR passes on Start (or manual via "Scan Passes" button)
  - Configurable pass name mapping (compatible with any Blender render layer naming)
  - Temporal blend weight slider
  - Real-time frame preview (with linear→sRGB tonemapping for display)
  - Color-coded log panel with elapsed time and ETA

---

## Project Structure

```
TemporalDenoiser/
├── src/
│   ├── main.py                        # Entry point
│   ├── ui/
│   │   ├── main_window.py             # Main window (toolbar, preview, splitter)
│   │   └── widgets/
│   │       ├── settings_panel.py      # Settings sidebar
│   │       └── progress_panel.py      # Progress bar + log
│   ├── core/
│   │   ├── exr_handler.py             # OpenEXR multilayer read/write (OpenImageIO)
│   │   ├── motion_vectors.py          # Blender Vector pass conversion
│   │   ├── temporal_warp.py           # cv2.remap warping + temporal blend
│   │   └── pipeline.py               # Pipeline orchestrator + QThread worker
│   ├── denoisers/
│   │   ├── base.py                    # Abstract denoiser interface
│   │   └── optix_denoiser.py          # OptiX via C++ bridge subprocess
│   └── utils/
│       └── frame_sequence.py          # EXR frame sequence detection
└── optix_bridge/
    ├── CMakeLists.txt                 # CMake build for the C++ bridge
    └── optix_denoiser.cpp             # OptiX temporal denoiser bridge
```

---

## Requirements

### Python dependencies

```
PySide6 >= 6.6
numpy >= 1.26
opencv-python >= 4.9
OpenEXR >= 3.2
```

Install with:
```bash
pip install -e .
```

Or manually:
```bash
pip install PySide6 numpy opencv-python OpenEXR
```

### OptiX C++ bridge

Build requirements:
- NVIDIA Driver 565+
- CUDA Toolkit 12.x
- OptiX SDK 9.x — download from [developer.nvidia.com/designworks/optix/download](https://developer.nvidia.com/designworks/optix/download)
- CMake 3.25+
- Visual Studio 2022 (MSVC)

Build steps:
```bash
cd optix_bridge
cmake -B build -DOPTIX_PATH="C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0"
cmake --build build --config Release
```

The compiled `optix_bridge.exe` will be placed at `optix_bridge/build/Release/Release/`.

---

## Running

```bash
python src/main.py
```

---

## Usage

1. **Input folder** — select the folder containing your Blender EXR frame sequence (e.g. `render/frame0001.exr`, `frame0002.exr`, …).
2. **Output folder** — where denoised EXR files will be saved (created automatically).
3. **HDR mode** — keep enabled for linear float renders (Cycles default).
4. **EXR Pass Mapping** — click **Scan Passes** to auto-detect layers from the first frame, or let it run automatically on **Start**. Adjust the dropdown selections if needed.
5. **Temporal** — enable for temporal stability. The blend weight controls how much the warped previous frame is mixed into the output (0 = OptiX temporal only, higher values add an additional Python-side blend).
6. **Video Preview** — optionally enable to generate a `preview.mp4` in the output folder after all frames are processed. Set the desired FPS.
7. Click **Start**.

### Blender render setup

In Blender (Cycles), enable these passes in the **View Layer Properties → Passes** panel:
- **Data → Vector** (motion vectors)
- **Denoising → Denoising Data** (adds Denoising Normal and Denoising Albedo)
- **Denoising → Noisy Image** (the unfiltered beauty pass used as input)

Set the output format to **OpenEXR Multilayer**.

---

## Architecture

### Temporal pipeline (per frame)

```
EXR Load
    │
    ├── Noisy pass   ──────────────────────────────────────────────┐
    ├── Albedo pass  ──────────────────────────────────────────────┤
    ├── Normal pass  ──────────────────────────────────────────────┤→ OptiX Denoiser (C++ bridge)
    ├── Vector pass → convert to backward flow                     │
    │                    │                                         │
    │              warp prev denoised frame ──── previousOutput ───┘
    │                    │
    │              validity mask (disocclusion detection)
    │
    └── Temporal blend (current denoised × prev warped)
              │
         Save EXR output

After all frames:
    └── (optional) Video preview → preview.mp4
```

### Motion vector convention

Blender's Vector pass stores:
- **RG channels**: backward flow — where each pixel came from in the previous frame (pixel space)
- **BA channels**: forward flow — where each pixel goes in the next frame

This implementation uses the RG channels to warp the previous frame. The **Vector scale** setting can be set to `-1.0` if the motion appears inverted for a given render.

### Note on temporal blend weight

The OptiX temporal denoiser already accumulates temporal history internally. The **Blend weight** in the UI applies an *additional* Python-side blend on top of OptiX's output. For scenes with significant camera motion, keeping the blend weight low (< 0.15) is recommended to avoid ghosting artifacts.

---

## License

MIT
