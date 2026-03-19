# Temporal Denoiser

A desktop application for applying **temporal denoising** to Blender's OpenEXR Multilayer render sequences, using either **Intel Open Image Denoise (OIDN) 2.x** or **NVIDIA OptiX** as the denoising engine — selectable from the UI.

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
- **Two denoising engines**, switchable from the interface:
  - **OIDN 2.x** — Intel Open Image Denoise, loaded via ctypes from the official SDK. Supports CPU and CUDA (NVIDIA GPU).
  - **OptiX** — NVIDIA OptiX 8.x temporal denoiser, invoked via a compiled C++ bridge.
- **Temporal denoising**: warps the previous denoised frame using motion vectors and feeds it as a guide to the denoiser, reducing flickering across frames.
- **Dark-themed PySide6 GUI** with:
  - Folder pickers for input/output
  - Configurable pass name mapping (compatible with any Blender render layer naming)
  - Quality settings (High / Balanced / Fast)
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
│   │   ├── exr_handler.py             # OpenEXR multilayer read/write
│   │   ├── motion_vectors.py          # Blender Vector pass conversion
│   │   ├── temporal_warp.py           # cv2.remap warping + temporal blend
│   │   └── pipeline.py               # Pipeline orchestrator + QThread worker
│   ├── denoisers/
│   │   ├── base.py                    # Abstract denoiser interface
│   │   ├── oidn_denoiser.py           # OIDN 2.x via ctypes
│   │   └── optix_denoiser.py          # OptiX via C++ bridge subprocess
│   └── utils/
│       └── frame_sequence.py          # EXR frame sequence detection
└── optix_bridge/
    ├── CMakeLists.txt                 # CMake build for the C++ bridge
    └── optix_denoiser.cpp             # OptiX 8.x temporal denoiser bridge
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
pip install PySide6 numpy opencv-python OpenEXR
```

### OIDN SDK (required for OIDN engine)

OIDN is **not** on PyPI. Install the official prebuilt SDK:

1. Download from [github.com/RenderKit/oidn/releases](https://github.com/RenderKit/oidn/releases)
   - File: `oidn-2.x.x.x86_64.windows.zip`
2. Extract to a folder, e.g. `C:\oidn`
3. Set the environment variable:
   ```
   OIDN_PATH=C:\oidn
   ```
   On Windows 11: search for **"Edit the system environment variables"** → Environment Variables → New user variable.

The app will locate `OpenImageDenoise.dll` automatically via `%OIDN_PATH%\bin\`.

OIDN supports **CUDA** (NVIDIA GPU) and **CPU** modes, selectable from the UI.

### OptiX C++ bridge (required for OptiX engine)

Build requirements:
- NVIDIA Driver 565+
- CUDA Toolkit 12.6+
- OptiX SDK 8.1+ — download from [developer.nvidia.com/designworks/optix/download](https://developer.nvidia.com/designworks/optix/download)
- CMake 3.25+

Build steps:
```bash
cd optix_bridge
cmake -B build -DOPTIX_PATH="C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0"
cmake --build build --config Release
```

The compiled `optix_bridge.exe` will be placed at `optix_bridge/build/Release/`.

---

## Running

```bash
python src/main.py
```

---

## Usage

1. **Input folder** — select the folder containing your Blender EXR frame sequence (e.g. `render/frame0001.exr`, `frame0002.exr`, …).
2. **Output folder** — where denoised EXR files will be saved (created automatically).
3. **Engine** — choose OIDN or OptiX.
4. **EXR Pass Mapping** — enter the pass names exactly as they appear in the EXR file. The defaults (`Combined`, `Denoising Albedo`, `Denoising Normal`, `Vector`) match Blender's standard naming when the **Denoising** checkbox is enabled in the View Layer properties.
5. **Temporal** — enable for temporal stability. The blend weight controls how much the previous frame contributes (0 = single-frame only, 1 = maximum temporal stability).
6. Click **Start**.

### Blender render setup

In Blender (Cycles), enable these passes in the **View Layer Properties → Passes** panel:
- **Data → Vector** (motion vectors)
- **Denoising → Denoising Data** (adds Denoising Normal and Denoising Albedo)
- **Denoising → Noisy Image** (the unfiltered beauty pass to use as input)

Set the output format to **OpenEXR Multilayer**.

---

## Architecture

### Temporal pipeline (per frame)

```
EXR Load
    │
    ├── Noisy pass   ──────────────────────────────────────────────┐
    ├── Albedo pass  ──────────────────────────────────────────────┤
    ├── Normal pass  ──────────────────────────────────────────────┤→ Denoiser (OIDN / OptiX)
    ├── Vector pass → convert to backward flow                     │
    │                    │                                         │
    │              warp prev denoised frame ──── previousOutput ───┘
    │                    │
    │              validity mask (disocclusion detection)
    │
    └── Temporal blend (current denoised × prev warped)
              │
         Save EXR output
```

### Motion vector convention

Blender's Vector pass stores:
- **RG channels**: backward flow — where each pixel came from in the previous frame (pixel space)
- **BA channels**: forward flow — where each pixel goes in the next frame

This implementation uses the RG channels to warp the previous frame. The **Vector scale** setting can be set to `-1.0` if the motion appears inverted for a given render.

---

## License

MIT
