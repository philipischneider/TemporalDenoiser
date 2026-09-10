# dlssg_bridge — DLSS Frame Generation backend (abandoned — see "Why this was abandoned")

Alternative frame-interpolation backend using NVIDIA's real DLSS-G (the
"frame generator used in games"), instead of the analytic gather/splat
methods in `src/core/frame_interpolator.py`. Requires impersonating a live
D3D12 game renderer, because DLSS-G is delivered as a Streamline plugin that
hooks a real swap chain's `Present()` call — there is no offline/batch API.

**Status: DLSS-G frame generation was confirmed genuinely working** —
`DLSSGState::numFramesActuallyPresented` reporting `4` at a 4x multiplier
(matching the configured value exactly), the SL ImGui debug overlay's
`sl.dlss_g` panel active, and a visual marker test (magenta/cyan flashing
corner, alternates only on real frames) showing the marker correctly
holding/blending across generated frames instead of flickering in lockstep
with real motion. It worked on real Blender footage (a 91-frame EXR
sequence, "Hanged Revenant"), including with real per-frame camera data
exported from the source `.blend` file.

**The project was abandoned anyway**, for two independent reasons, either
one sufficient on its own:

1. **No readback API** (below) — there is no way to export a generated
   frame to disk, only to display it. This was known from early on.
2. **NVIDIA's licensing wall** — once generation actually started working,
   an on-screen message appeared stating that DLSS must be licensed for the
   title by contacting NVIDIA. This isn't a bug to route around; it's
   NVIDIA gating actual use of DLSS-G output behind a business
   relationship, for exactly the commercial-integration scenario this
   bridge doesn't have (a `dlssg_bridge.exe` batch tool isn't a licensed
   game).

Given (1) alone was already a hard wall for this project's actual goal
(batch EXR frame export), and (2) would block real use even if (1) didn't
exist, this backend is not being pursued further. Everything below is kept
as a detailed record of the investigation — the SDK integration bugs found
along the way are real and may be useful reference for anyone else
integrating Streamline, even though the path doesn't lead anywhere for this
project.

## Why this needs so much scaffolding

Unlike `optix_bridge/` (a stateless CLI: load buffers, call OptiX, write
output, exit), DLSS-G's only public entry point is
`IDXGISwapChain::Present()`, intercepted by a Streamline proxy swap chain.
So this bridge has to *be* a minimal D3D12 application — window, device,
command queue, swap chain — that "plays back" a pre-rendered EXR sequence
as if it were rendering it live, tags Streamline's required resources each
frame, and calls Present() to trigger frame generation.

It also needs real per-frame camera matrices (`sl::Constants`), which
Blender's rendered passes don't carry — see `tools/blender_export_camera.py`,
run separately *inside Blender* before this bridge can process a sequence
with camera motion. This part works: tested end-to-end with a real
`camera.json` exported from the source scene, re-keyed into
`camera_sequence.json` by `tools/export_dlssg_sequence.py`, and consumed by
`--visual-test-sequence`.

## Getting the SDK

The Streamline SDK (headers, libs, and the prebuilt signed `sl.dlss_g.dll`)
is public, no NVIDIA developer account needed:

1. Download a release zip from https://github.com/NVIDIA-RTX/Streamline/releases
   (this was written against v2.14.1).
2. Extract it somewhere, e.g. `C:\SDKs\streamline-sdk-v2.14.1`.
3. Point CMake at it: `cmake -B build -DSTREAMLINE_SDK_PATH="C:/SDKs/streamline-sdk-v2.14.1"`
   (same pattern as `-DOPTIX_PATH` for `optix_bridge/`).

No need to build Streamline from source — the release zip ships everything
needed except `sl.dlss_g.dll` itself cannot be rebuilt anyway (prebuilt-only,
per NVIDIA's own README).

For the debug overlay (see "The debugging tool that actually cracked this"),
you additionally need the **Development**-config DLLs from the same zip
(`bin/x64/development/*.dll`, includes `sl.imgui.dll` — not present in the
default `bin/x64/` Production set at all). CMake's post-build step copies
Production DLLs from `bin/x64/`; if you want the debug overlay, manually
copy `bin/x64/development/*.dll` over top after each build (CMakeLists.txt
doesn't have a switch for this — a manual step every time in this
investigation).

## Data flow (as originally planned — blocked, see "No readback API")

```
Python (pipeline.py)
  -> writes noisy/depth/vector .npy pairs + a config.json (frame N, N+1, camera.json path)
  -> spawns dlssg_bridge.exe
       -> uploads N and N+1 as D3D12 textures (color, depth, motion vectors)
       -> fills sl::Constants from camera.json (recalculateCameraMatrices() derives
          the temporal cross-frame matrices from consecutive frames' position/fwd/right)
       -> slSetTagForFrame(...) for depth + motion vectors
       -> Present() -> DLSS-G generates the intermediate frame(s)
       -> reads back the generated frame from the (proxy) back buffer
  -> writes output.npy, exits
  -> Python loads output.npy, saves as EXR (same as optix_denoiser.py's pattern)
```

**The last step ("reads back the generated frame") does not exist as a
public API** — see "No readback API" below. Everything above it is real and
was confirmed working, including the generation itself — the batch/export
intent this diagram describes is what's actually blocked.

## No readback API

`sl::DLSSGState` (the struct `slDLSSGGetState` fills in) has no field
pointing at the generated frame's texture/resource — only status info
(VRAM usage, frame counts, fences). `ProgrammingGuideDLSS_G.md` (1196 lines)
never mentions "readback", "capture", or "export". This isn't a documentation
gap found by searching — it reflects how the feature is architected: DLSS-G
presents generated frames directly to the swap chain, paced against vsync
for live display, with no synchronous per-generated-frame callback to the
app. Even screen-capture tools like OBS can't grab these frames (only
NVIDIA's own driver-level ShadowPlay can); someone asked NVIDIA's own forums
about using DLSS-G with Unreal's Movie Render Queue (an offline batch
renderer — the same category of use case as this project) and the answer
was that Frame Generation "focuses more on interactive real time
applications" while offline renderers are a fundamental mismatch.

This alone was enough to make the `config.json`/`.npy` batch path (the
original design) a dead end regardless of anything else. `--visual-test`/
`--visual-test-sequence` (below) were built specifically because they don't
need readback — they just show the result on screen.

## Visual test mode

```bash
dlssg_bridge.exe --visual-test [width height] [multiplier]   # defaults to 640x360, 2x
```

Opens a real on-screen window and presents a synthetic sequence (a bar
sliding across a checkerboard, generated in `generate_synthetic_sequence()`
— no Blender render needed) through DLSS-G, looping until the window is
closed or ESC is pressed. `multiplier` is 2/3/4/.../7 (this hardware's
`numFramesToGenerateMax` + 1 = 6, confirmed via `slDLSSGGetState`).

### Watching it on real footage instead of the synthetic pattern

```bash
python tools/export_dlssg_sequence.py <exr_folder> <npy_folder> \
    [--noisy Combined] [--depth Depth] [--vector Vector] [--camera camera.json]

dlssg_bridge.exe --visual-test-sequence <npy_folder> <width> <height> [multiplier]
```

`export_dlssg_sequence.py` reads an EXR sequence the same way the main app
does (`exr_handler.py`/`motion_vectors.py`) and writes
`color_%04d.npy`/`depth_%04d.npy`/`mvec_%04d.npy` triplets, sequentially
indexed regardless of the original frame numbers. `--visual-test-sequence`
loads that folder and runs it through the exact same DLSS-G path as
`--visual-test`, just real color/depth/motion-vector data instead of the
built-in moving bar. Tested end-to-end on a real 91-frame, 1080x1080 Blender
render.

**Camera data**: without `--camera`, the sequence uses a static default
camera — this visibly degrades quality on any footage with real camera
motion (confirmed on real footage in this investigation). With `--camera`
pointing at a `camera.json` from `tools/blender_export_camera.py` (run
separately *inside Blender*, save it next to the EXR sequence — it's
render metadata, not a bridge-specific artifact, so it belongs there and
not in the `.npy` output folder), the export script re-keys it into
`camera_sequence.json` (index-aligned with the npy files) and the bridge
uses real per-frame `sl::Constants`. Confirmed working end-to-end.

Width/height must match the EXR resolution — `--visual-test-sequence`
doesn't read it from the files, it's on you to pass the right numbers.

## The debugging tool that actually cracked this: SL ImGui overlay

For a long stretch of this investigation, every SL API call returned `eOk`
and yet a magenta/cyan corner-marker test (`stamp_marker()` — overwrites a
corner with a saturated color that flips only on real, disk-loaded frames;
picked so it can't be confused with scene content the way an earlier
black/white version could) showed the marker flickering in lockstep with
real motion — meaning displayed frame rate equalled real frame rate, i.e.
no generation, despite clean status codes everywhere.

The breakthrough tool was Streamline's own **Development-build ImGui debug
overlay** (`docs/Debugging - SL ImGUI (Realtime Data Inspection).md`,
Development DLLs only, `sl.imgui.dll`, feature `sl::kFeatureImGUI`).
Toggled with `Ctrl+Shift+Home` once loaded. It showed panels for
`sl.reflex`, `sl.interposer`, `sl.common` — but **no `sl.dlss_g` panel at
all**, and pressing the dlss_g-specific buffer-visualizer hotkey
(`Ctrl+Shift+Insert`) produced an explicit on-screen warning: *"debug mode
requires DLSS-G to be turned on."* This was the first piece of evidence, in
plain language from the plugin itself, that contradicted every `eOk` we'd
been getting — DLSS-G considered itself off regardless of what the API
calls reported.

### The actual fix: Reflex/PCL frame markers

The real requirement, undocumented anywhere in `ProgrammingGuideDLSS_G.md`:
DLSS-G's "am I actually engaged" state depends on Reflex seeing a genuine
per-frame marker sequence via `slPCLSetMarker` (declared in `sl_pcl.h`,
plugin `sl.pcl` — already loaded automatically as a Reflex dependency, no
extra `featuresToLoad` entry needed). Setting `ReflexOptions::mode =
eLowLatency` once is not enough on its own. Adding this per-frame sequence,
bracketing the existing per-frame work, is what made it work:

```cpp
slPCLSetMarker(sl::PCLMarker::eSimulationStart, *frameToken);
// ... build sl::Constants, slSetConstants ...
slPCLSetMarker(sl::PCLMarker::eSimulationEnd, *frameToken);
slPCLSetMarker(sl::PCLMarker::eRenderSubmitStart, *frameToken);
// ... record + submit the command list, tag depth/mvec ...
slPCLSetMarker(sl::PCLMarker::eRenderSubmitEnd, *frameToken);
// ... copy color into the back buffer ...
slPCLSetMarker(sl::PCLMarker::ePresentStart, *frameToken);
swapChain->Present(0, presentFlags);
slPCLSetMarker(sl::PCLMarker::ePresentEnd, *frameToken);
```

After adding this, `DLSSGState::numFramesActuallyPresented` immediately
started reporting `4` at a 4x multiplier (exactly matching the configured
value, on every frame) instead of `1`. The marker test flipped too: the
magenta/cyan corner started correctly holding/blending across generated
frames instead of flickering in lockstep with real motion. Both signals
agreed — this is what confirmed generation was genuinely happening, not
inferred from a status code.

(Re-asserting `slDLSSGSetOptions` every frame, tried as an intermediate
hypothesis before finding the marker requirement, made no difference on its
own — the markers were the actual fix.)

## Verified against the real SDK (v2.14.1), not just doc prose

The programming guide's code samples use some names that don't match the
actual v2.14.1 headers — checked by downloading the real release zip and
grepping `include/*.h` directly rather than trusting the markdown samples
verbatim:

- The guide's sample code tags motion vectors as `sl::kBufferTypeMvec`.
  The real constant in `sl_core_types.h` is **`sl::kBufferTypeMotionVectors`**.
- Confirmed real paths: `include/*.h`, `lib/x64/sl.interposer.lib`,
  `bin/x64/sl.interposer.dll` + `sl.dlss_g.dll` + `sl.common.dll` +
  `nvngx_dlssg.dll` (all four DLLs needed at runtime, not just the two
  `sl.*` ones) — plus `sl.reflex.dll` (see below) and, for the debug
  overlay only, `sl.imgui.dll` from the Development DLL set.
- Confirmed function signatures directly from `sl_core_api.h`: `slInit`,
  `slIsFeatureSupported`, `slSetFeatureLoaded`, `slSetTagForFrame`,
  `slSetConstants`, `slEvaluateFeature`, `slGetNewFrameToken`,
  `slAllocateResources`. `main.cpp` calls these with the real signatures.
- Feature enum is `sl::kFeatureDLSS_G` (`sl_core_types.h`), options struct
  is `sl::DLSSGOptions` with `sl::DLSSGMode` (`sl_dlss_g.h`).

## Fixed through a real build/run cycle (not just reading docs)

Every item below was caught by an actual MSVC error, a line in `sl.log`, or
a direct on-screen signal (marker test / debug overlay) — not something a
documentation read-through would have surfaced. Roughly in the order they
were found:

- **Free functions, not `sl::` members.** `slInit`, `slSetD3DDevice`,
  `slSetTagForFrame`, `slSetConstants`, `slIsFeatureSupported`,
  `slDLSSGSetOptions`, `slGetNewFrameToken`, `slShutdown` are all global
  functions (`sl_core_api.h` has no `namespace sl` wrapper around them) —
  only the *types* (`sl::Result`, `sl::Preferences`, ...) are namespaced.
- **`sl::recalculateCameraMatrices`**, not `sl::matrixHelpers::...` — it's a
  free function directly in `namespace sl` (`sl_matrix_helpers.h`).
- **`sl::Constants` in this SDK version has no `renderingGameFrames`
  field** — that name appears in some doc versions but not in v2.14.1's
  actual `sl_consts.h` (`kStructVersion2`).
- **`sl::kBufferTypeMotionVectors`**, not `sl::kBufferTypeMvec` — the
  DLSS-G programming guide's own code sample uses the latter, which doesn't
  exist in the real header.
- **`sl::ResourceType::eTex2d`**, not `Tex2d` (scoped enum, needs the `e`
  prefix like the rest of the SDK's enums).
- **`sl::Resource` has no 6-argument aggregate-style constructor.** It has
  real constructors: `Resource(type, native_ptr, state)` (3-arg, the one to
  use for D3D12) or `Resource(type, native_ptr, mem, view, state)` (5-arg,
  Vulkan-oriented). `state` is a `uint32_t` holding the resource's current
  `D3D12_RESOURCE_STATE` — it is not optional/nullable the way the doc's
  illustrative snippet implied.
- **`D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX` / `_PLACED_FOOTPRINT`**,
  not `D3D12_TEXTURE_COPY_LOCATION_*` — plain D3D12 API naming, unrelated
  to Streamline, just a name misremembered while writing the first draft.
- **DLSS-G hard-requires the Reflex plugin to be loaded alongside it.**
  Undocumented in `ProgrammingGuideDLSS_G.md` — found via `sl.log`:
  *"Plugin 'sl.dlss_g' will be unloaded since it requires plugin
  'sl.reflex' which is NOT loaded."* Fix: add `sl::kFeatureReflex` to
  `Preferences::featuresToLoad`, and copy `sl.reflex.dll` next to the exe.
- **NGX needs EITHER a real NVIDIA-issued `applicationId`, OR all three of
  `projectId` + `engine` + `engineVersion` together.** Confirmed by an
  NVIDIA staff reply on the dev forums (someone hit the exact same wall
  integrating DLSS into a custom Blender/Eevee-based renderer). Setting
  `projectId` alone is not enough — `sl.log` says *"Please provide correct
  application id when calling slInit"* until `engine`/`engineVersion` are
  also set. `projectId` must be a well-formed GUID string, not an arbitrary
  identifier — a non-hex placeholder silently breaks NGX init.
- **`slSetD3DDevice` must be called immediately after `D3D12CreateDevice`,
  before creating the command queue or swap chain** — calling it at the end
  of device setup (after queue+swapchain already exist) produces *"D3D or
  VK API hook is activated without device being created"* in `sl.log`.
- **`Preferences::flags` needs `PreferenceFlags::eUseFrameBasedResourceTagging`
  explicitly OR'd in** (not part of the struct's own defaults) or
  `slSetTagForFrame` fails outright with *"'slSetTagForFrame' SL API is
  called but 'PreferenceFlag::eUseFrameBasedResourceTagging' flag is not
  set!"*.
- **DLSS-G requires Reflex to be *actively enabled* (`slReflexSetOptions`
  with `mode = eLowLatency`), not just loaded as a plugin.** Loading
  `kFeatureReflex` alone leaves `ReflexOptions::mode` at its default
  (`eOff`), which triggers `DLSSGStatus::eFailReflexNotDetectedAtRuntime`
  ("Reflex must be turned on when DLSS-G is on") — a status you only see if
  you actually call `slDLSSGGetState` and check it.
- **Present with vsync ON (`Present(1, 0)`) silently prevented visible
  generation.** Switching to vsync-off + tearing (`Present(0,
  DXGI_PRESENT_ALLOW_TEARING)`, swap chain created with
  `DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING`) plus Reflex's own `frameLimitUs`
  for pacing (instead of a manual `Sleep`) was necessary groundwork, though
  not sufficient by itself (see next item).
- **The actual missing piece: `slPCLSetMarker` per-frame Reflex/PCL
  markers** — see "The actual fix" above. This is what took
  `numFramesActuallyPresented` from `1` to `4` at a 4x multiplier, and
  what the whole rest of this investigation was missing. Re-asserting
  `slDLSSGSetOptions` every frame (a reasonable-sounding intermediate guess)
  did not fix it on its own.

Debugging techniques that made all of this findable, roughly in order of
how much they revealed: (1) `Preferences::logLevel = eVerbose` +
`pathToLogsAndData = L"."` in code (not the `SL_LOG_LEVEL` env var, which
didn't surface these messages in this setup) — `sl.log` gets written next
to the exe with plugin-loading/validation errors; (2) a visual A/B marker
test (saturated corner color, flips only on real frames) since "motion
looks smooth" is not a reliable signal on its own — real footage at a
steady frame rate looks smooth with or without generation; (3) the SL
ImGui Development-build debug overlay, which gave a direct, explicit
statement from the plugin itself ("debug mode requires DLSS-G to be turned
on") instead of another status code to interpret.

## Known caveats (never fully re-verified after the marker fix, moot now)

These were flagged earlier in the investigation and never revisited once
the project moved to shutting the effort down — left here in case anyone
picks this back up:

- **`depthInverted`** (`sl::Constants`): whichever way Blender's Z pass
  actually goes (closer = smaller vs. closer = larger) has to match this
  flag exactly, or depth-based reprojection will treat foreground/background
  backwards.
- **`cameraFwd` sign / handedness**: `tools/blender_export_camera.py`
  exports Blender's native right-handed, Z-up world-space vectors directly.
  `recalculateCameraMatrices()` only cross-products Right x Fwd for Up, so
  it's self-consistent regardless of "true" world handedness — but if
  generated frames ever showed motion running backwards, negate `cameraFwd`
  first. Not something that came up on the tested footage.
- **No Hudless/UI buffers are tagged.** Blender renders have no HUD, so
  the whole "final color" *is* the hudless buffer conceptually — the guide's
  quality recommendations assume real games always provide these separately.
- **Motion vectors are Blender's backward flow only** (RG channels,
  `motion_vectors.py`'s existing convention) — never cross-checked against
  what DLSS-G actually expects beyond the pixel-space `mvecScale` guidance.

## Why this was abandoned

Two independent blockers, either sufficient alone:

1. **No readback API** (confirmed early, see above) — no way to export a
   generated frame to disk.
2. **NVIDIA's licensing requirement** (confirmed once generation actually
   started working) — an on-screen message stating DLSS must be licensed
   for the title, requiring contacting NVIDIA directly. This is a business/
   legal gate, not a technical one, and isn't something a batch command-line
   tool like this bridge is positioned to satisfy.

The technical investigation succeeded on its own terms — real DLSS-G frame
generation was reproduced on real Blender footage with real camera data,
and every SDK integration bug encountered along the way was found and
fixed through an actual build/run/log cycle. It just doesn't lead anywhere
usable for this project. If revisited, both blockers would need to not
apply — e.g. a future SDK version exposing readback, and a resolved
licensing relationship with NVIDIA — before this path is worth reopening.
For interpolation going forward, see `src/core/frame_interpolator.py`
(analytic gather/splat, uses Blender's own vectors directly) or the RIFE
integration path discussed earlier in this project's history.
