/**
 * dlssg_bridge.cpp
 *
 * DLSS Frame Generation backend for the interpolation pipeline. Unlike
 * optix_bridge.cpp (a stateless CLI around a CUDA call), DLSS-G's only entry
 * point is IDXGISwapChain::Present(), intercepted by a Streamline proxy swap
 * chain -- so this executable has to run a minimal, real D3D12
 * device/swapchain/present loop to drive it.
 *
 * Usage:
 *   dlssg_bridge.exe config.json
 *
 * config.json fields (flat, matching optix_bridge.cpp's convention -- the
 * Python side is responsible for resolving the right camera.json entries
 * for frame N and N+1 and flattening them here):
 *
 *   width, height,
 *   color_n, color_n1       (path to float32 .npy, shape HxWx3 -- frame N and N+1)
 *   depth_n, depth_n1       (path to float32 .npy, shape HxW)
 *   mvec_n, mvec_n1         (path to float32 .npy, shape HxWx2 -- pixel-space,
 *                            same backward-flow convention as motion_vectors.py)
 *   depth_inverted          (true/false -- see README "Known-risky spots")
 *   cam_pos_n / cam_fwd_n / cam_right_n            ([x,y,z] arrays, frame N)
 *   cam_pos_n1 / cam_fwd_n1 / cam_right_n1         ([x,y,z] arrays, frame N+1)
 *   cam_fov, cam_aspect, cam_near, cam_far         (shared; from camera.json)
 *   num_frames_to_generate  (1 = 2x, 2 = 3x, 3 = 4x -- matches DLSSGOptions)
 *   output_prefix           (generated frames written to
 *                            "<output_prefix>_0.npy", "_1.npy", ...)
 */

#define NOMINMAX  // windows.h's max/min macros collide with std::max/std::min
#define WIN32_LEAN_AND_MEAN

#include <sl.h>
#include <sl_consts.h>
#include <sl_core_types.h>
#include <sl_dlss_g.h>
#include <sl_reflex.h>
#include <sl_pcl.h>
#include <sl_matrix_helpers.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <thread>
#include <filesystem>
#include <chrono>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// NumPy .npy load/save -- verbatim from optix_bridge/optix_denoiser.cpp so
// both bridges speak the exact same interop format to the Python side.
// ---------------------------------------------------------------------------
static std::vector<float> load_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open npy: " + path);

    char magic[7]; f.read(magic, 6); magic[6] = 0;
    if (std::string(magic) != "\x93NUMPY")
        throw std::runtime_error("Not a .npy file: " + path);
    uint8_t major, minor;
    f.read((char*)&major, 1); f.read((char*)&minor, 1);

    uint16_t hlen;
    f.read((char*)&hlen, 2);
    std::string header(hlen, ' ');
    f.read(header.data(), hlen);

    auto data_start = f.tellg();
    f.seekg(0, std::ios::end);
    auto data_end = f.tellg();
    size_t nbytes = (size_t)(data_end - data_start);
    f.seekg(data_start);

    std::vector<float> data(nbytes / sizeof(float));
    f.read((char*)data.data(), nbytes);
    return data;
}

static void save_npy(const std::string& path, const std::vector<float>& data,
                      int h, int w, int c) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write: " + path);

    std::ostringstream hdr;
    hdr << "{'descr': '<f4', 'fortran_order': False, 'shape': ("
        << h << ", " << w << ", " << c << "), }";
    std::string hdr_str = hdr.str();
    size_t header_len = hdr_str.size() + 1;
    size_t pad = (64 - ((10 + header_len) % 64)) % 64;
    hdr_str += std::string(pad, ' ');
    hdr_str += '\n';

    const char magic[] = "\x93NUMPY\x01\x00";
    f.write(magic, 8);
    uint16_t hl = (uint16_t)hdr_str.size();
    f.write((char*)&hl, 2);
    f.write(hdr_str.data(), hl);
    f.write((char*)data.data(), data.size() * sizeof(float));
}

// ---------------------------------------------------------------------------
// Flat config.json parser -- extends optix_denoiser.cpp's approach (still no
// real nesting) with float3 array values, e.g. "cam_pos_n": [1.0, 2.0, 3.0].
// ---------------------------------------------------------------------------
static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n\"");
    size_t b = s.find_last_not_of(" \t\r\n\"");
    if (a == std::string::npos) return "";
    return s.substr(a, b - a + 1);
}

struct Config {
    int width = 0, height = 0;
    std::string color_n, color_n1;
    std::string depth_n, depth_n1;
    std::string mvec_n, mvec_n1;
    bool depth_inverted = false;
    float cam_pos_n[3]{}, cam_fwd_n[3]{}, cam_right_n[3]{};
    float cam_pos_n1[3]{}, cam_fwd_n1[3]{}, cam_right_n1[3]{};
    float cam_fov = 0, cam_aspect = 0, cam_near = 0.1f, cam_far = 1000.0f;
    uint32_t num_frames_to_generate = 1;
    std::string output_prefix;
};

static void parse_float3(const std::string& arr_text, float out[3]) {
    // arr_text is the raw "[a, b, c]" slice, brackets included.
    std::string inner = arr_text.substr(1, arr_text.size() - 2);
    std::stringstream ss(inner);
    std::string tok;
    for (int i = 0; i < 3 && std::getline(ss, tok, ','); ++i)
        out[i] = std::stof(trim(tok));
}

static Config parse_config(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open config: " + path);
    std::string text((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());

    Config cfg;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t kstart = text.find('"', pos);
        if (kstart == std::string::npos) break;
        size_t kend = text.find('"', kstart + 1);
        std::string key = text.substr(kstart + 1, kend - kstart - 1);
        size_t colon = text.find(':', kend);
        size_t vstart = colon + 1;
        while (vstart < text.size() && std::isspace((unsigned char)text[vstart])) ++vstart;

        std::string value;
        if (text[vstart] == '"') {
            size_t vend = text.find('"', vstart + 1);
            value = text.substr(vstart + 1, vend - vstart - 1);
            pos = vend + 1;
        } else if (text[vstart] == '[') {
            size_t vend = text.find(']', vstart);
            value = text.substr(vstart, vend - vstart + 1);
            pos = vend + 1;
        } else {
            size_t vend = text.find_first_of(",}", vstart);
            value = trim(text.substr(vstart, vend - vstart));
            pos = vend + 1;
        }

        if (key == "width") cfg.width = std::stoi(value);
        else if (key == "height") cfg.height = std::stoi(value);
        else if (key == "color_n") cfg.color_n = value;
        else if (key == "color_n1") cfg.color_n1 = value;
        else if (key == "depth_n") cfg.depth_n = value;
        else if (key == "depth_n1") cfg.depth_n1 = value;
        else if (key == "mvec_n") cfg.mvec_n = value;
        else if (key == "mvec_n1") cfg.mvec_n1 = value;
        else if (key == "depth_inverted") cfg.depth_inverted = (value == "true");
        else if (key == "cam_pos_n") parse_float3(value, cfg.cam_pos_n);
        else if (key == "cam_fwd_n") parse_float3(value, cfg.cam_fwd_n);
        else if (key == "cam_right_n") parse_float3(value, cfg.cam_right_n);
        else if (key == "cam_pos_n1") parse_float3(value, cfg.cam_pos_n1);
        else if (key == "cam_fwd_n1") parse_float3(value, cfg.cam_fwd_n1);
        else if (key == "cam_right_n1") parse_float3(value, cfg.cam_right_n1);
        else if (key == "cam_fov") cfg.cam_fov = std::stof(value);
        else if (key == "cam_aspect") cfg.cam_aspect = std::stof(value);
        else if (key == "cam_near") cfg.cam_near = std::stof(value);
        else if (key == "cam_far") cfg.cam_far = std::stof(value);
        else if (key == "num_frames_to_generate") cfg.num_frames_to_generate = (uint32_t)std::stoi(value);
        else if (key == "output_prefix") cfg.output_prefix = value;
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Minimal Win32 window -- DXGI swap chains need a real HWND. `visible`
// controls whether it's a real on-screen window (visual test mode, so a
// human can actually see the generated frames DLSS-G presents -- there is
// no documented API to read them back, see dlssg_bridge/README.md) or kept
// off-screen (WS_POPUP, no WS_VISIBLE) for the original batch-export intent.
// ---------------------------------------------------------------------------
static HWND create_window(int width, int height, bool visible) {
    WNDCLASSEXW wc{ sizeof(WNDCLASSEXW) };
    wc.lpfnWndProc = DefWindowProcW;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"DlssgBridgeWindow";
    RegisterClassExW(&wc);

    DWORD style = visible ? (WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME) : WS_POPUP;
    RECT rect{ 0, 0, width, height };
    AdjustWindowRect(&rect, style, FALSE);

    HWND hwnd = CreateWindowExW(
        0, wc.lpszClassName, L"dlssg_bridge -- DLSS-G visual test", style,
        CW_USEDEFAULT, CW_USEDEFAULT, rect.right - rect.left, rect.bottom - rect.top,
        nullptr, nullptr, wc.hInstance, nullptr);
    if (!hwnd) throw std::runtime_error("CreateWindowExW failed");
    if (visible) {
        ShowWindow(hwnd, SW_SHOW);
        UpdateWindow(hwnd);
    }
    return hwnd;
}

#define HR_CHECK(call) \
    do { HRESULT hr_ = (call); if (FAILED(hr_)) { \
        std::ostringstream oss_; oss_ << #call << " failed, hr=0x" << std::hex << hr_; \
        throw std::runtime_error(oss_.str()); } } while(0)

// ---------------------------------------------------------------------------
// D3D12 device + command queue + swap chain.
//
// slInit() must run BEFORE this -- Streamline's interposer DLL hooks
// CreateDXGIFactory/D3D12CreateDevice/CreateSwapChain at the DLL-export
// level, so the device/factory/swapchain created here transparently become
// SL-proxied objects as long as sl.interposer.dll is loaded (via slInit)
// first and linked/loaded ahead of the real d3d12.dll/dxgi.dll exports
// resolving -- this is the "automatic hooking" path, the alternative to
// ProgrammingGuideManualHooking.md's explicit slUpgradeInterface() approach.
// ---------------------------------------------------------------------------
struct D3D12Context {
    ComPtr<IDXGIFactory6> factory;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<IDXGISwapChain3> swapChain;
    ComPtr<ID3D12CommandAllocator> cmdAlloc;
    ComPtr<ID3D12GraphicsCommandList> cmdList;
    ComPtr<ID3D12Fence> fence;
    HANDLE fenceEvent = nullptr;
    uint64_t fenceValue = 0;
    bool tearingSupported = false;
    static const UINT kBackBufferCount = 2;

    void wait_for_gpu() {
        const uint64_t v = ++fenceValue;
        HR_CHECK(queue->Signal(fence.Get(), v));
        if (fence->GetCompletedValue() < v) {
            HR_CHECK(fence->SetEventOnCompletion(v, fenceEvent));
            WaitForSingleObject(fenceEvent, INFINITE);
        }
    }
};

static D3D12Context create_d3d12_context(HWND hwnd, int width, int height) {
    D3D12Context ctx;

    UINT factoryFlags = 0;
#ifdef _DEBUG
    ComPtr<ID3D12Debug> debugController;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
        debugController->EnableDebugLayer();
    factoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
#endif
    HR_CHECK(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&ctx.factory)));

    // Pick the first hardware adapter that supports D3D12 (the RTX card).
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; ctx.factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                         IID_PPV_ARGS(&ctx.device))))
            break;
        ctx.device.Reset();
    }
    if (!ctx.device) throw std::runtime_error("No D3D12-capable hardware adapter found");

    // Must happen immediately after device creation, before the command
    // queue/swap chain -- confirmed via sl.log: "D3D or VK API hook is
    // activated without device being created, did you forget to call
    // slSetD3DDevice ... or trying to use another SL API before setting the
    // device?" fired when this call was deferred to the end of this
    // function (after queue/swapchain creation).
    slSetD3DDevice(ctx.device.Get());

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    HR_CHECK(ctx.device->CreateCommandQueue(&qd, IID_PPV_ARGS(&ctx.queue)));

    // Games running DLSS-G conventionally present with vsync OFF + tearing
    // allowed -- FG paces the actual display timing itself; forcing a hard
    // vsync wait via Present(1,0) may be why generation silently no-ops
    // despite DLSSGStatus::eOk (untested theory, being tried here).
    BOOL allowTearing = FALSE;
    ctx.factory->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing, sizeof(allowTearing));

    DXGI_SWAP_CHAIN_DESC1 scd{};
    scd.Width = width;
    scd.Height = height;
    scd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.SampleDesc.Count = 1;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.BufferCount = D3D12Context::kBackBufferCount;
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    scd.Flags = allowTearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
    ctx.tearingSupported = allowTearing;

    ComPtr<IDXGISwapChain1> swapChain1;
    // NOTE: this call is what Streamline's interposer intercepts to install
    // its proxy swap chain -- as long as slInit() ran first and
    // sl.interposer.dll's exports shadow the real dxgi.dll ones (standard
    // DLL-export-hooking; see ProgrammingGuide.md section on automatic
    // hooking), ctx.factory here is already the SL-patched factory.
    HR_CHECK(ctx.factory->CreateSwapChainForHwnd(
        ctx.queue.Get(), hwnd, &scd, nullptr, nullptr, &swapChain1));
    HR_CHECK(swapChain1.As(&ctx.swapChain));

    HR_CHECK(ctx.device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&ctx.cmdAlloc)));
    HR_CHECK(ctx.device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT, ctx.cmdAlloc.Get(), nullptr,
        IID_PPV_ARGS(&ctx.cmdList)));
    HR_CHECK(ctx.cmdList->Close());

    HR_CHECK(ctx.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&ctx.fence)));
    ctx.fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!ctx.fenceEvent) throw std::runtime_error("CreateEventW failed");

    return ctx;
}

// ---------------------------------------------------------------------------
// Upload a float32 HxWxC buffer into a D3D12 default-heap Tex2D, via an
// intermediate upload heap. `channels` is the source data's channel count;
// `format` is the destination texture format (may pad, e.g. 3->4 for RGBA).
// ---------------------------------------------------------------------------
static ComPtr<ID3D12Resource> upload_texture(
    D3D12Context& ctx, const std::vector<float>& data,
    int width, int height, int channels, DXGI_FORMAT format) {

    const UINT bytesPerPixel = 4 /* R32 */ * (format == DXGI_FORMAT_R32G32_FLOAT ? 2 :
                                                format == DXGI_FORMAT_R32_FLOAT ? 1 : 4);

    D3D12_HEAP_PROPERTIES defaultHeap{ D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC texDesc{};
    texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Width = width;
    texDesc.Height = height;
    texDesc.DepthOrArraySize = 1;
    texDesc.MipLevels = 1;
    texDesc.Format = format;
    texDesc.SampleDesc.Count = 1;
    texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

    ComPtr<ID3D12Resource> tex;
    HR_CHECK(ctx.device->CreateCommittedResource(
        &defaultHeap, D3D12_HEAP_FLAG_NONE, &texDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&tex)));

    // Pack source data (which may be 3-channel RGB) into the destination's
    // channel layout, padding with 0 (or 1 for alpha) as needed.
    const UINT dstChannels = (format == DXGI_FORMAT_R32G32B32A32_FLOAT) ? 4 :
                              (format == DXGI_FORMAT_R32G32_FLOAT) ? 2 : 1;
    std::vector<float> packed((size_t)width * height * dstChannels, 0.0f);
    for (size_t px = 0; px < (size_t)width * height; ++px) {
        for (UINT c = 0; c < (UINT)channels && c < dstChannels; ++c)
            packed[px * dstChannels + c] = data[px * channels + c];
        if (dstChannels == 4 && channels < 4)
            packed[px * 4 + 3] = 1.0f;  // alpha
    }

    UINT64 rowPitch = (UINT64)width * dstChannels * 4;
    UINT64 alignedRowPitch = (rowPitch + 255) & ~255ULL;  // D3D12_TEXTURE_DATA_PITCH_ALIGNMENT
    UINT64 uploadSize = alignedRowPitch * height;

    D3D12_HEAP_PROPERTIES uploadHeap{ D3D12_HEAP_TYPE_UPLOAD };
    D3D12_RESOURCE_DESC bufDesc{};
    bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufDesc.Width = uploadSize;
    bufDesc.Height = 1;
    bufDesc.DepthOrArraySize = 1;
    bufDesc.MipLevels = 1;
    bufDesc.Format = DXGI_FORMAT_UNKNOWN;
    bufDesc.SampleDesc.Count = 1;
    bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ComPtr<ID3D12Resource> uploadBuf;
    HR_CHECK(ctx.device->CreateCommittedResource(
        &uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uploadBuf)));

    uint8_t* mapped = nullptr;
    HR_CHECK(uploadBuf->Map(0, nullptr, (void**)&mapped));
    for (int y = 0; y < height; ++y) {
        memcpy(mapped + y * alignedRowPitch,
               (uint8_t*)packed.data() + (size_t)y * width * dstChannels * 4,
               (size_t)width * dstChannels * 4);
    }
    uploadBuf->Unmap(0, nullptr);

    HR_CHECK(ctx.cmdAlloc->Reset());
    HR_CHECK(ctx.cmdList->Reset(ctx.cmdAlloc.Get(), nullptr));

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = tex.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = uploadBuf.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Offset = 0;
    src.PlacedFootprint.Footprint.Format = format;
    src.PlacedFootprint.Footprint.Width = width;
    src.PlacedFootprint.Footprint.Height = height;
    src.PlacedFootprint.Footprint.Depth = 1;
    src.PlacedFootprint.Footprint.RowPitch = (UINT)alignedRowPitch;

    ctx.cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = tex.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                                   | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    ctx.cmdList->ResourceBarrier(1, &barrier);

    HR_CHECK(ctx.cmdList->Close());
    ID3D12CommandList* lists[] = { ctx.cmdList.Get() };
    ctx.queue->ExecuteCommandLists(1, lists);
    ctx.wait_for_gpu();  // keeps uploadBuf alive long enough; simplest correct
                         // option for a batch tool with no perf pressure

    return tex;
}

// ---------------------------------------------------------------------------
// Build sl::Constants for one frame. cameraViewToClip is a standard
// perspective projection built from fovY/aspect/near/far; the rest of the
// temporal matrices (clipToCameraView, clipToPrevClip, prevClipToClip) are
// derived by the SDK's own recalculateCameraMatrices() helper from this +
// cameraPos/Fwd/Right, using internal static state to track "previous frame"
// -- explicitly documented as fine for single-viewport sequential use, which
// is exactly this bridge's usage pattern (one process per frame pair, called
// strictly in frame order by the Python pipeline).
// ---------------------------------------------------------------------------
static sl::Constants build_constants(
    const float pos[3], const float fwd[3], const float right[3],
    float fovY, float aspect, float nearZ, float farZ,
    float mvecScaleX, float mvecScaleY, bool depthInverted) {
    sl::Constants consts{};
    consts.cameraPos = { pos[0], pos[1], pos[2] };
    consts.cameraFwd = { fwd[0], fwd[1], fwd[2] };
    consts.cameraRight = { right[0], right[1], right[2] };
    consts.cameraFOV = fovY;
    consts.cameraAspectRatio = aspect;
    consts.cameraNear = nearZ;
    consts.cameraFar = farZ;
    consts.mvecScale = { mvecScaleX, mvecScaleY };
    consts.depthInverted = depthInverted ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    consts.cameraMotionIncluded = sl::Boolean::eTrue;
    consts.motionVectors3D = sl::Boolean::eFalse;
    consts.motionVectorsDilated = sl::Boolean::eFalse;
    consts.motionVectorsJittered = sl::Boolean::eFalse;
    consts.reset = sl::Boolean::eFalse;
    consts.orthographicProjection = sl::Boolean::eFalse;
    consts.jitterOffset = { 0.0f, 0.0f };  // Blender renders are not TAA-jittered
    consts.cameraPinholeOffset = { 0.0f, 0.0f };  // optional, but the
        // validator warns if left at the struct's "invalid" sentinel

    // Standard right-handed perspective projection (D3D convention, depth
    // range [0,1]), row-major to match sl::float4x4's documented layout.
    const float f = 1.0f / std::tan(fovY * 0.5f);
    sl::float4x4 proj{};
    proj[0] = sl::float4(f / aspect, 0, 0, 0);
    proj[1] = sl::float4(0, f, 0, 0);
    proj[2] = sl::float4(0, 0, farZ / (nearZ - farZ), -1);
    proj[3] = sl::float4(0, 0, (nearZ * farZ) / (nearZ - farZ), 0);
    consts.cameraViewToClip = proj;

    sl::recalculateCameraMatrices(consts);

    return consts;
}

// ---------------------------------------------------------------------------
// Upload a float32 HxWx3 linear-space buffer as an sRGB-tonemapped
// R8G8B8A8_UNORM texture -- the format swap chain back buffers actually use,
// needed so this can be CopyResource'd directly into the current back
// buffer (D3D12 copies require matching formats; no implicit conversion).
// ---------------------------------------------------------------------------
static ComPtr<ID3D12Resource> upload_color_as_backbuffer_source(
    D3D12Context& ctx, const std::vector<float>& data, int width, int height) {

    D3D12_HEAP_PROPERTIES defaultHeap{ D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC texDesc{};
    texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Width = width;
    texDesc.Height = height;
    texDesc.DepthOrArraySize = 1;
    texDesc.MipLevels = 1;
    texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texDesc.SampleDesc.Count = 1;
    texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

    ComPtr<ID3D12Resource> tex;
    HR_CHECK(ctx.device->CreateCommittedResource(
        &defaultHeap, D3D12_HEAP_FLAG_NONE, &texDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&tex)));

    std::vector<uint8_t> packed((size_t)width * height * 4, 255);
    for (size_t px = 0; px < (size_t)width * height; ++px) {
        for (int c = 0; c < 3; ++c) {
            float linear = data[px * 3 + c];
            float srgb = linear <= 0.0031308f ? linear * 12.92f
                                               : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
            packed[px * 4 + c] = (uint8_t)std::lround(std::clamp(srgb, 0.0f, 1.0f) * 255.0f);
        }
    }

    UINT64 rowPitch = (UINT64)width * 4;
    UINT64 alignedRowPitch = (rowPitch + 255) & ~255ULL;
    UINT64 uploadSize = alignedRowPitch * height;

    D3D12_HEAP_PROPERTIES uploadHeap{ D3D12_HEAP_TYPE_UPLOAD };
    D3D12_RESOURCE_DESC bufDesc{};
    bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufDesc.Width = uploadSize;
    bufDesc.Height = 1;
    bufDesc.DepthOrArraySize = 1;
    bufDesc.MipLevels = 1;
    bufDesc.Format = DXGI_FORMAT_UNKNOWN;
    bufDesc.SampleDesc.Count = 1;
    bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ComPtr<ID3D12Resource> uploadBuf;
    HR_CHECK(ctx.device->CreateCommittedResource(
        &uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uploadBuf)));

    uint8_t* mapped = nullptr;
    HR_CHECK(uploadBuf->Map(0, nullptr, (void**)&mapped));
    for (int y = 0; y < height; ++y)
        memcpy(mapped + y * alignedRowPitch, packed.data() + (size_t)y * width * 4, (size_t)width * 4);
    uploadBuf->Unmap(0, nullptr);

    HR_CHECK(ctx.cmdAlloc->Reset());
    HR_CHECK(ctx.cmdList->Reset(ctx.cmdAlloc.Get(), nullptr));

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = tex.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = uploadBuf.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Offset = 0;
    src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    src.PlacedFootprint.Footprint.Width = width;
    src.PlacedFootprint.Footprint.Height = height;
    src.PlacedFootprint.Footprint.Depth = 1;
    src.PlacedFootprint.Footprint.RowPitch = (UINT)alignedRowPitch;

    ctx.cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = tex.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    ctx.cmdList->ResourceBarrier(1, &barrier);

    HR_CHECK(ctx.cmdList->Close());
    ID3D12CommandList* lists[] = { ctx.cmdList.Get() };
    ctx.queue->ExecuteCommandLists(1, lists);
    ctx.wait_for_gpu();

    return tex;
}

// ---------------------------------------------------------------------------
// Copy a COPY_SOURCE-state color texture into the swap chain's current back
// buffer, transitioning the back buffer PRESENT -> COPY_DEST -> PRESENT.
// ---------------------------------------------------------------------------
static void copy_to_backbuffer(D3D12Context& ctx, ID3D12Resource* colorSrc) {
    UINT idx = ctx.swapChain->GetCurrentBackBufferIndex();
    ComPtr<ID3D12Resource> backbuffer;
    HR_CHECK(ctx.swapChain->GetBuffer(idx, IID_PPV_ARGS(&backbuffer)));

    HR_CHECK(ctx.cmdAlloc->Reset());
    HR_CHECK(ctx.cmdList->Reset(ctx.cmdAlloc.Get(), nullptr));

    D3D12_RESOURCE_BARRIER toDst{};
    toDst.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    toDst.Transition.pResource = backbuffer.Get();
    toDst.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    toDst.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    toDst.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    ctx.cmdList->ResourceBarrier(1, &toDst);

    ctx.cmdList->CopyResource(backbuffer.Get(), colorSrc);

    D3D12_RESOURCE_BARRIER toPresent = toDst;
    std::swap(toPresent.Transition.StateBefore, toPresent.Transition.StateAfter);
    ctx.cmdList->ResourceBarrier(1, &toPresent);

    HR_CHECK(ctx.cmdList->Close());
    ID3D12CommandList* lists[] = { ctx.cmdList.Get() };
    ctx.queue->ExecuteCommandLists(1, lists);
    ctx.wait_for_gpu();
}

static void tag_frame(const sl::FrameToken& frame, ID3D12Resource* depth,
                       ID3D12Resource* mvec, ID3D12GraphicsCommandList* cmdList) {
    sl::Extent fullExtent{};  // zero-initialized = "whole resource", per guide

    // 3-arg ctor: (type, native pointer, D3D12_RESOURCE_STATE as uint32_t).
    // State must match what upload_texture() actually left the resource in.
    const uint32_t srvState =
        (uint32_t)(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    sl::Resource depthRes(sl::ResourceType::eTex2d, depth, srvState);
    sl::Resource mvecRes(sl::ResourceType::eTex2d, mvec, srvState);

    sl::ResourceTag depthTag{ &depthRes, sl::kBufferTypeDepth,
                               sl::ResourceLifecycle::eValidUntilPresent, &fullExtent };
    sl::ResourceTag mvecTag{ &mvecRes, sl::kBufferTypeMotionVectors,
                              sl::ResourceLifecycle::eValidUntilPresent, &fullExtent };

    sl::ResourceTag tags[] = { depthTag, mvecTag };
    if (slSetTagForFrame(frame, sl::ViewportHandle{ 0 }, tags, 2, cmdList) != sl::Result::eOk)
        throw std::runtime_error("slSetTagForFrame failed");
}

// ---------------------------------------------------------------------------
// Synthetic moving-pattern sequence -- since there's no real Blender render
// available where this was built, this generates a simple, self-consistent
// test sequence (vertical bar sliding left to right over a checkerboard) so
// a human can actually see whether DLSS-G produces plausible in-between
// motion. Camera is static (no camera motion; cameraMotionIncluded still
// requires *some* valid pos/fwd/right, so those are set to sane defaults).
// ---------------------------------------------------------------------------
struct SyntheticFrame {
    std::vector<float> color;  // HxWx3
    std::vector<float> depth;  // HxW
    std::vector<float> mvec;   // HxWx2, backward flow (matches motion_vectors.py convention)
};

static std::vector<SyntheticFrame> generate_synthetic_sequence(int w, int h, int numFrames, float pixelsPerFrame) {
    std::vector<SyntheticFrame> frames(numFrames);
    const int barWidth = std::max(4, w / 16);

    for (int f = 0; f < numFrames; ++f) {
        SyntheticFrame& sf = frames[f];
        sf.color.assign((size_t)w * h * 3, 0.0f);
        sf.depth.assign((size_t)w * h, 50.0f);       // background, "far"
        sf.mvec.assign((size_t)w * h * 2, 0.0f);

        int barX = (int)std::lround(f * pixelsPerFrame) % w;

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                size_t px = (size_t)y * w + x;
                bool checker = ((x / 16) + (y / 16)) % 2 == 0;
                float bg = checker ? 0.15f : 0.05f;
                sf.color[px * 3 + 0] = bg;
                sf.color[px * 3 + 1] = bg;
                sf.color[px * 3 + 2] = bg;

                if (x >= barX && x < barX + barWidth) {
                    sf.color[px * 3 + 0] = 0.9f;
                    sf.color[px * 3 + 1] = 0.2f;
                    sf.color[px * 3 + 2] = 0.1f;
                    sf.depth[px] = 5.0f;  // "near" -- the bar is in front of the background

                    // Backward flow: this pixel's value came from
                    // (x - pixelsPerFrame) in the previous frame -- matches
                    // motion_vectors.py's blender_vector_to_backward_flow
                    // convention (RG = backward motion).
                    sf.mvec[px * 2 + 0] = pixelsPerFrame;
                    sf.mvec[px * 2 + 1] = 0.0f;
                }
            }
        }
    }
    return frames;
}

// ---------------------------------------------------------------------------
// Real-footage sequence loading for --visual-test-sequence -- reads
// color_%04d.npy / depth_%04d.npy / mvec_%04d.npy triplets written by
// tools/export_dlssg_sequence.py (sequential index, regardless of the
// original EXR frame numbers), and the optional camera_sequence.json it
// writes alongside them (a plain JSON array, index-aligned with the npy
// files -- not the per-Blender-frame-number dict tools/blender_export_camera.py
// produces; the export script does that re-keying).
// ---------------------------------------------------------------------------
struct SeqCamera {
    float pos[3], fwd[3], right[3];
    float fov, aspect, near_, far_;
};

static std::vector<SyntheticFrame> load_npy_sequence(const std::filesystem::path& folder, int& outW, int& outH) {
    std::vector<SyntheticFrame> frames;
    for (int i = 0; ; ++i) {
        char idx[8];
        snprintf(idx, sizeof(idx), "%04d", i);
        auto colorPath = folder / (std::string("color_") + idx + ".npy");
        auto depthPath = folder / (std::string("depth_") + idx + ".npy");
        auto mvecPath = folder / (std::string("mvec_") + idx + ".npy");
        if (!std::filesystem::exists(colorPath)) break;

        SyntheticFrame sf;
        sf.color = load_npy(colorPath.string());
        sf.depth = std::filesystem::exists(depthPath) ? load_npy(depthPath.string())
                                                        : std::vector<float>();
        sf.mvec = std::filesystem::exists(mvecPath) ? load_npy(mvecPath.string())
                                                      : std::vector<float>();
        frames.push_back(std::move(sf));
    }
    if (frames.empty())
        throw std::runtime_error("No color_0000.npy found in " + folder.string() +
                                  " -- run tools/export_dlssg_sequence.py first");

    // Infer width/height from the color buffer size (HxWx3).
    size_t pixels = frames[0].color.size() / 3;
    // Caller already knows the intended width/height (passed on the command
    // line); this is just a sanity check that the files match.
    (void)pixels;
    return frames;
}

// Minimal parser for camera_sequence.json's fixed shape: a JSON array of
// {"pos":[x,y,z],"fwd":[x,y,z],"right":[x,y,z],"fov":f,"aspect":f,"near":f,"far":f}
// objects. Not a general JSON parser -- assumes export_dlssg_sequence.py's
// exact output format, same spirit as parse_config() above.
static std::vector<SeqCamera> load_camera_sequence(const std::filesystem::path& path) {
    std::vector<SeqCamera> cams;
    if (!std::filesystem::exists(path)) return cams;

    std::ifstream f(path);
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    size_t pos = 0;
    while (true) {
        size_t objStart = text.find('{', pos);
        if (objStart == std::string::npos) break;
        size_t objEnd = text.find('}', objStart);
        if (objEnd == std::string::npos) break;
        std::string obj = text.substr(objStart, objEnd - objStart + 1);
        pos = objEnd + 1;

        SeqCamera cam{};
        auto findArr = [&](const char* key, float out[3]) {
            size_t k = obj.find(std::string("\"") + key + "\"");
            if (k == std::string::npos) return;
            size_t br = obj.find('[', k);
            size_t brEnd = obj.find(']', br);
            parse_float3(obj.substr(br, brEnd - br + 1), out);
        };
        auto findScalar = [&](const char* key) -> float {
            size_t k = obj.find(std::string("\"") + key + "\"");
            if (k == std::string::npos) return 0.0f;
            size_t colon = obj.find(':', k);
            size_t vend = obj.find_first_of(",}", colon);
            return std::stof(trim(obj.substr(colon + 1, vend - colon - 1)));
        };
        findArr("pos", cam.pos);
        findArr("fwd", cam.fwd);
        findArr("right", cam.right);
        cam.fov = findScalar("fov");
        cam.aspect = findScalar("aspect");
        cam.near_ = findScalar("near");
        cam.far_ = findScalar("far");
        cams.push_back(cam);
    }
    return cams;
}

// ---------------------------------------------------------------------------
// Stamps a blinking corner marker into a COPY of the real frame's color
// buffer, alternating solid white/black each REAL frame (never touching
// depth/mvec, so it doesn't affect DLSS-G's actual inputs). Diagnostic only:
// generated (interpolated) frames are synthesized by DLSS-G, not copied from
// disk, so they cannot reproduce this sharp on/off flip -- if generation is
// really happening, the marker should visibly blend/hold between real
// frames rather than flip in lockstep with the rest of the scene's motion.
// ---------------------------------------------------------------------------
static std::vector<float> stamp_marker(const std::vector<float>& color, int width, int height, int frameIndex) {
    std::vector<float> out = color;
    // Saturated magenta vs. cyan -- unlike black/white, neither blends into
    // typical scene lighting/background, so both states stay clearly
    // distinguishable from the underlying footage regardless of its
    // brightness.
    bool even = (frameIndex % 2 == 0);
    float r = even ? 1.0f : 0.0f;
    float g = even ? 0.0f : 1.0f;
    float b = 1.0f;
    int marker = std::max(8, width / 20);
    for (int y = 0; y < marker && y < height; ++y) {
        for (int x = 0; x < marker && x < width; ++x) {
            size_t px = ((size_t)y * width + x) * 3;
            out[px + 0] = r;
            out[px + 1] = g;
            out[px + 2] = b;
        }
    }
    return out;
}

static void pump_messages() {
    MSG msg;
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}

static int run_visual_test(int width, int height, std::vector<SyntheticFrame> sequence,
                            std::vector<SeqCamera> cameras = {}, uint32_t numFramesToGenerate = 1) {
    sl::Preferences pref{};
    pref.showConsole = true;
    pref.logLevel = sl::LogLevel::eVerbose;
    pref.pathToLogsAndData = L".";
    pref.renderAPI = sl::RenderAPI::eD3D12;
    pref.flags = pref.flags | sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    // kFeatureImGUI: debug overlay, Development-build DLLs only (won't load
    // in Production, per Streamline's own docs) -- toggle on-screen with
    // Ctrl+Shift+Home once running, dlss_g's buffer visualizer with
    // Ctrl+Shift+Insert. This is how we get direct visibility into whether
    // dlss_g is actually generating, instead of inferring it from status
    // codes and counters that turned out to be misleading.
    sl::Feature featuresToLoad[] = { sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeatureImGUI };
    pref.featuresToLoad = featuresToLoad;
    pref.numFeaturesToLoad = 3;
    pref.projectId = "51f5f296-55d5-4240-be4a-0e7f7637b7b1";
    pref.engine = sl::EngineType::eCustom;
    pref.engineVersion = "1.0.0";
    if (slInit(pref) != sl::Result::eOk)
        throw std::runtime_error("slInit failed -- is sl.interposer.dll next to the exe?");

    HWND hwnd = create_window(width, height, /*visible=*/true);
    D3D12Context ctx = create_d3d12_context(hwnd, width, height);

    LUID adapterLuid = ctx.device->GetAdapterLuid();
    sl::AdapterInfo adapterInfo{};
    adapterInfo.deviceLUID = reinterpret_cast<uint8_t*>(&adapterLuid);
    adapterInfo.deviceLUIDSizeInBytes = sizeof(adapterLuid);
    if (slIsFeatureSupported(sl::kFeatureDLSS_G, adapterInfo) != sl::Result::eOk)
        throw std::runtime_error("DLSS-G not supported on this adapter/driver");

    sl::DLSSGOptions options{};
    options.mode = sl::DLSSGMode::eOn;
    options.numFramesToGenerate = numFramesToGenerate;  // 1=2x, 2=3x, 3=4x, ... (max reported via DLSSGState::numFramesToGenerateMax)
    options.colorWidth = width;
    options.colorHeight = height;
    options.mvecDepthWidth = width;
    options.mvecDepthHeight = height;
    options.numBackBuffers = D3D12Context::kBackBufferCount;
    sl::ViewportHandle viewport{ 0 };
    // NOTE: also re-asserted every frame inside the loop below -- the debug
    // overlay reported "debug mode requires DLSS-G to be turned on" despite
    // this call succeeding here, before any frame/viewport state exists yet.
    // Untested hypothesis: options need to be reasserted per-frame like
    // slSetConstants/slSetTagForFrame, not set once up front.
    if (slDLSSGSetOptions(viewport, options) != sl::Result::eOk)
        throw std::runtime_error("slDLSSGSetOptions failed");

    // DLSSGStatus::eFailReflexNotDetectedAtRuntime exists specifically
    // because DLSS-G silently disables itself if Reflex is loaded but not
    // actively turned on -- loading the plugin (Preferences::featuresToLoad)
    // is necessary but not sufficient, it must also be explicitly enabled.
    sl::ReflexOptions reflexOptions{};
    reflexOptions.mode = sl::ReflexMode::eLowLatency;
    // Cap real-frame submission at 20 fps via Reflex's own limiter (not a
    // manual Sleep) -- with vsync off there's nothing else pacing the loop,
    // and this is the documented way to pace a DLSS-G app since it's
    // designed to work alongside Reflex's frame timing, not fight it.
    reflexOptions.frameLimitUs = 50000;
    if (slReflexSetOptions(reflexOptions) != sl::Result::eOk)
        throw std::runtime_error("slReflexSetOptions failed");

    const float defaultPos[3] = { 0, 0, 0 };
    const float defaultFwd[3] = { 0, 0, -1 };
    const float defaultRight[3] = { 1, 0, 0 };
    bool haveCameraSequence = cameras.size() == sequence.size() && !cameras.empty();
    if (!cameras.empty() && !haveCameraSequence) {
        std::cerr << "dlssg_bridge: camera sequence size (" << cameras.size()
                  << ") doesn't match frame count (" << sequence.size()
                  << ") -- falling back to a static default camera.\n";
    }

    std::cerr << "dlssg_bridge visual test: presenting " << sequence.size()
              << " real frames with DLSS-G " << (numFramesToGenerate + 1) << "x in between"
              << (haveCameraSequence ? " (using per-frame camera data)" : " (static default camera)")
              << ". Close the window or press ESC to stop.\n";

    bool quit = false;
    for (int loop = 0; !quit; ++loop) {
        for (size_t i = 0; i < sequence.size() && !quit; ++i) {
            pump_messages();
            if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) { quit = true; break; }
            if (!IsWindow(hwnd)) { quit = true; break; }

            const SyntheticFrame& sf = sequence[i];

            auto markedColor = stamp_marker(sf.color, width, height, (int)i);
            auto colorTex = upload_color_as_backbuffer_source(ctx, markedColor, width, height);
            auto depthTex = sf.depth.empty()
                ? upload_texture(ctx, std::vector<float>((size_t)width * height, 50.0f), width, height, 1, DXGI_FORMAT_R32_FLOAT)
                : upload_texture(ctx, sf.depth, width, height, 1, DXGI_FORMAT_R32_FLOAT);
            auto mvecTex = sf.mvec.empty()
                ? upload_texture(ctx, std::vector<float>((size_t)width * height * 2, 0.0f), width, height, 2, DXGI_FORMAT_R32G32_FLOAT)
                : upload_texture(ctx, sf.mvec, width, height, 2, DXGI_FORMAT_R32G32_FLOAT);

            sl::Constants consts = haveCameraSequence
                ? build_constants(cameras[i].pos, cameras[i].fwd, cameras[i].right,
                                   cameras[i].fov, cameras[i].aspect, cameras[i].near_, cameras[i].far_,
                                   1.0f / width, 1.0f / height, /*depthInverted=*/false)
                : build_constants(defaultPos, defaultFwd, defaultRight, 0.8f, (float)width / height,
                                   0.1f, 1000.0f, 1.0f / width, 1.0f / height, /*depthInverted=*/false);

            sl::FrameToken* frameToken = nullptr;
            if (slGetNewFrameToken(frameToken) != sl::Result::eOk)
                throw std::runtime_error("slGetNewFrameToken failed");

            // Reflex/PCL markers -- untested hypothesis that DLSS-G's
            // internal "am I actually engaged" state depends on Reflex
            // seeing a real per-frame marker sequence, not just
            // ReflexOptions::mode being set once. sl.pcl is already loaded
            // (a Reflex dependency, confirmed in sl.log) so no extra
            // feature-loading needed, just the calls themselves.
            slPCLSetMarker(sl::PCLMarker::eSimulationStart, *frameToken);

            if (slSetConstants(consts, *frameToken, viewport) != sl::Result::eOk)
                throw std::runtime_error("slSetConstants failed");

            slPCLSetMarker(sl::PCLMarker::eSimulationEnd, *frameToken);
            slPCLSetMarker(sl::PCLMarker::eRenderSubmitStart, *frameToken);

            HR_CHECK(ctx.cmdAlloc->Reset());
            HR_CHECK(ctx.cmdList->Reset(ctx.cmdAlloc.Get(), nullptr));
            tag_frame(*frameToken, depthTex.Get(), mvecTex.Get(), ctx.cmdList.Get());
            HR_CHECK(ctx.cmdList->Close());
            ID3D12CommandList* lists[] = { ctx.cmdList.Get() };
            ctx.queue->ExecuteCommandLists(1, lists);
            ctx.wait_for_gpu();

            slPCLSetMarker(sl::PCLMarker::eRenderSubmitEnd, *frameToken);

            copy_to_backbuffer(ctx, colorTex.Get());

            // Reassert every frame -- testing the hypothesis that this
            // doesn't "stick" from a single call made before any frame/
            // viewport state exists.
            if (slDLSSGSetOptions(viewport, options) != sl::Result::eOk)
                throw std::runtime_error("slDLSSGSetOptions failed (per-frame)");

            slPCLSetMarker(sl::PCLMarker::ePresentStart, *frameToken);
            UINT presentFlags = ctx.tearingSupported ? DXGI_PRESENT_ALLOW_TEARING : 0;
            HR_CHECK(ctx.swapChain->Present(0, presentFlags));  // vsync OFF + tearing --
                // conventional DLSS-G present mode; FG paces display timing itself
            slPCLSetMarker(sl::PCLMarker::ePresentEnd, *frameToken);

            sl::DLSSGState state{};
            if (slDLSSGGetState(viewport, state, &options) == sl::Result::eOk) {
                if (loop == 0 && i < 30) {
                    std::cerr << "  frame " << i << ": status=" << (uint32_t)state.status
                              << " (0=eOk) numFramesActuallyPresented=" << state.numFramesActuallyPresented
                              << " numFramesToGenerateMax=" << state.numFramesToGenerateMax << "\n";
                }
            }
        }
        if (loop == 0) std::cerr << "dlssg_bridge visual test: looping until closed...\n";
    }

    slShutdown();
    return 0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: dlssg_bridge.exe config.json\n"
                     "       dlssg_bridge.exe --visual-test [width height] [multiplier]\n"
                     "       dlssg_bridge.exe --visual-test-sequence <folder> [width height] [multiplier]\n"
                     "       multiplier: 2 (default), 3, 4, ... up to DLSSGState::numFramesToGenerateMax + 1\n";
        return 1;
    }

    if (std::string(argv[1]) == "--visual-test") {
        int w = argc > 2 ? std::atoi(argv[2]) : 640;
        int h = argc > 3 ? std::atoi(argv[3]) : 360;
        int multiplier = argc > 4 ? std::atoi(argv[4]) : 2;
        try {
            return run_visual_test(w, h, generate_synthetic_sequence(w, h, 60, 1.0f), {},
                                    (uint32_t)std::max(1, multiplier - 1));
        } catch (const std::exception& e) {
            std::cerr << "dlssg_bridge error: " << e.what() << "\n";
            return 1;
        }
    }

    if (std::string(argv[1]) == "--visual-test-sequence") {
        if (argc < 3) {
            std::cerr << "Usage: dlssg_bridge.exe --visual-test-sequence <folder> [width height] [multiplier]\n";
            return 1;
        }
        std::filesystem::path folder = argv[2];
        int w = argc > 3 ? std::atoi(argv[3]) : 0;
        int h = argc > 4 ? std::atoi(argv[4]) : 0;
        int multiplier = argc > 5 ? std::atoi(argv[5]) : 2;
        try {
            int dummyW = 0, dummyH = 0;
            auto sequence = load_npy_sequence(folder, dummyW, dummyH);
            if (w == 0 || h == 0)
                throw std::runtime_error("width/height required, e.g. --visual-test-sequence <folder> 1920 1080 "
                                          "(must match the EXR resolution export_dlssg_sequence.py read)");
            auto cameras = load_camera_sequence(folder / "camera_sequence.json");
            return run_visual_test(w, h, std::move(sequence), std::move(cameras),
                                    (uint32_t)std::max(1, multiplier - 1));
        } catch (const std::exception& e) {
            std::cerr << "dlssg_bridge error: " << e.what() << "\n";
            return 1;
        }
    }

    try {
        Config cfg = parse_config(argv[1]);

        sl::Preferences pref{};
        pref.showConsole = true;
        pref.logLevel = sl::LogLevel::eVerbose;
        pref.pathToLogsAndData = L".";
        pref.renderAPI = sl::RenderAPI::eD3D12;
        // slSetTagForFrame requires this flag explicitly -- default flags
        // don't include it (confirmed via sl.log: "'slSetTagForFrame' SL API
        // is called but 'PreferenceFlag::eUseFrameBasedResourceTagging' flag
        // is not set!"). OR'd onto the struct's own defaults rather than
        // replacing them.
        pref.flags = pref.flags | sl::PreferenceFlags::eUseFrameBasedResourceTagging;
        // DLSS-G hard-requires the Reflex plugin to be loaded alongside it
        // (confirmed via sl.log: "Plugin 'sl.dlss_g' will be unloaded since
        // it requires plugin 'sl.reflex' which is NOT loaded") -- not
        // mentioned anywhere in the DLSS-G programming guide's own text.
        sl::Feature featuresToLoad[] = { sl::kFeatureDLSS_G, sl::kFeatureReflex };
        pref.featuresToLoad = featuresToLoad;
        pref.numFeaturesToLoad = 2;
        // NGX requires EITHER a real NVIDIA-issued applicationId OR the
        // combination of projectId + engine + engineVersion (confirmed by
        // an NVIDIA staff reply on the dev forums -- "Unable to find NGX
        // context" / "provide correct application id" happens if any of
        // the three is missing, projectId alone is not enough).
        pref.projectId = "51f5f296-55d5-4240-be4a-0e7f7637b7b1";
        pref.engine = sl::EngineType::eCustom;
        pref.engineVersion = "1.0.0";
        if (slInit(pref) != sl::Result::eOk)
            throw std::runtime_error("slInit failed -- is sl.interposer.dll next to the exe?");

        HWND hwnd = create_window(cfg.width, cfg.height, /*visible=*/false);
        D3D12Context ctx = create_d3d12_context(hwnd, cfg.width, cfg.height);

        LUID adapterLuid = ctx.device->GetAdapterLuid();
        sl::AdapterInfo adapterInfo{};
        adapterInfo.deviceLUID = reinterpret_cast<uint8_t*>(&adapterLuid);
        adapterInfo.deviceLUIDSizeInBytes = sizeof(adapterLuid);
        if (slIsFeatureSupported(sl::kFeatureDLSS_G, adapterInfo) != sl::Result::eOk)
            throw std::runtime_error("DLSS-G not supported on this adapter/driver");

        sl::DLSSGOptions options{};
        options.mode = sl::DLSSGMode::eOn;
        options.numFramesToGenerate = cfg.num_frames_to_generate;
        options.colorWidth = cfg.width;
        options.colorHeight = cfg.height;
        options.mvecDepthWidth = cfg.width;
        options.mvecDepthHeight = cfg.height;
        options.numBackBuffers = D3D12Context::kBackBufferCount;
        sl::ViewportHandle viewport{ 0 };
        if (slDLSSGSetOptions(viewport, options) != sl::Result::eOk)
            throw std::runtime_error("slDLSSGSetOptions failed");

        // --- Upload frame N and N+1's color/depth/mvec buffers ---
        auto color_n_data = load_npy(cfg.color_n);
        auto color_n1_data = load_npy(cfg.color_n1);
        auto depth_n_data = load_npy(cfg.depth_n);
        auto mvec_n_data = load_npy(cfg.mvec_n);

        // R8G8B8A8_UNORM (not the float format upload_texture() defaults
        // to elsewhere) -- CopyResource into the backbuffer requires a
        // matching format, see upload_color_as_backbuffer_source().
        auto colorTexN = upload_color_as_backbuffer_source(ctx, color_n_data, cfg.width, cfg.height);
        auto depthTex = upload_texture(ctx, depth_n_data, cfg.width, cfg.height, 1, DXGI_FORMAT_R32_FLOAT);
        auto mvecTex = upload_texture(ctx, mvec_n_data, cfg.width, cfg.height, 2, DXGI_FORMAT_R32G32_FLOAT);
        (void)color_n1_data;  // unused now that there's no readback to present a "next" frame for

        sl::Constants constsN = build_constants(
            cfg.cam_pos_n, cfg.cam_fwd_n, cfg.cam_right_n,
            cfg.cam_fov, cfg.cam_aspect, cfg.cam_near, cfg.cam_far,
            1.0f / cfg.width, 1.0f / cfg.height, cfg.depth_inverted);

        sl::FrameToken* frameToken = nullptr;
        if (slGetNewFrameToken(frameToken) != sl::Result::eOk)
            throw std::runtime_error("slGetNewFrameToken failed");

        if (slSetConstants(constsN, *frameToken, viewport) != sl::Result::eOk)
            throw std::runtime_error("slSetConstants failed");

        HR_CHECK(ctx.cmdAlloc->Reset());
        HR_CHECK(ctx.cmdList->Reset(ctx.cmdAlloc.Get(), nullptr));
        tag_frame(*frameToken, depthTex.Get(), mvecTex.Get(), ctx.cmdList.Get());
        HR_CHECK(ctx.cmdList->Close());
        ID3D12CommandList* lists[] = { ctx.cmdList.Get() };
        ctx.queue->ExecuteCommandLists(1, lists);
        ctx.wait_for_gpu();

        // copy_to_backbuffer() + Present() below are confirmed working (see
        // run_visual_test() and README "Visual test mode") -- vsync-off +
        // tearing is required, Present(1,0) silently produces no visible
        // generation. Included here for consistency even though this batch
        // path is blocked on a different, unrelated problem: there is no
        // known public API to read the generated frame back into a buffer
        // afterward (see README "No readback API"). Nothing past this point
        // can produce the .npy this function is meant to output.
        copy_to_backbuffer(ctx, colorTexN.Get());
        UINT presentFlags = ctx.tearingSupported ? DXGI_PRESENT_ALLOW_TEARING : 0;
        HR_CHECK(ctx.swapChain->Present(0, presentFlags));

        std::cerr << "dlssg_bridge: reached end of first pass; frame generation itself "
                     "works (see --visual-test) but there is no known API to read the "
                     "generated frame back for export -- see README \"No readback API\"\n";

        slShutdown();
        return 1;

    } catch (const std::exception& e) {
        std::cerr << "dlssg_bridge error: " << e.what() << "\n";
        return 1;
    }
}
