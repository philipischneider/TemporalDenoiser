/**
 * optix_bridge.cpp
 *
 * Standalone OptiX 8.x temporal denoiser bridge.
 *
 * Usage:
 *   optix_bridge.exe config.json
 *
 * config.json fields:
 *   width, height, hdr, temporal,
 *   noisy   (path to float32 .npy, shape HxWx3),
 *   albedo  (optional, float32 HxWx3),
 *   normal  (optional, float32 HxWx3),
 *   prev_output (optional, float32 HxWx3),
 *   flow    (optional, float32 HxWx2),
 *   output  (path where float32 HxWx3 will be written)
 */

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>

// Minimal JSON parser using nlohmann/json embedded subset
// (we use a simpler hand-rolled parser to avoid extra dependencies)

// --------------------------------------------------------------------------
// Tiny JSON reader (keys we need only)
// --------------------------------------------------------------------------
struct Config {
    int width = 0;
    int height = 0;
    bool hdr = true;
    bool temporal = true;
    std::string noisy;
    std::string albedo;
    std::string normal;
    std::string prev_output;
    std::string flow;
    std::string output;
};

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n\"");
    size_t b = s.find_last_not_of(" \t\r\n\"");
    if (a == std::string::npos) return "";
    return s.substr(a, b - a + 1);
}

static Config parse_config(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open config: " + path);
    std::string text((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());

    Config cfg;
    // Super minimal parser for flat JSON with string/int/bool values
    size_t pos = 0;
    while (pos < text.size()) {
        size_t kstart = text.find('"', pos);
        if (kstart == std::string::npos) break;
        size_t kend = text.find('"', kstart + 1);
        std::string key = text.substr(kstart + 1, kend - kstart - 1);
        size_t colon = text.find(':', kend);
        size_t vstart = colon + 1;
        while (vstart < text.size() && std::isspace(text[vstart])) ++vstart;

        std::string value;
        if (text[vstart] == '"') {
            size_t vend = text.find('"', vstart + 1);
            value = text.substr(vstart + 1, vend - vstart - 1);
            pos = vend + 1;
        } else {
            size_t vend = text.find_first_of(",}", vstart);
            value = trim(text.substr(vstart, vend - vstart));
            pos = vend + 1;
        }

        if (key == "width")        cfg.width       = std::stoi(value);
        else if (key == "height")  cfg.height      = std::stoi(value);
        else if (key == "hdr")     cfg.hdr         = (value == "true");
        else if (key == "temporal") cfg.temporal   = (value == "true");
        else if (key == "noisy")       cfg.noisy       = value;
        else if (key == "albedo")      cfg.albedo      = value;
        else if (key == "normal")      cfg.normal      = value;
        else if (key == "prev_output") cfg.prev_output = value;
        else if (key == "flow")        cfg.flow        = value;
        else if (key == "output")      cfg.output      = value;
    }
    return cfg;
}

// --------------------------------------------------------------------------
// NumPy .npy loader (float32 only, C-contiguous)
// --------------------------------------------------------------------------
static std::vector<float> load_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open npy: " + path);

    // Magic + version
    char magic[7]; f.read(magic, 6); magic[6] = 0;
    if (std::string(magic) != "\x93NUMPY")
        throw std::runtime_error("Not a .npy file: " + path);
    uint8_t major, minor;
    f.read((char*)&major, 1); f.read((char*)&minor, 1);

    uint16_t hlen;
    f.read((char*)&hlen, 2);
    std::string header(hlen, ' ');
    f.read(header.data(), hlen);

    // Read all remaining bytes as float32
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

    // Write numpy header for float32 array of shape (h, w, c)
    std::ostringstream hdr;
    hdr << "{'descr': '<f4', 'fortran_order': False, 'shape': ("
        << h << ", " << w << ", " << c << "), }";
    std::string hdr_str = hdr.str();
    // Pad to multiple of 64 (v1.0 spec)
    size_t header_len = hdr_str.size() + 1; // +1 for newline
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

// --------------------------------------------------------------------------
// CUDA helpers
// --------------------------------------------------------------------------
#define CUDA_CHECK(call) \
    do { cudaError_t e = (call); if (e != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e)); } while(0)

#define OPTIX_CHECK(call) \
    do { OptixResult r = (call); if (r != OPTIX_SUCCESS) \
        throw std::runtime_error(std::string("OptiX error: ") + optixGetErrorName(r)); } while(0)

struct CudaBuffer {
    CUdeviceptr ptr = 0;
    size_t size = 0;

    void alloc(size_t bytes) {
        size = bytes;
        CUDA_CHECK(cudaMalloc((void**)&ptr, bytes));
    }
    void upload(const void* src) {
        CUDA_CHECK(cudaMemcpy((void*)ptr, src, size, cudaMemcpyHostToDevice));
    }
    void download(void* dst) const {
        CUDA_CHECK(cudaMemcpy(dst, (void*)ptr, size, cudaMemcpyDeviceToHost));
    }
    void free() {
        if (ptr) { cudaFree((void*)ptr); ptr = 0; }
    }
};

static OptixImage2D make_optix_image(CUdeviceptr ptr, int w, int h, OptixPixelFormat fmt) {
    OptixImage2D img = {};
    img.data = ptr;
    img.width = w;
    img.height = h;
    img.rowStrideInBytes = w * (fmt == OPTIX_PIXEL_FORMAT_FLOAT2 ? 8 : 12);
    img.pixelStrideInBytes = (fmt == OPTIX_PIXEL_FORMAT_FLOAT2 ? 8 : 12);
    img.format = fmt;
    return img;
}

// --------------------------------------------------------------------------
// main
// --------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: optix_bridge config.json\n";
        return 1;
    }

    try {
        Config cfg = parse_config(argv[1]);
        int W = cfg.width, H = cfg.height;
        size_t npixels = (size_t)W * H;

        // --- Load input buffers ---
        auto noisy_data = load_npy(cfg.noisy);
        std::vector<float> albedo_data, normal_data, prev_data, flow_data;

        bool has_albedo = !cfg.albedo.empty();
        bool has_normal = !cfg.normal.empty();
        bool has_prev   = !cfg.prev_output.empty();
        bool has_flow   = !cfg.flow.empty();
        bool do_temporal = cfg.temporal && has_prev;

        if (has_albedo) albedo_data = load_npy(cfg.albedo);
        if (has_normal) normal_data = load_npy(cfg.normal);
        if (has_prev)   prev_data   = load_npy(cfg.prev_output);
        if (has_flow)   flow_data   = load_npy(cfg.flow);

        // --- CUDA init ---
        CUDA_CHECK(cudaFree(0)); // lazy init
        CUstream stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // --- OptiX init ---
        OPTIX_CHECK(optixInit());

        OptixDeviceContext optix_ctx;
        CUcontext cu_ctx;
        cuCtxGetCurrent(&cu_ctx);
        OptixDeviceContextOptions ctx_opts = {};
        ctx_opts.logCallbackFunction = nullptr;
        ctx_opts.logCallbackLevel = 0;
        OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &ctx_opts, &optix_ctx));

        // --- Denoiser setup ---
        OptixDenoiserOptions denoiser_opts = {};
        denoiser_opts.guideAlbedo = has_albedo ? 1 : 0;
        denoiser_opts.guideNormal = has_normal ? 1 : 0;

        OptixDenoiserModelKind model_kind = do_temporal
            ? OPTIX_DENOISER_MODEL_KIND_TEMPORAL
            : OPTIX_DENOISER_MODEL_KIND_HDR;

        OptixDenoiser denoiser;
        OPTIX_CHECK(optixDenoiserCreate(optix_ctx, model_kind, &denoiser_opts, &denoiser));

        OptixDenoiserSizes sizes;
        OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser, W, H, &sizes));

        CudaBuffer state_buf, scratch_buf;
        state_buf.alloc(sizes.stateSizeInBytes);
        scratch_buf.alloc(sizes.withoutOverlapScratchSizeInBytes);

        OPTIX_CHECK(optixDenoiserSetup(
            denoiser, stream,
            W, H,
            state_buf.ptr, sizes.stateSizeInBytes,
            scratch_buf.ptr, sizes.withoutOverlapScratchSizeInBytes
        ));

        // --- Upload buffers ---
        CudaBuffer noisy_buf, albedo_buf, normal_buf, prev_buf, flow_buf, output_buf, hdr_intensity_buf;
        std::vector<float> output_data(npixels * 3, 0.0f);

        noisy_buf.alloc(npixels * 3 * sizeof(float));
        noisy_buf.upload(noisy_data.data());
        output_buf.alloc(npixels * 3 * sizeof(float));

        if (has_albedo) { albedo_buf.alloc(npixels * 3 * sizeof(float)); albedo_buf.upload(albedo_data.data()); }
        if (has_normal) { normal_buf.alloc(npixels * 3 * sizeof(float)); normal_buf.upload(normal_data.data()); }
        if (has_prev)   { prev_buf.alloc(npixels * 3 * sizeof(float));   prev_buf.upload(prev_data.data()); }
        if (has_flow)   { flow_buf.alloc(npixels * 2 * sizeof(float));   flow_buf.upload(flow_data.data()); }
        hdr_intensity_buf.alloc(sizeof(float));

        // --- Build layer structs ---
        OptixDenoiserLayer layer = {};
        layer.input  = make_optix_image(noisy_buf.ptr,  W, H, OPTIX_PIXEL_FORMAT_FLOAT3);
        layer.output = make_optix_image(output_buf.ptr, W, H, OPTIX_PIXEL_FORMAT_FLOAT3);
        if (do_temporal && has_prev) {
            layer.previousOutput = make_optix_image(prev_buf.ptr, W, H, OPTIX_PIXEL_FORMAT_FLOAT3);
        }

        OptixDenoiserGuideLayer guide = {};
        if (has_albedo) guide.albedo = make_optix_image(albedo_buf.ptr, W, H, OPTIX_PIXEL_FORMAT_FLOAT3);
        if (has_normal) guide.normal = make_optix_image(normal_buf.ptr, W, H, OPTIX_PIXEL_FORMAT_FLOAT3);
        if (do_temporal && has_flow) guide.flow = make_optix_image(flow_buf.ptr, W, H, OPTIX_PIXEL_FORMAT_FLOAT2);

        // --- Invoke ---
        OptixDenoiserParams params = {};
        params.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
        if (cfg.hdr) {
            OPTIX_CHECK(optixDenoiserComputeIntensity(
                denoiser, stream, &layer.input,
                hdr_intensity_buf.ptr,
                scratch_buf.ptr, sizes.withoutOverlapScratchSizeInBytes
            ));
            params.hdrIntensity = hdr_intensity_buf.ptr;
        }

        OPTIX_CHECK(optixDenoiserInvoke(
            denoiser, stream, &params,
            state_buf.ptr, sizes.stateSizeInBytes,
            &guide, &layer, 1,
            0, 0,
            scratch_buf.ptr, sizes.withoutOverlapScratchSizeInBytes
        ));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        // --- Download & save ---
        output_buf.download(output_data.data());
        save_npy(cfg.output, output_data, H, W, 3);

        // --- Cleanup ---
        noisy_buf.free(); albedo_buf.free(); normal_buf.free();
        prev_buf.free(); flow_buf.free(); output_buf.free();
        hdr_intensity_buf.free(); state_buf.free(); scratch_buf.free();

        optixDenoiserDestroy(denoiser);
        optixDeviceContextDestroy(optix_ctx);
        cudaStreamDestroy(stream);

        std::cout << "OptiX denoising complete.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
