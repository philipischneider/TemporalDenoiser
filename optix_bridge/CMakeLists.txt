cmake_minimum_required(VERSION 3.25)
project(optix_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# -----------------------------------------------------------------------
# Find CUDA
# -----------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# -----------------------------------------------------------------------
# Find OptiX SDK
# Set OPTIX_PATH to the root of your OptiX installation, e.g.:
#   cmake -DOPTIX_PATH="C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0" ..
# or set the environment variable OPTIX_PATH
# -----------------------------------------------------------------------
if(NOT DEFINED OPTIX_PATH)
    if(DEFINED ENV{OPTIX_PATH})
        set(OPTIX_PATH "$ENV{OPTIX_PATH}")
    else()
        # Try common default locations on Windows
        file(GLOB _optix_candidates
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK *"
        )
        list(SORT _optix_candidates)
        list(REVERSE _optix_candidates)
        if(_optix_candidates)
            list(GET _optix_candidates 0 OPTIX_PATH)
        else()
            message(FATAL_ERROR
                "OptiX SDK not found. Set -DOPTIX_PATH=<path> or OPTIX_PATH env var.\n"
                "Download from: https://developer.nvidia.com/designworks/optix/download"
            )
        endif()
    endif()
endif()

message(STATUS "Using OptiX SDK at: ${OPTIX_PATH}")

# -----------------------------------------------------------------------
# Target
# -----------------------------------------------------------------------
add_executable(optix_bridge
    optix_denoiser.cpp
)

target_include_directories(optix_bridge PRIVATE
    "${OPTIX_PATH}/include"
    "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_libraries(optix_bridge PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
)

# OptiX itself is header-only; the runtime is loaded via the driver
# No link against optix.lib needed

set_target_properties(optix_bridge PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
)
