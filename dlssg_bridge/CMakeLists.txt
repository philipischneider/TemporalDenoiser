cmake_minimum_required(VERSION 3.25)
project(dlssg_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# -----------------------------------------------------------------------
# Find the Streamline SDK
# Set STREAMLINE_SDK_PATH to the extracted release zip root, e.g.:
#   cmake -DSTREAMLINE_SDK_PATH="C:/SDKs/streamline-sdk-v2.14.1" ..
# or set the environment variable STREAMLINE_SDK_PATH.
# Download from: https://github.com/NVIDIA-RTX/Streamline/releases
# (no NVIDIA developer account needed — the release zip ships headers, libs,
# and the prebuilt signed sl.dlss_g.dll)
# -----------------------------------------------------------------------
if(NOT DEFINED STREAMLINE_SDK_PATH)
    if(DEFINED ENV{STREAMLINE_SDK_PATH})
        set(STREAMLINE_SDK_PATH "$ENV{STREAMLINE_SDK_PATH}")
    else()
        message(FATAL_ERROR
            "Streamline SDK not found. Set -DSTREAMLINE_SDK_PATH=<path> or the "
            "STREAMLINE_SDK_PATH env var to the extracted release zip root.\n"
            "Download from: https://github.com/NVIDIA-RTX/Streamline/releases"
        )
    endif()
endif()

message(STATUS "Using Streamline SDK at: ${STREAMLINE_SDK_PATH}")

add_executable(dlssg_bridge
    main.cpp
)

target_include_directories(dlssg_bridge PRIVATE
    "${STREAMLINE_SDK_PATH}/include"
)

target_link_directories(dlssg_bridge PRIVATE
    "${STREAMLINE_SDK_PATH}/lib/x64"
)

target_link_libraries(dlssg_bridge PRIVATE
    sl.interposer
    d3d12
    dxgi
    d3dcompiler
)

set_target_properties(dlssg_bridge PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
)

# Streamline plugin DLLs (sl.interposer.dll, sl.dlss_g.dll, sl.common.dll,
# etc.) must sit next to the exe at runtime — Streamline discovers plugins by
# scanning the executable's own directory, same idea as OptiX's runtime DLL
# search but explicit rather than driver-mediated.
add_custom_command(TARGET dlssg_bridge POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${STREAMLINE_SDK_PATH}/bin/x64"
        "$<TARGET_FILE_DIR:dlssg_bridge>"
    COMMENT "Copying Streamline runtime DLLs next to dlssg_bridge.exe"
)
