cmake_minimum_required(VERSION 3.16)

# --- Source File Discovery ---
file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
list(FILTER srcs EXCLUDE REGEX "/work_group_config\\.cpp$")
list(FILTER srcs EXCLUDE REGEX "/m_power_config\\.cpp$")
list(FILTER srcs EXCLUDE REGEX "/execution_mode\\.cpp$")

# --- Target Definition ---
set(TARGET_NAME cmeansOA)

# ============================================================
# Hardware-specific SYCL options (compile opts vs link opts)
# ============================================================
set(SYCL_COMPILE_OPTS "")
set(SYCL_LINK_OPTS "")

if (HARDWARE_TARGET STREQUAL "GPU")
    # Target NVIDIA GPUs through the CUDA backend and the requested SM arch.
    message(STATUS "-- Configuring for NVIDIA GPU Target (CUDA) --")
    set(CUDA_GPU_ARCH "sm_89" CACHE STRING "NVIDIA CUDA GPU arch")
    list(APPEND SYCL_COMPILE_OPTS
        -fsycl-targets=nvptx64-nvidia-cuda
        -Xsycl-target-backend=nvptx64-nvidia-cuda
        --cuda-gpu-arch=${CUDA_GPU_ARCH}
        -fsycl-unnamed-lambda
        -Wno-linker-warnings
    )
    list(APPEND SYCL_LINK_OPTS
        -fsycl-targets=nvptx64-nvidia-cuda
        -Xsycl-target-backend=nvptx64-nvidia-cuda
        --cuda-gpu-arch=${CUDA_GPU_ARCH}
        -Wno-linker-warnings
    )

elseif(HARDWARE_TARGET STREQUAL "AMD_GPU")
    # ========================================================
    # AMD GPU configuration
    # ========================================================
    message(STATUS "-- Configuring for AMD GPU Target (ROCm/HIP) --")
    
    # Default to gfx1200; override AMD_GPU_ARCH with the actual device target.
    set(AMD_GPU_ARCH "gfx1200" CACHE STRING "AMD GPU offload arch (e.g., gfx1100, gfx1200, gfx90a)")
    set(AMD_TRIPLE "amdgcn-amd-amdhsa")

    # Compile the SYCL device image for the selected AMD offload architecture.
    list(APPEND SYCL_COMPILE_OPTS
        -fsycl-targets=${AMD_TRIPLE}
        -Xsycl-target-backend=${AMD_TRIPLE}
        --offload-arch=${AMD_GPU_ARCH}
        -fsycl-unnamed-lambda
        -Wno-linker-warnings
    )
    
    # Link against the same AMD target triple and architecture.
    list(APPEND SYCL_LINK_OPTS
        -fsycl-targets=${AMD_TRIPLE}
        -Xsycl-target-backend=${AMD_TRIPLE}
        --offload-arch=${AMD_GPU_ARCH}
        -Wno-linker-warnings
    )

elseif(HARDWARE_TARGET STREQUAL "INTEL_ARC_B580")
    # Use the oneAPI compiler's Level Zero JIT path for Intel Arc B580.
    message(STATUS "-- Configuring for Intel Arc B580 (Level Zero JIT) --")
    list(APPEND SYCL_COMPILE_OPTS -fsycl-unnamed-lambda -Wno-linker-warnings)
    list(APPEND SYCL_LINK_OPTS -Wno-linker-warnings)

elseif(HARDWARE_TARGET STREQUAL "FPGA")
    # Build the current FPGA path for the Intel FPGA emulator.
    message(STATUS "-- Configuring for FPGA Target (Emulator) --")
    list(APPEND SYCL_COMPILE_OPTS -fintelfpga)

else()
    message(STATUS "-- Configuring for CPU Target --")
endif()

# --- AMD out-of-tree compiler headers and runtime search path ---
if(HARDWARE_TARGET STREQUAL "AMD_GPU")
    set(_FCM_AMD_SYCL_SOURCE_INCLUDE
        "${FCM_AMD_SYCL_SOURCE_ROOT}/sycl/include")
    set(_FCM_AMD_SYCL_BUILD_INCLUDE
        "${FCM_AMD_SYCL_BUILD_ROOT}/include/sycl")
    foreach(_FCM_REQUIRED_AMD_DIRECTORY
            "${_FCM_AMD_SYCL_SOURCE_INCLUDE}"
            "${_FCM_AMD_SYCL_BUILD_INCLUDE}"
            "${FCM_ROCM_ROOT}")
        if(NOT IS_DIRECTORY "${_FCM_REQUIRED_AMD_DIRECTORY}")
            message(FATAL_ERROR
                "AMD_GPU required directory not found: "
                "'${_FCM_REQUIRED_AMD_DIRECTORY}'.")
        endif()
    endforeach()

    set(_FCM_AMD_RUNTIME_DIRS
        "${FCM_ROCM_ROOT}/lib"
        "${FCM_ROCM_ROOT}/lib64"
        "${FCM_AMD_SYCL_BUILD_ROOT}/lib"
    )
endif()

# --- oneDPL include path ---
set(FCM_ONEDPL_ROOT "$ENV{DPL_ROOT}" CACHE PATH
    "oneDPL installation root (the directory containing include/)")
find_path(FCM_ONEDPL_INCLUDE_DIR
    NAMES oneapi/dpl/random
    HINTS "${FCM_ONEDPL_ROOT}"
    PATH_SUFFIXES include
)
if(NOT FCM_ONEDPL_INCLUDE_DIR)
    message(FATAL_ERROR
        "oneDPL header 'oneapi/dpl/random' was not found. Source the oneAPI "
        "environment or set FCM_ONEDPL_ROOT=<oneDPL-root>.")
endif()
add_library(fcm_onedpl INTERFACE)
target_include_directories(
    fcm_onedpl SYSTEM INTERFACE "${FCM_ONEDPL_INCLUDE_DIR}"
)

# ============================================================
# Apply SYCL base flags + hardware-specific opts
# ============================================================
# Keep the hardware-specific device image contract reusable by test targets.
# Test executables that submit SYCL kernels must carry the same target triple,
# backend, architecture, and device-link options as cmeansOA.
add_library(fcm_sycl_target_contract INTERFACE)
target_compile_options(
    fcm_sycl_target_contract
    INTERFACE -fsycl ${SYCL_COMPILE_OPTS}
)
target_link_options(
    fcm_sycl_target_contract
    INTERFACE -fsycl ${SYCL_LINK_OPTS}
)

# --- Hardware macro definitions ---
if (HARDWARE_TARGET STREQUAL "GPU")
    target_compile_definitions(fcm_sycl_target_contract INTERFACE _GPU_)
elseif(HARDWARE_TARGET STREQUAL "AMD_GPU")
    target_compile_definitions(fcm_sycl_target_contract INTERFACE _GPU_ _AMD_GPU_)
elseif(HARDWARE_TARGET STREQUAL "INTEL_ARC_B580")
    target_compile_definitions(fcm_sycl_target_contract INTERFACE _GPU_ _INTEL_ARC_B580_)
elseif(HARDWARE_TARGET STREQUAL "FPGA")
    target_compile_definitions(
        fcm_sycl_target_contract INTERFACE _FPGA_ FPGA_EMULATOR
    )
else()
    target_compile_definitions(fcm_sycl_target_contract INTERFACE _CPU_)
endif()

# --- Production targets ---
add_executable(${TARGET_NAME} ${srcs})
add_library(fcm_work_group_config STATIC
    src/work_group_config.cpp
    include/cmeans/work_group_config.h
)
add_library(fcm_m_power_config STATIC
    src/m_power_config.cpp
    include/cmeans/m_power_config.h
)
add_library(fcm_execution_mode STATIC
    src/execution_mode.cpp
    include/cmeans/execution_mode.h
)

target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)
target_include_directories(${TARGET_NAME} PUBLIC include)
target_include_directories(fcm_work_group_config PUBLIC include)
target_include_directories(fcm_m_power_config PUBLIC include)
target_include_directories(fcm_execution_mode PUBLIC include)

if(HARDWARE_TARGET STREQUAL "AMD_GPU")
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
        "${_FCM_AMD_SYCL_SOURCE_INCLUDE}"
        "${_FCM_AMD_SYCL_BUILD_INCLUDE}"
    )
    set_target_properties(${TARGET_NAME} PROPERTIES
        BUILD_RPATH "${_FCM_AMD_RUNTIME_DIRS}"
        INSTALL_RPATH "${_FCM_AMD_RUNTIME_DIRS}"
    )
endif()

target_compile_options(
    ${TARGET_NAME} PRIVATE -Wall -Wno-deprecated-declarations
)
target_compile_options(
    fcm_work_group_config PRIVATE -Wall -Wno-deprecated-declarations
)
target_compile_options(
    fcm_m_power_config PRIVATE -Wall -Wno-deprecated-declarations
)
target_compile_options(fcm_execution_mode PRIVATE -Wall -Wextra)

# --- Optional ITT/profiling definitions ---
if(FCM_ENABLE_ITT)
    set(FCM_ITT_ROOT "$ENV{VTUNE_PROFILER_DIR}" CACHE PATH
        "Intel VTune root containing ITT headers and library")
    find_path(FCM_ITT_INCLUDE_DIR
        NAMES ittnotify.h
        HINTS "${FCM_ITT_ROOT}"
        PATH_SUFFIXES include
    )
    find_library(FCM_ITT_LIBRARY
        NAMES ittnotify
        HINTS "${FCM_ITT_ROOT}"
        PATH_SUFFIXES lib64 lib
    )
    if(NOT FCM_ITT_INCLUDE_DIR OR NOT FCM_ITT_LIBRARY)
        message(FATAL_ERROR
            "FCM_ENABLE_ITT=ON requires ittnotify.h and libittnotify. "
            "Source the VTune environment or set FCM_ITT_ROOT.")
    endif()
    target_include_directories(
        ${TARGET_NAME} SYSTEM PRIVATE "${FCM_ITT_INCLUDE_DIR}"
    )
    target_compile_definitions(${TARGET_NAME} PRIVATE FCM_ENABLE_ITT=1)
    target_link_libraries(${TARGET_NAME} PRIVATE "${FCM_ITT_LIBRARY}")
endif()
if(FCM_ENABLE_PROFILING)
    target_compile_definitions(${TARGET_NAME} PRIVATE FCM_ENABLE_PROFILING=1)
endif()

# --- Final Linking ---
target_link_libraries(${TARGET_NAME} PRIVATE
    fcm_math_mode
    fcm_sycl_target_contract
    fcm_onedpl
    fcm_work_group_config
    fcm_m_power_config
    fcm_execution_mode
    util
)

target_link_libraries(fcm_work_group_config PUBLIC
    fcm_math_mode
    fcm_sycl_target_contract
)

target_link_libraries(fcm_m_power_config PUBLIC
    fcm_math_mode
    fcm_sycl_target_contract
)

target_link_libraries(fcm_execution_mode PUBLIC
    fcm_math_mode
)

add_executable(
    production_api_smoke_test
    src/cmeans.cpp
    ${CMAKE_SOURCE_DIR}/tests/production_api_smoke_test.cpp
)
target_include_directories(
    production_api_smoke_test
    PRIVATE
        include
)
target_compile_definitions(
    production_api_smoke_test
    PRIVATE FCM_SUPPRESS_MAIN=1
)
target_compile_features(
    production_api_smoke_test PRIVATE cxx_std_17
)
target_compile_options(
    production_api_smoke_test
    PRIVATE -Wall -Wno-deprecated-declarations
)
target_link_libraries(
    production_api_smoke_test
    PRIVATE
        fcm_sycl_target_contract
        fcm_onedpl
        fcm_math_mode
        fcm_work_group_config
        fcm_m_power_config
        fcm_execution_mode
        util
)
