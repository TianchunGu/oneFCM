# ---------------------------------------------------------------------------
# CMakeLists.txt for cmeansOneAPI Project
#
# This is the main configuration file for the project. It sets up the compiler,
# global build settings, custom options, and includes sub-project modules.
# ---------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

# --- Custom Build Options ---
# Add a cached variable named HARDWARE_TARGET to allow users to select the target hardware during configuration.
# CACHE STRING: Defines the variable type as a string and stores it in the CMake cache, so the user does not have to specify it every time.
# The default value is "CPU". The user can modify this option from the command line, e.g., `cmake -DHARDWARE_TARGET=GPU ..`
SET(HARDWARE_TARGET "CPU" CACHE STRING "Select hardware target (e.g., CPU, GPU, FPGA, INTEL_ARC_B580, AMD_GPU)")
set_property(CACHE HARDWARE_TARGET PROPERTY STRINGS
    CPU GPU INTEL_ARC_B580 AMD_GPU FPGA
)
set(_FCM_HARDWARE_TARGETS CPU GPU INTEL_ARC_B580 AMD_GPU FPGA)
if(NOT HARDWARE_TARGET IN_LIST _FCM_HARDWARE_TARGETS)
    message(FATAL_ERROR
        "Unsupported HARDWARE_TARGET='${HARDWARE_TARGET}'. Expected one of: "
        "CPU, GPU, INTEL_ARC_B580, AMD_GPU, FPGA.")
endif()

# --- AMD GPU toolchain selection ---
# AMD builds use an out-of-tree intel/llvm compiler and ROCm runtime. Keep all
# machine-specific locations explicit instead of embedding a developer path.
set(FCM_AMD_SYCL_SOURCE_ROOT "$ENV{SYCL_LLVM_ROOT}" CACHE PATH
    "intel/llvm source root for AMD_GPU builds")
set(FCM_AMD_SYCL_BUILD_ROOT "$ENV{SYCL_BUILD_ROOT}" CACHE PATH
    "intel/llvm build root for AMD_GPU builds")
set(FCM_AMD_SYCL_COMPILER "$ENV{FCM_AMD_SYCL_COMPILER}" CACHE FILEPATH
    "SYCL clang++ executable for AMD_GPU builds")
set(FCM_ROCM_ROOT "$ENV{ROCM_PATH}" CACHE PATH
    "ROCm installation root for AMD_GPU builds")

if(HARDWARE_TARGET STREQUAL "AMD_GPU")
    foreach(_FCM_REQUIRED_AMD_VARIABLE
            FCM_AMD_SYCL_SOURCE_ROOT
            FCM_AMD_SYCL_BUILD_ROOT
            FCM_ROCM_ROOT)
        if("${${_FCM_REQUIRED_AMD_VARIABLE}}" STREQUAL "")
            message(FATAL_ERROR
                "AMD_GPU requires ${_FCM_REQUIRED_AMD_VARIABLE}. Set it with "
                "-D${_FCM_REQUIRED_AMD_VARIABLE}=<path> or its documented "
                "environment hint before configuring.")
        endif()
    endforeach()

    if(NOT FCM_AMD_SYCL_COMPILER)
        set(FCM_AMD_SYCL_COMPILER
            "${FCM_AMD_SYCL_BUILD_ROOT}/bin/clang++")
    endif()
    if(NOT EXISTS "${FCM_AMD_SYCL_COMPILER}")
        message(FATAL_ERROR
            "AMD_GPU SYCL compiler not found at '${FCM_AMD_SYCL_COMPILER}'. "
            "Set FCM_AMD_SYCL_COMPILER to a valid clang++ executable.")
    endif()
    set(CMAKE_CXX_COMPILER "${FCM_AMD_SYCL_COMPILER}" CACHE FILEPATH
        "SYCL compiler for AMD HIP" FORCE)
endif()

# --- Compiler Selection ---
# Prefer the Intel oneAPI compiler for non-AMD builds when no compiler was
# selected explicitly. Sourcing setvars.sh makes icpx discoverable via PATH.
if(NOT HARDWARE_TARGET STREQUAL "AMD_GPU" AND NOT CMAKE_CXX_COMPILER)
    find_program(ONEAPI_ICPX NAMES icpx)
    if(ONEAPI_ICPX)
        set(CMAKE_CXX_COMPILER ${ONEAPI_ICPX} CACHE FILEPATH "DPC++ compiler" FORCE)
    else()
        message(FATAL_ERROR "icpx compiler not found; please source oneAPI setvars.sh or add it to PATH.")
    endif()
endif()

# --- Default Build Type ---
# If the user does not specify a build type via -DCMAKE_BUILD_TYPE=... when running cmake
# (e.g., Debug, Release, RelWithDebInfo, MinSizeRel),
if (NOT CMAKE_BUILD_TYPE)
    # then default it to "Release" to enable high-level optimizations (-O3 or -O3 -ffast-math).
    set(CMAKE_BUILD_TYPE Release)
endif()


# Enable Intel ITT markers that separate input, compute, and output regions.
option(FCM_ENABLE_ITT "Enable Intel ITT markers in main to aid kernel profiling" OFF)
option(FCM_ENABLE_PROFILING "Enable SYCL queue profiling and kernel timing" OFF)
# --- Output Directory Settings ---
# Set the output directory for all generated executables to the root of the build directory (${CMAKE_BINARY_DIR}).
# This makes it convenient to find and run the programs directly from the build directory without navigating into subdirectories.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# (Debug info) Uncomment this line to print the current path of the CMake script.
# message(${CMAKE_CURRENT_LIST_DIR}/cmake)


# --- Project Definition ---
# Define the project name as "cmeansOneAPI" and specify the primary language as C++ (CXX).
project(cmeansOneAPI LANGUAGES CXX)

# --- Project-scoped floating-point mode ---
# Keep the historical default (-O3 -ffast-math) while avoiding global flag
# leakage into unrelated or future targets.
option(
    FCM_ENABLE_FAST_MATH
    "Build oneFCM project targets with -ffast-math; disable for -fno-fast-math"
    ON
)

set(_FCM_VALID_BOOL_VALUES ON OFF TRUE FALSE YES NO Y N 1 0)
string(TOUPPER "${FCM_ENABLE_FAST_MATH}" _FCM_FAST_MATH_VALUE)
list(FIND _FCM_VALID_BOOL_VALUES "${_FCM_FAST_MATH_VALUE}" _FCM_BOOL_INDEX)
if(_FCM_BOOL_INDEX EQUAL -1)
    message(FATAL_ERROR
        "FCM_ENABLE_FAST_MATH must be a CMake boolean value, got "
        "'${FCM_ENABLE_FAST_MATH}'.")
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" _FCM_BUILD_TYPE_UPPER)
set(_FCM_BUILD_TYPE_FLAGS_VAR "CMAKE_CXX_FLAGS_${_FCM_BUILD_TYPE_UPPER}")
set(_FCM_GLOBAL_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} ${${_FCM_BUILD_TYPE_FLAGS_VAR}}")

if(FCM_ENABLE_FAST_MATH)
    if(_FCM_GLOBAL_CXX_FLAGS MATCHES "(^|[ ;])-fno-fast-math($|[ ;])")
        message(FATAL_ERROR
            "Global -fno-fast-math conflicts with FCM_ENABLE_FAST_MATH=ON. "
            "Remove the global flag and use the project option.")
    endif()
    set(FCM_MATH_MODE "fast")
    set(FCM_MATH_COMPILE_OPTION "-ffast-math")
else()
    if(_FCM_GLOBAL_CXX_FLAGS MATCHES "(^|[ ;])-ffast-math($|[ ;])")
        message(FATAL_ERROR
            "Global -ffast-math conflicts with FCM_ENABLE_FAST_MATH=OFF. "
            "Remove the global flag and use the project option.")
    endif()

    include(CheckCXXCompilerFlag)
    include(CheckCXXSourceCompiles)
    check_cxx_compiler_flag(
        "-fno-fast-math"
        FCM_HOST_SUPPORTS_FNO_FAST_MATH
    )
    if(NOT FCM_HOST_SUPPORTS_FNO_FAST_MATH)
        message(FATAL_ERROR
            "The selected C++ compiler does not accept -fno-fast-math.")
    endif()

    set(_FCM_REQUIRED_FLAGS_SAVED "${CMAKE_REQUIRED_FLAGS}")
    set(_FCM_REQUIRED_LINK_OPTIONS_SAVED "${CMAKE_REQUIRED_LINK_OPTIONS}")
    set(CMAKE_REQUIRED_FLAGS
        "${CMAKE_REQUIRED_FLAGS} -fsycl -fno-fast-math")
    set(CMAKE_REQUIRED_LINK_OPTIONS "-fsycl")
    check_cxx_source_compiles([=[
        #include <sycl/sycl.hpp>
        int main() {
          sycl::queue queue;
          int* value = sycl::malloc_shared<int>(1, queue);
          queue.single_task([=]() { *value = 1; }).wait();
          const int result = *value;
          sycl::free(value, queue);
          return result == 1 ? 0 : 1;
        }
    ]=] FCM_SYCL_SUPPORTS_FNO_FAST_MATH)
    set(CMAKE_REQUIRED_FLAGS "${_FCM_REQUIRED_FLAGS_SAVED}")
    set(CMAKE_REQUIRED_LINK_OPTIONS "${_FCM_REQUIRED_LINK_OPTIONS_SAVED}")
    if(NOT FCM_SYCL_SUPPORTS_FNO_FAST_MATH)
        message(FATAL_ERROR
            "The selected SYCL compiler cannot build host/device code with "
            "-fno-fast-math.")
    endif()

    set(FCM_MATH_MODE "no-fast")
    set(FCM_MATH_COMPILE_OPTION "-fno-fast-math")
endif()

add_library(fcm_math_mode INTERFACE)
target_compile_options(
    fcm_math_mode
    INTERFACE
        -O3
        ${FCM_MATH_COMPILE_OPTION}
)
message(STATUS
    "oneFCM math mode: ${FCM_MATH_MODE} "
    "(FCM_ENABLE_FAST_MATH=${FCM_ENABLE_FAST_MATH}, "
    "options=-O3 ${FCM_MATH_COMPILE_OPTION})")

# --- Include Subdirectories ---
# Sequentially add subdirectories to the build process. CMake will enter these directories
# and process their respective CMakeLists.txt files. This helps in modularizing the project.

# Add the "util" subdirectory, which contains utility functions or helper classes.
add_subdirectory(util)

# Add the "cmeans" subdirectory, which contains the core c-means algorithm implementation.
add_subdirectory(cmeans)

include(CTest)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
