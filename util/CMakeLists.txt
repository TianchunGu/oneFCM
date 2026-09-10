# ===========================================================================
# CMakeLists.txt for the 'util' library
#
# This file is responsible for defining and building the 'util' static library,
# which provides common utility functions and classes for the project.
# ===========================================================================

# --- 1. Source File Discovery ---
file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)


# --- 2. Create Static Library ---
add_library(util STATIC ${srcs})


# --- 3. Specify Include Directories ---
target_include_directories(util PUBLIC include)

# Keep CSV parsing/output under the same project-scoped floating-point mode as
# cmeansOA so the default build preserves the legacy input/output contract.
target_link_libraries(util PRIVATE fcm_math_mode)
