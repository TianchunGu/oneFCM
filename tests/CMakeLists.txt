function(add_fcm_cpp_test target source)
    add_executable(${target} ${source})
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_options(${target} PRIVATE -Wall -Wextra)
    target_link_libraries(${target} PRIVATE ${ARGN})
    add_test(NAME ${target} COMMAND ${target})
endfunction()

function(set_fcm_test_labels label)
    set_tests_properties(${ARGN} PROPERTIES LABELS "${label}")
endfunction()

add_fcm_cpp_test(
    execution_mode_config_test execution_mode_config_test.cpp
    fcm_execution_mode
)
add_fcm_cpp_test(
    dynamic_control_test dynamic_control_test.cpp
    fcm_execution_mode
)
add_fcm_cpp_test(
    normalization_layout_test normalization_layout_test.cpp
    fcm_sycl_target_contract fcm_math_mode util
)
add_fcm_cpp_test(
    normalization_nonfinite_test normalization_nonfinite_test.cpp
    fcm_sycl_target_contract fcm_math_mode util
)
add_fcm_cpp_test(
    work_group_selection_test work_group_selection_test.cpp
    fcm_work_group_config
)
add_fcm_cpp_test(
    m_power_config_test m_power_config_test.cpp
    fcm_m_power_config
)
function(add_public_header_compile_test target definition)
    add_fcm_cpp_test(
        ${target} public_headers_compile_test.cpp
        fcm_execution_mode fcm_m_power_config fcm_work_group_config
    )
    target_compile_definitions(${target} PRIVATE ${definition}=1)
endfunction()

add_public_header_compile_test(
    public_cmeans_header_compile_test FCM_TEST_CMEANS_HEADER
)
add_public_header_compile_test(
    public_execution_mode_header_compile_test FCM_TEST_EXECUTION_MODE_HEADER
)
add_public_header_compile_test(
    public_m_power_header_compile_test FCM_TEST_M_POWER_HEADER
)
add_public_header_compile_test(
    public_work_group_header_compile_test FCM_TEST_WORK_GROUP_HEADER
)

add_test(
    NAME production_api_smoke_test
    COMMAND $<TARGET_FILE:production_api_smoke_test>
)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

function(add_no_bytecode_python_test name)
    add_test(
        NAME ${name}
        COMMAND "${CMAKE_COMMAND}" -E env
                "PYTHONDONTWRITEBYTECODE=1"
                "${Python3_EXECUTABLE}" -B
                ${ARGN}
    )
endfunction()

add_no_bytecode_python_test(
    dataset_generator_contract_test
    "${CMAKE_CURRENT_SOURCE_DIR}/dataset_generator_contract_test.py"
    --root "${CMAKE_SOURCE_DIR}"
)
add_no_bytecode_python_test(
    container_contract_test
    "${CMAKE_CURRENT_SOURCE_DIR}/container_contract_test.py"
    --root "${CMAKE_SOURCE_DIR}"
)
add_no_bytecode_python_test(
    execution_mode_cli_contract_test
    "${CMAKE_CURRENT_SOURCE_DIR}/execution_mode_cli_contract_test.py"
    --binary "$<TARGET_FILE:cmeansOA>"
    --dataset "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tiny.csv"
)
add_no_bytecode_python_test(
    work_group_source_contract_test
    "${CMAKE_CURRENT_SOURCE_DIR}/work_group_source_contract_test.py"
    --source "${CMAKE_SOURCE_DIR}/cmeans/src/cmeans.cpp"
    --work-group-header
    "${CMAKE_SOURCE_DIR}/cmeans/include/cmeans/work_group_config.h"
)
add_no_bytecode_python_test(
    membership_init_timing_boundary_contract_test
    "${CMAKE_CURRENT_SOURCE_DIR}/membership_init_timing_boundary_contract_test.py"
    --source "${CMAKE_SOURCE_DIR}/cmeans/src/cmeans.cpp"
)
add_no_bytecode_python_test(
    m_power_public_regression_test
    "${CMAKE_CURRENT_SOURCE_DIR}/m_power_public_regression_test.py"
    --binary "$<TARGET_FILE:cmeansOA>"
    --dataset "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tiny.csv"
)
add_no_bytecode_python_test(
    m_power_source_contract_test
    "${CMAKE_CURRENT_SOURCE_DIR}/m_power_source_contract_test.py"
    --source "${CMAKE_SOURCE_DIR}/cmeans/src/cmeans.cpp"
    --config-header "${CMAKE_SOURCE_DIR}/cmeans/include/cmeans/m_power_config.h"
)
add_no_bytecode_python_test(
    production_release_boundary_test
    "${CMAKE_CURRENT_SOURCE_DIR}/production_release_boundary_test.py"
    --binary "$<TARGET_FILE:cmeansOA>"
    --dataset "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tiny.csv"
    --root-cmake "${CMAKE_SOURCE_DIR}/CMakeLists.txt"
    --cmeans-cmake "${CMAKE_SOURCE_DIR}/cmeans/CMakeLists.txt"
    --public-header "${CMAKE_SOURCE_DIR}/cmeans/include/cmeans/cmeans.h"
)
add_no_bytecode_python_test(
    public_surface_contract_test
    "${CMAKE_CURRENT_SOURCE_DIR}/public_surface_contract_test.py"
    --public-header "${CMAKE_SOURCE_DIR}/cmeans/include/cmeans/cmeans.h"
    --work-group-header
    "${CMAKE_SOURCE_DIR}/cmeans/include/cmeans/work_group_config.h"
    --work-group-source
    "${CMAKE_SOURCE_DIR}/cmeans/src/work_group_config.cpp"
    --production-source "${CMAKE_SOURCE_DIR}/cmeans/src/cmeans.cpp"
)
add_no_bytecode_python_test(
    public_cli_contract_test
    "${CMAKE_CURRENT_SOURCE_DIR}/public_cli_contract_test.py"
    --binary "$<TARGET_FILE:cmeansOA>"
    --dataset "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tiny.csv"
    --source "${CMAKE_SOURCE_DIR}/cmeans/src/cmeans.cpp"
    --execution-header
    "${CMAKE_SOURCE_DIR}/cmeans/include/cmeans/execution_mode.h"
    --m-power-header
    "${CMAKE_SOURCE_DIR}/cmeans/include/cmeans/m_power_config.h"
)
add_no_bytecode_python_test(
    readme_contract_test
    "${CMAKE_CURRENT_SOURCE_DIR}/readme_contract_test.py"
    --readme "${CMAKE_SOURCE_DIR}/README.md"
    --source "${CMAKE_SOURCE_DIR}/cmeans/src/cmeans.cpp"
    --public-header
    "${CMAKE_SOURCE_DIR}/cmeans/include/cmeans/cmeans.h"
    --m-power-source
    "${CMAKE_SOURCE_DIR}/cmeans/src/m_power_config.cpp"
)
add_no_bytecode_python_test(
    public_tree_contract_test
    "${CMAKE_CURRENT_SOURCE_DIR}/public_tree_contract_test.py"
    --root "${CMAKE_SOURCE_DIR}"
)
add_no_bytecode_python_test(
    public_comment_contract_test
    "${CMAKE_CURRENT_SOURCE_DIR}/public_comment_contract_test.py"
    --root "${CMAKE_SOURCE_DIR}"
)

set_fcm_test_labels(
    "static"
    public_cmeans_header_compile_test
    public_execution_mode_header_compile_test
    public_m_power_header_compile_test
    public_work_group_header_compile_test
    container_contract_test
    dataset_generator_contract_test
    work_group_source_contract_test
    membership_init_timing_boundary_contract_test
    m_power_source_contract_test
    public_surface_contract_test
    readme_contract_test
    public_tree_contract_test
    public_comment_contract_test
)

set_fcm_test_labels(
    "cpu"
    execution_mode_config_test
    dynamic_control_test
    normalization_layout_test
    normalization_nonfinite_test
    work_group_selection_test
    m_power_config_test
    execution_mode_cli_contract_test
    m_power_public_regression_test
)

set_fcm_test_labels(
    "cpu;smoke"
    production_api_smoke_test
    production_release_boundary_test
    public_cli_contract_test
)
