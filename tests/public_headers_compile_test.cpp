#if defined(FCM_TEST_CMEANS_HEADER)
#include <cmeans/cmeans.h>
#elif defined(FCM_TEST_EXECUTION_MODE_HEADER)
#include <cmeans/execution_mode.h>
#elif defined(FCM_TEST_M_POWER_HEADER)
#include <cmeans/m_power_config.h>
#elif defined(FCM_TEST_WORK_GROUP_HEADER)
#include <cmeans/work_group_config.h>
#else
#error "Select exactly one public header contract"
#endif

#include <type_traits>

#if defined(FCM_TEST_CMEANS_HEADER)
using PublicHeaderContract = decltype(&cmeans::fuzzy_c_means_standard);
static_assert(std::is_pointer_v<PublicHeaderContract>);
#elif defined(FCM_TEST_EXECUTION_MODE_HEADER)
static_assert(std::is_default_constructible_v<cmeans::ExecutionModeConfig>);
#elif defined(FCM_TEST_M_POWER_HEADER)
static_assert(std::is_default_constructible_v<cmeans::FuzzifierConfig>);
#elif defined(FCM_TEST_WORK_GROUP_HEADER)
static_assert(std::is_default_constructible_v<cmeans::WorkGroupRequest>);
static_assert(
    std::is_base_of_v<std::runtime_error, cmeans::WorkGroupExecutionError>);
#endif

int main() { return 0; }
