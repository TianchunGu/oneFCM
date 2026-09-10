#include <cmeans/cmeans.h>

#include <array>
#include <iostream>

int main() {
  try {
    sycl::queue queue{sycl::cpu_selector_v};
    constexpr int points = 4;
    constexpr int dims = 2;
    constexpr int clusters = 2;
    std::array<float, points * dims> data{
        0.0f, 0.0f, 1.0f, 1.0f,
        0.0f, 1.0f, 0.0f, 1.0f,
    };
    std::array<float, points * clusters> membership{};
    std::array<float, clusters * dims> centers{};
    std::array<int, points> labels{};
    float classify_time = 0.0f;
    float initialize_time = 0.0f;
    float total_iteration_time = 0.0f;
    float first_iteration_time = 0.0f;
    float other_iteration_time = 0.0f;
    float change_time = 0.0f;
    float membership_time = 0.0f;
    float center_time = 0.0f;

    cmeans::WorkGroupRequest request;
    request.mode = cmeans::WorkGroupRequestMode::LegacyDefault;
    request.center_membership_raw_requested = 512;
    request.change_raw_requested = 256;
    cmeans::IterationOutcome outcome;

    const int iterations = cmeans::fuzzy_c_means_standard(
        queue, clusters, points, cmeans::FuzzifierConfig{}, 550, 0, 50,
        0.2f, dims, labels.data(), membership.data(), data.data(),
        centers.data(), classify_time, initialize_time, total_iteration_time,
        first_iteration_time, other_iteration_time, request, clusters, points,
        change_time, membership_time, center_time,
        cmeans::ExecutionModeConfig{}, &outcome);

    if (iterations != 20 || outcome.actual_iterations != 20 ||
        outcome.change_checks != 19) {
      std::cerr << "production Fixed20 contract failed\n";
      return 1;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
