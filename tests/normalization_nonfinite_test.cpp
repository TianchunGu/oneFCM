#include <util/utils.h>

#include <iostream>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace {
int fail(const char* message) {
  std::cerr << message << '\n';
  return 1;
}
}  // namespace

int main() {
  const std::vector<float> valid = {0.5f, 0.5f, 0.5f, 0.5f};
  const std::uint32_t nan_bits = 0x7fc00000U;
  const std::uint32_t inf_bits = 0x7f800000U;
  float nan_value = 0.0f;
  float inf_value = 0.0f;
  std::memcpy(&nan_value, &nan_bits, sizeof(nan_value));
  std::memcpy(&inf_value, &inf_bits, sizeof(inf_value));
  auto nan_values = valid;
  nan_values[0] = nan_value;
  const auto nan_stats = util::compute_membership_normalization_stats(
      nan_values.data(), 2, 2, 1.0e-6);
  if (nan_stats.all_finite || nan_stats.normalized_within_tolerance ||
      nan_stats.nonfinite_count != 1 || nan_stats.finite_count != 3) {
    std::cerr << "NaN stats: all_finite=" << nan_stats.all_finite
              << " normalized=" << nan_stats.normalized_within_tolerance
              << " finite=" << nan_stats.finite_count
              << " nonfinite=" << nan_stats.nonfinite_count << '\n';
    return fail("NaN did not fail closed while preserving counts");
  }

  auto inf_values = valid;
  inf_values[3] = inf_value;
  const auto inf_stats = util::compute_membership_normalization_stats(
      inf_values.data(), 2, 2, 1.0e-6);
  if (inf_stats.all_finite || inf_stats.normalized_within_tolerance ||
      inf_stats.nonfinite_count != 1 || inf_stats.finite_count != 3) {
    std::cerr << "Inf stats: all_finite=" << inf_stats.all_finite
              << " normalized=" << inf_stats.normalized_within_tolerance
              << " finite=" << inf_stats.finite_count
              << " nonfinite=" << inf_stats.nonfinite_count << '\n';
    return fail("Inf did not fail closed while preserving counts");
  }

  const auto null_stats =
      util::compute_membership_normalization_stats(nullptr, 2, 2, 1.0e-6);
  if (null_stats.normalized_within_tolerance || null_stats.value_count != 0)
    return fail("null pointer did not fail closed");
  for (const auto dimensions :
       {std::pair<int, int>{0, 2}, {-1, 2}, {2, 0}, {2, -1}}) {
    const auto stats = util::compute_membership_normalization_stats(
        valid.data(), dimensions.first, dimensions.second, 1.0e-6);
    if (stats.normalized_within_tolerance || stats.value_count != 0)
      return fail("invalid dimensions did not fail closed");
  }
  const auto bad_tolerance = util::compute_membership_normalization_stats(
      valid.data(), 2, 2, -1.0);
  if (bad_tolerance.normalized_within_tolerance)
    return fail("negative tolerance did not fail closed");
  return 0;
}
