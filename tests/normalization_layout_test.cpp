#include <util/utils.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace {
bool close(double left, double right, double tolerance = 1.0e-12) {
  return std::fabs(left - right) <= tolerance;
}

int fail(const char* message) {
  std::cerr << message << '\n';
  return 1;
}
}  // namespace

int main() {
  constexpr int k = 3;
  constexpr int n = 5;
  const std::vector<float> membership = {
      0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
      0.2f, 0.3f, 0.4f, 0.1f, 0.2f,
      0.7f, 0.5f, 0.3f, 0.5f, 0.3f};
  const auto stats =
      util::compute_membership_normalization_stats(membership.data(), k, n,
                                                   1.0e-6);
  if (!stats.normalized_within_tolerance || !stats.all_finite)
    return fail("valid cluster-major fixture failed");
  if (stats.value_count != 15 || stats.finite_count != 15 ||
      stats.nonfinite_count != 0)
    return fail("valid fixture counts differ");
  if (!close(stats.membership_min, 0.1f) ||
      !close(stats.membership_max, 0.7f))
    return fail("membership range differs");

  bool legacy_contract_passes = true;
  for (int point = 0; point < n; ++point) {
    double sum = 0.0;
    for (int cluster = 0; cluster < k; ++cluster)
      sum += membership[k * point + cluster];
    legacy_contract_passes &= std::fabs(sum - 1.0) <= 1.0e-6;
  }
  if (legacy_contract_passes)
    return fail("fixture did not discriminate the legacy indexing");

  std::vector<float> bad_sum = membership;
  bad_sum[2 * n + 4] += 0.02f;
  const auto bad = util::compute_membership_normalization_stats(
      bad_sum.data(), k, n, 1.0e-6);
  if (bad.normalized_within_tolerance || bad.first_failed_point != 4)
    return fail("bad point sum did not fail");
  if (bad.max_abs_sum_error <= 0.019)
    return fail("max normalization error is too small");
  if (bad.mean_abs_sum_error <= 0.003)
    return fail("mean normalization error is too small");

  const double exact_boundary = bad.max_abs_sum_error;
  const auto at_boundary = util::compute_membership_normalization_stats(
      bad_sum.data(), k, n, exact_boundary);
  const auto below_boundary = util::compute_membership_normalization_stats(
      bad_sum.data(), k, n,
      std::nextafter(exact_boundary, 0.0));
  if (!at_boundary.normalized_within_tolerance ||
      below_boundary.normalized_within_tolerance)
    return fail("tolerance boundary is not inclusive/exact");
  if (!util::verify_normalization(membership.data(), k, n))
    return fail("compatibility wrapper rejected valid storage");
  return 0;
}
