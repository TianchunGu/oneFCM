#include <cmeans/m_power_config.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool near(float left, float right, float tolerance = 1.0e-6f) {
  return std::fabs(left - right) <= tolerance;
}

template <typename Function>
bool throws_configuration_error(Function function) {
  try {
    function();
  } catch (const cmeans::FuzzifierConfigurationError&) {
    return true;
  }
  return false;
}

}  // namespace

int main() {
  for (const std::string& value :
       {"1.5", "2.0", "2.2", "2.5", "3.0", "4.25", "2.5001"}) {
    const double parsed = cmeans::parse_fuzzifier(value);
    if (parsed <= 1.0)
      return 1;
  }
  for (const std::string& value :
       {"1", "1.0", "0", "-2", "NaN", "nan", "Inf", "+Inf", "-Inf",
        "", "text", "2.0abc", "1e9999"}) {
    if (!throws_configuration_error(
            [&]() { (void)cmeans::parse_fuzzifier(value); })) {
      std::cerr << "accepted invalid fuzzifier: " << value << '\n';
      return 1;
    }
  }

  const auto default_config = cmeans::resolve_fuzzifier_config(
      "2.0", 2.0, cmeans::MPowerPolicy::Auto, false, false);
  if (default_config.center_effective_path !=
          cmeans::CenterPowerPath::M200Square ||
      default_config.membership_effective_path !=
          cmeans::MembershipPowerPath::M200Identity) {
    return 1;
  }
  for (const std::string& text : {"2", "2.0", "2.0000"}) {
    const double value = cmeans::parse_fuzzifier(text);
    const auto config = cmeans::resolve_fuzzifier_config(
        text, value, cmeans::MPowerPolicy::Auto, true, false);
    if (config.center_effective_path !=
            cmeans::CenterPowerPath::M200Square ||
        config.membership_effective_path !=
            cmeans::MembershipPowerPath::M200Identity) {
      std::cerr << "exact public m=2 spelling did not specialize: " << text
                << '\n';
      return 1;
    }
  }

  const auto generic_config = cmeans::resolve_fuzzifier_config(
      "2.0", 2.0, cmeans::MPowerPolicy::Generic, true, false);
  if (generic_config.center_effective_path !=
          cmeans::CenterPowerPath::GenericPow ||
      generic_config.membership_effective_path !=
          cmeans::MembershipPowerPath::GenericPow) {
    return 1;
  }

  const auto alias_config = cmeans::resolve_fuzzifier_config(
      "2.0", 2.0, cmeans::MPowerPolicy::Auto, false, true);
  if (alias_config.effective_policy != cmeans::MPowerPolicy::Generic)
    return 1;

  const auto generic_alias_config = cmeans::resolve_fuzzifier_config(
      "2.0", 2.0, cmeans::MPowerPolicy::Generic, true, true);
  if (generic_alias_config.effective_policy != cmeans::MPowerPolicy::Generic)
    return 1;

  if (!throws_configuration_error([&]() {
        (void)cmeans::resolve_fuzzifier_config(
            "2.0", 2.0, cmeans::MPowerPolicy::Auto, true, true);
      })) {
    return 1;
  }

  for (const double value : {1.5, 2.5, 3.0, 2.5001, 2.2, 4.25}) {
    const auto config = cmeans::resolve_fuzzifier_config(
        std::to_string(value), value, cmeans::MPowerPolicy::Auto, true,
        false);
    if (config.center_effective_path !=
            cmeans::CenterPowerPath::GenericPow ||
        config.membership_effective_path !=
            cmeans::MembershipPowerPath::GenericPow) {
      return 1;
    }
  }

  for (const double value : {2.0001, 1.9999}) {
    const auto config = cmeans::resolve_fuzzifier_config(
        std::to_string(value), value, cmeans::MPowerPolicy::Auto, true,
        false);
    if (config.center_effective_path !=
            cmeans::CenterPowerPath::GenericPow ||
        config.membership_effective_path !=
            cmeans::MembershipPowerPath::GenericPow) {
      std::cerr << "near-m=2 value incorrectly specialized: " << value
                << '\n';
      return 1;
    }
  }

  const cmeans::CenterM200SquareEvaluator square;
  const cmeans::MembershipM200IdentityEvaluator identity;
  const cmeans::CenterGenericPowEvaluator generic_center{2.5f};
  const cmeans::MembershipGenericPowEvaluator generic_membership{
      1.0f / 1.5f};
  if (!near(square(0.5f), 0.25f) || !near(identity(0.25f), 0.25f) ||
      !near(generic_center(0.0f), 0.0f) ||
      !near(generic_center(0.5f), std::pow(0.5f, 2.5f)) ||
      !near(generic_membership(0.25f),
            std::pow(0.25f, 1.0f / 1.5f))) {
    return 1;
  }

  return 0;
}
