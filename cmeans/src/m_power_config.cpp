#include <cmeans/m_power_config.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace cmeans {
namespace {

bool ieee_finite_double(double value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const volatile std::uint64_t exponent =
      bits & 0x7ff0000000000000ULL;
  return exponent != 0x7ff0000000000000ULL;
}

bool ieee_finite_float(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const volatile std::uint32_t exponent = bits & 0x7f800000U;
  return exponent != 0x7f800000U;
}

bool is_exact_public_m2(double value) { return value == 2.0; }

}  // namespace

double parse_fuzzifier(const std::string& value) {
  if (value.empty())
    throw FuzzifierConfigurationError("--fuzzifier requires a value");
  std::size_t parsed = 0;
  double requested = 0.0;
  try {
    requested = std::stod(value, &parsed);
  } catch (const std::exception&) {
    throw FuzzifierConfigurationError("invalid --fuzzifier value: " + value);
  }
  if (parsed != value.size())
    throw FuzzifierConfigurationError(
        "--fuzzifier must consume the complete value: " + value);
  if (!ieee_finite_double(requested) || requested <= 1.0)
    throw FuzzifierConfigurationError(
        "--fuzzifier must be finite and greater than 1");
  const float effective = static_cast<float>(requested);
  if (!ieee_finite_float(effective) || effective <= 1.0f)
    throw FuzzifierConfigurationError(
        "--fuzzifier is outside the supported effective float range");
  return requested;
}

MPowerPolicy parse_m_power_policy(const std::string& value) {
  if (value == "auto")
    return MPowerPolicy::Auto;
  if (value == "generic")
    return MPowerPolicy::Generic;
  throw FuzzifierConfigurationError(
      "--m-power-policy must be auto or generic");
}

const char* m_power_policy_name(MPowerPolicy policy) {
  switch (policy) {
    case MPowerPolicy::Auto:
      return "auto";
    case MPowerPolicy::Generic:
      return "generic";
  }
  return "unknown";
}

const char* center_power_path_name(CenterPowerPath path) {
  switch (path) {
    case CenterPowerPath::M200Square:
      return "center:m200_square";
    case CenterPowerPath::GenericPow:
      return "center:generic_pow";
  }
  return "center:unknown";
}

const char* membership_power_path_name(MembershipPowerPath path) {
  switch (path) {
    case MembershipPowerPath::M200Identity:
      return "membership:m200_identity";
    case MembershipPowerPath::GenericPow:
      return "membership:generic_pow";
  }
  return "membership:unknown";
}

FuzzifierConfig resolve_fuzzifier_config(
    const std::string& requested_m_text,
    double requested_m,
    MPowerPolicy requested_policy,
    bool explicit_policy_present,
    bool force_generic_alias_present) {
  if (!ieee_finite_double(requested_m) || requested_m <= 1.0)
    throw FuzzifierConfigurationError(
        "requested fuzzifier must be finite and greater than 1");
  const float effective_m = static_cast<float>(requested_m);
  if (!ieee_finite_float(effective_m) || effective_m <= 1.0f)
    throw FuzzifierConfigurationError(
        "effective fuzzifier must be finite and greater than 1");
  if (force_generic_alias_present && explicit_policy_present &&
      requested_policy == MPowerPolicy::Auto) {
    throw MPowerPolicyConflictError(
        "explicit auto conflicts with deprecated --force-generic-m");
  }
  FuzzifierConfig config;
  config.requested_m_text = requested_m_text;
  config.requested_m = requested_m;
  config.effective_m = effective_m;
  config.membership_exponent_q = 1.0f / (effective_m - 1.0f);
  if (!ieee_finite_float(config.membership_exponent_q)) {
    throw FuzzifierConfigurationError(
        "membership exponent is not finite after float conversion");
  }
  config.requested_policy = requested_policy;
  config.explicit_policy_present = explicit_policy_present;
  config.force_generic_alias_present = force_generic_alias_present;
  config.effective_policy =
      force_generic_alias_present ? MPowerPolicy::Generic : requested_policy;
  const bool public_auto_m200 =
      config.effective_policy == MPowerPolicy::Auto &&
      is_exact_public_m2(requested_m);
  config.center_effective_path =
      public_auto_m200 ? CenterPowerPath::M200Square
                       : CenterPowerPath::GenericPow;
  config.membership_effective_path =
      public_auto_m200 ? MembershipPowerPath::M200Identity
                       : MembershipPowerPath::GenericPow;
  return config;
}

}  // namespace cmeans
