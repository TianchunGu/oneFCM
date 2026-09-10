#pragma once

#include <stdexcept>
#include <string>

#include <sycl/sycl.hpp>

namespace cmeans {

// Auto selects exact multiplication/identity evaluators only for m == 2;
// Generic uses sycl::pow for both center and membership exponents.
enum class MPowerPolicy { Auto, Generic };

enum class CenterPowerPath {
  M200Square,
  GenericPow
};

enum class MembershipPowerPath {
  M200Identity,
  GenericPow
};

class FuzzifierConfigurationError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

class MPowerPolicyConflictError : public FuzzifierConfigurationError {
 public:
  using FuzzifierConfigurationError::FuzzifierConfigurationError;
};

struct FuzzifierConfig {
  std::string requested_m_text = "2.0";
  double requested_m = 2.0;
  float effective_m = 2.0f;
  float membership_exponent_q = 1.0f;
  MPowerPolicy requested_policy = MPowerPolicy::Auto;
  MPowerPolicy effective_policy = MPowerPolicy::Auto;
  bool explicit_policy_present = false;
  bool force_generic_alias_present = false;
  CenterPowerPath center_effective_path = CenterPowerPath::M200Square;
  MembershipPowerPath membership_effective_path =
      MembershipPowerPath::M200Identity;
};

double parse_fuzzifier(const std::string& value);
MPowerPolicy parse_m_power_policy(const std::string& value);
const char* m_power_policy_name(MPowerPolicy policy);
const char* center_power_path_name(CenterPowerPath path);
const char* membership_power_path_name(MembershipPowerPath path);

// Resolve public fuzzifier arguments once on the host before kernel dispatch.
FuzzifierConfig resolve_fuzzifier_config(
    const std::string& requested_m_text,
    double requested_m,
    MPowerPolicy requested_policy,
    bool explicit_policy_present,
    bool force_generic_alias_present);

struct CenterM200SquareEvaluator {
  float operator()(float membership) const { return membership * membership; }
};

struct CenterGenericPowEvaluator {
  float exponent = 2.0f;
  float operator()(float membership) const {
    return sycl::pow(membership, exponent);
  }
};

struct MembershipM200IdentityEvaluator {
  float operator()(float ratio) const { return ratio; }
};

struct MembershipGenericPowEvaluator {
  float exponent = 1.0f;
  float operator()(float ratio) const { return sycl::pow(ratio, exponent); }
};

}  // namespace cmeans
