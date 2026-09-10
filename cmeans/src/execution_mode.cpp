#include <cmeans/execution_mode.h>
#include <cmeans/convergence.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace cmeans {
namespace {

int parse_bounded_integer(const std::string& value,
                          const char* option,
                          int minimum) {
  std::size_t parsed = 0;
  long long result = 0;
  try {
    result = std::stoll(value, &parsed, 10);
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string(option) +
                                " requires an integer");
  }
  if (parsed != value.size()) {
    throw std::invalid_argument(std::string(option) +
                                " requires an integer");
  }
  if (result < minimum || result > std::numeric_limits<int>::max()) {
    throw std::out_of_range(std::string(option) +
                            " is outside the supported range");
  }
  return static_cast<int>(result);
}

bool finite_float(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x7f800000U) != 0x7f800000U;
}

}  // namespace

IterationProtocol parse_iteration_protocol(const std::string& value) {
  if (value == "fixed20")
    return IterationProtocol::Fixed20;
  if (value == "native-dynamic")
    return IterationProtocol::NativeDynamic;
  throw std::invalid_argument(
      "--iteration-protocol must be fixed20 or native-dynamic");
}

const char* iteration_protocol_name(IterationProtocol protocol) {
  switch (protocol) {
    case IterationProtocol::Fixed20:
      return "fixed20";
    case IterationProtocol::NativeDynamic:
      return "native-dynamic";
  }
  throw std::logic_error("unknown iteration protocol");
}

const char* stop_reason_name(StopReason reason) {
  switch (reason) {
    case StopReason::Fixed20Complete:
      return "fixed20_complete";
    case StopReason::NativeConverged:
      return "native_converged";
    case StopReason::MaxIterReached:
      return "max_iter_reached";
  }
  throw std::logic_error("unknown stop reason");
}

int parse_max_iter(const std::string& value) {
  return parse_bounded_integer(value, "--max-iter", 1);
}

int parse_seed(const std::string& value) {
  return parse_bounded_integer(value, "--seed", 0);
}

void validate_execution_mode(const ExecutionModeConfig& config,
                             float threshold) {
  if (config.protocol == IterationProtocol::Fixed20) {
    if (config.max_iter != 20) {
      throw std::invalid_argument(
          "fixed20 requires --max-iter 20 when explicitly supplied");
    }
    return;
  }
  if (!config.max_iter_explicit) {
    throw std::invalid_argument(
        "native-dynamic requires an explicit --max-iter");
  }
  if (config.max_iter <= 0) {
    throw std::invalid_argument("native-dynamic requires --max-iter > 0");
  }
  if (!finite_float(threshold) || threshold <= 0.0f) {
    throw std::invalid_argument("native-dynamic requires -e > 0");
  }
}

bool should_stop_after_iteration(IterationProtocol protocol,
                                 int zero_based_iteration,
                                 float center_change,
                                 float threshold) {
  return protocol == IterationProtocol::NativeDynamic &&
         zero_based_iteration > 0 &&
         center_change_meets_strict_threshold(center_change, threshold);
}

StopReason completed_stop_reason(IterationProtocol protocol,
                                 bool native_converged) {
  if (protocol == IterationProtocol::Fixed20)
    return StopReason::Fixed20Complete;
  return native_converged ? StopReason::NativeConverged
                          : StopReason::MaxIterReached;
}

}  // namespace cmeans
