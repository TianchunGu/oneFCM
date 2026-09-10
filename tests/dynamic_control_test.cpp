#include <cmeans/convergence.h>
#include <cmeans/execution_mode.h>

#include <stdexcept>

namespace {

void require(bool condition) {
  if (!condition)
    throw std::runtime_error("dynamic control contract assertion failed");
}

cmeans::IterationOutcome simulate(cmeans::IterationProtocol protocol,
                                  int max_iter,
                                  float threshold,
                                  float change_value) {
  cmeans::IterationOutcome outcome;
  bool converged = false;
  const int loop_limit =
      protocol == cmeans::IterationProtocol::Fixed20 ? 20 : max_iter;
  for (int iter = 0; iter < loop_limit; ++iter) {
    ++outcome.actual_iterations;
    const float value = iter > 0 ? change_value : -1.0f;
    if (iter > 0) {
      ++outcome.change_checks;
      outcome.final_center_change = value;
    }
    if (cmeans::should_stop_after_iteration(protocol, iter, value,
                                            threshold)) {
      converged = true;
      break;
    }
  }
  outcome.stop_reason = cmeans::completed_stop_reason(protocol, converged);
  return outcome;
}

}  // namespace

int main() {
  constexpr float epsilon = 0.25f;
  require(cmeans::center_change_meets_strict_threshold(0.24f, epsilon));
  require(!cmeans::center_change_meets_strict_threshold(0.25f, epsilon));
  require(!cmeans::center_change_meets_strict_threshold(0.26f, epsilon));

  require(!cmeans::should_stop_after_iteration(
      cmeans::IterationProtocol::NativeDynamic, 0, 0.0f, epsilon));
  require(cmeans::should_stop_after_iteration(
      cmeans::IterationProtocol::NativeDynamic, 1, 0.24f, epsilon));
  require(!cmeans::should_stop_after_iteration(
      cmeans::IterationProtocol::NativeDynamic, 1, epsilon, epsilon));

  const auto fixed = simulate(cmeans::IterationProtocol::Fixed20, 100,
                              1000000.0f, 0.0f);
  require(fixed.actual_iterations == 20);
  require(fixed.change_checks == 19);
  require(fixed.stop_reason == cmeans::StopReason::Fixed20Complete);

  const auto converged = simulate(cmeans::IterationProtocol::NativeDynamic,
                                  100, epsilon, 0.24f);
  require(converged.actual_iterations == 2);
  require(converged.change_checks == 1);
  require(converged.stop_reason == cmeans::StopReason::NativeConverged);

  const auto equal = simulate(cmeans::IterationProtocol::NativeDynamic, 3,
                              epsilon, epsilon);
  require(equal.actual_iterations == 3);
  require(equal.change_checks == 2);
  require(equal.stop_reason == cmeans::StopReason::MaxIterReached);

  const auto capped = simulate(cmeans::IterationProtocol::NativeDynamic, 4,
                               epsilon, 0.26f);
  require(capped.actual_iterations == 4);
  require(capped.change_checks == 3);
  require(capped.stop_reason == cmeans::StopReason::MaxIterReached);
  return 0;
}
