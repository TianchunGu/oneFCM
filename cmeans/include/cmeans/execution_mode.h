#pragma once

#include <string>

namespace cmeans {

// Fixed20 always owns exactly 20 iterations. NativeDynamic owns early stopping
// after the first comparable center update and otherwise respects max_iter.
enum class IterationProtocol {
  Fixed20,
  NativeDynamic,
};

enum class StopReason {
  Fixed20Complete,
  NativeConverged,
  MaxIterReached,
};

struct IterationOutcome {
  // change_checks is one less than actual_iterations because iteration zero
  // establishes the first center matrix rather than comparing two matrices.
  int actual_iterations = 0;
  int change_checks = 0;
  float final_center_change = -1.0f;
  StopReason stop_reason = StopReason::Fixed20Complete;
};

struct ExecutionModeConfig {
  IterationProtocol protocol = IterationProtocol::Fixed20;
  int max_iter = 20;
  int seed = 550;
  bool protocol_explicit = false;
  bool max_iter_explicit = false;
  bool seed_explicit = false;
};

IterationProtocol parse_iteration_protocol(const std::string& value);
const char* iteration_protocol_name(IterationProtocol protocol);
const char* stop_reason_name(StopReason reason);
int parse_max_iter(const std::string& value);
int parse_seed(const std::string& value);
void validate_execution_mode(const ExecutionModeConfig& config,
                             float threshold);
bool should_stop_after_iteration(IterationProtocol protocol,
                                 int zero_based_iteration,
                                 float center_change,
                                 float threshold);
StopReason completed_stop_reason(IterationProtocol protocol,
                                 bool native_converged);

}  // namespace cmeans
