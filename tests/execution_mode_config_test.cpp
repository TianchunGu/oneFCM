#include <cmeans/execution_mode.h>

#include <stdexcept>
#include <string>

namespace {

void require(bool condition) {
  if (!condition)
    throw std::runtime_error("execution-mode contract assertion failed");
}

template <typename Function>
void expect_failure(Function function) {
  bool failed = false;
  try {
    function();
  } catch (const std::exception&) {
    failed = true;
  }
  require(failed);
}

}  // namespace

int main() {
  cmeans::ExecutionModeConfig legacy;
  require(legacy.protocol == cmeans::IterationProtocol::Fixed20);
  require(legacy.max_iter == 20);
  require(legacy.seed == 550);
  cmeans::validate_execution_mode(legacy, 0.001f);

  require(cmeans::parse_iteration_protocol("fixed20") ==
          cmeans::IterationProtocol::Fixed20);
  require(cmeans::parse_iteration_protocol("native-dynamic") ==
          cmeans::IterationProtocol::NativeDynamic);
  expect_failure([] { cmeans::parse_iteration_protocol("dynamic"); });

  cmeans::ExecutionModeConfig fixed20_explicit;
  fixed20_explicit.max_iter = cmeans::parse_max_iter("20");
  fixed20_explicit.max_iter_explicit = true;
  cmeans::validate_execution_mode(fixed20_explicit, 0.001f);
  fixed20_explicit.max_iter = 100;
  expect_failure([&] {
    cmeans::validate_execution_mode(fixed20_explicit, 0.001f);
  });

  cmeans::ExecutionModeConfig dynamic;
  dynamic.protocol = cmeans::IterationProtocol::NativeDynamic;
  dynamic.protocol_explicit = true;
  expect_failure([&] { cmeans::validate_execution_mode(dynamic, 0.001f); });
  dynamic.max_iter = cmeans::parse_max_iter("100");
  dynamic.max_iter_explicit = true;
  cmeans::validate_execution_mode(dynamic, 0.001f);
  expect_failure([&] { cmeans::validate_execution_mode(dynamic, 0.0f); });

  require(cmeans::parse_seed("700") == 700);
  require(cmeans::parse_seed("0") == 0);
  expect_failure([] { cmeans::parse_seed("-1"); });
  expect_failure([] { cmeans::parse_seed("1x"); });
  expect_failure([] { cmeans::parse_max_iter("0"); });
  return 0;
}
