#include <cmeans/work_group_config.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace {

using cmeans::KernelWorkGroupSelection;
using cmeans::WorkGroupCapabilities;
using cmeans::WorkGroupRequest;
using cmeans::WorkGroupRequestMode;
using cmeans::WorkGroupSelectionInput;

bool require(bool condition, const std::string& message) {
  if (!condition)
    std::cerr << message << '\n';
  return condition;
}

KernelWorkGroupSelection select(const WorkGroupRequest& request,
                                const WorkGroupCapabilities& capabilities,
                                const std::string& kernel,
                                const std::string& path,
                                std::size_t raw,
                                std::size_t required = 0,
                                std::size_t reserve = 0,
                                std::size_t residency = 0,
                                bool explicit_strict = false) {
  WorkGroupSelectionInput input;
  input.kernel_id = kernel;
  input.path = path;
  input.raw_requested = raw;
  input.request_source = "test";
  input.local_mem_required_bytes = required;
  input.local_mem_reserve_bytes = reserve;
  input.target_residency = residency;
  input.explicit_strict = explicit_strict;
  return cmeans::select_kernel_work_group(request, capabilities, input);
}

std::size_t frozen_legacy_oracle(const WorkGroupCapabilities& capabilities,
                                 std::size_t raw,
                                 std::size_t required) {
  const std::size_t alignment =
      capabilities.supported_sub_group_sizes.empty()
          ? 1
          : capabilities.supported_sub_group_sizes.back();
  std::size_t candidate = raw == 0 ? (capabilities.is_cpu ? 512 : 256) : raw;
  if (alignment > 1)
    candidate = std::max(alignment, (candidate / alignment) * alignment);
  candidate = std::min(candidate, capabilities.device_max_work_group_size);
  if (required == 0)
    return candidate;
  const std::size_t effective =
      capabilities.local_mem_capacity_bytes > 128
          ? capabilities.local_mem_capacity_bytes - 128
          : capabilities.local_mem_capacity_bytes;
  std::size_t best = alignment;
  bool found = false;
  for (std::size_t current = candidate; current >= alignment;) {
    if (required <= effective) {
      if (!found || current > best) {
        best = current;
        found = true;
      }
      if (required * 2 <= effective)
        return current;
    }
    current /= 2;
    current = (current / alignment) * alignment;
  }
  return found ? best : alignment;
}

}  // namespace

int main() {
  WorkGroupCapabilities cpu{true, 1024, {8, 16}, 65536};
  WorkGroupCapabilities gpu{false, 1024, {8, 16, 32}, 65536};
  WorkGroupRequest legacy{WorkGroupRequestMode::LegacyDefault, 512, 256};
  WorkGroupRequest defaults;
  WorkGroupRequest explicit_request{WorkGroupRequestMode::ExplicitStrict, 0,
                                    0};
  bool ok = true;

  ok &= require(select(defaults, cpu, "membership", "A", 0).effective ==
                    512,
                "CPU zero-hint default did not select 512");
  ok &= require(select(defaults, gpu, "membership", "A", 0).effective ==
                    256,
                "GPU zero-hint default did not select 256");

  ok &= require(select(legacy, cpu, "center_phase1", "center", 512).effective ==
                    512,
                "CPU legacy center/membership source did not preserve 512");
  ok &= require(select(legacy, cpu, "change_check", "reduction", 256).effective ==
                    256,
                "CPU legacy change source did not preserve 256");
  for (std::size_t value : {std::size_t{128}, std::size_t{256},
                            std::size_t{512}}) {
    auto result =
        select(explicit_request, gpu, "membership", "A", value);
    ok &= require(result.status == "selected" && result.effective == value &&
                      !result.adjusted,
                  "explicit request was not exact");
  }
  ok &= require(
      select(explicit_request, gpu, "membership", "A", 0).status ==
          "unsupported",
      "explicit zero was not rejected");
  ok &= require(
      select(explicit_request, WorkGroupCapabilities{false, 256, {32}, 65536},
             "membership", "A", 512)
              .adjustment_reason == "explicit_request_exceeds_device_max",
      "explicit request over device max was not rejected");
  ok &= require(
      select(explicit_request, gpu, "membership", "B", 128, 70000, 1024, 2)
              .adjustment_reason ==
          "local_memory_required_exceeds_device_capacity",
      "local-memory overflow was not rejected");
  const auto near_local_memory_limit =
      select(explicit_request, gpu, "membership", "B", 128, 64512, 1024, 1);
  ok &= require(near_local_memory_limit.status == "selected" &&
                    near_local_memory_limit.target_residency_met,
                "near-capacity local-memory request was not selected");

  const auto production_only =
      select(legacy, gpu, "membership", "B", 256, 4096, 1024, 2);
  const auto production_only_again =
      select(legacy, gpu, "membership", "B", 256, 4096, 1024, 2);
  ok &= require(production_only.effective == production_only_again.effective &&
                    production_only.status == production_only_again.status &&
                    production_only.adjustment_reason ==
                        production_only_again.adjustment_reason,
                "repeated production selection changed the decision");
  const auto low_residency =
      select(explicit_request, gpu, "membership", "B", 128, 40000, 1024, 2);
  ok &= require(low_residency.status == "selected" &&
                    !low_residency.target_residency_met,
                "one-group fit with low residency was not allowed/marked");
  ok &= require(select(legacy, cpu, "membership", "A", 250).effective == 240,
                "legacy subgroup alignment differs");
  const auto independent =
      select(legacy, gpu, "membership", "A", 16, 0, 0, 0, true);
  ok &= require(independent.status == "selected" &&
                    independent.effective == 16 &&
                    independent.request_mode == "explicit-strict",
                "per-kernel explicit request was not exact");
  ok &= require(
      select(legacy, WorkGroupCapabilities{true, 256, {8, 16}, 65536},
             "membership", "A", 512)
              .effective == 256,
      "legacy device-max adjustment differs");

  std::vector<WorkGroupSelectionInput> fixtures(3);
  fixtures[0].kernel_id = "membership";
  fixtures[0].path = "A";
  fixtures[0].raw_requested = 512;
  fixtures[0].request_source = "test";
  fixtures[1] = fixtures[0];
  fixtures[1].path = "B";
  fixtures[1].local_mem_required_bytes = 4096;
  fixtures[1].local_mem_reserve_bytes = 1024;
  fixtures[1].target_residency = 2;
  fixtures[2] = fixtures[0];
  fixtures[2].path = "C";
  fixtures[2].local_mem_required_bytes = 8192;
  fixtures[2].local_mem_reserve_bytes = 2048;
  fixtures[2].target_residency = 2;
  fixtures[2].tile_d = 64;
  for (const auto& fixture : fixtures) {
    const auto result =
        cmeans::select_kernel_work_group(legacy, cpu, fixture);
    ok &= require(result.path == fixture.path &&
                      result.local_mem_required_bytes ==
                          fixture.local_mem_required_bytes,
                  "Path A/B/C local-byte selection metadata differs");
  }

  WorkGroupSelectionInput full_tile = fixtures[1];
  full_tile.tile_k = 32;
  full_tile.tile_d = 17;
  full_tile.tiles_k = 2;
  full_tile.tiles_d = 1;
  full_tile.full_tile = true;
  const auto full_tile_result =
      cmeans::select_kernel_work_group(legacy, gpu, full_tile);
  ok &= require(full_tile_result.tile_k == 32 &&
                    full_tile_result.tile_d == 17 &&
                    full_tile_result.tiles_k == 2 &&
                    full_tile_result.tiles_d == 1 &&
                    full_tile_result.full_tile,
                "full-tile selection metadata differs");

  WorkGroupSelectionInput tiled_d = full_tile;
  tiled_d.tile_d = 16;
  tiled_d.tiles_d = 2;
  tiled_d.full_tile = false;
  const auto tiled_d_result =
      cmeans::select_kernel_work_group(legacy, gpu, tiled_d);
  ok &= require(tiled_d_result.tile_k == 32 &&
                    tiled_d_result.tile_d == 16 &&
                    tiled_d_result.tiles_k == 2 &&
                    tiled_d_result.tiles_d == 2 &&
                    !tiled_d_result.full_tile,
                "tiled-D selection metadata differs");

  for (std::size_t raw : {std::size_t{7}, std::size_t{250},
                          std::size_t{512}, std::size_t{2048}}) {
    for (std::size_t required :
         {std::size_t{0}, std::size_t{1024}, std::size_t{40000},
          std::size_t{70000}}) {
      const auto result =
          select(legacy, cpu, "center_phase1", "center", raw, required);
      ok &= require(result.effective ==
                        frozen_legacy_oracle(cpu, raw, required),
                    "legacy selector differs from frozen oracle");
    }
  }

  return ok ? 0 : 1;
}
