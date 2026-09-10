#include <cmeans/work_group_config.h>

#include <algorithm>
#include <limits>
#include <utility>

namespace cmeans {
namespace {

bool safe_residency_fit(std::size_t required,
                        std::size_t reserve,
                        std::size_t residency,
                        std::size_t capacity) {
  if (required == 0 || residency == 0)
    return true;
  if (required > (std::numeric_limits<std::size_t>::max() - reserve) /
                     residency)
    return false;
  return required * residency + reserve <= capacity;
}

std::size_t alignment_for(const WorkGroupCapabilities& capabilities) {
  if (capabilities.supported_sub_group_sizes.empty())
    return 1;
  return capabilities.supported_sub_group_sizes.back();
}

std::size_t legacy_select(const WorkGroupCapabilities& capabilities,
                          std::size_t raw_requested,
                          std::size_t local_mem_required_bytes) {
  const std::size_t alignment = alignment_for(capabilities);
  std::size_t candidate = raw_requested;
  if (candidate == 0)
    candidate = capabilities.is_cpu ? 512 : 256;
  if (alignment > 1) {
    candidate =
        std::max(alignment, (candidate / alignment) * alignment);
  }
  candidate =
      std::min(candidate, capabilities.device_max_work_group_size);

  if (local_mem_required_bytes == 0)
    return candidate;

  constexpr std::size_t selector_reserve = 128;
  const std::size_t effective_local_mem =
      capabilities.local_mem_capacity_bytes > selector_reserve
          ? capabilities.local_mem_capacity_bytes - selector_reserve
          : capabilities.local_mem_capacity_bytes;
  constexpr std::size_t target_residency = 2;
  std::size_t best_safe_size = alignment;
  bool found_safe = false;
  std::size_t current = candidate;
  while (current >= alignment) {
    if (local_mem_required_bytes <= effective_local_mem) {
      if (!found_safe || current > best_safe_size) {
        best_safe_size = current;
        found_safe = true;
      }
      if (local_mem_required_bytes <=
          effective_local_mem / target_residency) {
        return current;
      }
    }
    current /= 2;
    current = (current / alignment) * alignment;
  }
  return found_safe ? best_safe_size : alignment;
}

}  // namespace

WorkGroupExecutionError::WorkGroupExecutionError(std::string kernel_id,
                                                 std::string stage,
                                                 const std::string& message)
    : std::runtime_error(message),
      kernel_id_(std::move(kernel_id)),
      stage_(std::move(stage)) {}

const char* work_group_request_mode_name(WorkGroupRequestMode mode) {
  switch (mode) {
    case WorkGroupRequestMode::LegacyDefault:
      return "legacy-default";
    case WorkGroupRequestMode::ExplicitStrict:
      return "explicit-strict";
  }
  return "unknown";
}

const char* work_group_selection_status_name(
    WorkGroupSelectionStatus status) {
  switch (status) {
    case WorkGroupSelectionStatus::Selected:
      return "selected";
    case WorkGroupSelectionStatus::Unsupported:
      return "unsupported";
  }
  return "unknown";
}

WorkGroupCapabilities query_work_group_capabilities(
    const sycl::queue& queue) {
  const sycl::device device = queue.get_device();
  WorkGroupCapabilities capabilities;
  capabilities.is_cpu = device.is_cpu();
  capabilities.device_max_work_group_size =
      device.get_info<sycl::info::device::max_work_group_size>();
  capabilities.supported_sub_group_sizes =
      device.get_info<sycl::info::device::sub_group_sizes>();
  capabilities.local_mem_capacity_bytes =
      device.get_info<sycl::info::device::local_mem_size>();
  return capabilities;
}

KernelWorkGroupSelection select_kernel_work_group(
    const WorkGroupRequest& request,
    const WorkGroupCapabilities& capabilities,
    const WorkGroupSelectionInput& input) {
  KernelWorkGroupSelection result;
  result.kernel_id = input.kernel_id;
  const bool explicit_strict =
      input.explicit_strict ||
      request.mode == WorkGroupRequestMode::ExplicitStrict;
  result.request_mode = work_group_request_mode_name(
      explicit_strict ? WorkGroupRequestMode::ExplicitStrict
                      : WorkGroupRequestMode::LegacyDefault);
  result.request_source = input.request_source;
  result.path = input.path;
  result.raw_requested = input.raw_requested;
  result.device_max_work_group_size =
      capabilities.device_max_work_group_size;
  result.supported_sub_group_sizes =
      capabilities.supported_sub_group_sizes;
  result.alignment_sub_group_size = alignment_for(capabilities);
  result.local_mem_capacity_bytes =
      capabilities.local_mem_capacity_bytes;
  result.local_mem_required_bytes = input.local_mem_required_bytes;
  result.local_mem_reserve_bytes = input.local_mem_reserve_bytes;
  result.legacy_selector_reserve_bytes =
      input.local_mem_required_bytes > 0 ? 128 : 0;
  result.target_residency = input.target_residency;
  result.tile_k = input.tile_k;
  result.tile_d = input.tile_d;
  result.tiles_k = input.tiles_k;
  result.tiles_d = input.tiles_d;
  result.full_tile = input.full_tile;
  result.target_residency_met = safe_residency_fit(
      input.local_mem_required_bytes, input.local_mem_reserve_bytes,
      input.target_residency, capabilities.local_mem_capacity_bytes);

  if (explicit_strict) {
    if (input.raw_requested == 0) {
      result.status =
          work_group_selection_status_name(WorkGroupSelectionStatus::Unsupported);
      result.adjustment_reason = "explicit_request_must_be_positive";
      return result;
    }
    if (input.raw_requested > capabilities.device_max_work_group_size) {
      result.status =
          work_group_selection_status_name(WorkGroupSelectionStatus::Unsupported);
      result.adjustment_reason = "explicit_request_exceeds_device_max";
      return result;
    }
    if (input.local_mem_required_bytes >
        capabilities.local_mem_capacity_bytes) {
      result.status =
          work_group_selection_status_name(WorkGroupSelectionStatus::Unsupported);
      result.adjustment_reason =
          "local_memory_required_exceeds_device_capacity";
      return result;
    }
    result.effective = input.raw_requested;
    result.adjusted = false;
    result.adjustment_reason = "explicit_exact";
    result.status =
        work_group_selection_status_name(WorkGroupSelectionStatus::Selected);
    return result;
  }

  result.effective = legacy_select(capabilities, input.raw_requested,
                                   input.local_mem_required_bytes);
  result.adjusted = result.effective != input.raw_requested;
  if (!result.adjusted) {
    result.adjustment_reason = "legacy_request_preserved";
  } else if (input.raw_requested >
             capabilities.device_max_work_group_size) {
    result.adjustment_reason = "legacy_device_max_adjustment";
  } else if (input.raw_requested < result.alignment_sub_group_size) {
    result.adjustment_reason = "legacy_sub_group_minimum_adjustment";
  } else {
    result.adjustment_reason =
        "legacy_sub_group_or_resource_adjustment";
  }
  result.status =
      work_group_selection_status_name(WorkGroupSelectionStatus::Selected);
  return result;
}

}  // namespace cmeans
