#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

namespace cmeans {

// LegacyDefault may adjust historical defaults to device limits. ExplicitStrict
// either preserves the requested size exactly or reports Unsupported.
enum class WorkGroupRequestMode { LegacyDefault, ExplicitStrict };

enum class WorkGroupSelectionStatus {
  Selected,
  Unsupported
};

struct WorkGroupRequest {
  WorkGroupRequestMode mode = WorkGroupRequestMode::LegacyDefault;
  std::size_t center_membership_raw_requested = 0;
  std::size_t change_raw_requested = 0;
  std::size_t center_raw_requested = 0;
  std::size_t membership_raw_requested = 0;
  bool center_explicit = false;
  bool membership_explicit = false;
  bool change_explicit = false;
};

struct WorkGroupCapabilities {
  bool is_cpu = false;
  std::size_t device_max_work_group_size = 0;
  std::vector<std::size_t> supported_sub_group_sizes;
  std::size_t local_mem_capacity_bytes = 0;
};

struct WorkGroupSelectionInput {
  std::string kernel_id;
  std::string path;
  std::size_t raw_requested = 0;
  std::string request_source;
  std::size_t local_mem_required_bytes = 0;
  std::size_t local_mem_reserve_bytes = 0;
  std::size_t target_residency = 0;
  std::size_t tile_k = 0;
  std::size_t tile_d = 0;
  std::size_t tiles_k = 0;
  std::size_t tiles_d = 0;
  bool full_tile = false;
  bool explicit_strict = false;
};

struct KernelWorkGroupSelection {
  std::string kernel_id;
  std::string request_mode;
  std::string request_source;
  std::string path;
  std::size_t raw_requested = 0;
  std::size_t effective = 0;
  std::size_t device_max_work_group_size = 0;
  std::vector<std::size_t> supported_sub_group_sizes;
  std::size_t alignment_sub_group_size = 1;
  std::size_t local_mem_capacity_bytes = 0;
  std::size_t local_mem_required_bytes = 0;
  std::size_t local_mem_reserve_bytes = 0;
  std::size_t legacy_selector_reserve_bytes = 0;
  std::size_t target_residency = 0;
  bool target_residency_met = true;
  std::size_t tile_k = 0;
  std::size_t tile_d = 0;
  std::size_t tiles_k = 0;
  std::size_t tiles_d = 0;
  bool full_tile = false;
  bool adjusted = false;
  std::string adjustment_reason;
  std::string status;
};

class WorkGroupExecutionError : public std::runtime_error {
 public:
  WorkGroupExecutionError(std::string kernel_id,
                          std::string stage,
                          const std::string& message);

  const std::string& kernel_id() const noexcept { return kernel_id_; }
  const std::string& stage() const noexcept { return stage_; }

 private:
  std::string kernel_id_;
  std::string stage_;
};

const char* work_group_request_mode_name(WorkGroupRequestMode mode);
const char* work_group_selection_status_name(WorkGroupSelectionStatus status);

WorkGroupCapabilities query_work_group_capabilities(const sycl::queue& queue);

// Select a per-kernel size against device, subgroup, local-memory, and tile
// constraints; the caller owns any Unsupported failure policy.
KernelWorkGroupSelection select_kernel_work_group(
    const WorkGroupRequest& request,
    const WorkGroupCapabilities& capabilities,
    const WorkGroupSelectionInput& input);

}  // namespace cmeans
