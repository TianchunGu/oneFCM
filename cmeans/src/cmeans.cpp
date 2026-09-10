#include <cmeans/cmeans.h>
#include <cmeans/convergence.h>
#include <cmeans/execution_mode.h>
#include <util/utils.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <oneapi/dpl/random>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#if FCM_ENABLE_ITT
#include <ittnotify.h>
static __itt_domain* g_fcm_itt_domain = __itt_domain_create("fcm");
static __itt_string_handle* g_fcm_itt_read =
    __itt_string_handle_create("fcm_read_data");
static __itt_string_handle* g_fcm_itt_compute =
    __itt_string_handle_create("fcm_compute");
static __itt_string_handle* g_fcm_itt_write =
    __itt_string_handle_create("fcm_write_output");
#define FCM_ITT_TASK_BEGIN(handle) \
  __itt_task_begin(g_fcm_itt_domain, __itt_null, __itt_null, handle)
#define FCM_ITT_TASK_END() __itt_task_end(g_fcm_itt_domain)
#else
#define FCM_ITT_TASK_BEGIN(handle)
#define FCM_ITT_TASK_END()
#endif

namespace fs = std::filesystem;

namespace {
[[maybe_unused]] bool ieee_finite_float(const float& value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x7f800000U) != 0x7f800000U;
}

}

namespace cmeans {

// Return device profiling nanoseconds when profiling is enabled; otherwise
// preserve the public timing interface with a zero value.
inline float get_kernel_execution_time(const sycl::event& event) {
#if FCM_ENABLE_PROFILING
  auto start_time =
      event.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end_time =
      event.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<float>(end_time - start_time);
#else
  (void)event;
  return 0.0f;
#endif
}

inline std::size_t round_up(std::size_t value, std::size_t factor) {
  if (factor == 0)
    return value;
  return ((value + factor - 1) / factor) * factor;
}

KernelWorkGroupSelection select_work_group_or_throw(
    const sycl::queue& queue,
    const WorkGroupRequest& request,
    WorkGroupSelectionInput input) {
  const WorkGroupCapabilities capabilities =
      query_work_group_capabilities(queue);
  const KernelWorkGroupSelection selection =
      select_kernel_work_group(request, capabilities, input);
  if (selection.status !=
      work_group_selection_status_name(WorkGroupSelectionStatus::Selected)) {
    throw WorkGroupExecutionError(
        selection.kernel_id, "host_selection",
        selection.adjustment_reason.empty() ? "unsupported work-group request"
                                            : selection.adjustment_reason);
  }
  return selection;
}

[[noreturn]] void rethrow_work_group_failure(
    const std::string& kernel_id,
    const std::string& stage,
    const std::string& category,
    const std::string& message) {
  throw WorkGroupExecutionError(kernel_id, stage, message);
}

sycl::event parallel_initialize_membership_matrix(sycl::queue& q,
                                                  float* membership_matrix_dev,
                                                  int old_num_clusters,
                                                  int old_num_points,
                                                  uint32_t seed,
                                                  float low,
                                                  float high,
                                                  float& init_time) {
  // One work-item owns one point. It draws K deterministic values from the
  // point-indexed engine and normalizes the cluster-major [K, N] column.
  sycl::event event = q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<1>(old_num_points), [=](sycl::id<1> idx) {
      const std::size_t i = idx[0];
      const std::size_t K = static_cast<std::size_t>(old_num_clusters);
      const std::size_t N = static_cast<std::size_t>(old_num_points);

      oneapi::dpl::minstd_rand engine(seed, i);
      oneapi::dpl::uniform_real_distribution<float> distr(low, high);

      float sum = 0.0f;

      for (std::size_t j = 0; j < K; ++j) {
        float val = distr(engine);
        sum += val;

        membership_matrix_dev[j * N + i] = val;
      }

      float scale = (sum > 1e-9f) ? (1.0f / sum) : 0.0f;

      for (std::size_t j = 0; j < K; ++j) {
        std::size_t index = j * N + i;

        membership_matrix_dev[index] *= scale;
      }
    });
  });

  // Initialization completes before the iteration wall-clock timer starts.
  event.wait_and_throw();

#if FCM_ENABLE_PROFILING
  init_time += get_kernel_execution_time(event);
#endif

  return event;
}

template <typename CenterPowerEvaluator>
void update_cluster_centers(sycl::queue& q,
                            int num_dims,
                            CenterPowerEvaluator center_power,
                            float* membership_current_dev,
                            const float* data_points_dev,
                            float* centers_next_dev,
                            const WorkGroupRequest& work_group_request,
                            int old_num_clusters,
                            int old_num_points,
                            float* numerators_dev,
                            float* denominators_dev,
                            float& update_cluster_centers_time) {
  using namespace sycl;

  // Input membership and data layouts are [K, N] and [D, N]. The two output
  // accumulators hold numerator[c, d] = sum(u[c, p]^m * x[d, p]) and
  // denominator[c] = sum(u[c, p]^m) before center normalization.

  const std::size_t K = static_cast<std::size_t>(old_num_clusters);
  const std::size_t D = static_cast<std::size_t>(num_dims);
  const std::size_t N = static_cast<std::size_t>(old_num_points);

  const std::size_t numerators_size = K * D;

  const auto dev = q.get_device();
  const std::size_t local_mem_bytes =
      dev.get_info<info::device::local_mem_size>();

  const std::size_t reserve_bytes = 1024;

  const std::size_t target_residency = 2;

  // Tile selection maximizes covered [K, D] area while reserving 1024 bytes
  // and budgeting local memory for two resident work-groups.

  std::size_t lm_budget_floats = 0;
  if (local_mem_bytes > reserve_bytes) {
    lm_budget_floats = (local_mem_bytes - reserve_bytes) / sizeof(float);
    if (lm_budget_floats >= target_residency)
      lm_budget_floats /=
          target_residency;
    else
      lm_budget_floats = 0;
  }

  std::size_t tile_D = std::min<std::size_t>(D, 32);
  std::size_t tile_K = std::min<std::size_t>(K, 1);

  if (lm_budget_floats == 0) {

    tile_D = std::min<std::size_t>(D, 16);
    tile_K = 1;
  } else {

    const std::size_t candidates[] = {64, 32, 16, 8, 4, 2, 1};
    std::size_t best_score = 0;

    for (std::size_t cand_D : candidates) {
      if (cand_D > D)
        continue;

      const std::size_t per_cluster_cost = cand_D + 1;
      if (per_cluster_cost == 0)
        continue;

      const std::size_t cap_K = lm_budget_floats / per_cluster_cost;
      if (cap_K == 0)
        continue;

      const std::size_t cand_K = std::min<std::size_t>(K, cap_K);

      const std::size_t score = cand_D * cand_K;
      if (score > best_score) {
        best_score = score;
        tile_D = cand_D;
        tile_K = cand_K;
      }
    }

    tile_D = std::max<std::size_t>(1, tile_D);
    tile_K = std::max<std::size_t>(1, tile_K);
  }

  const std::size_t local_bytes_phase1 = tile_K * (tile_D + 1) * sizeof(float);

  WorkGroupSelectionInput selection_input;
  selection_input.kernel_id = "center_phase1";
  selection_input.path = "phase1_tiled_reduction";
  selection_input.raw_requested =
      work_group_request.center_explicit
          ? work_group_request.center_raw_requested
          : work_group_request.center_membership_raw_requested;
  selection_input.explicit_strict = work_group_request.center_explicit;
  selection_input.request_source =
      work_group_request.center_explicit
          ? "cli_center_wg_size"
          : (work_group_request.mode == WorkGroupRequestMode::ExplicitStrict
          ? "cli_work_group_size"
          : (q.get_device().is_cpu()
                 ? "legacy_cpu_center_membership_default"
                 : "legacy_gpu_center_membership_default"));
  selection_input.local_mem_required_bytes = local_bytes_phase1;
  selection_input.local_mem_reserve_bytes = reserve_bytes;
  selection_input.target_residency = target_residency;
  selection_input.tile_k = tile_K;
  selection_input.tile_d = tile_D;
  const KernelWorkGroupSelection selection =
      select_work_group_or_throw(q, work_group_request, selection_input);
  const std::size_t lsize_phase1 = selection.effective;

  std::string failure_stage = "center_phase0_memset";
  try {
  // Phase 0 clears global accumulators. Phase 1 reduces point contributions
  // through work-group-local atomics before one global atomic per tile entry.
  sycl::event init_num =
      q.memset(numerators_dev, 0, numerators_size * sizeof(float));

  sycl::event init_den = q.memset(denominators_dev, 0, K * sizeof(float));

  const std::size_t padded_num_points = round_up(N, lsize_phase1);

  failure_stage = "center_phase1_submit";
  sycl::event eventPhase1 = q.submit([&](handler& h) {

    h.depends_on({init_num, init_den});

    local_accessor<float, 1> local_denominators(range<1>(tile_K), h);

    local_accessor<float, 1> local_numerators(range<1>(tile_K * tile_D), h);

    range<1> global_range(padded_num_points);
    range<1> local_range(lsize_phase1);
    nd_range<1> execution_range(global_range, local_range);

    h.parallel_for(execution_range, [=](nd_item<1> item) {

      const std::size_t lid = item.get_local_id(0);
      const std::size_t lsize = item.get_local_range(0);
      const std::size_t j =
          item.get_global_id(0);

      auto sg = item.get_sub_group();

      for (std::size_t k_base = 0; k_base < K; k_base += tile_K) {

        const std::size_t cur_tile_K = sycl::min(tile_K, K - k_base);

        for (std::size_t i = lid; i < cur_tile_K; i += lsize)
          local_denominators[i] = 0.0f;

        item.barrier(access::fence_space::local_space);

        for (std::size_t d_base = 0; d_base < D; d_base += tile_D) {
          const std::size_t cur_tile_D = sycl::min(tile_D, D - d_base);

          for (std::size_t i = lid; i < cur_tile_K * cur_tile_D; i += lsize)
            local_numerators[i] = 0.0f;

          item.barrier(access::fence_space::local_space);

          for (std::size_t coff = 0; coff < cur_tile_K; ++coff) {
            const std::size_t c = k_base + coff;

            float u_pow_m = 0.0f;
            if (j < N) {
              const float u = membership_current_dev[c * N + j];

              u_pow_m = center_power(u);
            }

            if (d_base == 0) {

              float sg_den =
                  sycl::reduce_over_group(sg, u_pow_m, sycl::plus<float>());

              if (sg.get_local_id() == 0) {
                using WGLocalAtomic =
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                     sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>;
                WGLocalAtomic atm(local_denominators[coff]);
                atm.fetch_add(sg_den);
              }
            }

            for (std::size_t toff = 0; toff < cur_tile_D; ++toff) {
              const std::size_t d = d_base + toff;

              float val = 0.0f;
              if (j < N) {

                const float x = data_points_dev[d * N + j];
                val = u_pow_m * x;
              }

              float sg_num =
                  sycl::reduce_over_group(sg, val, sycl::plus<float>());

              if (sg.get_local_id() == 0) {
                using WGLocalAtomic =
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                     sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>;
                const std::size_t lidx = coff * cur_tile_D + toff;
                WGLocalAtomic atm(local_numerators[lidx]);
                atm.fetch_add(sg_num);
              }
            }
          }

          item.barrier(access::fence_space::local_space);

          for (std::size_t i = lid; i < cur_tile_K * cur_tile_D; i += lsize) {
            const float val = local_numerators[i];

            const std::size_t coff = i / cur_tile_D;
            const std::size_t toff = i % cur_tile_D;

            const std::size_t c = k_base + coff;
            const std::size_t d = d_base + toff;

            const std::size_t gidx = c * D + d;

            using DevAtomic =
                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>;
            DevAtomic atm(numerators_dev[gidx]);
            atm.fetch_add(val);
          }

          item.barrier(access::fence_space::local_space);
        }

        for (std::size_t coff = lid; coff < cur_tile_K; coff += lsize) {
          const float val = local_denominators[coff];
          const std::size_t c = k_base + coff;

          using DevAtomic =
              sycl::atomic_ref<float, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::global_space>;
          DevAtomic atm(denominators_dev[c]);
          atm.fetch_add(val);
        }

        item.barrier(access::fence_space::local_space);
      }
    });
  });

  failure_stage = "center_phase2_submit";
  // Phase 2 normalizes each [K, D] numerator by its cluster denominator.
  sycl::event eventFinal = q.submit([&](handler& h) {
    h.depends_on(eventPhase1);

    h.parallel_for(range<1>(numerators_size), [=](id<1> idx) {
      const std::size_t i = idx[0];
      const std::size_t c = i / D;

      const float den = denominators_dev[c];

      centers_next_dev[i] = (den > 1e-9f) ? (numerators_dev[i] / den) : 0.0f;
    });
  });

#if FCM_ENABLE_PROFILING

  failure_stage = "center_phase1_wait";
  eventPhase1.wait_and_throw();
  failure_stage = "center_phase2_wait";
  eventFinal.wait_and_throw();
  update_cluster_centers_time += get_kernel_execution_time(eventPhase1) +
                                 get_kernel_execution_time(eventFinal);
#else
  failure_stage = "center_phase2_wait";
  eventFinal.wait_and_throw();
#endif
  } catch (const sycl::exception& error) {
    rethrow_work_group_failure(
        "center_phase1", failure_stage, error.code().category().name(),
        error.what());
  }
}

template <typename MembershipPowerEvaluator>
void update_membership_values(sycl::queue& q,
                              int num_clusters,
                              int num_points,
                              MembershipPowerEvaluator membership_power,
                              int num_dims,
                              float* membership_next_dev,
                              const float* data_points_dev,
                              const float* centers_next_dev,
                              const WorkGroupRequest& work_group_request,
                              int old_num_clusters,
                              int old_num_points,
                              float& time) {
  using namespace sycl;

  // Read dimension-major data [D, N] and cluster-major centers [K, D], then
  // write cluster-major memberships [K, N]. Squared-distance ratios use the
  // exponent q = 1 / (m - 1).

  const size_t K = static_cast<size_t>(num_clusters);
  const size_t N = static_cast<size_t>(old_num_points);
  const size_t D = static_cast<size_t>(num_dims);

  const float eps = 1e-9f;

  auto select_membership = [&](const std::string& path,
                               std::size_t local_required,
                               std::size_t reserve,
                               std::size_t tile_k,
                               std::size_t tile_d,
                               std::size_t tiles_k,
                               std::size_t tiles_d,
                               bool full_tile) {
    WorkGroupSelectionInput input;
    input.kernel_id = "membership";
    input.path = path;
    input.raw_requested =
        work_group_request.membership_explicit
            ? work_group_request.membership_raw_requested
            : work_group_request.center_membership_raw_requested;
    input.explicit_strict = work_group_request.membership_explicit;
    input.request_source =
        work_group_request.membership_explicit
            ? "cli_membership_wg_size"
            : (work_group_request.mode == WorkGroupRequestMode::ExplicitStrict
            ? "cli_work_group_size"
            : (q.get_device().is_cpu()
                   ? "legacy_cpu_center_membership_default"
                   : "legacy_gpu_center_membership_default"));
    input.local_mem_required_bytes = local_required;
    input.local_mem_reserve_bytes = reserve;
    input.target_residency = local_required == 0 ? 0 : 2;
    input.tile_k = tile_k;
    input.tile_d = tile_d;
    input.tiles_k = tiles_k;
    input.tiles_d = tiles_d;
    input.full_tile = full_tile;
    return select_work_group_or_throw(q, work_group_request, input);
  };

  if (K <= 32) {
    // Scheme A assigns one point to each work-item and keeps all K squared
    // distances in a bounded private array; it requires no local memory.
    const KernelWorkGroupSelection selection =
        select_membership("A", 0, 0, 0, 0, 0, 0, false);
    const size_t lsize = selection.effective;

    const size_t padded_num_points = round_up(N, lsize);

    std::string failure_stage = "membership_submit";
    try {
    event e = q.submit([&](handler& h) {
      h.parallel_for(
          nd_range<1>(range<1>(padded_num_points), range<1>(lsize)),
          [=](nd_item<1> item) {

            const size_t j = item.get_global_id(0);

            if (j >= N)
              return;

            float dist_sq[32];

            int match = -1;

            for (int i = 0; i < (int)K; ++i) {
              float d2 = 0.0f;

              for (int k = 0; k < (int)D; ++k) {

                const float c_val = centers_next_dev[i * (int)D + k];
                const float diff = data_points_dev[k * (int)N + j] - c_val;
                d2 += diff * diff;
              }

              if (d2 < eps) {
                match = i;
                break;
              }

              dist_sq[i] = d2;
            }

            if (match != -1) {
              for (int i = 0; i < (int)K; ++i) {
                membership_next_dev[i * (int)N + j] =
                    (i == match) ? 1.0f : 0.0f;
              }
              return;
            }

            for (int i = 0; i < (int)K; ++i) {

              const float di = sycl::fmax(dist_sq[i], eps);
              float sum = 0.0f;

              for (int k = 0; k < (int)K; ++k) {
                const float ratio =
                    di / sycl::fmax(dist_sq[k], eps);
                sum += membership_power(ratio);
              }

              membership_next_dev[i * (int)N + j] = 1.0f / sum;
            }
          });
    });

#if FCM_ENABLE_PROFILING
    failure_stage = "membership_wait";
    e.wait_and_throw();
    time += get_kernel_execution_time(e);
#else
    failure_stage = "membership_wait";
    e.wait_and_throw();
#endif
    } catch (const sycl::exception& error) {
      rethrow_work_group_failure(
          "membership", failure_stage, error.code().category().name(),
          error.what());
    }
    return;
  }

  constexpr size_t kSchemeBPrivateDistanceBound = 32;
  constexpr size_t kSchemeBLocalMemReserveBytes = 1024;
  constexpr size_t kSchemeBTargetResidency = 2;
  constexpr size_t kSchemeBTileKCandidates[] = {32, 16, 8, 4, 2, 1};

  // Scheme B handles K > 32 with resource-aware [K, D] center tiles. Its two
  // passes compute a stable normalization sum, then recompute tiles to write
  // [K, N] memberships without persistent distance scratch storage.

  const size_t local_mem_capacity =
      q.get_device().get_info<info::device::local_mem_size>();
  const size_t center_tile_budget =
      local_mem_capacity > kSchemeBLocalMemReserveBytes
          ? (local_mem_capacity - kSchemeBLocalMemReserveBytes) /
                kSchemeBTargetResidency
          : 0;

  auto ceil_div = [](size_t value, size_t divisor) {
    return value / divisor + (value % divisor != 0 ? 1 : 0);
  };

  size_t selected_tile_k = 0;
  size_t selected_tile_d = 0;
  size_t best_tile_rounds = std::numeric_limits<size_t>::max();
  size_t best_tile_area = 0;

  for (const size_t candidate_tile_k : kSchemeBTileKCandidates) {
    const size_t bytes_per_feature = candidate_tile_k * sizeof(float);
    const size_t max_tile_d =
        bytes_per_feature == 0 ? 0 : center_tile_budget / bytes_per_feature;
    const size_t candidate_tile_d = std::min(D, max_tile_d);
    if (candidate_tile_d == 0)
      continue;

    const size_t candidate_tiles_k = ceil_div(K, candidate_tile_k);
    const size_t candidate_tiles_d = ceil_div(D, candidate_tile_d);
    const size_t candidate_rounds = candidate_tiles_k * candidate_tiles_d;
    const size_t candidate_area = candidate_tile_k * candidate_tile_d;

    const bool better =
        candidate_rounds < best_tile_rounds ||
        (candidate_rounds == best_tile_rounds &&
         (candidate_area > best_tile_area ||
          (candidate_area == best_tile_area &&
           (candidate_tile_d > selected_tile_d ||
            (candidate_tile_d == selected_tile_d &&
             candidate_tile_k > selected_tile_k)))));
    if (better) {
      selected_tile_k = candidate_tile_k;
      selected_tile_d = candidate_tile_d;
      best_tile_rounds = candidate_rounds;
      best_tile_area = candidate_area;
    }
  }

  if (selected_tile_k == 0 || selected_tile_d == 0) {
    throw WorkGroupExecutionError(
        "membership", "host_selection",
        "no legal Scheme-B (T_k,T_d) tile satisfies the local-memory contract");
  }

  const size_t selected_tiles_k = ceil_div(K, selected_tile_k);
  const size_t selected_tiles_d = ceil_div(D, selected_tile_d);
  const size_t center_tile_bytes =
      selected_tile_k * selected_tile_d * sizeof(float);
  const bool full_tile = selected_tile_k == K && selected_tile_d == D;
  const KernelWorkGroupSelection selection = select_membership(
      "B", center_tile_bytes, kSchemeBLocalMemReserveBytes, selected_tile_k,
      selected_tile_d, selected_tiles_k, selected_tiles_d, full_tile);
  const size_t lsize = selection.effective;
  const size_t padded_num_points = round_up(N, lsize);
  constexpr std::uint32_t kQuietNaNBits = 0x7fc00000U;

  std::string failure_stage = "membership_submit";
  try {
    event e = q.submit([&](handler& h) {
      local_accessor<float, 1> local_center_tile(
          range<1>(selected_tile_k * selected_tile_d), h);

      h.parallel_for(
          nd_range<1>(range<1>(padded_num_points), range<1>(lsize)),
          [=](nd_item<1> item) {
            const size_t j = item.get_global_id(0);
            const size_t lid = item.get_local_id(0);
            const size_t lrange = item.get_local_range(0);
            const bool is_valid_point = j < N;

            float normalization_scale = 1.0f;
            float normalization_sum = 0.0f;

            for (size_t cluster_start = 0; cluster_start < K;
                 cluster_start += selected_tile_k) {
              const size_t current_tile_k =
                  sycl::min(selected_tile_k, K - cluster_start);
              float partial_distances[kSchemeBPrivateDistanceBound];
              for (size_t tile_cluster = 0;
                   tile_cluster < selected_tile_k; ++tile_cluster) {
                partial_distances[tile_cluster] = 0.0f;
              }

              for (size_t feature_start = 0; feature_start < D;
                   feature_start += selected_tile_d) {
                const size_t current_tile_d =
                    sycl::min(selected_tile_d, D - feature_start);
                const size_t tile_elements = current_tile_k * current_tile_d;

                for (size_t index = lid; index < tile_elements;
                     index += lrange) {
                  const size_t tile_cluster = index / current_tile_d;
                  const size_t tile_feature = index % current_tile_d;
                  local_center_tile[index] =
                      centers_next_dev[(cluster_start + tile_cluster) * D +
                                       feature_start + tile_feature];
                }
                item.barrier(access::fence_space::local_space);

                if (is_valid_point) {
                  for (size_t tile_cluster = 0;
                       tile_cluster < current_tile_k; ++tile_cluster) {
                    float partial = partial_distances[tile_cluster];
                    for (size_t tile_feature = 0;
                         tile_feature < current_tile_d; ++tile_feature) {
                      const size_t feature = feature_start + tile_feature;
                      const float diff = data_points_dev[feature * N + j] -
                                         local_center_tile[
                                             tile_cluster * current_tile_d +
                                             tile_feature];
                      partial += diff * diff;
                    }
                    partial_distances[tile_cluster] = partial;
                  }
                }

                item.barrier(access::fence_space::local_space);
              }

              if (is_valid_point) {
                for (size_t tile_cluster = 0;
                     tile_cluster < current_tile_k; ++tile_cluster) {
                  const float distance =
                      sycl::fmax(partial_distances[tile_cluster], eps);
                  const std::uint32_t distance_bits =
                      sycl::bit_cast<std::uint32_t>(distance);
                  if ((distance_bits & 0x7f800000U) == 0x7f800000U)
                    continue;
                  if (normalization_sum == 0.0f) {
                    normalization_scale = distance;
                    normalization_sum = 1.0f;
                  } else if (distance < normalization_scale) {
                    normalization_sum =
                        normalization_sum * membership_power(
                                                distance /
                                                normalization_scale) +
                        1.0f;
                    normalization_scale = distance;
                  } else {
                    normalization_sum += membership_power(
                        normalization_scale / distance);
                  }
                }
              }
            }

            for (size_t cluster_start = 0; cluster_start < K;
                 cluster_start += selected_tile_k) {
              const size_t current_tile_k =
                  sycl::min(selected_tile_k, K - cluster_start);
              float partial_distances[kSchemeBPrivateDistanceBound];
              for (size_t tile_cluster = 0;
                   tile_cluster < selected_tile_k; ++tile_cluster) {
                partial_distances[tile_cluster] = 0.0f;
              }

              for (size_t feature_start = 0; feature_start < D;
                   feature_start += selected_tile_d) {
                const size_t current_tile_d =
                    sycl::min(selected_tile_d, D - feature_start);
                const size_t tile_elements = current_tile_k * current_tile_d;

                for (size_t index = lid; index < tile_elements;
                     index += lrange) {
                  const size_t tile_cluster = index / current_tile_d;
                  const size_t tile_feature = index % current_tile_d;
                  local_center_tile[index] =
                      centers_next_dev[(cluster_start + tile_cluster) * D +
                                       feature_start + tile_feature];
                }
                item.barrier(access::fence_space::local_space);

                if (is_valid_point) {
                  for (size_t tile_cluster = 0;
                       tile_cluster < current_tile_k; ++tile_cluster) {
                    float partial = partial_distances[tile_cluster];
                    for (size_t tile_feature = 0;
                         tile_feature < current_tile_d; ++tile_feature) {
                      const size_t feature = feature_start + tile_feature;
                      const float diff = data_points_dev[feature * N + j] -
                                         local_center_tile[
                                             tile_cluster * current_tile_d +
                                             tile_feature];
                      partial += diff * diff;
                    }
                    partial_distances[tile_cluster] = partial;
                  }
                }

                item.barrier(access::fence_space::local_space);
              }

              if (is_valid_point) {
                for (size_t tile_cluster = 0;
                     tile_cluster < current_tile_k; ++tile_cluster) {
                  const float distance =
                      sycl::fmax(partial_distances[tile_cluster], eps);
                  const size_t cluster = cluster_start + tile_cluster;
                  const std::uint32_t distance_bits =
                      sycl::bit_cast<std::uint32_t>(distance);
                  if (normalization_sum == 0.0f ||
                      (distance_bits & 0x7f800000U) == 0x7f800000U) {
                    membership_next_dev[cluster * N + j] =
                        sycl::bit_cast<float>(kQuietNaNBits);
                  } else {
                    const float scaled_weight = membership_power(
                        normalization_scale / distance);
                    membership_next_dev[cluster * N + j] =
                        scaled_weight / normalization_sum;
                  }
                }
              }
            }
          });
    });

#if FCM_ENABLE_PROFILING
    failure_stage = "membership_wait";
    e.wait_and_throw();
    time += get_kernel_execution_time(e);
#else
    failure_stage = "membership_wait";
    e.wait_and_throw();
#endif
  } catch (const sycl::exception& error) {
    rethrow_work_group_failure(
        "membership", failure_stage, error.code().category().name(),
        error.what());
  }
}

float change_matrix(sycl::queue& q,
                    const float* curr,
                    const float* next,
                    int d,
                    int old_k,
                    int old_n,
                    float& time,
                    float* max_diff,
                    const WorkGroupRequest& work_group_request) {
  (void)old_n;

  // Reduce the maximum absolute element-wise change across [K, D] centers.
  // The execution-control layer separately applies the strict < threshold rule.
  const std::size_t total_items = static_cast<std::size_t>(old_k) * d;
  WorkGroupSelectionInput selection_input;
  selection_input.kernel_id = "change_check";
  selection_input.path = "reduction";
  selection_input.raw_requested = work_group_request.change_raw_requested;
  selection_input.explicit_strict = work_group_request.change_explicit;
  selection_input.request_source =
      work_group_request.change_explicit
          ? "cli_change_wg_size"
          : (work_group_request.mode == WorkGroupRequestMode::ExplicitStrict
          ? "cli_work_group_size"
          : "legacy_change_default");
  const KernelWorkGroupSelection selection =
      select_work_group_or_throw(q, work_group_request, selection_input);
  const std::size_t work_group_size = selection.effective;
  const std::size_t global_size =
      (total_items + work_group_size - 1) / work_group_size * work_group_size;

  std::string failure_stage = "change_memset";
  try {

  sycl::event init = q.memset(max_diff, 0, sizeof(float));
  failure_stage = "change_submit";
  sycl::event e = q.submit([&](sycl::handler& h) {
    h.depends_on(init);

    auto red = sycl::reduction(max_diff, sycl::maximum<float>());

    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size),
                                     sycl::range<1>(work_group_size)),
                   red, [=](sycl::nd_item<1> item, auto& maxr) {
                     const std::size_t idx = item.get_global_id(0);
                     float diff = 0.0f;
                     if (idx < total_items) {
                       diff = sycl::fabs(next[idx] - curr[idx]);
                     }
                     maxr.combine(diff);
                   });
  });

  float res = 0.0f;
  failure_stage = "change_wait_copy";
  q.memcpy(&res, max_diff, sizeof(float)).wait_and_throw();

#if FCM_ENABLE_PROFILING
  time += get_kernel_execution_time(e);
#endif
  return res;
  } catch (const sycl::exception& error) {
    rethrow_work_group_failure(
        "change_check", failure_stage, error.code().category().name(),
        error.what());
  }
}

void classify(sycl::queue& q,
              const float* u,
              int* l,
              int ok,
              int on,
              float& time) {
  // Select the maximum membership for each point and emit one-based cluster
  // labels in the range 1..K.
  sycl::event e = q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<1>(on), [=](sycl::id<1> idx) {
      int id = idx[0];
      float max_val = -1.0f;
      int label = 0;
      for (int i = 0; i < ok; ++i) {
        float val = u[i * on + id];
        if (val > max_val) {
          max_val = val;
          label = i + 1;
        }
      }
      l[id] = label;
    });
  });
#if FCM_ENABLE_PROFILING
  e.wait();
  time += get_kernel_execution_time(e);
#endif
}

template <typename CenterPowerEvaluator, typename MembershipPowerEvaluator>
int fuzzy_c_means_standard_impl(sycl::queue& q,
                           int k,
                           int n,
                           const FuzzifierConfig& fuzzifier,
                           CenterPowerEvaluator center_power,
                           MembershipPowerEvaluator membership_power,
                           int seed,
                           int low,
                           int high,
                           float threshold,
                           int d,
                           int* label,
                           float* u,
                           float* data,
                           float* centers,
                           float& t_cls,
                           float& t_init,
                           float& t_total,
                           float& t_first_iter,
                           float& t_other_iters,
                           const WorkGroupRequest& work_group_request,
                           int old_k,
                           int old_n,
                           float& t_change,
                           float& t_up_u,
                           float& t_up_c,
                           const ExecutionModeConfig& execution_mode,
                           IterationOutcome* iteration_outcome) {
  validate_execution_mode(execution_mode, threshold);
  size_t pts_sz = (size_t)old_n * d;
  size_t u_sz = (size_t)old_k * old_n;
  size_t c_sz = (size_t)old_k * d;

  float* d_data = sycl::malloc_device<float>(pts_sz, q);
  float* d_u_curr = sycl::malloc_device<float>(u_sz, q);
  float* d_u_next = sycl::malloc_device<float>(u_sz, q);
  float* d_c_curr = sycl::malloc_device<float>(c_sz, q);
  float* d_c_next = sycl::malloc_device<float>(c_sz, q);
  int* d_label = sycl::malloc_device<int>(old_n, q);
  float* d_num = sycl::malloc_device<float>(c_sz, q);
  float* d_den = sycl::malloc_device<float>(old_k, q);
  float* d_diff = sycl::malloc_device<float>(1, q);

  q.memcpy(d_data, data, pts_sz * sizeof(float)).wait();

  parallel_initialize_membership_matrix(q, d_u_curr, old_k, old_n, seed, low,
                                        high, t_init);

  // Iteration wall time excludes initialization and includes only the loop;
  // classification and final copies follow after the loop completes.
  auto start = std::chrono::steady_clock::now();
  const int loop_limit =
      execution_mode.protocol == IterationProtocol::Fixed20
          ? 20
          : execution_mode.max_iter;
  int executed_iters = 0;
  int change_checks = 0;
  float final_center_change = -1.0f;
  bool native_converged = false;

  for (int iter = 0; iter < loop_limit; ++iter) {
    ++executed_iters;
    auto iter_start = std::chrono::steady_clock::now();

    update_cluster_centers(q, d, center_power, d_u_curr, d_data, d_c_next,
                           work_group_request, old_k, old_n, d_num, d_den,
                           t_up_c);

    update_membership_values(q, k, n, membership_power, d, d_u_next, d_data, d_c_next,
                             work_group_request, old_k, old_n, t_up_u);

    float value = -1.0f;
    if (iter > 0) {
      value = change_matrix(q, d_c_curr, d_c_next, d, old_k, old_n, t_change,
                            d_diff, work_group_request);
      ++change_checks;
      final_center_change = value;
    }
    std::cout << "Iteration " << iter + 1
              << ", center change value=" << value << "\n";

    std::swap(d_u_curr, d_u_next);
    std::swap(d_c_curr, d_c_next);

    if (iter == 0) {
      q.wait();
      auto iter_end = std::chrono::steady_clock::now();
      t_first_iter +=
          std::chrono::duration<float, std::milli>(iter_end - iter_start)
              .count();
    }
    if (should_stop_after_iteration(execution_mode.protocol, iter, value,
                                    threshold)) {
      native_converged = true;
      break;
    }
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  float loop_time_ms =
      std::chrono::duration<float, std::milli>(end - start).count();
  t_total += loop_time_ms;
  float non_first_time = loop_time_ms - t_first_iter;
  if (non_first_time > 0.0f)
    t_other_iters += non_first_time;

  classify(q, d_u_curr, d_label, old_k, old_n, t_cls);
  q.memcpy(centers, d_c_curr, c_sz * sizeof(float));
  q.memcpy(label, d_label, old_n * sizeof(int));
  q.memcpy(u, d_u_curr, u_sz * sizeof(float)).wait();

  if (iteration_outcome != nullptr) {
    iteration_outcome->actual_iterations = executed_iters;
    iteration_outcome->change_checks = change_checks;
    iteration_outcome->final_center_change = final_center_change;
    iteration_outcome->stop_reason =
        completed_stop_reason(execution_mode.protocol, native_converged);
  }

  sycl::free(d_data, q);
  sycl::free(d_u_curr, q);
  sycl::free(d_u_next, q);
  sycl::free(d_c_curr, q);
  sycl::free(d_c_next, q);
  sycl::free(d_label, q);
  sycl::free(d_num, q);
  sycl::free(d_den, q);
  sycl::free(d_diff, q);

  return executed_iters;
}

int fuzzy_c_means_standard_dispatch(
    sycl::queue& q,
    int k,
    int n,
    const FuzzifierConfig& fuzzifier,
    int seed,
    int low,
    int high,
    float threshold,
    int d,
    int* label,
    float* u,
    float* data,
    float* centers,
    float& t_cls,
    float& t_init,
    float& t_total,
    float& t_first_iter,
    float& t_other_iters,
    const WorkGroupRequest& work_group_request,
    int old_k,
    int old_n,
    float& t_change,
    float& t_up_u,
    float& t_up_c,
    const ExecutionModeConfig& execution_mode,
    IterationOutcome* iteration_outcome) {
  // Resolve the host-side fuzzifier policy once, then instantiate the exact
  // m=2 evaluators or generic pow evaluators for the complete iteration path.
  auto execute = [&](auto center_power, auto membership_power) {
    return fuzzy_c_means_standard_impl(
        q, k, n, fuzzifier, center_power, membership_power, seed, low, high,
        threshold, d, label, u, data, centers, t_cls, t_init, t_total,
        t_first_iter, t_other_iters, work_group_request, old_k, old_n, t_change,
        t_up_u, t_up_c, execution_mode, iteration_outcome);
  };

  auto dispatch_membership = [&](auto center_power) {
    switch (fuzzifier.membership_effective_path) {
      case MembershipPowerPath::M200Identity:
        return execute(center_power, MembershipM200IdentityEvaluator{});
      case MembershipPowerPath::GenericPow:
        return execute(
            center_power,
            MembershipGenericPowEvaluator{fuzzifier.membership_exponent_q});
    }
    throw FuzzifierConfigurationError("unknown membership power path");
  };

  switch (fuzzifier.center_effective_path) {
    case CenterPowerPath::M200Square:
      return dispatch_membership(CenterM200SquareEvaluator{});
    case CenterPowerPath::GenericPow:
      return dispatch_membership(
          CenterGenericPowEvaluator{fuzzifier.effective_m});
  }
  throw FuzzifierConfigurationError("unknown center power path");
}

int fuzzy_c_means_standard(sycl::queue& q,
                           int k,
                           int n,
                           const FuzzifierConfig& fuzzifier,
                           int seed,
                           int low,
                           int high,
                           float threshold,
                           int d,
                           int* label,
                           float* u,
                           float* data,
                           float* centers,
                           float& t_cls,
                           float& t_init,
                           float& t_total,
                           float& t_first_iter,
                           float& t_other_iters,
                           const WorkGroupRequest& work_group_request,
                           int old_k,
                           int old_n,
                           float& t_change,
                           float& t_up_u,
                           float& t_up_c,
                           const ExecutionModeConfig& execution_mode,
                           IterationOutcome* iteration_outcome) {
  return fuzzy_c_means_standard_dispatch(
      q, k, n, fuzzifier, seed, low, high, threshold, d, label, u, data,
      centers, t_cls, t_init, t_total, t_first_iter, t_other_iters,
      work_group_request, old_k, old_n, t_change, t_up_u, t_up_c,
      execution_mode, iteration_outcome);
}

}

#ifndef FCM_SUPPRESS_MAIN
int main(int argc, char* argv[]) {
  int num_clusters = -1;
  int num_points = -1;
  int num_dims = -1;
  float threshold = -1.0f;
  std::string dataset_path_str = "";
  std::string output_label_path = "";
  std::size_t explicit_work_group_size = 0;
  bool has_explicit_work_group_size = false;
  std::size_t explicit_center_work_group_size = 0;
  std::size_t explicit_membership_work_group_size = 0;
  std::size_t explicit_change_work_group_size = 0;
  bool has_explicit_center_work_group_size = false;
  bool has_explicit_membership_work_group_size = false;
  bool has_explicit_change_work_group_size = false;
  std::string requested_fuzzifier_text = "2.0";
  double requested_fuzzifier = 2.0;
  cmeans::MPowerPolicy requested_m_power_policy =
      cmeans::MPowerPolicy::Auto;
  bool explicit_m_power_policy_present = false;
  bool force_generic_m_alias_present = false;
  cmeans::FuzzifierConfig fuzzifier_config;
  cmeans::ExecutionModeConfig execution_mode_config;

  if (argc < 13) {
    std::cerr << "Usage: " << argv[0]
              << " -i <dataset-path> -o <output-label-path> "
                 "-c <number-of-clusters> -e <threshold> "
                 "-n <number-of-points> -d <number-of-dimensions> "
                 "[--iteration-protocol fixed20|native-dynamic] "
                 "[--max-iter <positive-integer>] [--seed <integer>] "
                 "[--fuzzifier <finite-value-greater-than-1>] "
                 "[--m-power-policy auto|generic] [--force-generic-m] "
                 "[--work-group-size <positive-integer>] "
                 "[--center-wg-size <positive-integer>] "
                 "[--membership-wg-size <positive-integer>] "
                 "[--change-wg-size <positive-integer>]"
              << "\nCompatibility alias: --force-generic-m is equivalent to "
                 "--m-power-policy generic."
              << std::endl;
    return 1;
  }
  std::set<std::string> allowed_options = {
      "-i", "-o", "-c", "-e", "-n", "-d", "--work-group-size",
      "--center-wg-size", "--membership-wg-size", "--change-wg-size",
      "--fuzzifier", "--m-power-policy", "--iteration-protocol",
      "--max-iter", "--seed"};
  std::set<std::string> seen_options;
  try {
    for (int i = 1; i < argc;) {
      const std::string arg = argv[i];
      if (arg == "--force-generic-m") {
        if (force_generic_m_alias_present)
          throw std::invalid_argument(
              "duplicate option: --force-generic-m");
        force_generic_m_alias_present = true;
        ++i;
        continue;
      }
      if (allowed_options.count(arg) == 0)
        throw std::invalid_argument("unknown option: " + arg);
      if (!seen_options.insert(arg).second)
        throw std::invalid_argument("duplicate option: " + arg);
      if (i + 1 >= argc || allowed_options.count(argv[i + 1]) != 0 ||
          std::string(argv[i + 1]) == "--force-generic-m") {
        if (arg == "--fuzzifier") {
          requested_fuzzifier_text = "<missing>";
        }
        throw std::invalid_argument("missing value for option: " + arg);
      }
      const std::string val = argv[i + 1];
      if (val.empty())
        throw std::invalid_argument("missing value for option: " + arg);

      std::size_t parsed = 0;
      if (arg == "-i") {
        dataset_path_str = val;
        parsed = val.size();
      } else if (arg == "-o") {
        output_label_path = val;
        parsed = val.size();
      } else if (arg == "-c") {
        num_clusters = std::stoi(val, &parsed);
      } else if (arg == "-e") {
        threshold = std::stof(val, &parsed);
      } else if (arg == "-n") {
        num_points = std::stoi(val, &parsed);
      } else if (arg == "-d") {
        num_dims = std::stoi(val, &parsed);
      } else if (arg == "--iteration-protocol") {
        execution_mode_config.protocol =
            cmeans::parse_iteration_protocol(val);
        execution_mode_config.protocol_explicit = true;
        parsed = val.size();
      } else if (arg == "--max-iter") {
        execution_mode_config.max_iter = cmeans::parse_max_iter(val);
        execution_mode_config.max_iter_explicit = true;
        parsed = val.size();
      } else if (arg == "--seed") {
        execution_mode_config.seed = cmeans::parse_seed(val);
        execution_mode_config.seed_explicit = true;
        parsed = val.size();
      } else if (arg == "--work-group-size" ||
                 arg == "--center-wg-size" ||
                 arg == "--membership-wg-size" ||
                 arg == "--change-wg-size") {
        const unsigned long long parsed_value = std::stoull(val, &parsed);
        if (parsed_value == 0 ||
            parsed_value >
                static_cast<unsigned long long>(
                    std::numeric_limits<int>::max()) ||
            parsed_value >
                static_cast<unsigned long long>(
                    std::numeric_limits<std::size_t>::max())) {
          throw std::out_of_range(
              arg + " is outside the supported positive range");
        }
        const std::size_t value = static_cast<std::size_t>(parsed_value);
        if (arg == "--work-group-size") {
          explicit_work_group_size = value;
          has_explicit_work_group_size = true;
        } else if (arg == "--center-wg-size") {
          explicit_center_work_group_size = value;
          has_explicit_center_work_group_size = true;
        } else if (arg == "--membership-wg-size") {
          explicit_membership_work_group_size = value;
          has_explicit_membership_work_group_size = true;
        } else {
          explicit_change_work_group_size = value;
          has_explicit_change_work_group_size = true;
        }
      } else if (arg == "--fuzzifier") {
        requested_fuzzifier_text = val;
        requested_fuzzifier = cmeans::parse_fuzzifier(val);
        parsed = val.size();
      } else if (arg == "--m-power-policy") {
        requested_m_power_policy = cmeans::parse_m_power_policy(val);
        explicit_m_power_policy_present = true;
        parsed = val.size();
      }
      if (parsed != val.size())
        throw std::invalid_argument("malformed value for option " + arg);
      i += 2;
    }
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << std::endl;
    return 1;
  }

  try {
    fuzzifier_config = cmeans::resolve_fuzzifier_config(
        requested_fuzzifier_text, requested_fuzzifier,
        requested_m_power_policy, explicit_m_power_policy_present,
        force_generic_m_alias_present);
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << std::endl;
    return 1;
  }

  const std::set<std::string> required_options = {
      "-i", "-o", "-c", "-e", "-n", "-d"};
  for (const std::string& option : required_options) {
    if (seen_options.count(option) == 0) {
      std::cerr << "Error: missing required option " << option << std::endl;
      return 1;
    }
  }
  if (num_clusters <= 0 || num_points <= 0 || num_dims <= 0 ||
      !ieee_finite_float(threshold) || threshold <= 0.0f ||
      dataset_path_str.empty() ||
      output_label_path.empty()) {
    std::cerr << "Error: invalid arguments." << std::endl;
    return 1;
  }
  try {
    cmeans::validate_execution_mode(execution_mode_config, threshold);
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << std::endl;
    return 1;
  }

#ifdef _FPGA_
  sycl::queue q(
      sycl::ext::intel::fpga_emulator_selector_v,
#if FCM_ENABLE_PROFILING
      sycl::property_list {
        sycl::property::queue::enable_profiling{},
            sycl::property::queue::in_order {}
      }
#else
      sycl::property_list{sycl::property::queue::in_order{}}
#endif
  );
#endif

#ifdef _GPU_
  sycl::queue q(
      sycl::gpu_selector_v,
#if FCM_ENABLE_PROFILING
      sycl::property_list {
        sycl::property::queue::enable_profiling{},
            sycl::property::queue::in_order {}
      }
#else
      sycl::property_list{sycl::property::queue::in_order{}}
#endif
  );
#endif

#ifdef _CPU_
  sycl::queue q(
      sycl::cpu_selector_v,
#if FCM_ENABLE_PROFILING
      sycl::property_list {
        sycl::property::queue::enable_profiling{},
            sycl::property::queue::in_order {}
      }
#else
      sycl::property_list{sycl::property::queue::in_order{}}
#endif
  );
#endif

  std::cout << "Device:"
            << q.get_device().template get_info<sycl::info::device::name>()
            << "\n";

  cmeans::WorkGroupRequest work_group_request;
  if (has_explicit_work_group_size) {
    work_group_request.mode =
        cmeans::WorkGroupRequestMode::ExplicitStrict;
    work_group_request.center_membership_raw_requested =
        explicit_work_group_size;
    work_group_request.change_raw_requested = explicit_work_group_size;
  } else {
    work_group_request.mode =
        cmeans::WorkGroupRequestMode::LegacyDefault;
    work_group_request.center_membership_raw_requested =
        q.get_device().is_cpu() ? 512 : 256;
    work_group_request.change_raw_requested = 256;
  }
  if (has_explicit_center_work_group_size) {
    work_group_request.center_raw_requested =
        explicit_center_work_group_size;
    work_group_request.center_explicit = true;
  }
  if (has_explicit_membership_work_group_size) {
    work_group_request.membership_raw_requested =
        explicit_membership_work_group_size;
    work_group_request.membership_explicit = true;
  }
  if (has_explicit_change_work_group_size) {
    work_group_request.change_raw_requested =
        explicit_change_work_group_size;
    work_group_request.change_explicit = true;
  }

  const int old_num_points = num_points;
  const int old_num_clusters = num_clusters;

  const int seed = execution_mode_config.seed;
  int low = 0;
  int high = 50;

  float read_data_time = 0.0f, write_data_time = 0.0f, classify_time = 0.0f;
  float initialize_membership_matrix_time = 0.0f, total_iteration_time = 0.0f;
  float update_membership_matrix_time = 0.0f,
        update_cluster_centers_time = 0.0f;
  float change_matrix_time = 0.0f;
  float first_iteration_time = 0.0f;
  float other_iteration_time = 0.0f;

  std::vector<float> data_points(static_cast<std::size_t>(old_num_points) *
                                 num_dims);
  std::vector<float> membership_matrix(
      static_cast<std::size_t>(old_num_clusters) * old_num_points);
  std::vector<float> centers(static_cast<std::size_t>(old_num_clusters) *
                             num_dims);
  std::vector<int> label(old_num_points);

  FCM_ITT_TASK_BEGIN(g_fcm_itt_read);
  auto read_data_start_time = std::chrono::steady_clock::now();
  util::read_data_transposed(dataset_path_str, data_points.data(),
                             old_num_points, num_dims);
  auto read_data_end_time = std::chrono::steady_clock::now();
  FCM_ITT_TASK_END();
  read_data_time = std::chrono::duration<float, std::milli>(
                       read_data_end_time - read_data_start_time)
                       .count();
  std::cout << "Input data loaded and transposed by util." << "\n";

  auto start_time = std::chrono::steady_clock::now();

  FCM_ITT_TASK_BEGIN(g_fcm_itt_compute);
  int count = 0;
  cmeans::IterationOutcome iteration_outcome;
  try {
    count = cmeans::fuzzy_c_means_standard(
        q, num_clusters, num_points, fuzzifier_config, seed, low, high,
        threshold, num_dims,
        label.data(), membership_matrix.data(), data_points.data(),
        centers.data(), classify_time, initialize_membership_matrix_time,
        total_iteration_time, first_iteration_time, other_iteration_time,
        work_group_request, old_num_clusters, old_num_points,
        change_matrix_time, update_membership_matrix_time,
        update_cluster_centers_time, execution_mode_config,
        &iteration_outcome);
  } catch (const cmeans::WorkGroupExecutionError& error) {
    std::cerr << "Error: work-group failure kernel=" << error.kernel_id()
              << " stage=" << error.stage()
              << " message=" << error.what() << std::endl;
    return 2;
  }
  std::cout << "Iteration protocol: "
            << cmeans::iteration_protocol_name(execution_mode_config.protocol)
            << "\nActual iterations: " << iteration_outcome.actual_iterations
            << "\nStop reason: "
            << cmeans::stop_reason_name(iteration_outcome.stop_reason)
            << "\n";
  FCM_ITT_TASK_END();

  auto end_time = std::chrono::steady_clock::now();
  const double raw_fcm_wall_ms_total =
      std::chrono::duration<double, std::milli>(end_time - start_time).count();
  const float fcm_time = static_cast<float>(raw_fcm_wall_ms_total);

  FCM_ITT_TASK_BEGIN(g_fcm_itt_write);
  auto write_data_start_time = std::chrono::steady_clock::now();

  util::write_data_of_data_point_transposed(output_label_path.c_str(),
                                            data_points.data(), label.data(),
                                            old_num_points, num_dims);

  fs::path out_label_p(output_label_path);
  fs::path out_dir =
      out_label_p.has_parent_path() ? out_label_p.parent_path() : fs::path(".");
  std::string centers_out_path = (out_dir / "centers.csv").string();
  std::string time_out_path = (out_dir / "time.csv").string();

  std::cout << "Writing labels to: " << output_label_path << std::endl;
  std::cout << "Writing cluster centers to: " << centers_out_path << std::endl;
  std::cout << "Writing timing data to: " << time_out_path << std::endl;

  util::write_data_of_center(centers_out_path.c_str(), centers.data(),
                             old_num_clusters, num_dims);

  auto write_data_end_time = std::chrono::steady_clock::now();
  FCM_ITT_TASK_END();
  write_data_time = std::chrono::duration<float, std::milli>(
                        write_data_end_time - write_data_start_time)
                        .count();

  if (count > 0) {
    update_cluster_centers_time =
        update_cluster_centers_time / count / 1000000.0f;
    update_membership_matrix_time =
        update_membership_matrix_time / count / 1000000.0f;
    const int change_calls = std::max(0, count - 1);
    if (change_calls > 0) {
      change_matrix_time = change_matrix_time / change_calls / 1000000.0f;
    } else {
      change_matrix_time = 0.0f;
    }
    total_iteration_time = total_iteration_time / count;
    if (count > 1)
      other_iteration_time = other_iteration_time / (count - 1);
  }
  initialize_membership_matrix_time /= 1000000.0f;
  classify_time /= 1000000.0f;

  util::write_data_of_time(
      time_out_path.c_str(), read_data_time, write_data_time, fcm_time,
      initialize_membership_matrix_time, total_iteration_time,
      first_iteration_time, other_iteration_time, classify_time,
      update_membership_matrix_time, update_cluster_centers_time,
      change_matrix_time, count, std::max(0, count - 1));

  util::print_data_of_time(
      read_data_time, write_data_time, fcm_time,
      initialize_membership_matrix_time, total_iteration_time,
      first_iteration_time, other_iteration_time, classify_time,
      update_membership_matrix_time, update_cluster_centers_time,
      change_matrix_time, count, std::max(0, count - 1));

  std::cout << "----------- Dataset information -----------" << std::endl;
  std::cout << "Dataset: "
            << fs::path(dataset_path_str).filename().string() << std::endl;
  std::cout << "Points: " << old_num_points << std::endl;
  std::cout << "Clusters: " << old_num_clusters << std::endl;
  std::cout << "Dimensions: " << num_dims << std::endl;
  std::cout << "-------------------------------------------" << std::endl
            << std::endl;

  return 0;
}
#endif
