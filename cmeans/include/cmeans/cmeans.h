#pragma once

#include <cmeans/execution_mode.h>
#include <cmeans/m_power_config.h>
#include <cmeans/work_group_config.h>
#include <sycl/sycl.hpp>

namespace cmeans {

/**
 * Run fuzzy c-means clustering on the supplied SYCL queue.
 *
 * The data buffer uses dimension-major layout `[dimension][point]`; the
 * membership buffer uses cluster-major layout `[cluster][point]`; centers use
 * row-major `[cluster][dimension]`. All buffers are caller-owned and must be
 * sized for the original (unpadded) cluster and point counts.
 *
 * @param q SYCL queue that selects the execution device.
 * @param num_clusters Active cluster count.
 * @param num_points Active point count.
 * @param fuzzifier Validated fuzzifier and m-power dispatch configuration.
 * @param seed Membership-initialization seed.
 * @param low Lower bound for the initialization distribution.
 * @param high Upper bound for the initialization distribution.
 * @param threshold Positive convergence threshold; Fixed20 validates but does
 *        not use it for early stopping.
 * @param num_dims Number of dimensions per point.
 * @param label Output array of one-based cluster labels in the range 1..K.
 * @param membership_matrix Input/output cluster-major membership matrix.
 * @param data_points Input dimension-major data matrix.
 * @param centers Output row-major center matrix.
 * @param classify_time Classification-kernel time accumulator.
 * @param initialize_membership_matrix_time Initialization-kernel time
 *        accumulator.
 * @param total_iteration_time Iteration wall-time accumulator.
 * @param first_iteration_time First-iteration wall-time accumulator.
 * @param other_iteration_time Later-iteration wall-time accumulator.
 * @param work_group_request Global and per-kernel work-group requests.
 * @param old_num_clusters Original, unpadded cluster count.
 * @param old_num_points Original, unpadded point count.
 * @param change_membership_matrix_time Center-change-kernel time accumulator.
 * @param update_membership_matrix_time Membership-update-kernel time
 *        accumulator.
 * @param update_cluster_centers_time Center-update-kernel time accumulator.
 * @param execution_mode Fixed20 or Native Dynamic execution configuration.
 * @param iteration_outcome Optional completed-iteration outcome.
 * @return Number of completed iterations.
 */
int fuzzy_c_means_standard(sycl::queue& q,
                           int num_clusters,
                           int num_points,
                           const FuzzifierConfig& fuzzifier,
                           int seed,
                           int low,
                           int high,
                           float threshold,
                           int num_dims,
                           int* label,
                           float* membership_matrix,
                           float* data_points,
                           float* centers,
                           float& classify_time,
                           float& initialize_membership_matrix_time,
                           float& total_iteration_time,
                           float& first_iteration_time,
                           float& other_iteration_time,
                           const WorkGroupRequest& work_group_request,
                           int old_num_clusters,
                           int old_num_points,
                           float& change_membership_matrix_time,
                           float& update_membership_matrix_time,
                           float& update_cluster_centers_time,
                           const ExecutionModeConfig& execution_mode =
                               ExecutionModeConfig{},
                           IterationOutcome* iteration_outcome = nullptr);

}  // namespace cmeans
