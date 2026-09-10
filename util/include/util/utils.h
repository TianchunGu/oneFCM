#pragma once

#include <cmath>  // std::fabs
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <functional>  // std::plus
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

namespace util {
struct MembershipNormalizationStats {
  std::size_t value_count = 0;
  std::size_t finite_count = 0;
  std::size_t nonfinite_count = 0;
  double membership_min = 0.0;
  double membership_max = 0.0;
  double max_abs_sum_error = std::numeric_limits<double>::max();
  double mean_abs_sum_error = std::numeric_limits<double>::max();
  bool all_finite = false;
  bool normalized_within_tolerance = false;
  int first_failed_point = -1;
  double first_failed_sum = 0.0;
  double first_failed_error = std::numeric_limits<double>::max();
};

// Summarize finite values and per-point sums in cluster-major [K, N]
// membership storage. Invalid inputs fail closed with a non-normalized result.
MembershipNormalizationStats compute_membership_normalization_stats(
    const float* membership_matrix,
    int num_clusters,
    int num_points,
    double tolerance);

// Append delimiter-separated fields from str to res.
void string_split(string str, char split, vector<string>& res);

// Skip the first CSV row and read at most [num_points, num_dims] values into
// point-major storage [N, D].
void read_data(string fname, float* data_points, int num_points, int num_dims);

// Require a CSV header and N valid finite data rows with at least D fields,
// ignore extra rows and fields, and write dimension-major storage [D, N].
void read_data_transposed(const std::string& filename,
                          float* data,
                          int num_points,
                          int num_dims);

// Print dimension-major [D, N] data one point per line.
void print_data_points(float* data_points, int num_points, int num_dims);

// Write dimension-major [D, N] data as point-major CSV rows with labels.
void write_data_of_data_point_transposed(const char* filename,
                                         const float* data_col_major,
                                         const int* label,
                                         int num_points,
                                         int num_dims);

// Check cluster-major [K, N] memberships with a fixed tolerance of 1e-6.
bool verify_normalization(const float* membership_matrix,
                          int old_num_clusters,
                          int old_num_points);

// Print the flat membership buffer using the function's legacy row breaks.
void print_membership_matrix(float* membership_matrix,
                             int num_clusters,
                             int num_points);

// Print cluster-major centers [K, D].
void print_centers(float* centers, int num_clusters, int num_dims);

// Print point-major rows whose last element is treated as the class value.
void print_final_data_points(float* data_points, int num_points, int num_dims);

// Write point-major [N, D] data followed by one label column.
void write_data_of_data_point(string fname,
                              float* data_points,
                              int* label,
                              int num_points,
                              int num_dims);

// Write point-major center rows [num_points, num_dims].
void write_data_of_center(string fname,
                          float* data_points,
                          int num_points,
                          int num_dims);

// Write the fixed 14-column timing CSV schema.
void write_data_of_time(string fname,
                        float read_data_time,
                        float write_data_time,
                        float fcm_time,
                        float initialize_membership_matrix_time,
                        float total_iteration_time,
                        float first_iteration_time,
                        float other_iteration_time,
                        float classify_time,
                        float update_membership_values_time,
                        float update_cluster_centers_time,
                        float change_membership_matrix_time,
                        int iteration_count,
                        int change_calls);

// Print the timing summary while preserving the legacy runtime messages.
void print_data_of_time(float read_data_time,
                        float write_data_time,
                        float fcm_time,
                        float initialize_membership_matrix_time,
                        float total_iteration_time,
                        float first_iteration_time,
                        float other_iteration_time,
                        float classify_time,
                        float update_membership_values_time,
                        float update_cluster_centers_time,
                        float change_membership_matrix_time,
                        int iteration_count,
                        int change_calls);

}  // namespace util
