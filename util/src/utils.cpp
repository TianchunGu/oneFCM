#include <util/utils.h>
#include <cstdint>
#include <cstring>
#include <iomanip>

namespace util {
namespace {
[[noreturn]] void fail_input_csv(const std::string& message) {
  std::cerr << "Error: " << message << std::endl;
  std::exit(1);
}

std::size_t csv_column_count(const std::string& line) {
  std::size_t columns = 1;
  for (const char character : line) {
    if (character == ',')
      ++columns;
  }
  return columns;
}

bool ascii_space(char character) {
  return character == ' ' || character == '\t' || character == '\r' ||
         character == '\n' || character == '\f' || character == '\v';
}

char ascii_lower(char character) {
  if (character >= 'A' && character <= 'Z')
    return static_cast<char>(character + ('a' - 'A'));
  return character;
}

bool starts_with_nonfinite_literal(const std::string& cell) {
  std::size_t position = 0;
  while (position < cell.size() && ascii_space(cell[position]))
    ++position;
  if (position < cell.size() &&
      (cell[position] == '+' || cell[position] == '-')) {
    ++position;
  }
  if (cell.size() - position < 3)
    return false;
  const std::string prefix = {
      ascii_lower(cell[position]), ascii_lower(cell[position + 1]),
      ascii_lower(cell[position + 2])};
  return prefix == "inf" || prefix == "nan";
}

std::uint32_t ieee_bits(const float* value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(*value));
  std::memcpy(&bits, value, sizeof(bits));
  return bits;
}

bool ieee_finite(double value) {
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}
}

void string_split(string str, char split, vector<string>& res) {
  if (str == "")
    return;
  // A trailing delimiter lets the loop emit the final field.
  string strs = str + split;
  size_t pos = strs.find(split);

  while (pos != strs.npos) {
    string temp = strs.substr(0, pos);
    res.push_back(temp);

    strs = strs.substr(pos + 1, strs.size());
    pos = strs.find(split);
  }
}

void read_data(string filename,
               float* data_points,
               int num_points,
               int num_dims) {

  std::ifstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Error: Could not open the file - " << filename << std::endl;
    exit(1);
  }

  std::string line;
  // Public datasets use one header row; both CSV readers skip it.
  if (file.good()) {
    std::getline(file, line);
  }
  int point_index = 0;

  while (std::getline(file, line) && point_index < num_points) {
    std::stringstream ss(line);
    std::string value_str;
    int dim_index = 0;

    while (std::getline(ss, value_str, ',') && dim_index < num_dims) {
      try {

        float value = std::stof(value_str);

        data_points[point_index * num_dims + dim_index] = value;

        dim_index++;
      } catch (const std::invalid_argument& ia) {
        std::cerr << "Invalid argument: " << ia.what() << " at line "
                  << point_index + 1 << std::endl;
        exit(1);
      } catch (const std::out_of_range& oor) {
        std::cerr << "Out of range: " << oor.what() << " at line "
                  << point_index + 1 << std::endl;
        exit(1);
      }
    }
    point_index++;
  }

  if (point_index != num_points) {
    std::cerr << "Warning: Expected to read " << num_points
              << " points, but only found " << point_index << " in the file."
              << std::endl;
  }

  file.close();
}

void read_data_transposed(const std::string& filename,
                          float* data,
                          int num_points,
                          int num_dims) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    exit(1);
  }

  std::string line;
  int point_idx = 0;

  if (!std::getline(file, line)) {
    fail_input_csv("input CSV is missing a header row.");
  }
  if (csv_column_count(line) < static_cast<std::size_t>(num_dims)) {
    fail_input_csv("header has fewer than " + std::to_string(num_dims) +
                   " columns.");
  }

  while (point_idx < num_points && std::getline(file, line)) {
    std::stringstream ss(line);
    std::string cell;
    int dim_idx = 0;

    while (std::getline(ss, cell, ',') && dim_idx < num_dims) {
      float value = 0.0f;
      std::size_t parsed = 0;
      try {
        value = std::stof(cell, &parsed);
      } catch (const std::invalid_argument&) {
        fail_input_csv("invalid numeric value at row " +
                       std::to_string(point_idx + 1) + ", column " +
                       std::to_string(dim_idx + 1) + ".");
      } catch (const std::out_of_range&) {
        fail_input_csv("invalid numeric value at row " +
                       std::to_string(point_idx + 1) + ", column " +
                       std::to_string(dim_idx + 1) + ".");
      }

      while (parsed < cell.size() && ascii_space(cell[parsed]))
        ++parsed;
      if (parsed != cell.size()) {
        fail_input_csv("invalid numeric value at row " +
                       std::to_string(point_idx + 1) + ", column " +
                       std::to_string(dim_idx + 1) + ".");
      }
      if (starts_with_nonfinite_literal(cell)) {
        fail_input_csv("non-finite numeric value at row " +
                       std::to_string(point_idx + 1) + ", column " +
                       std::to_string(dim_idx + 1) + ".");
      }

      const std::size_t index = static_cast<std::size_t>(dim_idx) * num_points +
                                point_idx;
      data[index] = value;
      dim_idx++;
    }
    if (dim_idx < num_dims) {
      fail_input_csv("row " + std::to_string(point_idx + 1) +
                     " has fewer than " + std::to_string(num_dims) +
                     " columns.");
    }
    point_idx++;
  }

  if (point_idx < num_points) {
    fail_input_csv("expected " + std::to_string(num_points) +
                   " data rows but only found " +
                   std::to_string(point_idx) + ".");
  }

  file.close();
}

void print_data_points(float* data_points, int num_points, int num_dims) {

  if (data_points == nullptr) {
    std::cerr << "print_data_points Error: Null pointer provided." << std::endl;
    exit(1);
  }
  for (int i = 0; i < num_points; i++) {
    std::cout << "X" << i + 1 << ": (";
    for (int j = 0; j < num_dims; j++) {
      if (j != num_dims - 1) {
        std::cout << data_points[j * num_points + i] << ", ";
      } else {
        std::cout << data_points[j * num_points + i];
      }
    }
    std::cout << ")" << std::endl;
  }
}

MembershipNormalizationStats compute_membership_normalization_stats(
    const float* membership_matrix,
    int num_clusters,
    int num_points,
    double tolerance) {
  MembershipNormalizationStats stats;
  if (membership_matrix == nullptr || num_clusters <= 0 || num_points <= 0 ||
      !ieee_finite(tolerance) || tolerance < 0.0) {
    return stats;
  }

  stats.value_count = static_cast<std::size_t>(num_clusters) *
                      static_cast<std::size_t>(num_points);
  stats.max_abs_sum_error = 0.0;
  stats.mean_abs_sum_error = 0.0;
  double error_sum = 0.0;
  bool have_finite_value = false;

  // Membership is cluster-major, so each point is strided by num_points.
  for (int point = 0; point < num_points; ++point) {
    double point_sum = 0.0;
    bool point_finite = true;
    for (int cluster = 0; cluster < num_clusters; ++cluster) {
      const float* value_address =
          membership_matrix +
          static_cast<std::size_t>(cluster) * num_points + point;
      const std::uint32_t value_bits = ieee_bits(value_address);
      if ((value_bits & 0x7f800000U) == 0x7f800000U) {
        ++stats.nonfinite_count;
        point_finite = false;
        continue;
      }
      float raw_value = 0.0f;
      std::memcpy(&raw_value, &value_bits, sizeof(raw_value));
      const double value = raw_value;
      ++stats.finite_count;
      point_sum += value;
      if (!have_finite_value) {
        stats.membership_min = value;
        stats.membership_max = value;
        have_finite_value = true;
      } else {
        stats.membership_min = std::min(stats.membership_min, value);
        stats.membership_max = std::max(stats.membership_max, value);
      }
    }

    const double error = point_finite ? std::fabs(point_sum - 1.0) : 0.0;
    if (point_finite) {
      stats.max_abs_sum_error = std::max(stats.max_abs_sum_error, error);
      error_sum += error;
    }
    if (stats.first_failed_point < 0 && error > tolerance) {
      stats.first_failed_point = point;
      stats.first_failed_sum =
          point_finite ? point_sum : 0.0;
      stats.first_failed_error =
          point_finite ? error : std::numeric_limits<double>::max();
    } else if (stats.first_failed_point < 0 && !point_finite) {
      stats.first_failed_point = point;
      stats.first_failed_sum = 0.0;
      stats.first_failed_error = std::numeric_limits<double>::max();
    }
  }

  stats.all_finite = stats.nonfinite_count == 0;
  if (stats.all_finite) {
    stats.mean_abs_sum_error = error_sum / static_cast<double>(num_points);
  } else {
    stats.max_abs_sum_error = std::numeric_limits<double>::max();
    stats.mean_abs_sum_error = std::numeric_limits<double>::max();
  }
  stats.normalized_within_tolerance =
      stats.all_finite && stats.max_abs_sum_error <= tolerance;
  return stats;
}

bool verify_normalization(const float* membership_matrix,
                          int old_num_clusters,
                          int old_num_points) {
  constexpr double kTolerance = 1.0e-6;
  const MembershipNormalizationStats stats =
      compute_membership_normalization_stats(
          membership_matrix, old_num_clusters, old_num_points, kTolerance);
  if (!stats.normalized_within_tolerance) {
    std::cerr << "Membership normalization failed: point="
              << stats.first_failed_point << ", sum=" << stats.first_failed_sum
              << ", error=" << stats.first_failed_error << std::endl;
    return false;
  }
  return true;
}

void write_data_of_data_point_transposed(const char* filename,
                                         const float* data_col_major,
                                         const int* label,
                                         int num_points,
                                         int num_dims) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << " for writing."
              << std::endl;
    return;
  }

  // Six decimal places preserve the public CSV output contract.
  file << std::fixed << std::setprecision(6);

  for (int i = 0; i < num_points; ++i) {

    for (int j = 0; j < num_dims; ++j) {

      // Convert the dimension-major input index to a point-major CSV row.
      size_t idx = static_cast<size_t>(j) * num_points + i;

      file << data_col_major[idx];
      file << ",";
    }

    if (label != nullptr) {
      file << label[i];
    } else {
      file << "0";
    }

    file << "\n";
  }

  file.close();

}

void print_membership_matrix(float* membership_matrix,
                             int num_clusters,
                             int num_points) {
  if (membership_matrix == nullptr) {
    std::cerr << "print_membership_matrix Error: Null pointer provided."
              << std::endl;
    exit(1);
  }

  for (int i = 0; i < num_clusters * num_points; i++) {

    std::cout << i << ": " << membership_matrix[i] << "    ";

    if (i % num_clusters == num_clusters - 1 ||
        i == num_clusters * num_points - 1) {

      std::cout << std::endl;
    }
  }

  std::cout << "***************************************************************"
               "***********************"
            << std::endl;
}

void print_centers(float* centers, int num_clusters, int num_dims) {
  if (centers == nullptr) {
    std::cerr << "print_centers Error: Null pointer provided." << std::endl;
    exit(1);
  }
  for (int i = 0; i < num_clusters; i++) {
    std::cout << "Center " << i + 1 << ": (";
    for (int j = 0; j < num_dims; j++) {
      if (j != num_dims - 1) {
        std::cout << centers[i * num_dims + j] << ",";
      } else {
        std::cout << centers[i * num_dims + j];
      }
    }
    std::cout << ")" << std::endl;
  }
}

void print_final_data_points(float* data_points, int num_points, int num_dims) {
  if (data_points == nullptr) {
    std::cerr << "Error: Null pointer provided." << std::endl;
    exit(1);
  }

  for (int i = 0; i < num_points; i++) {
    std::cout << "X" << i + 1 << ":(";
    for (int j = 0; j < num_dims - 1; j++) {

      if (j != num_dims - 2) {
        std::cout << data_points[i * num_dims + j] << ",";
      } else {
        std::cout << data_points[i * num_dims + j];
      }
    }
    std::cout << ")    class:" << data_points[(i + 1) * num_dims - 1]
              << std::endl;
  }
}

void write_data_of_center(string fname,
                          float* data_points,
                          int num_points,
                          int num_dims) {

  ofstream dataFile;
  dataFile.open(fname, ios::out | ios::trunc);

  if (!dataFile.is_open()) {
    std::cerr << "Error: Unable to open file for writing - " << fname
              << std::endl;
    exit(1);
  }
  std::cout << "Writing cluster centers..." << std::endl;
  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < num_dims; j++) {
      if (j != num_dims - 1) {
        dataFile << data_points[i * num_dims + j] << ",";
      } else {

        if (i * num_dims + j == num_points * num_dims - 1) {
          dataFile << data_points[i * num_dims + j];
        } else {
          dataFile << data_points[i * num_dims + j] << std::endl;
        }
      }
    }
  }

  dataFile.close();
  std::cout << "Output written successfully." << std::endl;
}

void write_data_of_data_point(string fname,
                              float* data_points,
                              int* label,
                              int num_points,
                              int num_dims) {

  ofstream dataFile;
  dataFile.open(fname, ios::out | ios::trunc);

  if (!dataFile.is_open()) {
    std::cerr << "Error: Unable to open file for writing - " << fname
              << std::endl;
    exit(1);
  }
  std::cout << "Writing data points..." << std::endl;
  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j <= num_dims; j++) {
      if (j < num_dims) {
        dataFile << data_points[i * num_dims + j] << ",";
      } else {
        if (i * num_dims + j == num_points * num_dims) {
          dataFile << label[i];
        } else {
          dataFile << label[i] << std::endl;
        }
      }
    }
  }

  dataFile.close();
  std::cout << "Output written successfully." << std::endl;
}

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
                        float change_matrix_time,
                        int iteration_count,
                        int change_calls) {

  ofstream dataFile;
  dataFile.open(fname, ios::out | ios::trunc);

  if (!dataFile.is_open()) {
    std::cerr << "Error: Unable to open file for writing - " << fname
              << std::endl;
    exit(1);
  }
  std::cout << "Writing timing data..." << std::endl;

  const float fcm_avg_per_iter_time =
      (iteration_count > 0) ? (fcm_time / iteration_count) : 0.0f;

  // The fixed schema distinguishes wall-clock totals, per-iteration wall
  // times, per-call kernel profiling means, and iteration counters.
  dataFile << "read_data_time_ms"
           << ","
           << "write_data_time_ms"
           << ","
           << "fcm_time_ms_total"
           << ","
           << "fcm_time_ms_avg_per_iter"
           << ","
           << "iter_wall_time_ms_avg"
           << ","
           << "first_iter_wall_time_ms"
           << ","
           << "other_iter_wall_time_ms_avg"
           << ","
           << "init_membership_kernel_time_ms"
           << ","
           << "classify_kernel_time_ms"
           << ","
           << "update_membership_kernel_time_ms_avg"
           << ","
           << "update_cluster_centers_kernel_time_ms_avg"
           << ","
           << "change_matrix_kernel_time_ms_avg"
           << ","
           << "iteration_count"
           << ","
           << "change_calls"
           << std::endl;

  dataFile << std::fixed << std::setprecision(6) << read_data_time << ","
           << write_data_time << "," << fcm_time << "," << fcm_avg_per_iter_time
           << "," << total_iteration_time << "," << first_iteration_time << ","
           << other_iteration_time << ","
           << initialize_membership_matrix_time << "," << classify_time << ","
           << update_membership_values_time << "," << update_cluster_centers_time
           << "," << change_matrix_time << "," << iteration_count << ","
           << change_calls << std::endl;

  dataFile.close();
  std::cout << "Output written successfully." << std::endl;
}

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
                        float change_matrix_time,
                        int iteration_count,
                        int change_calls) {
  const float fcm_avg_per_iter_time =
      (iteration_count > 0) ? (fcm_time / iteration_count) : 0.0f;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Input reading time (wall, ms): " << read_data_time << std::endl;
  std::cout << "Output writing time (wall, ms): " << write_data_time << std::endl;
  std::cout << "Total FCM time (wall, ms, including copies, initialization, "
               "iterations, and classification): " << fcm_time
            << std::endl;
  std::cout << "Average FCM time (wall, ms/iter, approximate=fcm_total/iter): "
            << fcm_avg_per_iter_time << std::endl;

  std::cout << "Iterations (count): " << iteration_count << std::endl;
  std::cout << "change_matrix calls (count): " << change_calls << std::endl;

  std::cout << "Average iteration time (wall, ms/iter): " << total_iteration_time
            << std::endl;
  std::cout << "First iteration time (wall, ms): " << first_iteration_time
            << std::endl;
  std::cout << "Average time excluding the first iteration (wall, ms/iter): "
            << other_iteration_time
            << std::endl;

  std::cout << "Membership initialization time (kernel profiling, ms/call): "
            << initialize_membership_matrix_time << std::endl;
  std::cout << "Classification time (kernel profiling, ms/call): " << classify_time
            << std::endl;
  std::cout << "Membership update time (kernel profiling, ms/call): "
            << update_membership_values_time << std::endl;
  std::cout << "Cluster center update time (kernel profiling, ms/call): "
            << update_cluster_centers_time << std::endl;
  std::cout << "change_matrix time (kernel profiling, ms/call): "
            << change_matrix_time << std::endl;
}

}
