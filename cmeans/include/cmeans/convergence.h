#pragma once

namespace cmeans {

inline bool center_change_meets_strict_threshold(float center_change,
                                                 float threshold) {
  return center_change < threshold;
}

}  // namespace cmeans
