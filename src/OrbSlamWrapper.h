#pragma once
#include <opencv2/core/core.hpp>
#include <string>

// Forward declaration only.
// We do NOT include System.h here to keep C++17 code safe from g2o.
namespace ORB_SLAM3 {
  class System;
}

class OrbSlamWrapper {
public:
  OrbSlamWrapper(const std::string& vocab_path, const std::string& settings_path, bool use_viewer);
  ~OrbSlamWrapper();

  cv::Mat track(const cv::Mat& img, double timestamp);
  void reset();

private:
  // Pointer to incomplete type (allowed in header)
  ORB_SLAM3::System* system_;
};