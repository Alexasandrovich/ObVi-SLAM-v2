#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class VisualOdometry {
public:
  VisualOdometry(double fx, double fy, double cx, double cy);
  cv::Mat add_frame(const cv::Mat& img); // Возвращает 4x4 Pose Delta

private:
  cv::Mat K_;
  cv::Ptr<cv::ORB> orb_;
  cv::Mat last_img_, last_des_;
  std::vector<cv::KeyPoint> last_kp_;
  bool initialized_ = false;
};