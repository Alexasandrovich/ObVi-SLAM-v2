#include "OrbSlamWrapper.h"

// Include the heavy libraries ONLY in the .cpp file
#include <System.h>
#include <Converter.h>
#include <opencv2/core/eigen.hpp>

OrbSlamWrapper::OrbSlamWrapper(const std::string& vocab_path, const std::string& settings_path, bool use_viewer) {
  // Initialize the system here
  system_ = new ORB_SLAM3::System(vocab_path, settings_path, ORB_SLAM3::System::MONOCULAR, use_viewer);
}

OrbSlamWrapper::~OrbSlamWrapper() {
  if (system_) {
    system_->Shutdown();
    delete system_;
    system_ = nullptr;
  }
}

cv::Mat OrbSlamWrapper::track(const cv::Mat& img, double timestamp) {
  if (!system_) return cv::Mat();

  // Call ORB-SLAM3
  // Note: TrackMonocular returns Sophus::SE3f in modern versions
  Sophus::SE3f Tcw_sophus = system_->TrackMonocular(img, timestamp);

  // Convert to Eigen matrix
  Eigen::Matrix4f Tcw_eigen = Tcw_sophus.matrix();

  // Invert to get Camera Pose (Twc)
  Eigen::Matrix4f Twc_eigen = Tcw_eigen.inverse();

  // Convert to OpenCV Mat
  cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
  for(int i=0; i<4; i++)
    for(int j=0; j<4; j++)
      pose.at<double>(i,j) = (double)Twc_eigen(i,j);

  return pose;
}

void OrbSlamWrapper::reset() {
  if (system_) system_->Reset();
}