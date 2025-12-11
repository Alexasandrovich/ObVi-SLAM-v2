#include "OrbSlamWrapper.h"

// Include the heavy libraries ONLY in the .cpp file
#include <System.h>
#include <Converter.h>
#include <opencv2/core/eigen.hpp>

OrbSlamWrapper::OrbSlamWrapper(const std::string& vocab_path, const std::string& settings_path, bool use_viewer) {
  system_ = nullptr;
  try {
    std::cout << "Creating ORB_SLAM3::System..." << std::endl;
    system_ = new ORB_SLAM3::System(vocab_path, settings_path, ORB_SLAM3::System::MONOCULAR, use_viewer);
    std::cout << "ORB_SLAM3::System created successfully" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Exception creating ORB_SLAM3: " << e.what() << std::endl;
    system_ = nullptr;
  } catch (...) {
    std::cerr << "Unknown exception creating ORB_SLAM3" << std::endl;
    system_ = nullptr;
  }
}

OrbSlamWrapper::~OrbSlamWrapper() {
  if (system_) {
    system_->Shutdown();
    delete system_;
    system_ = nullptr;
  }
}

cv::Mat OrbSlamWrapper::track(const cv::Mat& img, double timestamp) {
  if (!system_) {
    std::cerr << "ORB-SLAM system is null!" << std::endl;
    return cv::Mat();
  }

  if (img.empty()) {
    std::cerr << "Input image is empty!" << std::endl;
    return cv::Mat();
  }

  try {
    Sophus::SE3f Tcw_sophus = system_->TrackMonocular(img, timestamp);

    Eigen::Matrix4f Tcw_eigen = Tcw_sophus.matrix();
    Eigen::Matrix4f Twc_eigen = Tcw_eigen.inverse();

    cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
    for(int i=0; i<4; i++)
      for(int j=0; j<4; j++)
        pose.at<double>(i,j) = (double)Twc_eigen(i,j);

    return pose;
  } catch (const std::exception& e) {
    std::cerr << "Exception in track: " << e.what() << std::endl;
    return cv::Mat();
  }
}

void OrbSlamWrapper::reset() {
  if (system_) system_->Reset();
}