#include "VisualOdometry.h"

VisualOdometry::VisualOdometry(double fx, double fy, double cx, double cy) {
  K_ = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
  orb_ = cv::ORB::create(2000);
}

cv::Mat VisualOdometry::add_frame(const cv::Mat& img) {
  cv::Mat gray;
  if (img.channels() == 3) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); else gray = img;

  std::vector<cv::KeyPoint> kp;
  cv::Mat des;
  orb_->detectAndCompute(gray, cv::noArray(), kp, des);

  cv::Mat T = cv::Mat::eye(4, 4, CV_64F);

  if (initialized_ && !last_des_.empty() && !des.empty()) {
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(last_des_, des, matches);

    std::vector<cv::Point2f> pts1, pts2;
    for (auto& m : matches) {
      if (m.distance < 50) {
        pts1.push_back(last_kp_[m.queryIdx].pt);
        pts2.push_back(kp[m.trainIdx].pt);
      }
    }

    if (pts1.size() > 10) {
      cv::Mat E, mask;
      E = cv::findEssentialMat(pts2, pts1, K_, cv::RANSAC, 0.999, 1.0, mask);
      if (!E.empty() && E.rows == 3 && E.cols == 3) {
        cv::Mat R, t;
        cv::recoverPose(E, pts2, pts1, K_, R, t, mask);
        R.copyTo(T(cv::Rect(0,0,3,3)));
        t.copyTo(T(cv::Rect(3,0,1,3)));
      }
    }
  }

  last_img_ = gray;
  last_kp_ = kp;
  last_des_ = des;
  initialized_ = true;
  return T;
}