#include "obvi/odometry/GlimOdom.hpp"

// GLIM Headers
#include <glim/util/config.hpp>
#include <glim/util/time_keeper.hpp>
#include <glim/preprocess/cloud_preprocessor.hpp>
#include <glim/odometry/async_odometry_estimation.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>

#include <spdlog/spdlog.h>
#include <mutex>

namespace obvi {

  struct GlimOdom::Impl {
    std::unique_ptr<glim::TimeKeeper> time_keeper;
    std::unique_ptr<glim::CloudPreprocessor> preprocessor;
    std::unique_ptr<glim::AsyncOdometryEstimation> odometry;

    Eigen::Matrix4d current_pose = Eigen::Matrix4d::Identity();
    std::mutex pose_mutex;

    Impl(const std::string& config_path) {
      glim::GlobalConfig::instance(config_path);
      time_keeper.reset(new glim::TimeKeeper);
      preprocessor.reset(new glim::CloudPreprocessor);

      glim::Config config_odom(glim::GlobalConfig::get_config_path("config_odometry"));
      std::string odom_so = config_odom.param<std::string>("odometry_estimation", "so_name", "libodometry_estimation_cpu.so");

      auto odom_base = glim::OdometryEstimationBase::load_module(odom_so);
      if (!odom_base) throw std::runtime_error("Failed to load GLIM module: " + odom_so);

      odometry.reset(new glim::AsyncOdometryEstimation(odom_base, false)); // false = no IMU
    }

    void update() {
      std::vector<glim::EstimationFrame::ConstPtr> est, marg;
      odometry->get_results(est, marg);

      if (!est.empty()) {
        std::lock_guard<std::mutex> lock(pose_mutex);
        current_pose = est.back()->T_world_lidar.matrix();
      }
    }
  };

  GlimOdom::GlimOdom(const std::string& config_path)
          : impl_(std::make_unique<Impl>(config_path)) {}

  GlimOdom::~GlimOdom() = default;

  void GlimOdom::insert_cloud(double timestamp, const float* data, size_t num_points) {
    // Convert flat float array -> Vector4d points
    std::vector<Eigen::Vector4d> points(num_points);
    for(size_t i=0; i<num_points; ++i) {
      points[i] << data[i*4], data[i*4+1], data[i*4+2], 1.0;
      // intensity ignored for now or stored separately
    }

    auto frame = std::make_shared<gtsam_points::PointCloudCPU>(points);
    auto glim_raw = std::make_shared<glim::RawPoints>();
    glim_raw->points = frame->points;
    glim_raw->stamp = timestamp;

    impl_->time_keeper->process(glim_raw);
    auto pre = impl_->preprocessor->preprocess(glim_raw);
    impl_->odometry->insert_frame(pre);

    impl_->update();
  }

  PoseMatrix GlimOdom::get_pose() {
    impl_->update();
    std::lock_guard<std::mutex> lock(impl_->pose_mutex);
    return impl_->current_pose;
  }

} // namespace obvi