#include "obvi/odometry/GlimOdom.hpp"

// GLIM & GTSAM Points Headers
#include <gtsam/geometry/Pose3.h>
#include <glim/util/config.hpp>
#include <glim/util/time_keeper.hpp>
#include <glim/preprocess/cloud_preprocessor.hpp>
#include <glim/odometry/async_odometry_estimation.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>

#include <spdlog/spdlog.h>
#include <mutex>
#include <string>

namespace obvi {

  struct GlimOdom::Impl {
    std::unique_ptr<glim::TimeKeeper> time_keeper;
    std::unique_ptr<glim::CloudPreprocessor> preprocessor;
    std::unique_ptr<glim::AsyncOdometryEstimation> odometry;

    Eigen::Matrix4d current_pose = Eigen::Matrix4d::Identity();
    std::mutex pose_mutex;
    Eigen::Isometry3d T_body_lidar;
    bool use_transform = false;

    Impl(const std::string& config_path, const std::vector<double>& extrinsics) {
      // Загружаем конфиги
      if (!config_path.empty()) {
        glim::GlobalConfig::instance(config_path);
      }

      time_keeper.reset(new glim::TimeKeeper);
      preprocessor.reset(new glim::CloudPreprocessor);

      // Загружаем модуль одометрии
      // В реальном конфиге должно быть прописано имя .so файла
      // Здесь хардкодим fallback, если конфиг пустой
      std::string odom_so = "libodometry_estimation_ct.so";

      // Пытаемся прочитать из конфига, если он есть
      try {
        glim::Config config_odom(glim::GlobalConfig::get_config_path("config_odometry"));
        odom_so = config_odom.param<std::string>("odometry_estimation", "so_name", "libodometry_estimation_cpu.so");
      } catch (...) {
        spdlog::warn("Config not found, using default CPU odometry");
      }

      auto odom_base = glim::OdometryEstimationBase::load_module(odom_so);
      if (!odom_base) {
        throw std::runtime_error("Failed to load GLIM module: " + odom_so +
                                 ". Check LD_LIBRARY_PATH and if glim is installed.");
      }

      odometry.reset(new glim::AsyncOdometryEstimation(odom_base, false));

      // настройки экстринсиков лидара для дальнейшего выравнивания с ОС камеры
      if (extrinsics.size() == 6)
      {
        double x = extrinsics[0];
        double y = extrinsics[1];
        double z = extrinsics[2];
        double roll   = extrinsics[3] * M_PI / 180.0;
        double pitch  = extrinsics[4] * M_PI / 180.0;
        double yaw    = extrinsics[5] * M_PI / 180.0;

        T_body_lidar = Eigen::Isometry3d::Identity();
        T_body_lidar.translation() << x, y, z;
        T_body_lidar.linear() = (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                                 Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                                 Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())).toRotationMatrix();
        if(!T_body_lidar.isApprox(Eigen::Isometry3d::Identity()))
        {
          use_transform = true;
        }
      }else{
        throw std::runtime_error("Failed to parse extrincics for lidar - must have 6 numbers, but it has " + std::to_string(extrinsics.size()));
      }

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

  GlimOdom::GlimOdom(const std::string& config_path, const std::vector<double>& extrinsics)
          : impl_(std::make_unique<Impl>(config_path, extrinsics)) {}

  GlimOdom::~GlimOdom() = default;

  void GlimOdom::insert_cloud(double timestamp, const float* data, size_t num_points) {
    // Convert flat float array -> Vector4d points
    std::vector<Eigen::Vector4d> points(num_points);
    for(size_t i=0; i<num_points; ++i) {
      // x, y, z, 1.0
      points[i] << data[i*4], data[i*4+1], data[i*4+2], 1.0;
    }

    auto glim_raw = std::make_shared<glim::RawPoints>();
    glim_raw->stamp = timestamp;
    glim_raw->points.resize(num_points);

    if(impl_->use_transform)
    {
      // использование нетриваиальных экстринсиков
      for(size_t i = 0; i < num_points; ++i)
      {
        Eigen::Vector3d p(data[i*4], data[i*4+1], data[i*4+2]);
        p = impl_->T_body_lidar * p;
        glim_raw->points[i] << p.x(), p.y(), p.z(), 1.0;
      }
    }else{
      for(size_t i=0; i<num_points; ++i) {
        glim_raw->points[i] << data[i*4], data[i*4+1], data[i*4+2], 1.0;
      }
    }

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