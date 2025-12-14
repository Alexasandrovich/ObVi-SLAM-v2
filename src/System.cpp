#include "obvi/System.hpp"
#include "obvi/odometry/GlimOdom.hpp"
#include "obvi/mapping/SemanticGraph.hpp"
#include <iostream>

namespace obvi {

  class System::Impl {
  public:
    std::unique_ptr<GlimOdom> odom;
    std::unique_ptr<SemanticGraph> mapper;

    Impl(const std::string& config, const std::string& glim_conf) {
      odom = std::make_unique<GlimOdom>(glim_conf);
      mapper = std::make_unique<SemanticGraph>();
    }
  };

  System::System(const std::string& config_file, const std::string& glim_config_path)
          : impl_(std::make_unique<Impl>(config_file, glim_config_path)) {}

  System::~System() = default;

  void System::process(double timestamp,
                       const std::vector<float>& lidar_data,
                       const std::vector<int>& obs_classes,
                       const std::vector<std::vector<double>>& obs_coords)
  {
    // 1. Odometry
    // Stride 4 assumed: [x,y,z,i]
    if (!lidar_data.empty()) {
      size_t num_points = lidar_data.size() / 4;
      impl_->odom->insert_cloud(timestamp, lidar_data.data(), num_points);
    }

    PoseMatrix pose = impl_->odom->get_pose();

    // 2. Mapping
    // Преобразуем вектора Python в структуры C++
    std::vector<LandmarkObservation> obs;
    for (size_t i = 0; i < obs_classes.size(); ++i) {
      Eigen::Vector3d p(obs_coords[i][0], obs_coords[i][1], obs_coords[i][2]);
      obs.push_back({obs_classes[i], p});
    }

    impl_->mapper->update(pose, obs);
  }

  std::vector<double> System::get_current_pose_vec() {
    PoseMatrix p = impl_->odom->get_pose();

    // Mat -> Pose3 -> Quaternion
    gtsam::Pose3 pose(p);
    auto t = pose.translation();
    auto q = pose.rotation().toQuaternion();

    return {t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w()};
  }

  std::vector<std::vector<double>> System::get_map_objects() {
    auto objects = impl_->mapper->get_map();
    std::vector<std::vector<double>> result;
    for(const auto& obj : objects) {
      result.push_back({
                               (double)obj.id, (double)obj.class_id,
                               obj.global_pos.x(), obj.global_pos.y(), obj.global_pos.z()
                       });
    }
    return result;
  }

} // namespace obvi