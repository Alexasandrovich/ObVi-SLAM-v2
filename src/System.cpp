#include "obvi/System.hpp"
#include "obvi/odometry/GlimOdom.hpp"
#include "obvi/mapping/SemanticGraph.hpp"
#include <gtsam/geometry/Pose3.h>

#include <iostream>

namespace obvi {

// Скрытая реализация (Pimpl)
  class System::Impl {
  public:
    std::unique_ptr<GlimOdom> odom;
    std::unique_ptr<SemanticGraph> mapper;

    Impl(const std::string& config, const std::string& glim_conf, const std::vector<double>& extrinsics) {
      // Инициализируем модули
      odom = std::make_unique<GlimOdom>(glim_conf, extrinsics);
      mapper = std::make_unique<SemanticGraph>();
    }
  };

// Конструктор
  System::System(const std::string& config_file,
                 const std::string& glim_config_path,
                 const std::vector<double>& lidar_extrinsics)
          : impl_(std::make_unique<Impl>(config_file, glim_config_path, lidar_extrinsics)) {}

// Деструктор (обязателен в cpp при использовании unique_ptr и Pimpl)
  System::~System() = default;

// Основной метод обработки
  void System::process(double timestamp,
                       const std::vector<float>& lidar_data,
                       const std::vector<int>& obs_classes,
                       const std::vector<std::vector<double>>& obs_coords)
  {
    // 1. Odometry Step (GLIM)
    // Данные лидара приходят как плоский массив float [x, y, z, i, ...]
    if (!lidar_data.empty()) {
      // Предполагаем stride = 4 (x,y,z,intensity)
      size_t num_points = lidar_data.size() / 4;
      impl_->odom->insert_cloud(timestamp, lidar_data.data(), num_points);
    }

    // Получаем текущую позу от одометрии (T_world_body)
    PoseMatrix pose = impl_->odom->get_pose();

    // 2. Mapping Step (GTSAM)
    // Преобразуем вектора Python (списки списков) в структуры C++
    std::vector<LandmarkObservation> obs;
    for (size_t i = 0; i < obs_classes.size(); ++i) {
      if (obs_coords[i].size() >= 3) {
        Eigen::Vector3d p(obs_coords[i][0], obs_coords[i][1], obs_coords[i][2]);
        obs.push_back({obs_classes[i], p});
      }
    }

    // Обновляем граф
    impl_->mapper->update(pose, obs);
  }

// Геттер позы для Python (x, y, z, qx, qy, qz, qw)
  std::vector<double> System::get_current_pose_vec() {
    PoseMatrix p = impl_->odom->get_pose();

    // Используем GTSAM для удобной конвертации матрицы в кватернион
    gtsam::Pose3 pose(p);
    auto t = pose.translation();
    auto q = pose.rotation().toQuaternion(); // Возвращает Eigen::Quaternion

    return {t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w()};
  }

// Геттер карты для Python
  std::vector<std::vector<double>> System::get_map_objects() {
    auto objects = impl_->mapper->get_map();
    std::vector<std::vector<double>> result;
    result.reserve(objects.size());

    for(const auto& obj : objects) {
      result.push_back({
                               (double)obj.id,
                               (double)obj.class_id,
                               obj.global_pos.x(),
                               obj.global_pos.y(),
                               obj.global_pos.z()
                       });
    }
    return result;
  }

} // namespace obvi