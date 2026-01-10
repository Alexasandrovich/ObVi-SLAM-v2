#pragma once
#include <memory>
#include <vector>
#include <string>
#include "obvi/types.hpp"

namespace obvi {

  class System {
  public:
    // config_file: путь к главному yaml конфигу
    System(const std::string& config_file,
           const std::string& glim_config_path,
           const std::vector<double>& lidar_extrinsics);
    ~System();

    // Основной цикл обработки
    // lidar_data: плоский массив float [x,y,z,i, x,y,z,i...]
    // obs_classes: список классов объектов
    // obs_coords: список координат объектов [[x,y,z], [x,y,z]...]
    void process(double timestamp,
                 const std::vector<float>& lidar_data,
                 const std::vector<int>& obs_classes,
                 const std::vector<std::vector<double>>& obs_coords);

    // Геттеры для визуализации
    std::vector<double> get_current_pose_vec(); // [x,y,z, qx,qy,qz,qw]
    std::vector<std::vector<double>> get_map_objects(); // [[id, class, x, y, z], ...]

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
  };

} // namespace obvi