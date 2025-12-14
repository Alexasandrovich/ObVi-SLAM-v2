#pragma once
#include <vector>
#include <Eigen/Dense>

namespace obvi {

  // 4x4 Матрица Позы
  using PoseMatrix = Eigen::Matrix4d;

  // Наблюдение объекта (приходит из Python)
  struct LandmarkObservation {
    int class_id;       // Класс (дерево, столб)
    Eigen::Vector3d local_pos; // Координаты [x,y,z] относительно камеры/робота
    // Можно добавить ковариацию, если Python её считает
  };

  // Глобальный объект карты (для отрисовки)
  struct MapObject {
    int id;
    int class_id;
    Eigen::Vector3d global_pos;
  };

} // namespace obvi