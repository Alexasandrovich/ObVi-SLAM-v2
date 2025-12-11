#pragma once
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/inference/Symbol.h>

using namespace gtsam;
using symbol_shorthand::X;
using symbol_shorthand::L;

class SlamKernel {
private:
  ISAM2 isam_;
  NonlinearFactorGraph graph_;
  Values initial_estimates_;

  boost::shared_ptr<Cal3_S2> K_;

  Pose3 current_pose_;
  int frame_id_ = -1;

  noiseModel::Diagonal::shared_ptr pose_noise_;
  noiseModel::Isotropic::shared_ptr measurement_noise_;

public:
  SlamKernel(double fx, double fy, double cx, double cy) {
    K_ = boost::make_shared<Cal3_S2>(fx, fy, 0.0, cx, cy);
    // vo_ удален

    ISAM2Params params;
    params.relinearizeThreshold = 0.1;
    isam_ = ISAM2(params);

    // Шум позы (очень маленький, так как мы верим ORB-SLAM)
    pose_noise_ = noiseModel::Diagonal::Sigmas((Vector(6) << 0.01, 0.01, 0.01, 0.05, 0.05, 0.05).finished());
    // Шум измерений (пиксели)
    measurement_noise_ = noiseModel::Isotropic::Sigma(2, 2.0);
  }

  // Метод для работы с ORB-SLAM
  void process_with_external_pose(const std::vector<double>& external_pose_vec,
                                  const std::vector<std::tuple<int, float, float>>& detections) {
    if (external_pose_vec.size() != 16) return;

    frame_id_++;

    // 1. Конвертация позы от ORB-SLAM
    Eigen::Matrix4d pose_mat;
    int k=0;
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) pose_mat(i,j) = external_pose_vec[k++];

    Pose3 current_pose_orb(pose_mat);

    // Добавляем фактор PriorPose (верим траектории ORB-SLAM)
    graph_.add(PriorFactor<Pose3>(X(frame_id_), current_pose_orb, pose_noise_));
    initial_estimates_.insert(X(frame_id_), current_pose_orb);
    current_pose_ = current_pose_orb;

    // 2. Обработка объектов
    for (const auto& det : detections) {
      int id = std::get<0>(det);
      double u = std::get<1>(det);
      double v = std::get<2>(det);
      Point2 measured_uv(u, v);

      if (!isam_.valueExists(L(id)) && !initial_estimates_.exists(L(id))) {
        // Инициализация (луч на условные 10 метров/единиц)
        Point2 normalized_xy = K_->calibrate(measured_uv);
        double depth = 10.0;
        Point3 camera_point(normalized_xy.x() * depth, normalized_xy.y() * depth, depth);
        Point3 global_point = current_pose_.transformFrom(camera_point);

        initial_estimates_.insert(L(id), global_point);
        // Слабый прайор на позицию объекта
        graph_.add(PriorFactor<Point3>(L(id), global_point, noiseModel::Isotropic::Sigma(3, 50.0)));
      }

      graph_.add(GenericProjectionFactor<Pose3, Point3, Cal3_S2>(
              measured_uv, measurement_noise_, X(frame_id_), L(id), K_
      ));
    }

    // 3. Оптимизация
    isam_.update(graph_, initial_estimates_);
    graph_.resize(0);
    initial_estimates_.clear();
  }

  std::vector<double> get_landmark_pos(int id) {
    if (isam_.valueExists(L(id))) {
      Point3 p = isam_.calculateEstimate<Point3>(L(id));
      return {p.x(), p.y(), p.z()};
    }
    return {};
  }
};