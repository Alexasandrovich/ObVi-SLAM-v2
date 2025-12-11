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

#include "VisualOdometry.h"

using namespace gtsam;
using symbol_shorthand::X; // Robot Pose
using symbol_shorthand::L; // Landmark (Tree/Pole)

class SlamKernel {
private:
  ISAM2 isam_;
  NonlinearFactorGraph graph_;
  Values initial_estimates_;

  boost::shared_ptr<Cal3_S2> K_;
  std::unique_ptr<VisualOdometry> vo_;

  Pose3 current_pose_; // Накапливаемая поза (Dead Reckoning)
  int frame_id_ = -1;

  // Noise Models
  noiseModel::Diagonal::shared_ptr pose_noise_; // Для Prior
  noiseModel::Diagonal::shared_ptr odom_noise_; // Для VO
  noiseModel::Isotropic::shared_ptr measurement_noise_; // Для детекций

public:
  SlamKernel(double fx, double fy, double cx, double cy) {
    K_ = boost::make_shared<Cal3_S2>(fx, fy, 0.0, cx, cy);
    vo_ = std::make_unique<VisualOdometry>(fx, fy, cx, cy);

    ISAM2Params params;
    params.relinearizeThreshold = 0.1;
    isam_ = ISAM2(params);

    pose_noise_ = noiseModel::Diagonal::Sigmas((Vector(6) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished());
    odom_noise_ = noiseModel::Diagonal::Sigmas((Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());
    measurement_noise_ = noiseModel::Isotropic::Sigma(2, 2.0); // 2 pixels error
  }

  void process(const cv::Mat& img, const std::vector<std::tuple<int, float, float>>& detections) {
    frame_id_++;

    // 1. Visual Odometry Step
    cv::Mat T_cv = vo_->add_frame(img);
    Eigen::Matrix4d T_eigen;
    for(int i=0; i<4; i++)
      for(int j=0; j<4; j++)
        T_eigen(i,j) = T_cv.at<double>(i,j);

    Pose3 relative_pose(T_eigen);

    // 2. Add Pose to Graph
    if (frame_id_ == 0) {
      current_pose_ = Pose3();
      graph_.add(PriorFactor<Pose3>(X(0), current_pose_, pose_noise_));
      initial_estimates_.insert(X(0), current_pose_);
    } else {
      current_pose_ = current_pose_.compose(relative_pose);
      graph_.add(BetweenFactor<Pose3>(X(frame_id_-1), X(frame_id_), relative_pose, odom_noise_));
      initial_estimates_.insert(X(frame_id_), current_pose_);
    }

    // 3. Add Landmarks (Detections)
    for (const auto& det : detections) {
      int id = std::get<0>(det);
      double u = std::get<1>(det);
      double v = std::get<2>(det);
      Point2 measured_uv(u, v);

      // Если новый объект - инициализируем
      if (!isam_.valueExists(L(id)) && !initial_estimates_.exists(L(id))) {
        // calibrate переводит пиксели (u,v) в нормализованные координаты (x,y)
        Point2 normalized_xy = K_->calibrate(measured_uv);

        // Мы не знаем глубину. Для старта предполагаем фиксированную глубину 5м.
        // GTSAM потом подвинет точку, когда увидит её с другого ракурса (параллакс).
        // Создаем луч на глубину 5 метров: [x*Z, y*Z, Z]
        double depth = 5.0;
        Point3 camera_point(normalized_xy.x() * depth, normalized_xy.y() * depth, depth);

        // Переводим в глобальные координаты
        Point3 global_point = current_pose_.transformFrom(camera_point);

        initial_estimates_.insert(L(id), global_point);

        // Добавляем слабый Prior на высоту (Z), предполагая, что дерево растет из земли
        // Это "парк" - земля примерно на одном уровне.
        graph_.add(PriorFactor<Point3>(L(id), global_point, noiseModel::Isotropic::Sigma(3, 10.0)));
      }

      // Добавляем ребро проекции
      graph_.add(GenericProjectionFactor<Pose3, Point3, Cal3_S2>(
              measured_uv, measurement_noise_, X(frame_id_), L(id), K_
      ));
    }

    // 4. Optimize
    isam_.update(graph_, initial_estimates_);
    graph_.resize(0);
    initial_estimates_.clear();

    // Update current pose estimate
    current_pose_ = isam_.calculateEstimate<Pose3>(X(frame_id_));
  }

  // Getters for Viz
  std::vector<double> get_pose() {
    auto t = current_pose_.translation();
    return {t.x(), t.y(), t.z()};
  }

  std::vector<std::tuple<int, double, double, double>> get_map() {
    std::vector<std::tuple<int, double, double, double>> map;
    // Перебираем оптимизированные лэндмарки (это дорого, но для дебага ок)
    // В реальном коде лучше кешировать IDs
    // Здесь просто вернем заглушку или последние добавленные,
    // так как итерация по ISAM сложнее.
    // Для MVP можно вызывать calculateEstimate для конкретных ID из Python.
    return map;
  }

  std::vector<double> get_landmark_pos(int id) {
    if (isam_.valueExists(L(id))) {
      Point3 p = isam_.calculateEstimate<Point3>(L(id));
      return {p.x(), p.y(), p.z()};
    }
    return {};
  }
};