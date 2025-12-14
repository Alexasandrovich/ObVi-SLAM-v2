#include "obvi/mapping/SemanticGraph.hpp"

// GTSAM Headers
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/inference/Symbol.h>

using namespace gtsam;

namespace obvi {

  struct SemanticGraph::Impl {
    ISAM2 isam;
    NonlinearFactorGraph graph;
    Values initial_estimates;

    Pose3 prev_pose;
    bool initialized = false;
    int pose_cnt = 0;
    int next_landmark_id = 0;

    // Data Association (упрощенная)
    struct TrackedLandmark {
      int id;
      int class_id;
      Point3 pos; // Global
    };
    std::vector<TrackedLandmark> landmarks;

    noiseModel::Diagonal::shared_ptr odom_noise;
    noiseModel::Isotropic::shared_ptr meas_noise;

    Impl() {
      ISAM2Params params;
      params.relinearizeThreshold = 0.1;
      isam = ISAM2(params);

      // Шум одометрии (верим GLIM, но не слепо)
      odom_noise = noiseModel::Diagonal::Sigmas((Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());
      // Шум измерений (визуальная глубина может шуметь)
      meas_noise = noiseModel::Isotropic::Sigma(3, 0.5); // 0.5m error
    }

    int find_correspondence(const Point3& global_pos, int class_id) {
      // Простейший Nearest Neighbor
      double min_dist = 2.0; // Порог матчинга (метры)
      int best_id = -1;

      for (const auto& lm : landmarks) {
        if (lm.class_id != class_id) continue;
        double d = distance3(lm.pos, global_pos);
        if (d < min_dist) {
          min_dist = d;
          best_id = lm.id;
        }
      }
      return best_id;
    }
  };

  SemanticGraph::SemanticGraph() : impl_(std::make_unique<Impl>()) {}
  SemanticGraph::~SemanticGraph() = default;

  void SemanticGraph::update(const PoseMatrix& odom_pose_mat, const std::vector<LandmarkObservation>& observations) {
    Pose3 current_pose(odom_pose_mat);

    // 1. Add Pose Factors
    if (!impl_->initialized) {
      // Первый кадр: фиксируем в начале координат GLIM
      auto prior_noise = noiseModel::Diagonal::Sigmas((Vector(6) << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4).finished());
      impl_->graph.add(PriorFactor<Pose3>(symbol_shorthand::X(0), current_pose, prior_noise));
      impl_->initial_estimates.insert(symbol_shorthand::X(0), current_pose);
      impl_->initialized = true;
    } else {
      // Одометрия от предыдущего шага
      Pose3 odom_delta = impl_->prev_pose.between(current_pose);
      impl_->graph.add(BetweenFactor<Pose3>(symbol_shorthand::X(impl_->pose_cnt),
                                            symbol_shorthand::X(impl_->pose_cnt + 1),
                                            odom_delta, impl_->odom_noise));

      impl_->pose_cnt++;
      impl_->initial_estimates.insert(symbol_shorthand::X(impl_->pose_cnt), current_pose);
    }
    impl_->prev_pose = current_pose;

    // 2. Process Landmarks
    for (const auto& obs : observations) {
      Point3 local_point(obs.local_pos.x(), obs.local_pos.y(), obs.local_pos.z());
      Point3 global_point = current_pose.transformFrom(local_point);

      int lm_id = impl_->find_correspondence(global_point, obs.class_id);

      if (lm_id == -1) {
        // New Landmark
        lm_id = impl_->next_landmark_id++;
        impl_->landmarks.push_back({lm_id, obs.class_id, global_point});
        impl_->initial_estimates.insert(symbol_shorthand::L(lm_id), global_point);

        // Prior for new landmark (weak) to help init
        impl_->graph.add(PriorFactor<Point3>(symbol_shorthand::L(lm_id), global_point,
                                             noiseModel::Isotropic::Sigma(3, 10.0)));
      }

      // Add Measurement Factor (Relative position)
      // Мера: где находится Лэндмарк в системе координат Робота
      impl_->graph.add(BetweenFactor<Pose3, Point3>(
              symbol_shorthand::X(impl_->pose_cnt), symbol_shorthand::L(lm_id),
              local_point, impl_->meas_noise));
    }

    // 3. Optimize
    impl_->isam.update(impl_->graph, impl_->initial_estimates);
    impl_->graph.resize(0);
    impl_->initial_estimates.clear();

    // Update landmark estimates
    Values result = impl_->isam.calculateEstimate();
    for (auto& lm : impl_->landmarks) {
      if (result.exists(symbol_shorthand::L(lm.id))) {
        lm.pos = result.at<Point3>(symbol_shorthand::L(lm.id));
      }
    }
  }

  PoseMatrix SemanticGraph::get_optimized_pose() const {
    // В реальном SLAM берем optimized, но для стабильности пока возвращаем последнюю
    return impl_->prev_pose.matrix();
  }

  std::vector<MapObject> SemanticGraph::get_map() const {
    std::vector<MapObject> map_out;
    for (const auto& lm : impl_->landmarks) {
      map_out.push_back({lm.id, lm.class_id, Eigen::Vector3d(lm.pos.x(), lm.pos.y(), lm.pos.z())});
    }
    return map_out;
  }

} // namespace obvi