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
#include <gtsam/nonlinear/ExpressionFactor.h>
#include <gtsam/slam/expressions.h>

// logging
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

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

    struct TrackedLandmark {
      int id;
      int class_id;
      Point3 pos;
    };
    std::vector<TrackedLandmark> landmarks;

    noiseModel::Diagonal::shared_ptr odom_noise;
    noiseModel::Isotropic::shared_ptr meas_noise;

    Impl() {
      ISAM2Params params;
      params.relinearizeThreshold = 0.1;
      params.relinearizeSkip = 1;
      isam = ISAM2(params);

      // Шум одометрии (0.05 rad, 0.1 m)
      odom_noise = noiseModel::Diagonal::Sigmas((Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());
      // Шум измерений (0.5 m)
      meas_noise = noiseModel::Isotropic::Sigma(3, 0.5);

      // logging
      spdlog::set_level(spdlog::level::info);
      spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
    }

    int find_correspondence(const Point3& global_pos, int class_id) {
      double min_dist = 2.0;
      int best_id = -1;

      spdlog::debug("find_corr: searching for class {} at [{:.2f}, {:.2f}, {:.2f}]",
                    class_id, global_pos.x(), global_pos.y(), global_pos.z());
      for (const auto& lm : landmarks) {
        if (lm.class_id != class_id) continue;

        double d = distance3(lm.pos, global_pos);
        spdlog::debug(" -> canditate ID {} (Class {}, pos [{:.2f}, {:.2f}, {:.2f}]): Dist {:.2f}m",
                      lm.id, lm.class_id, lm.pos.x(), lm.pos.y(), lm.pos.z(), d);
        if (d < min_dist) {
          min_dist = d;
          best_id = lm.id;
        }
      }

      if (best_id != -1)
      {
        spdlog::debug(" => matched ID {}", best_id);
      }else
      {
        spdlog::debug(" => no match found. New object constructing...");
      }

      return best_id;
    }
  };

  SemanticGraph::SemanticGraph() : impl_(std::make_unique<Impl>()) {}
  SemanticGraph::~SemanticGraph() = default;

  void SemanticGraph::update(const PoseMatrix& odom_pose_mat, const std::vector<LandmarkObservation>& observations) {
    Pose3 current_pose(odom_pose_mat);

    // --- 1. Поза (Odom / Prior) ---
    if (!impl_->initialized) {
      auto prior_noise = noiseModel::Diagonal::Sigmas((Vector(6) << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4).finished());
      impl_->graph.add(PriorFactor<Pose3>(symbol_shorthand::X(0), current_pose, prior_noise));
      impl_->initial_estimates.insert(symbol_shorthand::X(0), current_pose);
      impl_->initialized = true;
    } else {
      Pose3 odom_delta = impl_->prev_pose.between(current_pose);
      impl_->graph.add(BetweenFactor<Pose3>(symbol_shorthand::X(impl_->pose_cnt),
                                            symbol_shorthand::X(impl_->pose_cnt + 1),
                                            odom_delta, impl_->odom_noise));
      impl_->pose_cnt++;
      impl_->initial_estimates.insert(symbol_shorthand::X(impl_->pose_cnt), current_pose);
    }
    impl_->prev_pose = current_pose;
    spdlog::debug("Current pose from odometry: [{:.2f}, {:.2f}, {:.2f}]",
                  current_pose.x(), current_pose.y(), current_pose.z());

    // --- 2. Объекты (Landmarks) ---
    int obj_idx = 0;
    for (const auto& obs : observations) {
      Point3 local_point(obs.local_pos.x(), obs.local_pos.y(), obs.local_pos.z());
      spdlog::debug("Local detection pos: [{:.2f}, {:.2f}, {:.2f}]",
                    local_point.x(), local_point.y(), local_point.z());

      Point3 global_point = current_pose.transformFrom(local_point);

      int lm_id = impl_->find_correspondence(global_point, obs.class_id);

      if (lm_id == -1) {
        lm_id = impl_->next_landmark_id++;
        impl_->landmarks.push_back({lm_id, obs.class_id, global_point});
        impl_->initial_estimates.insert(symbol_shorthand::L(lm_id), global_point);

        // Слабый прайор для инициализации
        impl_->graph.add(PriorFactor<Point3>(symbol_shorthand::L(lm_id), global_point,
                                             noiseModel::Isotropic::Sigma(3, 10.0)));
      }

      // Создаем выражения
      Expression<Pose3> T_wb(symbol_shorthand::X(impl_->pose_cnt));
      Expression<Point3> P_w(symbol_shorthand::L(lm_id));

      // 1. Используем правильное имя функции (transformTo вместо transform_to)
      // P_b = T_wb.inverse() * P_w
      Expression<Point3> P_b_predicted = transformTo(T_wb, P_w);

      // 2. Используем правильный порядок аргументов: (Шум, Измерение, Функция)
      impl_->graph.addExpressionFactor(impl_->meas_noise, local_point, P_b_predicted);
    }

    // --- 3. Оптимизация ---
    try {
      impl_->isam.update(impl_->graph, impl_->initial_estimates);
      impl_->graph.resize(0);
      impl_->initial_estimates.clear();

      Values result = impl_->isam.calculateEstimate();
      for (auto& lm : impl_->landmarks) {
        if (result.exists(symbol_shorthand::L(lm.id))) {
          lm.pos = result.at<Point3>(symbol_shorthand::L(lm.id));
        }
      }
    } catch (...) {
    }
  }

  PoseMatrix SemanticGraph::get_optimized_pose() const {
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