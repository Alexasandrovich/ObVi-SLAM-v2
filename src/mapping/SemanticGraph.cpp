#include "obvi/mapping/SemanticGraph.hpp"

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
#include <gtsam/linear/NoiseModel.h>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

#include <list>

using namespace gtsam;

namespace obvi {

  struct TrackerParams {
    int min_hits_to_confirm = 90;    // Порог превращения в постоянный объект
    int max_misses_to_archive = 30;  // Порог "потери" объекта (перевод в архив)
    double match_dist_thresh = 2.0; // Дистанция ассоциации (метры)
  };

  struct SemanticGraph::Impl {
    ISAM2 isam;
    NonlinearFactorGraph graph;
    Values initial_estimates;

    Pose3 prev_pose;
    bool initialized = false;
    int pose_cnt = 0;
    int next_landmark_id = 0;

    // --- Состояния трека ---
    enum TrackState {
      CANDIDATE, // Еще не подтвержден (может быть шумом)
      ACTIVE,    // Подтвержден и находится в поле зрения
      ARCHIVED   // Был подтвержден, но пропал из виду (остается на карте навсегда)
    };

    struct Track {
      TrackState state = CANDIDATE;
      int class_id;
      Point3 global_pos;

      int hit_count = 1;
      int miss_count = 0;

      int graph_id = -1;     // ID в GTSAM (для ACTIVE и ARCHIVED)

      Point3 last_local_measurement;
      bool has_measurement_this_frame = false;
    };

    std::list<Track> tracks;
    TrackerParams tracker_params;

    noiseModel::Diagonal::shared_ptr odom_noise;
    noiseModel::Robust::shared_ptr meas_noise;

    Impl() {
      ISAM2Params params;
      params.relinearizeThreshold = 0.01; // Сделал чувствительнее
      params.relinearizeSkip = 1;
      isam = ISAM2(params);

      odom_noise = noiseModel::Diagonal::Sigmas((Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());

      // Huber loss для защиты от выбросов детекции
      auto gaussian = noiseModel::Isotropic::Sigma(3, 0.5);
      meas_noise = noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(1.345), gaussian);

      spdlog::set_level(spdlog::level::info);
    }

    // --- Логика Трекера ---
    void update_tracks(const Pose3& current_pose, const std::vector<LandmarkObservation>& observations) {
      // Сброс флагов
      for(auto& t : tracks) {
        t.has_measurement_this_frame = false;
      }

      // 1. Data Association
      std::vector<bool> obs_matched(observations.size(), false);

      for (size_t i = 0; i < observations.size(); ++i) {
        Point3 local_pt(observations[i].local_pos.x(), observations[i].local_pos.y(), observations[i].local_pos.z());
        Point3 global_pt = current_pose.transformFrom(local_pt);

        Track* best_track = nullptr;
        double min_dist = tracker_params.match_dist_thresh;

        for (auto& track : tracks) {
          // Мы матчимся только к АКТИВНЫМ или КАНДИДАТАМ.
          // ARCHIVED треки мы игнорируем (чтобы создавать новые объекты поверх старых, если те пропали)
          if (track.state == ARCHIVED) continue;

          if (track.class_id != observations[i].class_id) continue;
          if (track.has_measurement_this_frame) continue;

          double d = distance3(track.global_pos, global_pt);
          if (d < min_dist) {
            min_dist = d;
            best_track = &track;
          }
        }

        if (best_track) {
          obs_matched[i] = true;
          best_track->has_measurement_this_frame = true;
          best_track->hit_count++;
          best_track->miss_count = 0;
          best_track->last_local_measurement = local_pt;

          // Обновляем global_pos вручную ТОЛЬКО если объект еще не в графе.
          // Если он уже ACTIVE (в графе), мы доверяем GTSAM и не трогаем позицию тут.
          if (best_track->state == CANDIDATE) {
            double alpha = 0.7;
            best_track->global_pos = alpha * global_pt + (1.0 - alpha) * best_track->global_pos;
          }
        }
      }

      // 2. Создание новых треков
      for (size_t i = 0; i < observations.size(); ++i) {
        if (!obs_matched[i]) {
          Point3 local_pt(observations[i].local_pos.x(), observations[i].local_pos.y(), observations[i].local_pos.z());

          Track new_track;
          new_track.state = CANDIDATE;
          new_track.class_id = observations[i].class_id;
          new_track.global_pos = current_pose.transformFrom(local_pt); // Начальная гипотеза
          new_track.last_local_measurement = local_pt;
          new_track.has_measurement_this_frame = true;

          tracks.push_back(new_track);
        }
      }

      // 3. Управление жизненным циклом (State Machine)
      auto it = tracks.begin();
      while (it != tracks.end()) {
        if (!it->has_measurement_this_frame) {
          it->miss_count++;
        }

        // Логика удаления/архивации
        if (it->miss_count > tracker_params.max_misses_to_archive) {

          if (it->state == CANDIDATE) {
            // Если это был шум (не успел стать активным) -> Удаляем навсегда
            it = tracks.erase(it);
            continue;
          }
          else if (it->state == ACTIVE) {
            // Если был активным, переводим в АРХИВ.
            // Он останется в списке tracks, но больше не будет матчиться и обновляться.
            it->state = ARCHIVED;
            spdlog::info("Landmark L{} archived (lost tracking)", it->graph_id);
          }
        }

        // Логика активации (Candidate -> Active)
        if (it->state == CANDIDATE && it->hit_count >= tracker_params.min_hits_to_confirm) {
          it->state = ACTIVE;
          it->graph_id = next_landmark_id++; // Выдаем ID для GTSAM

          // Добавляем в граф
          initial_estimates.insert(symbol_shorthand::L(it->graph_id), it->global_pos);

          // Прайор при инициализации (пожестче, 1.0м, чтобы не улетал сразу)
          graph.add(PriorFactor<Point3>(symbol_shorthand::L(it->graph_id), it->global_pos,
                                        noiseModel::Isotropic::Sigma(3, 1.0)));

          spdlog::info("Confirmed new Landmark L{} at [{:.2f}, {:.2f}, {:.2f}]",
                       it->graph_id, it->global_pos.x(), it->global_pos.y(), it->global_pos.z());
        }

        ++it;
      }
    }
  };

  SemanticGraph::SemanticGraph() : impl_(std::make_unique<Impl>()) {}
  SemanticGraph::~SemanticGraph() = default;

  void SemanticGraph::update(const PoseMatrix& odom_pose_mat, const std::vector<LandmarkObservation>& observations) {
    Pose3 current_pose(odom_pose_mat);

    // --- 1. Odometry Update ---
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

    // --- 2. Tracking Update ---
    impl_->update_tracks(current_pose, observations);

    // --- 3. Add Factors to Graph ---
    for (auto& track : impl_->tracks) {
      // Добавляем факторы ТОЛЬКО для АКТИВНЫХ треков, которые видны сейчас
      if (track.state == Impl::ACTIVE && track.has_measurement_this_frame) {

        Expression<Pose3> T_wb(symbol_shorthand::X(impl_->pose_cnt));
        Expression<Point3> P_w(symbol_shorthand::L(track.graph_id));
        Expression<Point3> P_b_predicted = transformTo(T_wb, P_w);

        impl_->graph.addExpressionFactor(impl_->meas_noise, track.last_local_measurement, P_b_predicted);
      }
    }

    // --- 4. Optimization ---
    try {
      impl_->isam.update(impl_->graph, impl_->initial_estimates);
      impl_->graph.resize(0);
      impl_->initial_estimates.clear();

      Values result = impl_->isam.calculateEstimate();

      // Обновляем позиции треков ИЗ РЕЗУЛЬТАТОВ ISAM
      // Это решает проблему рассинхрона
      for (auto& track : impl_->tracks) {
        if ((track.state == Impl::ACTIVE || track.state == Impl::ARCHIVED) &&
            result.exists(symbol_shorthand::L(track.graph_id))) {

          track.global_pos = result.at<Point3>(symbol_shorthand::L(track.graph_id));
        }
      }

    } catch (...) {}
  }

  PoseMatrix SemanticGraph::get_optimized_pose() const {
    return impl_->prev_pose.matrix();
  }

  std::vector<MapObject> SemanticGraph::get_map() const {
    std::vector<MapObject> map_out;
    // Выводим И Активные, И Архивные (все, что есть в графе)
    for (const auto& track : impl_->tracks) {
      if (track.state == Impl::ACTIVE || track.state == Impl::ARCHIVED) {
        map_out.push_back({track.graph_id, track.class_id,
                           Eigen::Vector3d(track.global_pos.x(), track.global_pos.y(), track.global_pos.z())});
      }
    }
    return map_out;
  }

} // namespace obvi