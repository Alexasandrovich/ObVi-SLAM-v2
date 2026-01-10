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
#include <gtsam/linear/NoiseModel.h>

// Logging
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

#include <list>

using namespace gtsam;

namespace obvi {

  // --- Настройки Трекера и Графа ---
  struct TrackerParams {
    // Жизненный цикл
    int min_hits_to_confirm = 90;    // Сколько кадров видеть, чтобы добавить в граф
    int max_misses_to_archive = 60;  // Сколько кадров не видеть, чтобы отправить в архив
    double match_dist_thresh = 2.0;  // Метры для ассоциации (NN)

    // Keyframing (чтобы не спамить факторами)
    double kf_trans_thresh = 0.5;    // Добавлять фактор, если робот сдвинулся на 0.5м
    double kf_rot_thresh = 0.2;      // Добавлять фактор, если робот повернулся на ~11 град
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
      CANDIDATE, // Копим статистику, фильтруем шум
      ACTIVE,    // В графе, обновляем измерениями
      ARCHIVED   // В графе, но потерян из виду (храним для замыкания петель)
    };

    struct Track {
      TrackState state = CANDIDATE;
      int class_id;
      Point3 global_pos;

      int hit_count = 1;
      int miss_count = 0;
      int graph_id = -1;     // ID лендмарка в GTSAM

      // Данные последнего кадра
      Point3 last_local_measurement;
      bool has_measurement_this_frame = false;

      // Keyframing: запоминаем позу, когда последний раз добавляли фактор
      Pose3 pose_at_last_factor;
      bool has_added_factor = false;
    };

    std::list<Track> tracks;
    TrackerParams tracker_params;

    noiseModel::Diagonal::shared_ptr odom_noise;
    noiseModel::Robust::shared_ptr meas_noise;

    Impl() {
      ISAM2Params params;
      params.relinearizeThreshold = 0.01; // Чувствительность к изменениям
      params.relinearizeSkip = 1;
      isam = ISAM2(params);

      // Шум одометрии (X, Y, Z, Roll, Pitch, Yaw)
      odom_noise = noiseModel::Diagonal::Sigmas((Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());

      // Шум измерений + Huber Loss (защита от "выбросов" глубины)
      // Huber линейно штрафует большие ошибки, а не квадратично
      auto gaussian = noiseModel::Isotropic::Sigma(3, 0.5);
      meas_noise = noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(1.345), gaussian);

      spdlog::set_level(spdlog::level::info);
    }

    // --- Логика Трекера (Frontend) ---
    void update_tracks(const Pose3& current_pose, const std::vector<LandmarkObservation>& observations) {
      // 1. Сброс флагов текущего кадра
      for(auto& t : tracks) {
        t.has_measurement_this_frame = false;
      }

      // 2. Data Association (Nearest Neighbor)
      std::vector<bool> obs_matched(observations.size(), false);

      for (size_t i = 0; i < observations.size(); ++i) {
        Point3 local_pt(observations[i].local_pos.x(), observations[i].local_pos.y(), observations[i].local_pos.z());
        Point3 global_pt = current_pose.transformFrom(local_pt);

        Track* best_track = nullptr;
        double min_dist = tracker_params.match_dist_thresh;

        for (auto& track : tracks) {
          // Игнорируем архивные, чтобы позволить создание новых треков на том же месте,
          // если старый объект "ушел" (для динамических сцен) или если это реинициализация.
          // (Для полноценного Loop Closure здесь нужна логика проверки ARCHIVED + ReID)
          if (track.state == ARCHIVED) continue;

          if (track.class_id != observations[i].class_id) continue;
          if (track.has_measurement_this_frame) continue; // 1 трек - 1 измерение

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

          // Обновляем позицию вручную ТОЛЬКО для кандидатов.
          // Для ACTIVE позицию определяет GTSAM.
          if (best_track->state == CANDIDATE) {
            double alpha = 0.7; // Сглаживание
            best_track->global_pos = alpha * global_pt + (1.0 - alpha) * best_track->global_pos;
          }
        }
      }

      // 3. Создание новых треков
      for (size_t i = 0; i < observations.size(); ++i) {
        if (!obs_matched[i]) {
          Point3 local_pt(observations[i].local_pos.x(), observations[i].local_pos.y(), observations[i].local_pos.z());

          Track new_track;
          new_track.state = CANDIDATE;
          new_track.class_id = observations[i].class_id;
          new_track.global_pos = current_pose.transformFrom(local_pt);
          new_track.last_local_measurement = local_pt;
          new_track.has_measurement_this_frame = true;
          // Инициализируем pose_at_last_factor текущей позой, чтобы отсчет пошел отсюда
          new_track.pose_at_last_factor = current_pose;

          tracks.push_back(new_track);
        }
      }

      // 4. Управление жизненным циклом (State Machine)
      auto it = tracks.begin();
      while (it != tracks.end()) {
        if (!it->has_measurement_this_frame) {
          it->miss_count++;
        }

        // Удаление или Архивация
        if (it->miss_count > tracker_params.max_misses_to_archive) {
          if (it->state == CANDIDATE) {
            // Шум -> Удаляем
            it = tracks.erase(it);
            continue;
          }
          else if (it->state == ACTIVE) {
            // Был подтвержден -> В Архив (остается на карте)
            it->state = ARCHIVED;
            spdlog::info("Landmark L{} archived (lost tracking)", it->graph_id);
          }
        }

        // Активация (Candidate -> Active)
        if (it->state == CANDIDATE && it->hit_count >= tracker_params.min_hits_to_confirm) {
          it->state = ACTIVE;
          it->graph_id = next_landmark_id++;

          // Вставка в граф
          initial_estimates.insert(symbol_shorthand::L(it->graph_id), it->global_pos);

          // Начальный Prior (якорь)
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

    // --- 1. Odometry (Backend) ---
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

    // --- 2. Tracking (Frontend) ---
    impl_->update_tracks(current_pose, observations);

    // --- 3. Factor Creation (Middle-end) ---
    for (auto& track : impl_->tracks) {
      // Работаем только с АКТИВНЫМИ, которые видны сейчас
      if (track.state == Impl::ACTIVE && track.has_measurement_this_frame) {

        bool need_add_factor = false;

        if (!track.has_added_factor) {
          // Первый раз для этого объекта
          need_add_factor = true;
        } else {
          // KEYFRAMING: Проверяем, достаточно ли сдвинулся робот
          Pose3 delta = track.pose_at_last_factor.between(current_pose);
          double d_trans = delta.translation().norm();
          double d_rot = delta.rotation().rpy().norm(); // Приближенно угол

          if (d_trans > impl_->tracker_params.kf_trans_thresh ||
              d_rot > impl_->tracker_params.kf_rot_thresh) {
            need_add_factor = true;
          }
        }

        if (need_add_factor) {
          spdlog::info("Add landmark #{} measurement", track.graph_id);
          // Smart Factor Projection: P_body = T_wb^(-1) * P_world
          Expression<Pose3> T_wb(symbol_shorthand::X(impl_->pose_cnt));
          Expression<Point3> P_w(symbol_shorthand::L(track.graph_id));
          Expression<Point3> P_b_predicted = transformTo(T_wb, P_w);

          impl_->graph.addExpressionFactor(impl_->meas_noise, track.last_local_measurement, P_b_predicted);

          // Обновляем состояние keyframe
          track.pose_at_last_factor = current_pose;
          track.has_added_factor = true;
        }
      }
    }

    // --- 4. Optimization ---
    try {
      impl_->isam.update(impl_->graph, impl_->initial_estimates);
      impl_->graph.resize(0);
      impl_->initial_estimates.clear();

      Values result = impl_->isam.calculateEstimate();

      // --- 5. Sync ---
      // Обновляем позиции треков ИЗ РЕЗУЛЬТАТОВ ISAM (Обратная связь)
      for (auto& track : impl_->tracks) {
        if ((track.state == Impl::ACTIVE || track.state == Impl::ARCHIVED) &&
            result.exists(symbol_shorthand::L(track.graph_id))) {

          track.global_pos = result.at<Point3>(symbol_shorthand::L(track.graph_id));
        }
      }

    } catch (const std::exception& e) {
      spdlog::error("ISAM Update Error: {}", e.what());
    }
  }

  PoseMatrix SemanticGraph::get_optimized_pose() const {
    return impl_->prev_pose.matrix();
  }

  std::vector<MapObject> SemanticGraph::get_map() const {
    std::vector<MapObject> map_out;
    for (const auto& track : impl_->tracks) {
      // Возвращаем и активные (зеленые), и архивные (серые/старые) объекты
      if (track.state == Impl::ACTIVE || track.state == Impl::ARCHIVED) {
        map_out.push_back({track.graph_id, track.class_id,
                           Eigen::Vector3d(track.global_pos.x(), track.global_pos.y(), track.global_pos.z())});
      }
    }
    return map_out;
  }

} // namespace obvi