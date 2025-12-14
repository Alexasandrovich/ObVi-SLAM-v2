#pragma once
#include "obvi/types.hpp"
#include <vector>
#include <memory>

namespace obvi {

  class SemanticGraph {
  public:
    SemanticGraph();
    ~SemanticGraph();

    // Обновление графа
    // odom_pose: текущая поза от GLIM (T_world_body)
    void update(const PoseMatrix& odom_pose, const std::vector<LandmarkObservation>& observations);

    // Получить текущую оптимизированную позу
    PoseMatrix get_optimized_pose() const;

    // Получить карту
    std::vector<MapObject> get_map() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
  };

} // namespace obvi