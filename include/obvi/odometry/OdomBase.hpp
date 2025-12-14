#pragma once
#include "obvi/types.hpp"
#include <vector>

namespace obvi {

  class OdomBase {
  public:
    virtual ~OdomBase() = default;

    // Вставка данных (timestamp + сырые точки x,y,z,intensity)
    virtual void insert_cloud(double timestamp, const float* data, size_t num_points) = 0;

    // Получить текущую позу (Thread-safe)
    virtual PoseMatrix get_pose() = 0;

    // Сброс (если нужно)
    virtual void reset() {}
  };

} // namespace obvi