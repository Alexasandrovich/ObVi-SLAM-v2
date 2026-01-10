#pragma once
#include "obvi/odometry/OdomBase.hpp"
#include <memory>
#include <string>
#include <vector>

namespace obvi {

  class GlimOdom : public OdomBase {
  public:
    // config_path: путь к папке с конфигами (где лежит config_odometry_ct.json)
    GlimOdom(const std::string& config_path, const std::vector<double>& extrinsics);
    ~GlimOdom() override;

    void insert_cloud(double timestamp, const float* data, size_t num_points) override;
    PoseMatrix get_pose() override;

  private:
    struct Impl; // Скрытая реализация
    std::unique_ptr<Impl> impl_;
  };

} // namespace obvi