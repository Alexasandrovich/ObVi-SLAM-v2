#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "obvi/System.hpp"

namespace py = pybind11;

PYBIND11_MODULE(obvi_cpp, m) {
  m.doc() = "ObVi-SLAM v3 Core Module";

  py::class_<obvi::System>(m, "System")
    .def(py::init<const std::string&,
                  const std::string&,
                  const std::vector<double>&>(),
            py::arg("config_file"),
            py::arg("glim_config_path"),
            py::arg("lidar_extrinsics"))

    .def("process", &obvi::System::process,
         "Process frame: timestamp, lidar_points (flat), object_classes, object_coords_3d")

    .def("get_pose", &obvi::System::get_current_pose_vec,
         "Get current robot pose as [x, y, z, qx, qy, qz, qw]")

    .def("get_map", &obvi::System::get_map_objects,
         "Get map objects as list of [id, class, x, y, z]");
}