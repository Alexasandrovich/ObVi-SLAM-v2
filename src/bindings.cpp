#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>

// Include HEADERS only
#include "SlamKernel.hpp"
#include "OrbSlamWrapper.h"

namespace py = pybind11;

cv::Mat numpy_to_mat(py::array_t<uint8_t>& input) {
  py::buffer_info buf = input.request();
  cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (void*)buf.ptr);
  return mat.clone();
}

PYBIND11_MODULE(park_slam_cpp, m) {
py::class_<SlamKernel>(m, "SlamKernel")
.def(py::init<double, double, double, double>())
.def("add_frame_data", &SlamKernel::process_with_external_pose)
.def("get_landmark_pos", &SlamKernel::get_landmark_pos);

py::class_<OrbSlamWrapper>(m, "OrbSlam")
.def(py::init<std::string, std::string, bool>())
.def("track", [](OrbSlamWrapper& self, py::array_t<uint8_t> img, double timestamp){
cv::Mat mat = numpy_to_mat(img);
cv::Mat pose = self.track(mat, timestamp);
std::vector<double> pose_vec;
if (!pose.empty()) {
for(int i=0; i<4; i++) for(int j=0; j<4; j++) pose_vec.push_back(pose.at<double>(i,j));
}
return pose_vec;
});
}