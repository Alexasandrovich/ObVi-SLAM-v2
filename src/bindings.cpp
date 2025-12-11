#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "SlamKernel.cpp"

namespace py = pybind11;

cv::Mat numpy_to_mat(py::array_t<uint8_t>& input) {
  py::buffer_info buf = input.request();
  cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (void*)buf.ptr);
  return mat.clone();
}

PYBIND11_MODULE(park_slam_cpp, m) {
py::class_<SlamKernel>(m, "SlamKernel")
.def(py::init<double, double, double, double>())
.def("process", [](SlamKernel& self, py::array_t<uint8_t> img, std::vector<std::tuple<int, float, float>> dets){
self.process(numpy_to_mat(img), dets);
})
.def("get_pose", &SlamKernel::get_pose)
.def("get_landmark_pos", &SlamKernel::get_landmark_pos);
}