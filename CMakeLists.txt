cmake_minimum_required(VERSION 3.10)
project(park_slam)
set(CMAKE_CXX_STANDARD 17)

find_package(GTSAM REQUIRED)
find_package(OpenCV REQUIRED)

execute_process(COMMAND python3 -m pybind11 --includes OUTPUT_VARIABLE PYBIND_INCLUDES)
string(STRIP ${PYBIND_INCLUDES} PYBIND_INCLUDES)

add_library(park_slam_cpp SHARED src/bindings.cpp)
target_include_directories(park_slam_cpp PRIVATE src/ ${GTSAM_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_compile_options(park_slam_cpp PRIVATE ${PYBIND_INCLUDES} -O3)
target_link_libraries(park_slam_cpp PRIVATE gtsam ${OpenCV_LIBS})
set_target_properties(park_slam_cpp PROPERTIES PREFIX "" SUFFIX ".so")