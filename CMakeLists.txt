cmake_minimum_required(VERSION 3.10)
project(park_slam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(GTSAM REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
        OUTPUT_VARIABLE PYBIND_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# ВАЖНО: Мы компилируем только bindings.cpp (он внутри включает SlamKernel.cpp)
# и VisualOdometry.cpp (он сам по себе).
# SlamKernel.cpp НЕ добавляем в список, так как он заинклужен в bindings.
set(SOURCES
        src/bindings.cpp
        src/VisualOdometry.cpp
)

add_library(park_slam_cpp SHARED ${SOURCES})

target_include_directories(park_slam_cpp PRIVATE
        src/
        ${GTSAM_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${Python3_INCLUDE_DIRS}
        ${PYBIND_INCLUDE_DIR}
)

target_compile_options(park_slam_cpp PRIVATE -O3)

target_link_libraries(park_slam_cpp PRIVATE
        gtsam
        ${OpenCV_LIBS}
        Python3::Python
)

set_target_properties(park_slam_cpp PROPERTIES PREFIX "" SUFFIX ".so")