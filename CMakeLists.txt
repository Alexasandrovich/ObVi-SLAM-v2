cmake_minimum_required(VERSION 3.10)
project(park_slam)

# Global settings
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(GTSAM REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Pangolin REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

# ORB-SLAM3 Paths
set(ORB_SLAM3_DIR $ENV{ORB_SLAM3_ROOT_DIR})

# --- TARGET 1: ORB-SLAM Wrapper (C++14) ---
add_library(orb_wrapper_lib STATIC src/OrbSlamWrapper.cpp)

# FORCE C++14 for this library to avoid g2o/Eigen errors
set_target_properties(orb_wrapper_lib PROPERTIES CXX_STANDARD 14)

target_include_directories(orb_wrapper_lib PUBLIC
        src/
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
        ${GLEW_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIR}
        ${ORB_SLAM3_DIR}
        ${ORB_SLAM3_DIR}/include
        ${ORB_SLAM3_DIR}/include/CameraModels
        ${ORB_SLAM3_DIR}/Thirdparty/Sophus
)

# Link ORB-SLAM3 library to our wrapper
find_library(ORB_SLAM3_LIB NAMES ORB_SLAM3 PATHS ${ORB_SLAM3_DIR}/lib NO_DEFAULT_PATH)
target_link_libraries(orb_wrapper_lib PRIVATE
        ${ORB_SLAM3_LIB}
        ${OpenCV_LIBS}
        ${Pangolin_LIBRARIES}
        ${GLEW_LIBRARIES}
        ${OPENGL_LIBRARIES}
)

# --- TARGET 2: Main Python Module (C++17 for GTSAM) ---
execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
        OUTPUT_VARIABLE PYBIND_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_library(park_slam_cpp SHARED src/bindings.cpp)

# FORCE C++17 for GTSAM code
set_target_properties(park_slam_cpp PROPERTIES CXX_STANDARD 17)

target_include_directories(park_slam_cpp PRIVATE
        src/
        ${GTSAM_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${Python3_INCLUDE_DIRS}
        ${PYBIND_INCLUDE_DIR}
)

# Link everything together
target_link_libraries(park_slam_cpp PRIVATE
        gtsam
        orb_wrapper_lib
        ${OpenCV_LIBS}
        Python3::Python
        ${Pangolin_LIBRARIES}
        ${GLEW_LIBRARIES}
        ${OPENGL_LIBRARIES}
)

set_target_properties(park_slam_cpp PROPERTIES PREFIX "" SUFFIX ".so")