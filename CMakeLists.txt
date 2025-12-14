cmake_minimum_required(VERSION 3.16)
project(obvi_slam_v3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Dependencies ---
find_package(CUDA REQUIRED)
find_package(GTSAM REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# GLIM и его зависимости
find_package(glim REQUIRED)
find_package(gtsam_points REQUIRED)
find_package(Iridescence REQUIRED)

# --- Include Directories ---
include_directories(
        include
        ${GTSAM_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${PYTHON_INCLUDE_DIRS}
        # GLIM инклуды подтянутся через таргет glim::glim
)

add_definitions(${PCL_DEFINITIONS})

# --- Build Core Library ---
add_library(obvi_core SHARED
        src/System.cpp
        src/odometry/GlimOdom.cpp
        src/mapping/SemanticGraph.cpp
)

target_link_libraries(obvi_core PUBLIC
        glim::glim
        gtsam_points::gtsam_points
        gtsam
        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
)

# --- Build Python Bindings ---
execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
        OUTPUT_VARIABLE PYBIND_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
include_directories(${PYBIND_INCLUDE_DIR})

add_library(obvi_cpp SHARED src/bindings.cpp)

target_link_libraries(obvi_cpp PRIVATE
        obvi_core
        Python3::Python
)

set_target_properties(obvi_cpp PROPERTIES PREFIX "" SUFFIX ".so")