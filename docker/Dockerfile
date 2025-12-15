FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ARG GTSAM_POINTS_REPOPATH=https://github.com/koide3/gtsam_points
ARG GTSAM_POINTS_BRANCH=master
ARG GLIM_REPOPATH=https://github.com/koide3/glim
ARG GLIM_BRANCH=master

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

# 1. Base Setup
RUN apt-get update && apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/UTC /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

# 2. ROS2 Humble
RUN apt-get update && apt-get install -y curl gnupg2 lsb-release && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    sh -c 'echo "deb [arch=amd64 allow-insecure=yes] http://packages.ros.org/ros2/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros2-latest.list' && \
    apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-humble-pcl-conversions \
    ros-humble-pcl-msgs \
    ros-humble-rosbag2-storage-mcap \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# 3. System Deps
RUN apt-get update && apt-get install -y \
    git cmake build-essential wget unzip \
    libomp-dev libboost-all-dev libmetis-dev libfmt-dev libspdlog-dev \
    libglm-dev libglfw3-dev \
    libpng-dev libjpeg-dev libtiff-dev \
    libpcl-dev \
    libeigen3-dev  \
    python3-pip python3-dev \
    libxcb-xinerama0 libxcb-cursor0 libx11-xcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/3rd_party

# 4. GTSAM (версия 4.3a0)
RUN git clone https://github.com/borglab/gtsam && \
    cd gtsam && \
    git checkout 4.3a0 && \
    mkdir build && cd build && \
    cmake .. -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
             -DGTSAM_BUILD_TESTS=OFF \
             -DGTSAM_WITH_TBB=OFF \
             -DGTSAM_USE_SYSTEM_EIGEN=ON \
             -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF && \
    make -j$(nproc) && \
    make install && \
    cd /root/3rd_party && rm -rf gtsam

# 5. Iridescence (Визуализация)
RUN git clone https://github.com/koide3/iridescence --recursive && \
    cd iridescence && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install && \
    cd /root/3rd_party && rm -rf iridescence

# 6. gtsam_points (CUDA)
RUN git clone https://github.com/koide3/gtsam_points && \
    cd gtsam_points && \
    mkdir build && cd build && \
    cmake .. -DBUILD_WITH_CUDA=ON && \
    make -j$(nproc) && \
    make install && \
    cd /root/3rd_party && rm -rf gtsam_points

# 7. GLIM
RUN git clone https://github.com/koide3/glim && \
    cd glim && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DBUILD_WITH_CUDA=ON \
             -DBUILD_WITH_VIEWER=ON \
             -DBUILD_WITH_MARCH_NATIVE=OFF \
             -DBUILD_WITH_OPENCV=OFF && \
    make -j$(nproc) && \
    make install && \
    # Копируем конфиги GLIM в системную папку, чтобы враппер мог их найти
    mkdir -p /usr/local/share/glim && \
    cp -r ../config /usr/local/share/glim/ && \
    cd /root/3rd_party && rm -rf glim

# Обновляем кэш библиотек
RUN ldconfig

# 8. Python Deps
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir \
    ultralytics \
    opencv-python-headless \
    pybind11 \
    pyyaml \
    mcap-ros2-support \
    open3d \
    numpy

WORKDIR /app
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh