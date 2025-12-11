FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# 1. System Dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget unzip \
    libboost-all-dev libtbb-dev \
    libopencv-dev python3-opencv \
    libeigen3-dev \
    libglew-dev \
    libssl-dev \
    libpng-dev libjpeg-dev libx11-dev libxext-dev \
    libepoxy-dev \
    libxcb-xinerama0 libxcb-cursor0 libx11-xcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

# 2. Build Pangolin
# Flags disable OpenEXR (problematic) and suppress warnings (-w)
RUN git clone https://github.com/stevenlovegrove/Pangolin.git && \
    cd Pangolin && \
    mkdir build && cd build && \
    cmake -DMSVC_SYMBOLS=OFF -DPROJECT_VERSION_SUFFIX=2 \
          -DCMAKE_CXX_FLAGS="-w" \
          -DCMAKE_DISABLE_FIND_PACKAGE_OpenEXR=TRUE \
          .. && \
    make -j$(nproc) && make install && \
    cd /tmp && rm -rf Pangolin

# 3. Build ORB-SLAM3
RUN git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git orbslam3 && \
    cd orbslam3 && \
    # FIX 1: Add unistd.h for usleep (C++ compilation error)
    sed -i '1i #include <unistd.h>' src/System.cc && \
    sed -i '1i #include <unistd.h>' src/Tracking.cc && \
    sed -i '1i #include <unistd.h>' src/LocalMapping.cc && \
    sed -i '1i #include <unistd.h>' src/LoopClosing.cc && \
    sed -i '1i #include <unistd.h>' src/Viewer.cc && \
    sed -i '1i #include <unistd.h>' src/Optimizer.cc && \
    # FIX 2: Downgrade OpenCV requirement from 4.4 to 4.2 (Ubuntu 20.04 default)
    sed -i 's/find_package(OpenCV 4.4)/find_package(OpenCV 4.2)/g' CMakeLists.txt && \
    # FIX 3: Force C++14 standard (fixes sigslot errors: enable_if_t, decay_t, etc.)
    sed -i 's/-std=c++11/-std=c++14/g' CMakeLists.txt && \
    chmod +x build.sh && \
    ./build.sh && \
    cd /tmp

# 4. Download Vocabulary
RUN mkdir -p /app/vocab && \
    wget https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/master/Vocabulary/ORBvoc.txt.tar.gz -O /app/vocab/ORBvoc.txt.tar.gz && \
    cd /app/vocab && tar -xf ORBvoc.txt.tar.gz && rm ORBvoc.txt.tar.gz

# 5. GTSAM
RUN git clone https://github.com/borglab/gtsam.git && \
    cd gtsam && git checkout 4.2a9 && \
    mkdir build && cd build && \
    cmake -DGTSAM_USE_SYSTEM_EIGEN=ON -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
          -DGTSAM_BUILD_PYTHON=OFF -DGTSAM_BUILD_EXAMPLES=OFF -DGTSAM_BUILD_TESTS=OFF .. && \
    make -j$(nproc) install && \
    cd /tmp && rm -rf gtsam

# 6. Python Dependencies
RUN pip install --no-cache-dir "numpy<2.0" ultralytics opencv-python-headless pybind11 pyyaml

WORKDIR /app
COPY . .

# 7. Environment variable for ORB-SLAM3
ENV ORB_SLAM3_ROOT_DIR="/tmp/orbslam3"

# Build Project
RUN rm -rf build && mkdir build && cd build && \
    cmake .. && make -j$(nproc)

ENV PYTHONPATH="${PYTHONPATH}:/app/build"