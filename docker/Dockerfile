FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# 1. Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    libboost-all-dev \
    libtbb-dev \
    libopencv-dev \
    libeigen3-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# 2. Сборка GTSAM
WORKDIR /tmp
RUN git clone https://github.com/borglab/gtsam.git && \
    cd gtsam && \
    git checkout 4.2a9 && \
    mkdir build && cd build && \
    cmake -DGTSAM_USE_SYSTEM_EIGEN=ON \
          -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
          -DGTSAM_BUILD_PYTHON=OFF \
          -DGTSAM_BUILD_EXAMPLES=OFF \
          -DGTSAM_BUILD_TESTS=OFF \
          .. && \
    make -j$(nproc) install && \
    cd / && rm -rf /tmp/gtsam

# 3. Python зависимости
RUN pip install --no-cache-dir ultralytics opencv-python-headless pybind11 pyyaml

# 4. Сборка проекта
WORKDIR /app
COPY . .

# Очищаем и собираем заново
RUN rm -rf build && mkdir build && cd build && \
    cmake .. && make -j$(nproc)

ENV PYTHONPATH="${PYTHONPATH}:/app/build"