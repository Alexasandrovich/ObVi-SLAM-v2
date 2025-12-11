FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# 1. Установка системных зависимостей
# Добавил wget и ca-certificates, иногда git ругается на сертификаты
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    libboost-all-dev \
    libtbb-dev \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# 2. Сборка GTSAM (Fix: добавил mkdir build)
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

# 4. Сборка нашего проекта
WORKDIR /app
COPY . .

# Чистим старый build если был скопирован, и собираем заново
RUN rm -rf build && mkdir build && cd build && \
    cmake .. && make -j$(nproc)

ENV PYTHONPATH="${PYTHONPATH}:/app/build"