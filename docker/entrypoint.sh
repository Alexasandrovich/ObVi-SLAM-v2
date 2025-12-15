#!/bin/bash
set -e

echo "========================================"
echo "   ObVi-SLAM v2: Container Start"
echo "========================================"

# --- 1. Настройка Окружения ---

# Активируем ROS2 Humble
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
fi

# Настройка путей для библиотек
# PYTHONPATH: чтобы питон видел наш скомпилированный модуль (build) и исходники (python)
export PYTHONPATH="/app/build:/app/python:$PYTHONPATH"

# LD_LIBRARY_PATH: чтобы система видела библиотеки GLIM/GTSAM в /usr/local/lib
export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"


# --- 2. Сборка Проекта (Build) ---

echo "--- [1/2] Checking Build Directory... ---"
if [ ! -d "/app/build" ]; then
    mkdir -p /app/build
fi

cd /app/build

# Запускаем cmake только если нет Makefile (или если нужно переконфигурировать)
# Это ускоряет перезапуск
if [ ! -f "Makefile" ]; then
    echo "--- Configuring CMake... ---"
    cmake .. -DCMAKE_BUILD_TYPE=Release
fi

echo "--- [2/2] Compiling C++ Core... ---"
# -j$(nproc) использует все ядра процессора
make -j$(nproc)


# --- 3. Запуск (Run) ---

echo "--- [3/3] Starting Application... ---"
cd /app

# Если в docker-compose передана команда (например /bin/bash), выполняем её.
# Если нет (по умолчанию) - запускаем main.py
if [ "$#" -gt 0 ]; then
    exec "$@"
else
    # -u отключает буферизацию вывода (чтобы логи видели сразу)
    exec python3 -u python/main.py
fi