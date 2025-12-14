#!/bin/bash
set -e

# Активируем ROS2 Humble
source /opt/ros/humble/setup.bash

# Если есть локальный workspace (для разработки)
if [ -f "/app/install/setup.bash" ]; then
    source /app/install/setup.bash
fi

# Настройка путей для библиотек (чтобы Python нашел .so)
export PYTHONPATH="${PYTHONPATH}:/app/build:/app/python"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

# Выполняем команду, переданную в docker-compose
exec "$@"