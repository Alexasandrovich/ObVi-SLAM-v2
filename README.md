Система семантического картографирования, объединяющая одометрию робота, компьютерное зрение и графовую оптимизацию.

## Структура проекта

```
./
├── CMakeLists.txt          # Сборка C++ ядра
├── run.sh                  # Скрипт запуска (Docker wrapper)
├── README.md
│
├── config/                 # Конфигурация
│   ├── config.yaml         # Основные параметры системы
│   ├── sensors.yaml        # Калибровка сенсоров (K, T_lidar_cam)
│   └── glim/
│       └── config_outdoor.json
│
├── data/                   # Входные данные (монтируется как volume)
│   ├── video.mp4           # Видеопоток
│   └── lidar.mcap          # ROS2 MCAP (/rslidar_points)
│
├── docker/                 # Окружение
│   ├── Dockerfile          # Образ (CUDA, GLIM, GTSAM)
│   ├── docker-compose.yml
│   └── entrypoint.sh
│
├── include/obvi/           # C++ заголовки
│   ├── System.hpp          # Главный фасад
│   ├── types.hpp           # Pose, Landmark
│   ├── odometry/
│   │   ├── OdomBase.hpp    # Интерфейс одометрии
│   │   └── GlimOdom.hpp    # Обёртка GLIM
│   └── mapping/
│       └── SemanticGraph.hpp
│
├── src/                    # C++ реализация
│   ├── System.cpp
│   ├── bindings.cpp        # Pybind11 → obvi_cpp
│   ├── odometry/
│   │   └── GlimOdom.cpp
│   └── mapping/
│       └── SemanticGraph.cpp
│
├── python/                 # Python frontend
│   ├── main.py             # Точка входа
│   ├── core/
│   │   ├── loader.py       # Синхронизация MP4 + MCAP
│   │   └── visualizer.py
│   └── perception/
│       ├── detector.py     # YOLOv8
│       └── geometry.py     # Расчёт глубины
│
└── thirdparty/             # Git submodules
    ├── glim/
    └── gtsam/
```

## Быстрый старт

```bash
# Сборка и запуск
./run.sh
```

## Зависимости

- CUDA 11.8+
- GLIM (LiDAR odometry)
- GTSAM (factor graph optimization)
- YOLOv8 (semantic segmentation)
- nvidia container-toolkit (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
