./
├── CMakeLists.txt                  # Основной файл сборки C++ ядра
├── run.sh                          # Единый скрипт запуска (Docker wrapper)
├── README.md                       # Инструкция
│
├── config/                         # Все конфигурационные файлы
│   ├── config.yaml                 # Главный конфиг (пути, параметры логики)
│   ├── sensors.yaml                # Калибровка (K камеры, T_lidar_cam, высота)
│   └── glim/                       # Конфиги для GLIM
│       └── config_outdoor.json
│
├── data/                           # Папка для данных (mount volume)
│   ├── video.mp4                   # Видеопоток
│   └── lidar.mcap                  # ROS2 MCAP с топиком (/rslidar_points)
│
├── docker/                         # Окружение
│   ├── Dockerfile                  # Сборка образа (CUDA, GLIM, GTSAM)
│   ├── docker-compose.yml          # Параметры запуска
│   └── entrypoint.sh               # Скрипт инициализации внутри контейнера
│
├── include/                        # C++ Заголовки (Интерфейсы)
│   └── obvi/
│       ├── System.hpp              # Главный класс-фасад (виден из Python)
│       ├── types.hpp               # Общие структуры (Pose, Landmark)
│       ├── odometry/
│       │   ├── OdomBase.hpp        # Абстрактный интерфейс одометрии
│       │   └── GlimOdom.hpp        # Заголовок обертки над GLIM
│       └── mapping/
│           └── SemanticGraph.hpp   # Заголовок графа (GTSAM)
│
├── src/                            # C++ Реализация
│   ├── System.cpp                  # Реализация фасада
│   ├── bindings.cpp                # Pybind11 (создает модуль obvi_cpp)
│   ├── odometry/
│   │   └── GlimOdom.cpp            # Реализация связи с GLIM API
│   └── mapping/
│       └── SemanticGraph.cpp       # Реализация факторов GTSAM
│
├── python/                         # Python Frontend
│   ├── main.py                     # Точка входа (цикл обработки)
│   ├── core/
│   │   ├── loader.py               # Синхронизатор MP4 и MCAP
│   │   └── visualizer.py           # Отрисовка карты и кадра
│   └── perception/
│       ├── detector.py             # YOLOv8 (сегментация)
│       └── geometry.py             # Расчет глубины
│
└── thirdparty/                     # Внешние библиотеки (git submodules)
    ├── glim/                       # Lidar Odometry lib
    └── gtsam/                      # Factor Graph lib
