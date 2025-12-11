obvi-slam-ng/
├── config/
│   ├── config.yaml          # Основные настройки (камера, пути весов)
│   └── logger.yaml          # Настройки логирования (spdlog)
├── data/                    # Сюда маунтим датасеты (EuRoC, TUM)
├── docker/
│   ├── Dockerfile           # Multi-stage build (CUDA support)
│   └── entrypoint.sh
├── src/                     # C++ Core (Backend)
│   ├── factors/             # Кастомные факторы GTSAM (ObjectFactor)
│   ├── map/                 # Структуры карты (Landmarks, Keyframes)
│   ├── optimizer/           # Обертка над GTSAM ISAM2
│   ├── bindings.cpp         # Pybind11 интерфейс
│   └── CMakeLists.txt
├── python/                  # Python Frontend
│   ├── obvi_slam/
│   │   ├── detectors/       # Абстракции для NN (YOLO, Detic)
│   │   ├── feature_extractors/ # ORB, SuperPoint
│   │   └── system.py        # Основной класс, связывающий Frontend и Backend
│   └── main.py              # Точка входа
├── thirdparty/              # GTSAM, nlohmann_json (через git submodule)
├── docker-compose.yml
├── run.sh                   # Единая точка запуска
└── README.md
