import sys
import yaml
import os
import cv2

# Путь к скомпилированной библиотеке
sys.path.append("/app/build")
import obvi_cpp

from core.dataset_loader import DatasetLoader
from core.visualizer import Visualizer
from perception.detector import ObjectDetector
from perception.geometry import VisualGeometry

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("=== ObVi-SLAM v2 Starting ===")

    # Конфиги
    cfg = load_config("/app/config/config.yaml")
    sensor_cfg = load_config("/app/config/sensors.yaml")

    # 1. Инициализация C++ ядра
    glim_config_path = "/app/config/glim"
    system = obvi_cpp.System(config_file="", glim_config_path=glim_config_path)

    # 2. Perception (Python)
    detector = ObjectDetector(cfg['ai']['model_path'], conf=cfg['ai']['conf_threshold'])

    geo = VisualGeometry(
        fx=sensor_cfg['camera']['fx'],
        fy=sensor_cfg['camera']['fy'],
        cx=sensor_cfg['camera']['cx'],
        cy=sensor_cfg['camera']['cy'],
        camera_height=sensor_cfg['camera']['height']
    )

    # 3. Data Loader
    loader = DatasetLoader(
        video_path=cfg['system']['video_path'],
        mcap_path=cfg['system']['mcap_path'],
        lidar_topic=sensor_cfg['lidar']['topic']
    )

    # 4. Visualizer
    viz = Visualizer()

    while True:
        # Чтение синхронизированной пары (Видео + Лидар)
        frame, lidar_data, timestamp = loader.next_frame()

        if frame is None:
            print("End of dataset.")
            break

        # --- 1. Perception Step ---
        detections = detector.detect(frame)

        obs_classes = []
        obs_coords = []

        for det in detections:
            u, v = det['keypoint']
            # Магия: 2D -> 3D без лидара
            pos_3d = geo.pixel_to_3d_ground(u, v)

            if pos_3d is not None:
                obs_classes.append(det['class_id'])
                obs_coords.append(pos_3d)

        # --- 2. Backend Step (C++) ---
        # lidar_data - это numpy array, pybind11 сам конвертирует в std::vector
        # obs_coords - это список списков, конвертируется в std::vector<std::vector<double>>

        # Если лидара нет для кадра (рассинхрон), шлем пустой список, GLIM экстраполирует
        lidar_flat = lidar_data if lidar_data is not None else []

        system.process(timestamp, lidar_flat, obs_classes, obs_coords)

        # --- 3. Visualization ---
        pose = system.get_pose() # [x, y, z, ...]
        map_objs = system.get_map() # [[id, cls, x, y, z], ...]

        vis_frame, vis_map = viz.draw(frame, pose, map_objs, detections)

        cv2.imshow("Camera", cv2.resize(vis_frame, (960, 540)))
        cv2.imshow("Map", vis_map)

        if cv2.waitKey(1) == 27:
            break

    loader.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()