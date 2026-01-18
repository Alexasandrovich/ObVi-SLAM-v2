import sys
import yaml
import os
import cv2
import time
import numpy as np
from common.profiler import ScopedTimer

sys.path.append("/app/build")
import obvi_cpp

from core.dataset_loader import DatasetLoader
from core.visualizer import Visualizer
from core.vis_3d import Visualizer3D
from perception.pipeline import AgnosticPerception

def load_config(path):
    if not os.path.exists(path):
        print(f"ERROR: Config file not found: {path}")
        return None
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("=== ObVi-SLAM v2 Starting ===")

    # 1. Загружаем основной конфиг (Python YAML)
    cfg = load_config("/app/config/config.yaml")
    if cfg is None: return

    # 2. Инициализация C++ (GLIM + GTSAM)
    # Путь к папке с конфигами GLIM
    glim_config_path = "/app/config/glim"
    lidar_ext = cfg['system'].get('lidar_extrinsics', [0.0]*6)
    print(f"Lidar extrinsics: {lidar_ext}")
    system = obvi_cpp.System(config_file="",
                             glim_config_path=glim_config_path,
                             lidar_extrinsics=lidar_ext)

    # 3. Perception
    print("Loading AI models...")
    perception = AgnosticPerception(
        sensors_config_path="/app/config/sensors.yaml",
        algo_config=cfg.get('perception', {})
    )

    # 4. Data Loader
    loader = DatasetLoader(
        video_path=cfg['system']['video_path'],
        mcap_path=cfg['system']['mcap_path'],
        lidar_topic=cfg['system']['lidar_topic']
    )

    viz_2d = Visualizer(cfg['system'])
    viz_3d = Visualizer3D()
    print("System initialized. Processing...")

    frame_count = 0
    start_time = time.time()

    while True:
        frame, lidar_data, timestamp = loader.next_frame()

        if frame is None:
            print(f"\nDataset finished. Total frames: {frame_count}")
            break

        # --- PERCEPTION ---
        with ScopedTimer("2. Perception Pipeline"):
            detected_objects, pcd_vis = perception.process(frame) # ONLY camera, NO lidar

        with ScopedTimer("2.1 Convert to Numpy"):
            depth_cloud = np.asarray(pcd_vis.points) if pcd_vis.has_points() else None
        obs_coords = []

        for det in detected_objects:
            obs_coords.append(det.center)

        # --- BACKEND ---
        lidar_flat = lidar_data if lidar_data is not None else []
        with ScopedTimer("3. Backend (C++)"):
            system.process(timestamp, lidar_flat, [], obs_coords)

        # --- VIZ ---
        with ScopedTimer("4. Visualization 2D"):
            pose = system.get_pose()
            map_objs = system.get_map()

            vis_frame, vis_map = viz_2d.draw(frame, pose, map_objs, detected_objects, lidar_flat)

        with ScopedTimer("5. Viz 3D Update"):
            viz_3d.update(depth_cloud)

        cv2.imshow("Camera", cv2.resize(vis_frame, (960, 540)))
        cv2.imshow("Map (Follow)", vis_map)

        if cv2.waitKey(1) == 27: break

        frame_count += 1
        if frame_count % 10 == 0:
            fps = frame_count / (time.time() - start_time)
            print(f"\rFrames: {frame_count} | FPS: {fps:.1f} | Objects: {len(map_objs)}", end="")

    loader.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()