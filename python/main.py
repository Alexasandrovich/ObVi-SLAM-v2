import sys
import cv2
import numpy as np
import yaml
import os
from ultralytics import YOLO

# Подключаем C++ модуль
sys.path.append("/app/build")
import park_slam_cpp
from tracker import ObjectTracker

def load_config(path):
    if not os.path.exists(path):
        print(f"Config file not found: {path}")
        sys.exit(1)
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # 1. Загрузка конфигурации
    config_path = "/app/config/config.yaml"
    print(f"Loading config from {config_path}...")
    cfg = load_config(config_path)

    cam_conf = cfg['camera']
    ai_conf = cfg['ai']
    sys_conf = cfg['system']

    print(f"Camera: {cam_conf['width']}x{cam_conf['height']}, fx={cam_conf['fx']}")

    # 2. Инициализация SLAM ядра (C++)
    # Передаем параметры камеры напрямую из конфига
    slam = park_slam_cpp.SlamKernel(
        cam_conf['fx'],
        cam_conf['fy'],
        cam_conf['cx'],
        cam_conf['cy']
    )

    # 3. Инициализация AI и Трекера
    print(f"Loading Model: {ai_conf['model_path']}")
    yolo = YOLO(ai_conf['model_path'])
    tracker = ObjectTracker()

    # 4. Видео поток
    video_path = sys_conf['video_path']
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return

    # Данные для отрисовки траектории
    traj_x, traj_z = [], []

    # Скейл карты
    scale = sys_conf['map_scale']
    map_size = sys_conf['map_size']
    center_offset = map_size // 2

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- 1. AI Detection ---
        results = yolo(frame, verbose=False, classes=ai_conf['target_classes'], conf=ai_conf['conf_threshold'])

        raw_dets = []
        for box in results[0].boxes:
            x, y, w, h = box.xywh.cpu().numpy()[0]
            # [top_left_x, top_left_y, w, h]
            raw_dets.append([x - w/2, y - h/2, w, h])

            # --- 2. Tracking ---
        tracked_objs = tracker.update(raw_dets)

        # Подготовка данных: id, center_u, center_v
        slam_input = []
        for obj in tracked_objs:
            slam_input.append((obj['id'], obj['center'][0], obj['center'][1]))

        # --- 3. SLAM Update (C++) ---
        # Передаем кадр и детекции в ядро
        slam.process(frame, slam_input)

        # --- 4. Visualization ---
        pose = slam.get_pose() # [x, y, z]

        # Отрисовка карты (Вид сверху)
        map_img = np.zeros((map_size, map_size, 3), dtype=np.uint8)

        # Координаты робота на карте
        rx = int(pose[0] * scale + center_offset)
        rz = int(center_offset - pose[2] * scale)

        cv2.circle(map_img, (rx, rz), 5, (0, 255, 0), -1)
        traj_x.append(rx)
        traj_z.append(rz)

        # Хвост траектории
        if len(traj_x) > 1:
            pts = np.array([list(zip(traj_x, traj_z))], np.int32)
            cv2.polylines(map_img, [pts], False, (0, 200, 0), 1)

        # Отрисовка объектов
        for obj in tracked_objs:
            oid = obj['id']
            pos3d = slam.get_landmark_pos(oid)

            # На видео
            u, v = int(obj['center'][0]), int(obj['center'][1])
            cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)

            # На карте
            if pos3d:
                lx = int(pos3d[0] * scale + center_offset)
                lz = int(center_offset - pos3d[2] * scale)
                cv2.circle(map_img, (lx, lz), 4, (0, 0, 255), -1)
                cv2.putText(map_img, f"{oid}", (lx+5, lz), 0, 0.4, (255,255,255), 1)

        # Ресайз для отображения (если видео огромное)
        disp_frame = cv2.resize(frame, (960, 540))

        cv2.imshow("Camera View", disp_frame)
        cv2.imshow("Vector Map", map_img)

        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()