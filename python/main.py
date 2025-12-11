import sys
import cv2
import numpy as np
import yaml
import os
import time
from ultralytics import YOLO

# Подключаем наш C++ модуль
sys.path.append("/app/build")
import park_slam_cpp
from tracker import ObjectTracker

def load_config(path):
    if not os.path.exists(path):
        print(f"Config file not found: {path}")
        return None
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # --- НАСТРОЙКИ ---
    config_path = "/app/config/config.yaml"
    vocab_path = "/app/vocab/ORBvoc.txt"
    orb_settings_path = "/app/config/orbslam_settings.yaml"

    cfg = load_config(config_path)
    if cfg is None:
        # Дефолтные значения если конфига нет
        ai_model = 'yolov8n.pt'
        video_source = '/app/data/video.mp4'
        fx, fy, cx, cy = 1174.0, 1174.0, 960.0, 540.0
    else:
        ai_model = cfg['ai']['model_path']
        video_source = cfg['system']['video_path']
        fx = cfg['camera']['fx']
        fy = cfg['camera']['fy']
        cx = cfg['camera']['cx']
        cy = cfg['camera']['cy']

    print("=== Launching Hybrid Object-Visual SLAM ===")

    # 1. Инициализация ORB-SLAM3 (Frontend & Tracking)
    # Третий параметр True включает Pangolin Viewer (родное окно)
    if not os.path.exists(vocab_path) or not os.path.exists(orb_settings_path):
        print("Error: ORBvoc.txt or orbslam_settings.yaml not found in /app/config/")
        return

    print(" -> Initializing ORB-SLAM3 (C++)... This might take a few seconds.")
    # use_viewer=True включает GUI
    orbslam = park_slam_cpp.OrbSlam(vocab_path, orb_settings_path, True)

    # 2. Инициализация GTSAM (Backend для объектов)
    print(" -> Initializing GTSAM Kernel...")
    slam_backend = park_slam_cpp.SlamKernel(fx, fy, cx, cy)

    # 3. Инициализация AI
    print(f" -> Loading YOLO: {ai_model}...")
    yolo = YOLO(ai_model)
    tracker = ObjectTracker() # Наш питоновский трекер

    # 4. Видео поток
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_source}")
        return

    # Данные для визуализации (Python Map)
    traj_x, traj_z = [], []
    map_size = 800
    scale = 20.0 # пикселей на метр

    print("=== System Started ===")
    print("Wait for ORB-SLAM initialization... (Video might pause briefly)")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- 1. ORB-SLAM3 TRACKING ---
        # Важно: timestamp должен быть в секундах
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Функция track возвращает матрицу 4x4 (списком из 16 элементов)
        # В это время в отдельном окне обновляется карта ORB-SLAM
        pose_vec = orbslam.track(frame, timestamp)

        is_tracked = (len(pose_vec) == 16)

        # --- 2. OBJECT DETECTION ---
        # Детектим даже если трек потерян, чтобы видеть боксы
        # Классы: 0-person, 2-car, 3-motorcycle, 5-bus, 7-truck, 13-bench
        results = yolo(frame, verbose=False, classes=[0, 1, 2, 3, 5, 7, 13], conf=0.4)

        raw_dets = []
        for box in results[0].boxes:
            x, y, w, h = box.xywh.cpu().numpy()[0]
            # [x_top_left, y_top_left, w, h]
            raw_dets.append([x - w/2, y - h/2, w, h])

            # Рисуем боксы на кадре
            p1 = (int(x - w/2), int(y - h/2))
            p2 = (int(x + w/2), int(y + h/2))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

        # --- 3. DATA ASSOCIATION (Tracker) ---
        tracked_objs = tracker.update(raw_dets)

        # --- 4. BACKEND UPDATE (GTSAM) ---
        current_slam_pose = [0, 0, 0] # x, y, z

        if is_tracked:
            # Готовим данные для C++
            # Список кортежей (id, u, v)
            slam_input = []
            for obj in tracked_objs:
                slam_input.append((obj['id'], obj['center'][0], obj['center'][1]))

            # Отправляем в GTSAM:
            # Поза от ORB (как Prior) + Детекции (как Factors) -> Оптимизация карты объектов
            slam_backend.add_frame_data(pose_vec, slam_input)

            # Получаем текущую позу (она должна совпадать с ORB, но достаем для проверки)
            # В данном пайплайне она почти равна pose_vec, так как мы жестко верим ORB-SLAM
            p = pose_vec
            # pose_vec - row-major matrix. Translation is elements 3, 7, 11
            current_slam_pose = [p[3], p[7], p[11]]

            status_text = "TRACKING"
            color_status = (0, 255, 0)
        else:
            status_text = "LOST"
            color_status = (0, 0, 255)

        # --- 5. VISUALIZATION (Python Side) ---
        # Рисуем карту объектов (вид сверху), дополняющую 3D вид ORB-SLAM
        map_img = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        center_offset = map_size // 2

        # Рисуем робота
        if is_tracked:
            # X - право, Y - вниз, Z - вперед (обычно в ORB-SLAM)
            # Но ORB-SLAM инициализирует мир произвольно.
            # Обычно для отрисовки карты берем X и Z.
            rx = int(current_slam_pose[0] * scale + center_offset)
            rz = int(center_offset - current_slam_pose[2] * scale)

            traj_x.append(rx)
            traj_z.append(rz)
            cv2.circle(map_img, (rx, rz), 5, (0, 255, 0), -1)

        # Рисуем траекторию
        if len(traj_x) > 1:
            pts = np.array([list(zip(traj_x, traj_z))], np.int32)
            cv2.polylines(map_img, [pts], False, (0, 200, 0), 1)

        # Рисуем объекты на карте (из GTSAM)
        for obj in tracked_objs:
            oid = obj['id']
            # Получаем 3D координату, которую посчитал GTSAM
            pos3d = slam_backend.get_landmark_pos(oid)

            # Подпись ID на видео
            u, v = int(obj['center'][0]), int(obj['center'][1])
            cv2.putText(frame, f"ID:{oid}", (u, v-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

            if pos3d:
                # Рисуем на карте
                lx = int(pos3d[0] * scale + center_offset)
                lz = int(center_offset - pos3d[2] * scale)
                cv2.circle(map_img, (lx, lz), 4, (0, 0, 255), -1)
                cv2.putText(map_img, f"{oid}", (lx+5, lz), 0, 0.4, (255,255,255), 1)

        # UI Info
        cv2.putText(frame, f"ORB-SLAM: {status_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color_status, 2)
        cv2.putText(frame, f"Objects: {len(tracked_objs)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Показываем окна
        # 1. Камера с боксами
        disp_frame = cv2.resize(frame, (960, 540))
        cv2.imshow("Semantic View (Python)", disp_frame)

        # 2. Векторная карта (дополнение к ORB Viewer)
        cv2.imshow("Object Map (GTSAM)", map_img)

        key = cv2.waitKey(1)
        if key == 27: # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()