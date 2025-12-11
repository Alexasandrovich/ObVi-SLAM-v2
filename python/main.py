import sys
import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# Подключаем C++ модуль
sys.path.append("/app/build")
import park_slam_cpp
from tracker import ObjectTracker

def main():
    # --- КОНФИГУРАЦИЯ КАМЕРЫ (ИЗ ТВОЕГО YAML) ---
    # K: [1174, 0, 960, 0, 1174, 540, 0, 0, 1]
    width, height = 1920, 1080
    fx, fy = 1174.0, 1174.0
    cx, cy = 960.0, 540.0

    # Высота установки камеры над землей (по Z или Y в зависимости от системы)
    # t: [0.0, 0.0, 1.52] -> Это нам пригодится для визуализации или scale correction
    camera_height = 1.52

    print(f"Start SLAM with Cam: {width}x{height}, fx={fx}")

    # 1. Init Modules
    # C++ Backend
    slam = park_slam_cpp.SlamKernel(fx, fy, cx, cy)

    # AI (Используем модель n - nano для скорости, или m/l для точности)
    # classes=[0...]: 0-person, 1-bicycle, 2-car, 3-motorcycle, 5-bus, 7-truck, 9-traffic light, 11-stop sign
    # В парке нам важны деревья, но в COCO нет деревьев. Обычно используют 'potted plant' или дообучают.
    # Для теста будем детектить машины/людей/лавочки (bench=13).
    print("Loading YOLO...")
    yolo = YOLO("yolov8n.pt")
    tracker = ObjectTracker()

    # Видео поток
    video_path = '/app/data/video.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return

    traj_x, traj_z = [], []

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Если видео не 1080p, лучше сделать resize или crop,
        # но если калибровка под 1080p - подаем как есть.

        # --- 1. AI Detection ---
        # Детектим: Person(0), Car(2), Bench(13), Backpack(24 - как камень :))
        results = yolo(frame, verbose=False, classes=[0, 2, 13])

        raw_dets = []
        for box in results[0].boxes:
            x, y, w, h = box.xywh.cpu().numpy()[0]
            # Формат: [top_left_x, top_left_y, w, h]
            raw_dets.append([x - w/2, y - h/2, w, h])

            # --- 2. Tracking ---
        tracked_objs = tracker.update(raw_dets)

        # Подготовка данных: id, center_u, center_v
        slam_input = []
        for obj in tracked_objs:
            slam_input.append((obj['id'], obj['center'][0], obj['center'][1]))

        # --- 3. SLAM Update (C++) ---
        slam.process(frame, slam_input)

        # --- 4. Visualization ---
        pose = slam.get_pose() # [x, y, z]

        # --- Отрисовка карты (Вид сверху) ---
        map_size = 800
        map_img = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        scale = 20.0 # пикселей на метр
        center_offset = map_size // 2

        # Рисуем робота
        # В GTSAM обычно Z - вверх, Y - право, X - вперед (или оптическая ось Z-вперед)
        # Для визуализации: X (slam) -> X (img), Z (slam) -> Y (img)
        rx = int(pose[0] * scale + center_offset)
        rz = int(center_offset - pose[2] * scale) # минус, т.к. Y на картинке вниз

        cv2.circle(map_img, (rx, rz), 5, (0, 255, 0), -1)

        traj_x.append(rx)
        traj_z.append(rz)

        # Рисуем хвост траектории
        if len(traj_x) > 1:
            pts = np.array([list(zip(traj_x, traj_z))], np.int32)
            cv2.polylines(map_img, [pts], False, (0, 200, 0), 1)

        # Рисуем объекты на карте
        for obj in tracked_objs:
            oid = obj['id']
            # Получаем 3D координату из SLAM
            pos3d = slam.get_landmark_pos(oid) # returns [x, y, z] or empty

            # На кадре (кружочек на объекте)
            u, v = int(obj['center'][0]), int(obj['center'][1])
            cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)

            # На карте (если SLAM уже посчитал его позицию)
            if pos3d:
                lx = int(pos3d[0] * scale + center_offset)
                lz = int(center_offset - pos3d[2] * scale)

                # Цвет кружка на карте
                cv2.circle(map_img, (lx, lz), 4, (0, 0, 255), -1)
                cv2.putText(map_img, f"{oid}", (lx+5, lz), 0, 0.4, (255,255,255), 1)

        # Ресайз фрейма для удобства просмотра, если экран маленький
        disp_frame = cv2.resize(frame, (960, 540))

        cv2.imshow("Camera View", disp_frame)
        cv2.imshow("SLAM Map", map_img)

        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()