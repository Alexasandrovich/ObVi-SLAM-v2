import cv2
import numpy as np

class Visualizer:
    def __init__(self, map_size=800, scale=20.0):
        self.map_size = map_size
        self.scale = scale # пикс/метр
        self.center = map_size // 2
        self.traj = []
        self.map_img = np.zeros((map_size, map_size, 3), dtype=np.uint8)

    def draw(self, frame, pose, map_objects, detections):
        # 1. Отрисовка детекций на кадре
        vis_frame = frame.copy()
        for det in detections:
            x, y, w, h = det['bbox']
            u, v = det['keypoint']

            # Бокс
            p1 = (int(x - w/2), int(y - h/2))
            p2 = (int(x + w/2), int(y + h/2))
            cv2.rectangle(vis_frame, p1, p2, (0, 255, 0), 2)

            # Точка касания земли
            cv2.circle(vis_frame, (u, v), 5, (0, 0, 255), -1)

        # 2. Отрисовка Карты (Вид сверху)
        # Очищаем карту (или оставляем след, если хотим накапливать)
        # self.map_img.fill(0)
        # Лучше рисовать поверх черного фона каждый раз заново, если объектов немного
        display_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)

        # Траектория
        tx = int(pose[0] * self.scale + self.center)
        tz = int(self.center - pose[2] * self.scale) # Z - это вперед в камере, но Y на карте

        self.traj.append((tx, tz))
        if len(self.traj) > 2:
            cv2.polylines(display_map, [np.array(self.traj)], False, (0, 255, 255), 1)

        # Робот
        cv2.circle(display_map, (tx, tz), 3, (0, 0, 255), -1)

        # Объекты карты (Landmarks)
        # obj: [id, class, x, y, z]
        for obj in map_objects:
            ox = int(obj[2] * self.scale + self.center)
            oz = int(self.center - obj[4] * self.scale)

            color = (0, 255, 0) # Green tree
            cv2.circle(display_map, (ox, oz), 4, color, -1)
            cv2.putText(display_map, str(int(obj[0])), (ox+5, oz), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255))

        return vis_frame, display_map