import cv2
import numpy as np

class Visualizer:
    def __init__(self, map_size=800, scale=20.0):
        self.map_size = map_size
        self.scale = scale # пикселей на метр
        self.center_uv = (map_size // 2, map_size // 2)

        # Храним траекторию в мировых координатах (X, Z)
        self.traj_world = []

        # Буфер для картинки
        self.map_img = np.zeros((map_size, map_size, 3), dtype=np.uint8)

    def world_to_pixel(self, wx, wz, robot_x, robot_z):
        """
        Переводит мировые координаты в пиксели так,
        чтобы (robot_x, robot_z) всегда были в центре экрана.
        """
        # Сдвиг относительно робота
        dx = wx - robot_x
        dz = wz - robot_z

        # X на карте - это X в мире (вправо)
        # Y на карте - это минус Z в мире (вверх)
        px = int(self.center_uv[0] + dx * self.scale)
        py = int(self.center_uv[1] - dz * self.scale)
        return px, py

    def draw(self, frame, pose, map_objects, detections, lidar_data=None):
        # 1. Отрисовка детекций на кадре (Без изменений)
        vis_frame = frame.copy()
        for det in detections:
            x, y, w, h = det['bbox']
            u, v = det['keypoint']
            p1, p2 = (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2))
            cv2.rectangle(vis_frame, p1, p2, (0, 255, 0), 2)
            cv2.circle(vis_frame, (u, v), 5, (0, 0, 255), -1)

        # 2. Отрисовка Карты (С чистого листа каждый кадр)
        self.map_img.fill(0)

        # Текущая поза робота
        rx, rz = pose[0], pose[2]

        # Добавляем в историю (прореживаем, если нужно)
        self.traj_world.append((rx, rz))

        # --- А. Отрисовка Лидара (Серые точки) ---
        if lidar_data is not None and len(lidar_data) > 0:
            # lidar_data - это массив N*4
            # Нам нужны X, Y (в системе GLIM Y - это лево, Z - верх? Зависит от лидара)
            # Обычно для авто: X-вперед, Y-влево.
            # На карте: X (робота) -> Y (картинки, вверх), Y (робота) -> -X (картинки, влево)
            # Но у нас система координат карты совпадает с GLIM.
            # Пусть GLIM Pose: X-global, Z-global.

            # Для скорости берем срез
            points = lidar_data.reshape(-1, 4)[::5]

            # Вращение точек лидара (Local -> Global)
            # pose содержит [x, y, z, qx, qy, qz, qw]
            # Нам нужен Yaw. q = [3,4,5,6]
            qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
            # Yaw (вращение вокруг Y или Z? В GLIM Z-up)
            yaw = np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))

            c, s = np.cos(yaw), np.sin(yaw)

            # Трансформация точек: P_global = R * P_local + T
            # Но мы рисуем относительно центра экрана (который и есть T).
            # Значит нам нужно просто повернуть точки локально и отрисовать от центра.

            # P_screen_x = center_x + (x*c - y*s) * scale
            # P_screen_y = center_y - (x*s + y*c) * scale (минус т.к. Y вверх на карте)

            lx = points[:, 0]
            ly = points[:, 1]

            # Векторизованный расчет пикселей
            px = (self.center_uv[0] + (lx * c - ly * s) * self.scale).astype(np.int32)
            py = (self.center_uv[1] - (lx * s + ly * c) * self.scale).astype(np.int32)

            # Фильтр границ
            valid = (px >= 0) & (px < self.map_size) & (py >= 0) & (py < self.map_size)
            self.map_img[py[valid], px[valid]] = (100, 100, 100)

        # --- Б. Отрисовка Траектории ---
        # Пересчитываем пиксели для всей траектории относительно ТЕКУЩЕГО положения робота
        if len(self.traj_world) > 1:
            # Конвертируем список в numpy для скорости
            traj_arr = np.array(self.traj_world)

            # (World - Robot) * scale
            # dx = traj_x - rx
            dx = (traj_arr[:, 0] - rx) * self.scale
            dz = (traj_arr[:, 1] - rz) * self.scale

            px = (self.center_uv[0] + dx).astype(np.int32)
            py = (self.center_uv[1] - dz).astype(np.int32)

            pts = np.column_stack((px, py))
            cv2.polylines(self.map_img, [pts], False, (0, 255, 255), 1)

        # --- В. Отрисовка Объектов ---
        for obj in map_objects:
            # obj: [id, class, x, y, z]
            ox, oz = obj[2], obj[4]
            u, v = self.world_to_pixel(ox, oz, rx, rz)

            # Рисуем только если в пределах экрана
            if 0 <= u < self.map_size and 0 <= v < self.map_size:
                cv2.circle(self.map_img, (u, v), 4, (0, 255, 0), -1)
                cv2.putText(self.map_img, str(int(obj[0])), (u+5, v),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Робот всегда в центре (Красный треугольник или круг)
        cv2.circle(self.map_img, self.center_uv, 5, (0, 0, 255), -1)

        # Информация
        cv2.putText(self.map_img, f"Pos: {rx:.1f}, {rz:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        return vis_frame, self.map_img