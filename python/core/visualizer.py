import cv2
import numpy as np

class Visualizer:
    def __init__(self, config):
        self.map_size = config.get('map_size', 800)
        self.scale = config.get('map_scale', 20.0)
        self.center_uv = (self.map_size // 2, int(self.map_size * 0.75))

        # 1. Офсет Вида (Решает проблему "куда едем")
        view_deg = config.get('view_offset_deg', 0.0)
        self.view_rad = np.radians(view_deg)

        # 2. Офсет Лидара (Решает проблему "повернутых стен")
        lidar_deg = config.get('lidar_align_deg', 0.0)
        self.lidar_align_rad = np.radians(lidar_deg)

        self.traj_global = []
        self.last_lidar_points = None

        self.map_img = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)

    def _rotate_local(self, x, y, angle_rad):
        """Простое 2D вращение"""
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        xr = x * c - y * s
        yr = x * s + y * c
        return xr, yr

    def draw(self, frame, pose, map_objects, detections, lidar_data=None):
        # 1. Детекции
        vis_frame = frame.copy()
        for det in detections:
            x, y, w, h = det['bbox']
            u, v = det['keypoint']
            cv2.rectangle(vis_frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 255, 0), 2)
            cv2.circle(vis_frame, (u, v), 4, (0, 0, 255), -1)

        # 2. Карта
        self.map_img.fill(0)

        # Распаковка GLIM позы (Global)
        gx, gy = pose[0], pose[1]
        qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
        global_yaw = np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))

        self.traj_global.append((gx, gy))
        # Мы хотим, чтобы на экране робот смотрел ВВЕРХ.
        # Значит, мы должны повернуть весь мир так, чтобы угол робота стал равен -90 градусов (или 0, если Y-вверх).
        # В OpenCV Y-вниз. Значит "Вверх" это ось -Y.

        # Угол отрисовки = Глобальный угол + Офсет пользователя
        draw_yaw = global_yaw + self.view_rad

        # Матрица поворота Мир -> Экран (вокруг робота)
        # Мы поворачиваем вектор (Point - Robot) на угол -draw_yaw.
        # Тогда ось "Вперед" робота совпадет с осью X локальной системы.
        # А потом мы X отображаем как "Вверх" (-Y экрана).

        c_draw = np.cos(-draw_yaw)
        s_draw = np.sin(-draw_yaw)

        def world_to_screen(wx, wy):
            dx = wx - gx
            dy = wy - gy

            # Проекция на оси робота
            # fwd - расстояние "впереди" робота
            # lft - расстояние "слева" от робота
            fwd = dx * c_draw - dy * s_draw
            lft = dx * s_draw + dy * c_draw

            # Экран: Fwd -> Up (-Y), Left -> Left (-X)
            u = int(self.center_uv[0] - lft * self.scale)
            v = int(self.center_uv[1] - fwd * self.scale)
            return u, v

        # --- А. Лидар ---
        if lidar_data is not None and len(lidar_data) > 0:
            self.last_lidar_points = lidar_data.reshape(-1, 4)[::3]

        if self.last_lidar_points is not None:
            lx = self.last_lidar_points[:, 0]
            ly = self.last_lidar_points[:, 1]
            align_angle = self.view_rad + self.lidar_align_rad
            lx_rot, ly_rot = self._rotate_local(lx, ly, align_angle)

            px = (self.center_uv[0] - ly_rot * self.scale).astype(np.int32)
            py = (self.center_uv[1] - lx_rot * self.scale).astype(np.int32)

            mask = (px >= 0) & (px < self.map_size) & (py >= 0) & (py < self.map_size)
            self.map_img[py[mask], px[mask]] = (120, 120, 120)

        # --- Б. Траектория ---
        if len(self.traj_global) > 1:
            recent = self.traj_global[-500:]
            pts = [world_to_screen(p[0], p[1]) for p in recent]
            cv2.polylines(self.map_img, [np.array(pts)], False, (0, 255, 255), 1)

        # --- В. Объекты ---
        for obj in map_objects:
            # Объекты в глобальных координатах GLIM
            u, v = world_to_screen(obj[2], obj[3])

            if 0 <= u < self.map_size and 0 <= v < self.map_size:
                cv2.circle(self.map_img, (u, v), 5, (0, 255, 0), -1)
                cv2.putText(self.map_img, f"{int(obj[0])}", (u+6, v), 0, 0.4, (255, 255, 255))

        # --- Г. Робот ---
        cx, cy = self.center_uv
        pts_robot = np.array([[cx, cy-15], [cx-10, cy+10], [cx, cy+5], [cx+10, cy+10]])
        cv2.fillPoly(self.map_img, [pts_robot], (0, 0, 255))

        return vis_frame, self.map_img