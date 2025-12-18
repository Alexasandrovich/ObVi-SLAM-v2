import cv2
import numpy as np

class Visualizer:
    def __init__(self, config):
        self.map_size = config.get('map_size', 800)
        self.scale = config.get('map_scale', 20.0)
        # Робот по центру
        self.center_uv = (self.map_size // 2, self.map_size // 2)

        # Офсеты (если лидар стоит криво)
        self.lidar_align_rad = np.radians(config.get('lidar_align_deg', 0.0))

        self.traj_global = []
        self.last_lidar_points = None
        self.map_img = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)

    def _rotate(self, x, y, angle):
        """Вращение вектора на угол"""
        c = np.cos(angle)
        s = np.sin(angle)
        return x * c - y * s, x * s + y * c

    def draw(self, frame, pose, map_objects, detections, lidar_data=None):
        # --- 1. Отрисовка на кадре (Камера) ---
        vis_frame = frame.copy()
        for det in detections:
            x, y, w, h = det['bbox']
            u, v = det['keypoint']
            cv2.rectangle(vis_frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 255, 0), 2)
            cv2.circle(vis_frame, (u, v), 4, (0, 0, 255), -1)

        # --- 2. Подготовка Карты ---
        self.map_img.fill(0)

        # Распаковка позы
        gx, gy = pose[0], pose[1]
        qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
        # Текущий курс робота в мире
        robot_yaw = np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))

        self.traj_global.append((gx, gy))
        def local_to_screen(lx, ly):
            u = int(self.center_uv[0] + lx * self.scale)
            v = int(self.center_uv[1] - ly * self.scale)
            return u, v

        # Функция: Глобальные (World) -> Локальные (Robot Frame)
        # Мы поворачиваем мир "навстречу" роботу (-robot_yaw)
        c_yaw = np.cos(-robot_yaw)
        s_yaw = np.sin(-robot_yaw)

        def global_to_local(wx, wy):
            dx = wx - gx
            dy = wy - gy
            # Проекция на оси робота
            lx = dx * c_yaw - dy * s_yaw # Проекция на направление движения (X)
            ly = dx * s_yaw + dy * c_yaw # Проекция на перпендикуляр (Y)
            return lx, ly

        # === А. ОТРИСОВКА ЛИДАРА ===
        # Лидар УЖЕ в локальных координатах (относительно центра робота)
        if lidar_data is not None and len(lidar_data) > 0:
            self.last_lidar_points = lidar_data.reshape(-1, 4)[::3]

        if self.last_lidar_points is not None:
            lx = self.last_lidar_points[:, 0]
            ly = self.last_lidar_points[:, 1]

            # 1. Корректируем физическую установку лидара (если он повернут на корпусе)
            if self.lidar_align_rad != 0:
                lx, ly = self._rotate(lx, ly, self.lidar_align_rad)

            # 2. Сразу на экран (он уже Local)
            # Векторизированная версия local_to_screen
            px = (self.center_uv[0] + lx * self.scale).astype(np.int32)
            py = (self.center_uv[1] - ly * self.scale).astype(np.int32)

            # Клиппинг и отрисовка
            mask = (px >= 0) & (px < self.map_size) & (py >= 0) & (py < self.map_size)
            self.map_img[py[mask], px[mask]] = (150, 150, 150) # Серый цвет точек

        # === Б. ТРАЕКТОРИЯ ===
        if len(self.traj_global) > 1:
            recent = self.traj_global[-500:] # Последние 500 точек
            pts_screen = []
            for p in recent:
                lx, ly = global_to_local(p[0], p[1]) # Сначала в локальные
                u, v = local_to_screen(-lx, -ly)
                pts_screen.append([u, v])

            cv2.polylines(self.map_img, [np.array(pts_screen)], False, (0, 255, 255), 1)

        # === В. ОБЪЕКТЫ (Детекции) ===
        for obj in map_objects:
            # obj: [id, conf, global_x, global_y, global_z]
            lx, ly = global_to_local(obj[2], obj[3])
            u, v = local_to_screen(lx, ly)

            if 0 <= u < self.map_size and 0 <= v < self.map_size:
                cv2.circle(self.map_img, (u, v), 5, (0, 255, 0), -1)
                # Вывод координат (X=Вперед, Y=Влево)
                cv2.putText(self.map_img, f"X:{lx:.1f} Y:{ly:.1f}", (u+10, v),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # === Г. РОБОТ ===
        cx, cy = self.center_uv
        pts_robot = np.array([
            [cx,     cy - 15],  # Нос (вверху)
            [cx - 8, cy + 10],  # Левое крыло
            [cx,     cy + 5],   # Центр сзади
            [cx + 8, cy + 10]   # Правое крыло
        ])
        cv2.fillPoly(self.map_img, [pts_robot], (0, 0, 255))

        return vis_frame, self.map_img