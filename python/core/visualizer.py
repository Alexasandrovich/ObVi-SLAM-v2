import cv2
import numpy as np

class Visualizer:
    def __init__(self, config):
        # 1. Настройки экрана
        self.W = config.get('map_size', 800)
        self.H = config.get('map_size', 800)
        self.S = config.get('map_scale', 20.0)
        self.cx = self.W // 2
        self.cy = self.H // 2

        # 2. Состояние
        self.traj = []
        self.max_traj_len = 2000
        self.last_lidar = None # Кэш для лидара

        # 3. Канвас
        self.canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        # 4. Цвета (BGR)
        self.C_BG       = (30, 30, 30)
        self.C_LIDAR    = (180, 180, 180) # Серый
        self.C_TRAJ     = (0, 255, 255)   # Желтый
        self.C_OBJ      = (0, 255, 0)     # Зеленый (из карты)
        self.C_RAW_DET  = (255, 255, 0)   # Голубой (сырая детекция)
        self.C_ROBOT    = (0, 0, 255)     # Красный

    def draw(self, frame, pose, map_objects, detections, lidar_points=None, raw_pts=None):
        """Главный метод отрисовки."""
        self.canvas.fill(30) # Очистка фона

        # 1. Обновляем данные (буферы, траекторию)
        self._update_buffers(pose, lidar_points)

        # 2. Считаем матрицу трансформации Мир -> Экран
        M_view = self._compute_view_matrix(pose)

        # 3. Рисуем слои
        self._draw_lidar()
        self._draw_raw_detections(raw_pts)
        self._draw_trajectory(M_view)
        self._draw_map_objects(map_objects, M_view)
        self._draw_robot()

        # 4. Рисуем наложения на камеру
        vis_frame = self._draw_camera_overlay(frame, detections)

        return vis_frame, self.canvas

    # ================= ЛОГИКА (БЭКЕНД) =================

    def _update_buffers(self, pose, lidar_points):
        """Обновление траектории и кэша лидара."""
        # Траектория
        self.traj.append((pose[0], pose[1]))
        if len(self.traj) > self.max_traj_len:
            self.traj.pop(0)

        # Лидар (обработка 1D -> 2D)
        if lidar_points is not None and len(lidar_points) > 0:
            pts = np.array(lidar_points, copy=False)
            if pts.ndim == 1:
                stride = 4 if pts.size % 4 == 0 else 3
                pts = pts.reshape(-1, stride)
            # Сохраняем только X, Y
            self.last_lidar = pts[:, :2]

    def _compute_view_matrix(self, pose):
        """
        Создает матрицу 2x3: World -> Screen.
        """
        gx, gy = pose[0], pose[1]
        qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]

        # Yaw из кватерниона
        yaw = np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        # 1. World -> Robot (Сдвиг + Поворот)
        c, s = np.cos(-yaw), np.sin(-yaw)

        tx = -gx * c + gy * s
        ty = -gx * s - gy * c

        M_world_to_rob = np.array([
            [c, -s, tx],
            [s,  c, ty],
            [0,  0,  1]
        ])

        # 2. Robot -> Screen
        # Мы используем Y робота как ось движения (вертикаль экрана).
        # Мы используем X робота как боковую ось (горизонталь экрана).

        M_rob_to_screen = np.array([
            [-self.S,  0,       self.cx],
            [0,       self.S,  self.cy],
            [0,       0,       1      ]
        ])

        return M_rob_to_screen @ M_world_to_rob


    # ================= ОТРИСОВКА (ФРОНТЕНД) =================

    def _draw_lidar(self):
        """Рисует лидар. Согласовано с новой матрицей (Y -> Up)."""
        if self.last_lidar is None: return

        # Локальная матрица (должна совпадать с M_rob_to_screen по логике)
        M_local = np.array([
            [-self.S,  0,       self.cx],
            [0,       self.S,  self.cy]
        ], dtype=np.float32)

        # Трансформация
        pts_src = self.last_lidar.reshape(-1, 1, 2).astype(np.float32)
        pts_dst = cv2.transform(pts_src, M_local).astype(np.int32)

        # Отрисовка
        pts = pts_dst.reshape(-1, 2)
        mask = (pts[:, 0] >= 0) & (pts[:, 0] < self.W) & \
               (pts[:, 1] >= 0) & (pts[:, 1] < self.H)

        valid_pts = pts[mask]
        self.canvas[valid_pts[:, 1], valid_pts[:, 0]] = self.C_LIDAR

    def _draw_trajectory(self, M_view):
        if len(self.traj) < 2: return

        pts_world = np.array(self.traj, dtype=np.float32).reshape(-1, 1, 2)
        pts_screen = cv2.transform(pts_world, M_view[:2]).astype(np.int32)

        cv2.polylines(self.canvas, [pts_screen], False, self.C_TRAJ, 1, cv2.LINE_AA)

    def _draw_map_objects(self, map_objects, M_view):
        for obj in map_objects: # [id, cls, x, y, z]
            # Точка 1x1x2
            pt_src = np.array([[[obj[2], obj[3]]]], dtype=np.float32)
            pt_dst = cv2.transform(pt_src, M_view[:2]).astype(np.int32)[0][0]

            u, v = pt_dst
            if 0 <= u < self.W and 0 <= v < self.H:
                cv2.circle(self.canvas, (u, v), 5, self.C_OBJ, -1)
                cv2.putText(self.canvas, f"{int(obj[0])}", (u+5, v-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    def _draw_robot(self):
        # Статичный треугольник в центре
        # Нос (cy-15), Крылья (cx-10, cx+10)
        pts = np.array([
            [self.cx, self.cy - 15],
            [self.cx - 10, self.cy + 10],
            [self.cx + 10, self.cy + 10]
        ])
        cv2.fillPoly(self.canvas, [pts], self.C_ROBOT)

    def _draw_raw_detections(self, raw_pts):
        """
        Рисует детекции, которые пришли прямо из pixel_to_3d_ground.
        Они в системе робота, поэтому используем ту же логику, что и для Лидара.
        """
        if raw_pts is None or len(raw_pts) == 0:
            return

        # Используем ту же локальную матрицу, что и для лидара
        M_local = np.array([
            [-self.S,  0,       self.cx],
            [0,       self.S,  self.cy]
        ], dtype=np.float32)

        # Подготовка точек (N, 1, 2)
        # Берем только X и Y из [x, y, z]
        pts_src = np.array(raw_pts)[:, :2].reshape(-1, 1, 2).astype(np.float32)

        # Трансформация
        pts_dst = cv2.transform(pts_src, M_local).astype(np.int32)

        # Рисуем крестики
        for pt in pts_dst:
            u, v = pt[0]
            # Рисуем, если в пределах экрана
            if 0 <= u < self.W and 0 <= v < self.H:
                # Рисуем голубой крестик (X)
                cv2.drawMarker(self.canvas, (u, v), self.C_RAW_DET,
                               markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

    def _draw_camera_overlay(self, frame, detections):
        vis = frame.copy()
        for det in detections:
            x, y, w, h = map(int, det['bbox'])
            cv2.rectangle(vis, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
        return vis