import open3d as o3d
import numpy as np

class Visualizer3D:
    def __init__(self, window_name="3D Perception", width=960, height=720):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name, width=width, height=height)

        # 1. Основное облако точек (пустое при старте)
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)

        # 2. Координатные оси (X=Red, Y=Green, Z=Blue) размером 1 метр
        # Помогает понять, где "верх", а где "вперед"
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        self.vis.add_geometry(axis)

        # Флаг инициализации камеры (чтобы один раз настроить зум)
        self.first_update = True

        # Настройки рендера (черный фон, толстые точки)
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        opt.point_size = 3.0

    def update(self, points_np, detected_objects=None):
        """
        points_np: (N, 3) numpy array - точки в мировых координатах
        detected_objects: список объектов (для отрисовки 3D боксов)
        """
        if points_np is None or len(points_np) == 0:
            self.vis.poll_events()
            self.vis.update_renderer()
            return

        # 1. Обновляем точки
        self.pcd.points = o3d.utility.Vector3dVector(points_np)

        # (Опционально) Раскраска по высоте (Z), чтобы было красивее
        # Z-axis color gradient (Purple to Yellow)
        z = points_np[:, 2]
        # Нормализация цветов от -1м до 3м
        colors = np.zeros_like(points_np)
        norm_z = np.clip((z + 1.0) / 4.0, 0, 1)
        colors[:, 0] = norm_z          # R
        colors[:, 1] = 1.0 - norm_z    # G
        colors[:, 2] = 1.0             # B
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        self.vis.update_geometry(self.pcd)

        # 2. Если нужно, можно добавить отрисовку боксов (пока опустим для скорости,
        # так как в Open3D добавлять/удалять геометрию каждый кадр медленно)

        # 3. Обновление камеры при первом запуске
        if self.first_update:
            self.vis.reset_view_point(True)
            self.first_update = False

        # 4. Прокрутка событий (ВАЖНО: это позволяет вращать мышкой)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()