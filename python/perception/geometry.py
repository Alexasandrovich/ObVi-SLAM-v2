import numpy as np

class VisualGeometry:
    def __init__(self, fx, fy, cx, cy, camera_height, pitch_deg=0.0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.h = camera_height # Высота камеры над землей (метры)
        self.pitch = np.radians(pitch_deg) # Наклон камеры вниз

    def pixel_to_3d_ground(self, u, v):
        """
        Проецирует пиксель (u, v) на плоскость земли.
        Возвращает [x, y, z] в системе координат камеры.
        """
        # Если точка выше горизонта - игнорируем
        # (Простейшая проверка для pitch=0: v > cy)
        if v <= self.cy:
            return None

        # 1. Вектор луча в системе камеры
        # ray = [ (u-cx)/fx, (v-cy)/fy, 1.0 ]
        ray_x = (u - self.cx) / self.fx
        ray_y = (v - self.cy) / self.fy
        ray_z = 1.0

        # 2. Учет наклона камеры (Pitch) - вращение вокруг оси X
        # Если камера смотрит прямо (pitch=0), то Y вниз, Z вперед.
        # Земля находится на Y = h.
        # Уравнение: Y_point = h.
        # Point = t * Ray. => t * ray_y = h => t = h / ray_y.

        # С учетом питча нужно повернуть луч, но для MVP считаем pitch=0
        if ray_y <= 0: return None # Луч вверх или параллельно

        dist_z = self.h / ray_y

        x = ray_x * dist_z
        y = self.h
        z = dist_z

        # Фильтр дальности (не берем то, что дальше 50м)
        if z > 50.0 or z < 1.0: return None

        return [x, y, z]