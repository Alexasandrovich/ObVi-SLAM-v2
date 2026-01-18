import open3d as o3d
from .geometry import Camera
from .depth import DepthModel
from .clustering import PointCloudProcessor
from common.profiler import ScopedTimer

class AgnosticPerception:
    def __init__(self, sensors_config_path, algo_config):
        # 1. Загружаем камеру и параметры
        self.camera = Camera()
        self.camera.read_from_yaml_file(sensors_config_path)

        # 2. Нейросеть
        self.depth_model = DepthModel()

        # 3. Кластеризатор
        self.processor = PointCloudProcessor(algo_config)

    def process(self, frame):
        # 1. Инференс
        with ScopedTimer("2.1 Inference"):
            depth_map = self.depth_model.infer(frame)

        # 2. Проекция (используя метод внутри camera!)
        with ScopedTimer("2.2 reproject_depth_map"):
            pts_world = self.camera.reproject_depth_map(depth_map)

        # 3. Конвертация в Open3D
        with ScopedTimer("2.3 convert to Open3D"):
            full_pcd = o3d.geometry.PointCloud()
            full_pcd.points = o3d.utility.Vector3dVector(pts_world)

        # 4. Кластеризация
        with ScopedTimer("2.4 clustering"):
            objects, debug_pcd = self.processor.extract_objects(full_pcd)

        return objects, debug_pcd