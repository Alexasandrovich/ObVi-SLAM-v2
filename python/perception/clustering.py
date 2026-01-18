import numpy as np
import open3d as o3d
from common.types import DetectedObject

class PointCloudProcessor:
    def __init__(self, cfg):
        self.voxel_size = cfg.get('voxel_size', 0.1)
        self.cluster_eps = cfg.get('cluster_eps', 0.5)
        self.min_points = cfg.get('min_points', 15)

    def extract_objects(self, pcd: o3d.geometry.PointCloud):
        if not pcd.has_points():
            return [], pcd

        # 1. Downsample (Важно для скорости)
        pcd = pcd.voxel_down_sample(self.voxel_size)

        # 2. Удаление пола
        # Предполагаем, что пол — самая большая плоскость
        if len(pcd.points) < 50: return [], pcd

        _, inliers = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=50)
        objects_pcd = pcd.select_by_index(inliers, invert=True)

        # 3. Кластеризация
        labels = np.array(objects_pcd.cluster_dbscan(
            eps=self.cluster_eps, min_points=self.min_points, print_progress=False
        ))

        detected_objs = []
        if len(labels) > 0:
            max_label = labels.max()
            for i in range(max_label + 1):
                idxs = np.where(labels == i)[0]
                cluster = objects_pcd.select_by_index(idxs)

                # Bounding Box
                bbox = cluster.get_axis_aligned_bounding_box()

                obj = DetectedObject(
                    id=0,
                    center=bbox.get_center(),
                    extent=bbox.get_extent(),
                    bbox_corners=np.asarray(bbox.get_box_points())
                )
                detected_objs.append(obj)

        return detected_objs, objects_pcd