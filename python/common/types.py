from dataclasses import dataclass
import numpy as np

@dataclass
class DetectedObject:
    """Результат работы пайплайна"""
    id: int
    center: np.ndarray    # (3,) x, y, z (в координатах World/Robot, как настроена Camera)
    extent: np.ndarray    # (3,) размеры dx, dy, dz
    bbox_corners: np.ndarray = None