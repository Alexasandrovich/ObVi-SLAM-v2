import torch
import numpy as np
import cv2

class DepthModel:
    def __init__(self, model_type="vits", device="cuda"):
        self.device = device
        # Здесь загрузка DepthAnything V2
        # Для примера код загрузки через Hub (или локальный путь)
        try:
            self.model = torch.hub.load('LiheYoung/depth_anything_v2', 'depth_anything_v2',
                                        encoder=model_type, trust_repo=True).to(device)
        except:
            print("Warning: Loading Dummy Depth Model")
            self.model = None

    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Input: RGB Image (H, W, 3)
        Output: Depth Map (H, W) float32 in meters
        """
        if self.model is None:
            return np.random.uniform(0.5, 10.0, image.shape[:2]).astype(np.float32)

        # DepthAnything infer returns relative depth usually,
        # Metric depth version returns meters.
        depth = self.model.infer_image(image)
        return depth.astype(np.float32)