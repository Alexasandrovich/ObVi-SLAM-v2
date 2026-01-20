import torch
import numpy as np
import cv2

class DepthModel:
    def __init__(self, model_type="vits", device="cuda"):
        """
        model_type: 'vits', 'vitb', 'vitl'.
        Для SLAM лучше 'vits' (быстрее) или 'vitb'. 'vitl' очень тяжелый.
        """
        self.device = device
        print(f"Loading Depth Model ({model_type})...")
        try:
            # trust_repo=True обязателен
            self.model = torch.hub.load(
                'LiheYoung/depth_anything_v2',
                'depth_anything_v2',
                encoder=model_type,
                trust_repo=True
            ).to(device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        image: BGR numpy array (OpenCV format)
        Returns: Depth in meters (float32)
        """
        if self.model is None:
            return np.random.uniform(0.5, 5.0, image.shape[:2]).astype(np.float32)

        # 1. BGR -> RGB (КРИТИЧНО!)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Инференс
        # DepthAnything внутри сам делает нормализацию и ресайз,
        # но ожидает RGB.
        # raw_depth: (H, W), где большие значения = БЛИЗКО
        with torch.no_grad():
            raw_depth = self.model.infer_image(img_rgb)

        # 3. Конвертация в Метры (Heuristic)
        # DepthAnything выдает значения условно от 0 до 255 (или float эквивалент).
        # Нам нужно инвертировать: Z = Scale / (disp + epsilon)
        # Scale = 50.0 - это подбираемый коэффициент "масштаба мира".
        # Можно подобрать экспериментально, чтобы пол был на уровне 0 (если камера на высоте H).

        epsilon = 0.1
        metric_depth = 40.0 / (raw_depth + epsilon)

        # Клиппинг, чтобы не улетало в бесконечность
        metric_depth = np.clip(metric_depth, 0.1, 20.0)

        return metric_depth.astype(np.float32)