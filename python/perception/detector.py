from ultralytics import YOLO
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_path, target_classes=None, conf=0.5):
        self.model = YOLO(model_path)
        # Если классы не переданы, берем дефолтный набор без людей (0)
        self.classes = target_classes if target_classes else [1, 2, 3, 5, 7, 13]
        self.conf = conf

    def detect(self, image):
        """
        Возвращает список:
        [{'class_id': int, 'bbox': [x,y,w,h], 'keypoint': (u, v)}, ...]
        """
        # ВАЖНО: Передаем classes прямо в model(), чтобы YOLO сразу фильтровала лишнее
        results = self.model(image, verbose=False, classes=self.classes, conf=self.conf)[0]
        detections = []

        if not results.masks:
            return detections

        # Ресайз масок может быть дорогим, делаем один раз
        img_h, img_w = image.shape[:2]

        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls)

            # Двойная проверка (на всякий случай)
            if cls_id not in self.classes:
                continue

            xywh = box.xywh.cpu().numpy()[0] # center_x, center_y, w, h

            # --- УМНЫЙ ПОИСК ТОЧКИ КАСАНИЯ ЗЕМЛИ ---
            mask_data = results.masks.data[i].cpu().numpy()
            mask_resized = cv2.resize(mask_data, (img_w, img_h))

            # Находим координаты маски
            y_indices, x_indices = np.where(mask_resized > 0.5)

            if len(y_indices) > 0:
                # Берем нижние 5% пикселей маски, чтобы усреднить шум
                # Сортируем по Y (высоте)
                sorted_indices = np.argsort(y_indices)
                num_bottom = max(1, int(len(sorted_indices) * 0.05)) # Берем 5% снизу

                bottom_idx = sorted_indices[-num_bottom:]

                # Среднее по X и Y среди нижних точек
                avg_u = int(np.mean(x_indices[bottom_idx]))
                avg_v = int(np.mean(y_indices[bottom_idx]))

                detections.append({
                    'class_id': cls_id,
                    'bbox': xywh,
                    'keypoint': (avg_u, avg_v) # Стабильная точка низа
                })

        return detections