from ultralytics import YOLO
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_path, target_classes=[0, 1, 2, 3, 5, 7], conf=0.5):
        self.model = YOLO(model_path)
        self.classes = target_classes # COCO IDs
        self.conf = conf

    def detect(self, image):
        """
        Возвращает список объектов:
        [{'class': int, 'bbox': [x,y,w,h], 'mask_bottom_point': (u, v)}, ...]
        """
        results = self.model(image, verbose=False, classes=self.classes, conf=self.conf)[0]
        detections = []

        if not results.masks:
            return detections

        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls)
            xywh = box.xywh.cpu().numpy()[0] # center_x, center_y, w, h

            # Работа с маской для поиска самой нижней точки (контакт с землей)
            mask_data = results.masks.data[i].cpu().numpy() # low-res mask
            # Масштабируем маску к размеру изображения
            mask_resized = cv2.resize(mask_data, (image.shape[1], image.shape[0]))

            # Находим нижний пиксель маски
            # (Ищем индексы, где маска > 0.5)
            y_indices, x_indices = np.where(mask_resized > 0.5)

            if len(y_indices) > 0:
                # Берем самую нижнюю точку (максимальный Y)
                max_y_idx = np.argmax(y_indices)
                u = int(x_indices[max_y_idx])
                v = int(y_indices[max_y_idx])

                detections.append({
                    'class_id': cls_id,
                    'bbox': xywh,
                    'keypoint': (u, v) # Точка касания земли
                })

        return detections