from ultralytics import YOLO
from .base import ObjectDetector

class YoloDetector(ObjectDetector):
    def __init__(self, weights_path: str):
        self.model = YOLO(weights_path)

    def detect(self, image):
        results = self.model(image, verbose=False)[0]
        detections = []
        for box in results.boxes:
            xywh = box.xywh.cpu().numpy()[0]
            conf = float(box.conf)
            cls = int(box.cls)
            if conf > 0.5:
                detections.append({
                    'bbox': xywh,
                    'class_id': cls,
                    'confidence': conf
                })
        return detections