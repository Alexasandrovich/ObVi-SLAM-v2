from abc import ABC, abstractmethod
import numpy as np

class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray):
        """
        Returns list of detections: 
        [{'bbox': [x,y,w,h], 'class_id': int, 'confidence': float}]
        """
        pass