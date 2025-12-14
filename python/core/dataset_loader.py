import cv2
import numpy as np
import struct
from mcap.reader import make_reader
from mcap_ros2.reader import read_ros2_messages

class DatasetLoader:
    def __init__(self, video_path, mcap_path, lidar_topic="/points_raw"):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0: self.fps = 30.0

        self.mcap_path = mcap_path
        self.lidar_topic = lidar_topic

        # Итератор по сообщениям лидара
        self.reader = make_reader(open(mcap_path, "rb"))
        self.ros_generator = read_ros2_messages(self.reader, topics=[lidar_topic])

        self.next_lidar_msg = None
        self.frame_idx = 0
        self.eof = False

    def _get_next_lidar_msg(self):
        try:
            return next(self.ros_generator)
        except StopIteration:
            return None

    def _parse_pointcloud(self, msg):
        """
        Парсит ROS2 PointCloud2 в плоский массив float [x,y,z,i, x,y,z,i...]
        Предполагаем стандартный формат: float32 x, y, z, intensity
        """
        data = msg.data
        # Обычно PointCloud2 имеет поля x,y,z + intensity/ring.
        # Для простоты и скорости MVP считаем, что структура плотная и это float32 (4 байта)
        # stride = msg.point_step

        # Преобразуем байты в numpy array float32
        # ВНИМАНИЕ: Это упрощенный парсер. В продакшене нужно смотреть msg.fields
        raw_floats = np.frombuffer(data, dtype=np.float32)
        return raw_floats

    def next_frame(self):
        """
        Возвращает: (image, lidar_data_flat, timestamp)
        """
        if not self.cap.isOpened(): return None, None, None

        ret, frame = self.cap.read()
        if not ret: return None, None, None

        # 1. Считаем время текущего кадра
        current_time = self.frame_idx / self.fps
        self.frame_idx += 1

        # 2. Ищем лидарный скан, соответствующий этому времени
        # Лидар обычно быстрее или асинхронен. Нам нужен скан, который >= current_time
        # или ближайший предыдущий.

        target_lidar_data = np.array([], dtype=np.float32)

        if not self.eof:
            if self.next_lidar_msg is None:
                self.next_lidar_msg = self._get_next_lidar_msg()

            # Проматываем старые сообщения
            while self.next_lidar_msg is not None:
                msg_time = self.next_lidar_msg.pu