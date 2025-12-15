import cv2
import numpy as np
import os
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

class DatasetLoader:
    def __init__(self, video_path, mcap_path, lidar_topic="/rslidar_points"):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0: self.fps = 30.0

        self.mcap_path = mcap_path
        self.lidar_topic = lidar_topic

        self.lidar_iter = None
        self.lidar_available = False
        self.eof_lidar = False

        # Таймстемпы для синхронизации
        self.lidar_start_time = None
        self.last_returned_lidar_time = -1.0 # Чтобы не дублировать сообщения

        # Буфер для "текущего" и "следующего" сообщения лидара (для интерполяции/выбора ближайшего)
        self.cached_lidar_msg = None

        if os.path.exists(mcap_path) and os.path.getsize(mcap_path) > 0:
            try:
                self.stream = open(mcap_path, "rb")
                self.reader = make_reader(self.stream, decoder_factories=[DecoderFactory()])

                summary = self.reader.get_summary()
                available_topics = [c.topic for c in summary.channels.values()]

                if lidar_topic not in available_topics:
                    print(f"WARNING: Topic {lidar_topic} not found! Available: {available_topics}")
                else:
                    self.lidar_iter = self.reader.iter_decoded_messages(topics=[lidar_topic])
                    self.lidar_available = True
                    print(f"Lidar initialized on {lidar_topic}. Frequency preservation mode: ON")

            except Exception as e:
                print(f"ERROR: MCAP init failed: {e}")
        else:
            print(f"WARNING: MCAP invalid: {mcap_path}")

        self.frame_idx = 0

    def _get_next_lidar_tuple(self):
        """Возвращает (schema, channel, message, ros_msg)"""
        if not self.lidar_available or self.eof_lidar:
            return None
        try:
            return next(self.lidar_iter)
        except StopIteration:
            self.eof_lidar = True
            return None
        except Exception as e:
            print(f"Read error: {e}")
            self.eof_lidar = True
            return None

    def _parse_pointcloud(self, ros_msg):
        """Robust PointCloud2 parser"""
        if ros_msg is None: return np.array([], dtype=np.float32)

        fields = {f.name: f for f in ros_msg.fields}
        if 'x' not in fields or 'y' not in fields or 'z' not in fields:
            return np.array([], dtype=np.float32)

        dtype_list = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
        offsets = [fields['x'].offset, fields['y'].offset, fields['z'].offset]
        names = ['x', 'y', 'z']

        if 'intensity' in fields:
            names.append('intensity')
            f_type = fields['intensity'].datatype
            # 7=Float32, 2=Uint8, 4=Uint16
            if f_type == 7: dtype_list.append(('intensity', '<f4'))
            elif f_type == 2: dtype_list.append(('intensity', 'u1'))
            elif f_type == 4: dtype_list.append(('intensity', 'u2'))
            else: dtype_list.append(('intensity', '<f4'))
            offsets.append(fields['intensity'].offset)

        point_dtype = np.dtype({
            'names': names,
            'formats': [d[1] for d in dtype_list],
            'offsets': offsets,
            'itemsize': ros_msg.point_step
        })

        try:
            cloud_arr = np.frombuffer(ros_msg.data, dtype=point_dtype)
        except ValueError:
            return np.array([], dtype=np.float32)

        # Фильтр NaN
        mask = np.isfinite(cloud_arr['x']) & np.isfinite(cloud_arr['y']) & np.isfinite(cloud_arr['z'])
        cloud_arr = cloud_arr[mask]

        if len(cloud_arr) == 0:
            return np.array([], dtype=np.float32)

        x = cloud_arr['x'].astype(np.float32)
        y = cloud_arr['y'].astype(np.float32)
        z = cloud_arr['z'].astype(np.float32)

        if 'intensity' in names:
            i = cloud_arr['intensity'].astype(np.float32)
        else:
            i = np.zeros_like(x)

        return np.column_stack((x, y, z, i)).flatten()

    def next_frame(self):
        """
        Возвращает:
        - frame: изображение (или None, если конец)
        - lidar_data: плоский массив точек ИЛИ пустой массив, если для этого кадра нет НОВОГО скана
        - timestamp: время видео
        """
        if not self.cap.isOpened(): return None, None, None

        ret, frame = self.cap.read()
        if not ret: return None, None, None

        # Текущее время видео (с 0.0)
        current_video_time = self.frame_idx / self.fps
        self.frame_idx += 1

        target_lidar_data = np.array([], dtype=np.float32)

        if self.lidar_available and not self.eof_lidar:
            # 1. Загружаем первое сообщение, если его нет
            if self.cached_lidar_msg is None:
                self.cached_lidar_msg = self._get_next_lidar_tuple()

                # Инициализация Offset (начало отсчета)
                if self.cached_lidar_msg is not None and self.lidar_start_time is None:
                    self.lidar_start_time = self.cached_lidar_msg[2].publish_time * 1e-9
                    print(f"[Loader] Sync start. Lidar T0: {self.lidar_start_time:.3f}")

            # 2. Ищем сообщение, которое ближе всего к current_video_time, но не убегаем далеко вперед
            # Нам нужно найти сообщение L, такое что L.time <= current_video_time < (NextL.time)
            # Или просто самое близкое.

            while self.cached_lidar_msg is not None:
                raw_time = self.cached_lidar_msg[2].publish_time * 1e-9
                lidar_rel_time = raw_time - self.lidar_start_time

                # Если сообщение слишком старое (отстает больше чем на 0.1с) - выкидываем, берем следующее
                if lidar_rel_time < current_video_time - 0.1:
                    self.cached_lidar_msg = self._get_next_lidar_tuple()
                    continue

                # Если сообщение "в будущем" относительно видео (больше чем на полкадра вперед)
                # То мы пока не готовы его отдать (ждем, пока видео его догонит)
                if lidar_rel_time > current_video_time + (0.5 / self.fps):
                    break # Ждем следующего next_frame()

                # --- ПОПАДАНИЕ ВО ВРЕМЕННОЕ ОКНО ---

                # Проверяем, не отдавали ли мы этот скан уже?
                # (Сохраняем частоту сенсора: 1 скан - 1 раз)
                if raw_time != self.last_returned_lidar_time:
                    target_lidar_data = self._parse_pointcloud(self.cached_lidar_msg[3])
                    self.last_returned_lidar_time = raw_time

                    # Мы отдали данные, теперь можно сдвинуть кеш на следующее сообщение
                    self.cached_lidar_msg = self._get_next_lidar_tuple()
                else:
                    # Это сообщение уже было отдано на предыдущем кадре видео.
                    # Значит, для ТЕКУЩЕГО кадра видео нового скана нет.
                    # Возвращаем пустоту.
                    pass

                break

        return frame, target_lidar_data, current_video_time

    def close(self):
        self.cap.release()
        if hasattr(self, 'stream'):
            self.stream.close()