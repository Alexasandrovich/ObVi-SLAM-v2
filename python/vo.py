import cv2
import numpy as np

class SimpleVisualOdometry:
    def __init__(self, cam_matrix):
        self.orb = cv2.ORB_create(3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.K = cam_matrix
        self.last_img = None
        self.last_kp = None
        self.last_des = None

    def track(self, img):
        # Конвертация в Grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        kp, des = self.orb.detectAndCompute(gray, None)

        if self.last_img is None:
            self.last_img = gray
            self.last_kp = kp
            self.last_des = des
            return np.eye(4) # Первая поза - единичная

        # Матчинг фич
        matches = self.bf.match(self.last_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        # Нужно хотя бы 5 точек для расчета движения
        if len(matches) < 10:
            return np.eye(4)

        # Извлекаем точки
        pts1 = np.float32([self.last_kp[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

        # Вычисляем Essential Matrix и движение
        E, mask = cv2.findEssentialMat(pts2, pts1, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        if E is None or E.shape != (3,3):
            return np.eye(4)

        _, R, t, mask = cv2.recoverPose(E, pts2, pts1, self.K)

        # Создаем матрицу трансформации 4x4
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten() # Важно: VO дает t с неизвестным масштабом!

        # Обновляем состояние
        self.last_img = gray
        self.last_kp = kp
        self.last_des = des

        return T