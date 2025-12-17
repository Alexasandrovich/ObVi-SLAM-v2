import cv2
import numpy as np
import math

# --- 1. Transformation Helpers ---

def get_rot_matrix_from_angles(angles):
    """
    Создает матрицу поворота из углов Эйлера.
    angles: np.array([roll, pitch, yaw]) в радианах
    """
    sin_x, cos_x = math.sin(angles[0]), math.cos(angles[0])
    sin_y, cos_y = math.sin(angles[1]), math.cos(angles[1])
    sin_z, cos_z = math.sin(angles[2]), math.cos(angles[2])

    R_x = np.array([[1, 0, 0],
                    [0, cos_x, -sin_x],
                    [0, sin_x, cos_x]])

    R_y = np.array([[cos_y, 0, sin_y],
                    [0, 1, 0],
                    [-sin_y, 0, cos_y]])

    R_z = np.array([[cos_z, -sin_z, 0],
                    [sin_z, cos_z, 0],
                    [0, 0, 1]])

    return np.dot(R_z, np.dot(R_y, R_x))

def get_affine_matrix(R, t):
    """
    Создает аффинную матрицу 4x4 из R (3x3) и t (3x1).
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

# --- 2. Full Camera Class (Cleaned up) ---

class Camera:
    def __init__(self, R=None, t=None, f=None, principal_point=None, D=None, sz=None):
        """
        Class Camera.
        xy axes of the camera are in the image axes and z axes are forward,
        with the camera position being y-forward, x-right, z-up.
        """
        if R is None or t is None or f is None or principal_point is None:
            self.R = R
            self.t = t
            self.K = None
            self.D = D
            self.sz = sz
        else:
            # [R|T] - extrinsic
            self.R = R
            if self.R.shape == (3, 1):  # pitch, roll, yaw
                self.R = get_rot_matrix_from_angles(np.array([np.deg2rad(self.R[0][0]),
                                                              np.deg2rad(self.R[1][0]),
                                                              np.deg2rad(self.R[2][0])]))
            self.t = t
            self.mapx, self.mapy = None, None

            # intrinsics
            self.K = np.array([[f, 0, principal_point[0]],
                               [0, f, principal_point[1]],
                               [0, 0, 1]], dtype=np.float32)

            self.D = D
            self.sz = sz
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.K, self.D, None, self.K,
                                                               tuple(self.sz), cv2.CV_32FC1)

        self.camera_axis_remap = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    def read_from_yaml_file(self, yaml_path):
        """
        Read camera config from yaml file (OpenCV format).
        """
        try:
            fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
            if not fs.isOpened():
                raise FileNotFoundError(f"Could not open {yaml_path}")

            self.K = np.array(fs.getNode("K").mat())
            self.D = np.array(fs.getNode("D").mat())

            self.R = np.array(fs.getNode("r").mat())
            # Handle rotation: if vector -> convert to matrix
            if self.R.shape == (3, 1) or self.R.shape == (1, 3):
                # В файле градусы
                angles_rad = np.deg2rad(self.R.flatten())
                self.R = get_rot_matrix_from_angles(angles_rad)

            self.t = np.array(fs.getNode("t").mat())

            sz_node = fs.getNode("sz")
            width = int(sz_node.at(0).real())
            height = int(sz_node.at(1).real())
            self.sz = [width, height]

            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                self.K, self.D, None, self.K, tuple(self.sz), cv2.CV_32FC1)

            fs.release()
            print(f"[Camera] Loaded config. Size: {self.sz}, Height(t_z): {self.t[2]}")

        except Exception as e:
            print(f"[Camera] Error parsing YAML {yaml_path}: {str(e)}")
            self.K = None

    def get_camera_axis_remap_R(self):
        return self.camera_axis_remap.T @ self.R

    def compute_projection_matrix(self, meter_height):
        """
        Computes the projection matrix used for reprojecting image points
        into world coordinates at a given height.
        """
        # 1. Получаем полную аффинную матрицу 4x4 (World -> Camera)
        affine_full = get_affine_matrix(self.get_camera_axis_remap_R(), self.t)

        # 2. P_extrinsic (3x4) = [R | t]
        affine_3x4 = affine_full[:3, :]

        # 3. Projection Matrix P = K * [R | t] -> (3x4)
        P = np.dot(self.K, affine_3x4)

        # 4. Construct solving matrix A (3x3)
        # We solve for X, Y given Z=meter_height.
        # [u*w, v*w, w]^T = P * [X, Y, Z, 1]^T
        # Z is fixed constant.
        A = np.zeros((3, 3))
        for k in range(3):
            A[k, 0] = P[k, 0] # Coeff for X
            A[k, 1] = P[k, 1] # Coeff for Y
            # P[k, 2] * Z + P[k, 3] * 1 becomes the constant term
            A[k, 2] = meter_height * P[k, 2] + P[k, 3]
        return A

    def reproject_pt_with_height(self, pixel, height_meter=0.0):
        """
        Reprojects a single pixel coordinate from image space to world space
        at a specified height.
        @return: [x, y, z] list or None
        """
        if self.K is None: return None

        h = height_meter
        A = self.compute_projection_matrix(h)

        # Solve A * [x, y, 1] = s * [u, v, 1]
        # Inverse mapping: [x, y, 1] ~ A_inv * [u, v, 1]
        try:
            xyw = np.linalg.inv(A) @ np.array([pixel[0], pixel[1], 1])
        except np.linalg.LinAlgError:
            return None

        if xyw[2] == 0:
            return None

        # Normalize homogeneous coords
        x = xyw[0] / xyw[2]
        y = xyw[1] / xyw[2]

        return [x, y, h]

    def undistort_image(self, img):
        return cv2.remap(img, self.mapx, self.mapy, interpolation=cv2.INTER_LINEAR)

# --- 3. Wrapper for your architecture ---

class VisualGeometry:
    def __init__(self, calib_path):
        """
        calib_path: path to sensors.yaml
        """
        self.camera = Camera()
        self.camera.read_from_yaml_file(calib_path)

    def pixel_to_3d_ground(self, u, v):
        """
        Calculates 3D position of a pixel on ground plane (Z=0).
        Returns [x, y, z] in World Frame (defined by calib).
        """
        pt3d = self.camera.reproject_pt_with_height((u, v), height_meter=0.0)

        if pt3d is None:
            return None

        # Basic filtering (e.g. don't pick points 1km away)
        dist = np.sqrt(pt3d[0]**2 + pt3d[1]**2)
        if dist > 50.0 or dist < 1.0:
            return None

        return pt3d