import cv2
import numpy as np
import math

# --- 1. Transformation Helpers (Recreating missing imports) ---

def get_rot_matrix_from_angles(angles):
    """
    angles: [pitch, roll, yaw] in radians
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
    Returns 3x4 matrix [R | t]
    """
    T = np.zeros((3, 4))
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

# --- 2. Camera Class (Ported from user snippet) ---

class Camera:
    def __init__(self, R=None, t=None, f=None, principal_point=None, D=None, sz=None, logger=None):
        """
        Class Camera.
        @param R: 3x3 rotation matrix
        @param t: 3-D translation vector
        @param f: focal length
        @param principal_point: optical center
        @param D: distortion
        @param sz: image size (width, height)
        """
        self.logger = logger
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
        """Read camera config from yaml file."""
        try:
            fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
            if not fs.isOpened():
                raise FileNotFoundError(f"Could not open {yaml_path}")

            self.K = np.array(fs.getNode("K").mat())
            self.D = np.array(fs.getNode("D").mat())
            self.R = np.array(fs.getNode("r").mat())

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
        except Exception as e:
            print(f"[Camera] Error parsing YAML {yaml_path}: {str(e)}")
            self.K = None

    def get_camera_axis_remap_R(self):
        return self.camera_axis_remap.T @ self.R

    def compute_projection_matrix(self, meter_height):
        # Using 3x4 affine matrix here to fix the shape mismatch error
        P = np.dot(self.K, get_affine_matrix(self.get_camera_axis_remap_R(), self.t))
        A = np.zeros((3, 3))
        for k in range(3):
            A[k, 0] = P[k, 0]
            A[k, 1] = P[k, 1]
            A[k, 2] = meter_height * P[k, 2] + P[k, 3]
        return A

    def reproject_pt_with_height(self, pixel, height_meter=0.0):
        if self.K is None: return None
        h = height_meter
        A = self.compute_projection_matrix(h)
        try:
            xyw = np.linalg.inv(A) @ np.array([pixel[0], pixel[1], 1])
        except np.linalg.LinAlgError:
            return None

        if xyw[2] == 0: return None

        return [xyw[0] / xyw[2], xyw[1] / xyw[2], h]

# --- 3. Wrapper for pipeline ---

class VisualGeometry:
    def __init__(self, calib_path):
        self.camera = Camera()
        self.camera.read_from_yaml_file(calib_path)

    def pixel_to_3d_ground(self, u, v):
        pt3d = self.camera.reproject_pt_with_height((u, v), height_meter=0.0)
        if pt3d is None: return None
        return [pt3d[0], pt3d[1], pt3d[2]]