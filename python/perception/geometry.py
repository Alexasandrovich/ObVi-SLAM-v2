import cv2
import numpy as np
import math

class Camera:
    def __init__(self, R=None, t=None, f=None, principal_point=None, D=None, sz=None, logger=None):
        """
        Class Camera.
        @param R: 3x3 rotation matrix rotates corresponding axes of each frame into each other (extrinsic parameter)
        @param t: 3-D translation vector defines relative positions of each frame (extrinsic parameter)
        @param f: focal length (intrinsic parameter)
        @param principal_point: the point on the image plane onto which the perspective center is projected (intrinsic parameter)
        @param D: geometric distortion introduced by the lens (intrinsic parameter)
        @param sz: image size (width, height)

        xy axes of the camera are in the image axes and z axes are forward,
        with the camera position being y-forward, x-right, z-up
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
        """
        Read camera config from yaml file.

        @param yaml_path: path to yaml file.
        """
        try:
            fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
            self.K = np.array(fs.getNode("K").mat())
            self.D = np.array(fs.getNode("D").mat())
            self.R = np.array(fs.getNode("r").mat())
            if self.R.shape == (3, 1):  # pitch, roll, yaw
                self.R = self.get_rot_matrix_from_angles(np.array([self.R[0][0],
                                                              self.R[1][0],
                                                              self.R[2][0]]))
            self.t = np.array(fs.getNode("t").mat())
            sz_node = fs.getNode("sz")
            width = int(sz_node.at(0).real())
            height = int(sz_node.at(1).real())
            self.sz = [width, height]

            self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.K, self.D, None, self.K, tuple(self.sz),
                                                               cv2.CV_32FC1)
        except Exception as e:
            print(f"An error occurred while parsing the YAML file: {yaml_path} - {str(e)}")

    def get_camera_axis_remap_R(self):
        """

        @return:
        """
        return self.camera_axis_remap.T @ self.R

    def project_pt(self, pt):
        """

        @param pt: 3D world point
        @return: uv coordinates of input 3D point
        """
        pt = self.get_camera_axis_remap_R() @ (pt - self.t)
        result = self.K @ pt
        if result[2]:  # not zero
            return result[0] / result[2], result[1] / result[2]
        else:
            return 0, 0

    def get_rot_matrix_from_angles(self, angles, order=None):
        """

        @param angles:
        @param order:
        @return:
        """
        cos_x, cos_y, cos_z = np.cos(angles)
        sin_x, sin_y, sin_z = np.sin(angles)

        R_z = np.array([[cos_z, -sin_z, 0],
                        [sin_z, cos_z, 0],
                        [0, 0, 1]])

        R_y = np.array([[cos_y, 0, sin_y],
                        [0, 1, 0],
                        [-sin_y, 0, cos_y]])

        R_x = np.array([[1, 0, 0],
                        [0, cos_x, -sin_x],
                        [0, sin_x, cos_x]])

        if order is None or any(i >= 3 for i in order):
            return np.dot(R_z, np.dot(R_x, R_y)).T

        Rs = [R_x, R_y, R_z]

        return np.dot(Rs[order[0]], np.dot(Rs[order[1]], Rs[order[2]]))

    def get_angles_from_rotation_matrix(self, R):
        """

        @param R:
        @return:
        """
        result = np.zeros(3)
        result[2] = np.arctan2(-R[0][1], R[1][1])
        result[1] = np.arctan2(-R[2][0], R[2][2])

        temp_arg = R[2][1]
        temp_arg = max(-1.0, min(temp_arg, 1.0))

        result[0] = np.arcsin(temp_arg)
        return result

    def get_affine_matrix(self, R, t):
        """
        Returns the affine matrix from rotation matrix and translation vector.

        @param R: 3x3 rotation matrix.
        @param t: 3x1 translation vector.
        @return: 3x3 affine matrix.
        """
        return np.hstack((R, -np.dot(R, t)))


    def project_pts(self, pts):
        """
        Projects multiple 3D world points to 2D image coordinates in a vectorized manner.

        @param pts: 3D world points as a NumPy array of shape (3, N).
        @return: tuple of (u, v) pixel coordinates as NumPy arrays of shape (N,).
        """
        # Transform world coordinates to camera coordinates: X_cam = R * (X_world - t)
        transformed_pts = self.get_camera_axis_remap_R() @ (pts - self.t)  # Shape: (3, N)

        # Project the camera coordinates onto the image plane using the intrinsic matrix K
        projected = self.K @ transformed_pts  # Shape: (3, N)

        # Avoid division by zero
        Z = projected[2, :] + 1e-6

        # Compute pixel coordinates
        u = projected[0, :] / Z
        v = projected[1, :] / Z

        return u, v

    def compute_projection_matrix(self, meter_height):
        """
        Computes the projection matrix used for reprojecting image points
        into world coordinates at a given height.

        @param meter_height: The height in meters at which the points will be reprojected.
        @return: A 3x3 projection matrix for transforming pixel coordinates into world coordinates.
        """
        P = np.dot(self.K, self.get_affine_matrix(self.get_camera_axis_remap_R(), self.t))
        A = np.zeros((3, 3))
        for k in range(3):
            A[k, 0] = P[k, 0]
            A[k, 1] = P[k, 1]
            A[k, 2] = meter_height * P[k, 2] + P[k, 3]
        return A

    def reproject_pt_with_height(self, pixel, height_meter=0.0):
        """
        Reprojects a single pixel coordinate from image space to world space
        at a specified height.

        @param pixel: pixel coordinates (x, y) in the image.
        @param height_meter: the height in meters at which the point is reprojected (default is 0.0).
        @return: world coordinates [X, Y, Z] at the given height. Returns None if the transformation is not possible.
        """
        h = height_meter
        A = self.compute_projection_matrix(h)
        xyw = np.linalg.inv(A) @ np.array([pixel[0], pixel[1], 1])
        if xyw[2] == 0:
            return None

        return [[xyw[0] / xyw[2]], [xyw[1] / xyw[2]], [h]]

    def reproject_pts_with_height(self, pixels, height_meter=0.0):
        """
        Reprojects a single pixel coordinate from image space to world space
        at a specified height.

        @param pixels: pixels coordinates (x, y) in the image.
        @param height_meter: the height in meters at which the point is reprojected (default is 0.0).
        @return: world coordinates [X, Y, Z] at the given height.
        """
        h = height_meter
        A = self.compute_projection_matrix(h)

        pixels_homogeneous = np.hstack((pixels, np.ones((pixels.shape[0], 1))))
        xyw = np.linalg.inv(A) @ pixels_homogeneous.T

        mask = xyw[2, :] != 0
        world_points = np.zeros((pixels.shape[0], 3))
        world_points[mask, 0] = xyw[0, mask] / xyw[2, mask]
        world_points[mask, 1] = xyw[1, mask] / xyw[2, mask]
        world_points[mask, 2] = h

        return world_points[mask]

    def calculate_horizon_y_coordinate(self):
        """
        Calculate the y-coordinate of the horizon in the image.
        This method assumes the camera's z-axis is pointing upwards.

        @return: y-coordinate of the horizon.
        """
        # Horizon vector in camera coordinate system
        horizon_vector = np.array([0, 1, 0])

        # Adjust horizon vector based on camera orientation
        # Using the camera's extrinsic rotation matrix
        horizon_vector_world = np.dot(self.get_camera_axis_remap_R().T, horizon_vector)

        # Project horizon vector onto image plane using intrinsic matrix
        # This projection does not require appending 1 to the horizon_vector_world since it's not a position vector but a direction vector.
        horizon_image = np.dot(self.K, horizon_vector_world)

        # Calculate y-coordinate of the horizon in the image
        # The z-component of the projected vector is not needed for calculating the y-coordinate directly.
        if horizon_image[2] != 0:
            horizon_y = horizon_image[1] / horizon_image[2]
        else:
            horizon_y = None

        return max(0, self.sz[1] - horizon_y)

    def calculate_front_dead_zone(self):
        """
        Calculate the y-coordinate in 3d, which corresponds bottom of the image.

        @return: y-coordinate of the front dead zone.
        """
        pt3d = self.reproject_pt_with_height([self.sz[0] / 2, self.sz[1]])
        if pt3d is None:
            raise ValueError("self.camera.reproject_pt_with_height returned None!")
        return pt3d[1][0]

    def calc_undistort_point(self, point):
        """
        Calculate undistortion point.

        @param point: distorted point in camera space.
        @return: undistorted point in camera space.
        """
        dist_x, dist_y = self.D
        r_squared = point[0] ** 2 + point[1] ** 2 # r^2 = (x^2 + y^2)
        undistorted_point = point * (1 + dist_x * r_squared + dist_y * r_squared ** 2)  # calculate undistortion
        undistorted_point[2] = 1  # ensure that last factor is still 1

        return undistorted_point

    def undistort_image(self, img):
        """
        Undistort image.

        @param img: image with distortion.
        @return: undistorted image.
        """
        return cv2.remap(img, self.mapx, self.mapy, interpolation=cv2.INTER_LINEAR)


class VisualGeometry:
    def __init__(self, calib_path):
        self.camera = Camera()
        self.camera.read_from_yaml_file(calib_path)

    def pixel_to_3d_ground(self, u, v):
        """
        Преобразует пиксель в 3D координаты в системе робота
        """
        pt3d = self.camera.reproject_pt_with_height((u, v), height_meter=0.0)
        if pt3d is None:
            return None

        return [-pt3d[0][0], -pt3d[1][0], pt3d[2][0]]