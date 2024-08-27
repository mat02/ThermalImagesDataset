"""Estimate head pose according to the facial landmarks"""
"""Modified by MK"""
import cv2
import numpy as np


class PoseEstimator:
    """Estimate head pose according to the facial landmarks"""

    def __init__(self, img_size=None, filename='model.txt'):
        self.size = img_size

        # 3D model points.
        self.model_points = self._get_full_model_points(filename)

        # Camera internals
        if img_size is None:
            print("WARNING! Image size hasn't been passed to pose estimator, use set_camera() first!")
        else:
            self.set_camera(width=self.size[0], height=self.size[1])

    def set_camera(self, width, height):
        # Camera internals
        self.focal_length = width
        self.camera_center = (width / 2, height / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = None
        self.t_vec = None

    def _get_full_model_points(self, filename='model.txt'):
        """Get 3D model points from file"""
        raw_value = []
        with open(filename) as file:
            for line in file:
                if line.lstrip().startswith("#"):
                    continue
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T

        # Transform the model into a front view.
        # model_points[:, 2] *= -1
        # model_points[:, 1] *= -1
        # model_points[:, 0] *= -1

        model_points = np.ascontiguousarray(model_points)

        return model_points

    def show_3d_model(self):
        from matplotlib import pyplot
        from mpl_toolkits.mplot3d import Axes3D
        fig = pyplot.figure()

        # ax = Axes3D(fig)
        ax = fig.add_subplot(111, projection='3d')

        x = self.model_points[:, 0]
        y = self.model_points[:, 1]
        z = self.model_points[:, 2]

        ax.scatter(x, y, z)
        ax.axis('square')
        pyplot.xlabel('x')
        pyplot.ylabel('y')
        # pyplot.show()
        fig.show()

    def solve_pose(self, image_points):
        """
        Solve pose from image points
        Return (rotation_vector, translation_vector) as pose.
        """
        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points, image_points, self.camera_matrix, self.dist_coeefs)
            self.r_vec = rotation_vector
            self.t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeefs,
            flags=cv2.SOLVEPNP_EPNP,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        return (rotation_vector, translation_vector)

    def draw_annotation_box(self, image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
        """Draw a 3D box as annotation of pose"""
        point_3d = []
        rear_size = 50
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 100
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))

        # Pitch
        point_3d.append((0, 0, rear_depth))
        point_3d.append((0, 0, front_depth))

        # Yaw
        point_3d.append((0, 0, rear_depth))
        point_3d.append((0, front_depth, rear_depth))

        # Roll
        point_3d.append((0, 0, rear_depth))
        point_3d.append((front_depth, 0, rear_depth))

        point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv2.polylines(image, [point_2d[:-6]], True, color, line_width, cv2.LINE_AA)

        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)

        self.draw_axes(image, rotation_vector, translation_vector)

    def draw_axis(self, img, R, t):
        points = np.float32(
            [[30, 0, 0], [0, 30, 0], [0, 0, 30], [0, 0, 0]]).reshape(-1, 3)

        axisPoints, _ = cv2.projectPoints(
            points, R, t, self.camera_matrix, self.dist_coeefs)

        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[0].ravel()), (255, 0, 0), 3)
        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[1].ravel()), (0, 255, 0), 3)
        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[2].ravel()), (0, 0, 255), 3)

    def draw_axes(self, img, R, t):
        img	= cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeefs, R, t, 30)

    def plot_skeleton_kpts(self, img, kpts, steps=1, orig_shape=None):
        #Plot the skeleton and keypoints for coco datatset
        # palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
        #                     [230, 230, 0], [255, 153, 255], [153, 204, 255],
        #                     [255, 102, 255], [255, 51, 255], [102, 178, 255],
        #                     [51, 153, 255], [255, 153, 153], [255, 102, 102],
        #                     [255, 51, 51], [153, 255, 153], [102, 255, 102],
        #                     [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
        #                     [255, 255, 255]])

        # Simplified pallete for 10 face keypoints
        palette = np.array([[200, 0, 128],
                            [0, 128, 128],
                            [0, 128, 255],
                            [128, 0, 128],
                            [0, 255, 50],
                            [255, 0, 255],
                            [255, 255, 100],
                            [0, 0, 255],
                            [67, 221, 255],
                            [20, 100, 255],
                            [100, 0, 100],
                            [100, 0, 100],
                            [0, 0, 255],
                            [0, 0, 255],
                            [0, 0, 255],
                            [0, 0, 255],
                            [0, 0, 255],
                            [0, 0, 255],
                            [100, 0, 100],
                            ])

        radius = 4
        num_kpts = len(kpts) // steps
        for kid in range(num_kpts):
            r, g, b = palette[kid]
            x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
            if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                if steps == 3:
                    conf = kpts[steps * kid + 2]
                    if conf < 0.5:
                        continue
                cv2.circle(img, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

    # def get_mouth(self, image, keypoints, out_size=(100, 100)):
    #     assert keypoints.shape[0] == 4
    #     # Output points for left, right, top, bottom mouth points
    #     dst = np.array([
    #         [0, out_size[1] // 2],
    #         [out_size[0], out_size[1] // 2],
    #         [out_size[0] // 2, 0],
    #         [out_size[0] // 2, out_size[1]],
    #     ])
    #     H, _ = cv2.findHomography(keypoints, dst)
    #     # H = cv2.getPerspectiveTransform(keypoints.astype('float32'), dst.astype('float32'))
    #     output_image = cv2.warpPerspective(image, H, out_size)
    #     return output_image

    def align_face(self, image, keypoints, rect, out_size=(100, 150), out_size_mouth=(64, 64)):
        assert keypoints.shape[0] == self.model_points.shape[0]
        # TODO:
        # - create homography matrix from existing camera pose?
        # - fix model coordinates in X axis to remove negation

        # Output points for left, right, top, bottom mouth points
        face_2d_model = self.model_points[:, 0:2].copy()
        face_2d_model[:, 0] *= -1
        H, _ = cv2.findHomography(keypoints, face_2d_model)

        rect = np.expand_dims(rect.reshape((2, -1)), axis=0)
        t_rect = np.squeeze(cv2.perspectiveTransform(rect, H))
        face_height = np.abs(face_2d_model[0, 1] - t_rect[0, 1])
        face_width = 120

        translation = [face_width / 2, face_height - face_2d_model[0, 1]]
        face_2d_model += translation
        H, _ = cv2.findHomography(keypoints, face_2d_model)

        output_image = cv2.warpPerspective(image, H, np.array((face_width, face_height)).astype(np.uint32))

        t_kpts = np.squeeze(cv2.perspectiveTransform(np.expand_dims(keypoints, axis=0), H))
        mouth_rect = np.array([t_kpts[8:10, 1], t_kpts[6:8, 0]]).flatten().astype(np.int32)
        mouth_image = output_image[mouth_rect[0]:mouth_rect[1], mouth_rect[2]:mouth_rect[3]]
        output_image = cv2.resize(output_image, out_size)
        mouth_image = cv2.resize(mouth_image, out_size_mouth)
        return output_image, mouth_image
