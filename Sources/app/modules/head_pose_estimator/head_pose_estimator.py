from PoseEstimator import PoseEstimator
from scipy.spatial.transform import Rotation as R


class HeadPoseEstimator:
    def __init__(self, config):
        self.config = config

    def estimate(self, image, keypoints):

        imgsz = image.shape[:2] # width, height

        pe = PoseEstimator(img_size=imgsz[::-1], filename=self.config["model"])

        rot_vec, trans_vec = pe.solve_pose(keypoints)

        return rot_vec, trans_vec

    def get_head_pose(self, rot_vec):
        rotationMtx = R.from_rotvec(rot_vec.flatten())
        pitch, roll, yaw = rotationMtx.as_euler('zxy', degrees=True)

        return yaw, pitch, roll
