import numpy as np
import cv2
from ultralytics import YOLO

import warnings

warnings.filterwarnings('ignore', module='ultralytics.yolo.engine.results.Boxes')

class YoloV8FaceDetector:
    def __init__(self, config):
        self.config = config

        self.model = YOLO(self.config["weights"])
        self.model.overrides["verbose"] = self.config.get("verbose", False)


    def convert_kpts_to_relative(self, kpts, xy):
        kpts = kpts - np.array(xy)
        return kpts

    def split_keypoints(self, kpts, split):
        return { name: kpts[idxs] for (name, idxs) in split.items() }

    def detect(self, image):

        predictions = self.model.predict(image, **self.config["args"])

        boxes = predictions[0].boxes.cpu().numpy()
        keypoints = predictions[0].keypoints.cpu().numpy()

        faces = []
        for i in range(boxes.shape[0]):
            if "kpts_split" in self.config:
                face = {
                    "xyxyn": boxes.xyxyn[i],
                    "xyxy": boxes.xyxy[i],
                    "conf": boxes.conf[i],
                    "kpts_split": {
                        "kpts_conf": self.split_keypoints(keypoints.conf[i], self.config["kpts_split"]),
                        "kpts_xyn": self.split_keypoints(keypoints.xyn[i], self.config["kpts_split"]),
                        "kpts_xy": self.split_keypoints(keypoints.xy[i], self.config["kpts_split"]),
                        "kpts_rxyn": self.split_keypoints(self.convert_kpts_to_relative(keypoints.xyn[i], boxes.xyxyn[i][:2]), self.config["kpts_split"]),
                        "kpts_rxy": self.split_keypoints(self.convert_kpts_to_relative(keypoints.xy[i], boxes.xyxy[i][:2]), self.config["kpts_split"]),
                    },
                    "kpts_conf": keypoints.conf[i],
                    "kpts_xyn": keypoints.xyn[i],
                    "kpts_xy": keypoints.xy[i],
                    "kpts_rxyn": self.convert_kpts_to_relative(keypoints.xyn[i], boxes.xyxyn[i][:2]),
                    "kpts_rxy": self.convert_kpts_to_relative(keypoints.xy[i], boxes.xyxy[i][:2]),
                }
            else:
                face = {
                    "xyxyn": boxes.xyxyn[i],
                    "xyxy": boxes.xyxy[i],
                    "conf": boxes.conf[i],
                    "kpts_conf": keypoints.conf[i],
                    "kpts_xyn": keypoints.xyn[i],
                    "kpts_xy": keypoints.xy[i],
                }
            faces.append(face)

        return faces