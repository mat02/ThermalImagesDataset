import cv2
import numpy as np
import torch
from collections import deque
from skimage.feature import hog
from skimage import exposure
from utils.vis import draw_keypoints, save_thermo_image
from models.yawnDetection.LSTMClassifier import LSTMClassifier


MOUTH_SIZE = (48, 48)

class YawnDetectionPipe:

    def __init__(self, conf):
        """
        This pipeline uses LSTM network for yawn detection.
        """
        self.conf = conf
        self.yawn_data = {}

        self.dil_kernel = np.ones((2, 2), np.uint8)

        weights = self.conf['model'].pop('weights')
        self.model = LSTMClassifier.load_from_checkpoint(
            weights,
            **self.conf['model']
        ).to('cuda')
        self.model.eval()

    def __call__(self, data):
        return self.detect(data)
    
    def detect(self, data):
        idx = data["idx"]
        image = data["image"][:, :, 0] if len(data["image"].shape) > 2 else data["image"]

        objects = data.get("tracked_objects", {})
        faces = data.get("tracked_faces", {})

        # TODO:
        # Skip if head pose is not suitable???
        # Reset metrics?

        for id, face in faces.items():
            obj = objects[id]
            existing_face = self.yawn_data.get(id, None)
            if not existing_face:
                 existing_face = {
                      "rect": deque([], maxlen=self.conf['model']['seq_len']),
                      "keypoints": deque([], maxlen=self.conf['model']['seq_len']), 
                      "features": deque([], maxlen=self.conf['model']['seq_len']),
                      "yawns": {},
                 }
                 self.yawn_data[id] = existing_face
            
            if "max_disappeared" in self.conf and obj["disappeared"] > self.conf["max_disappeared"]:
                # Too many missing frames, abort.
                del self.yawn_data[id]
            
            if obj["disappeared"] > 0:
                # Duplicate data if possible
                if len(existing_face["features"]) > 0:
                    existing_face["features"].append(existing_face["features"][-1])
                else:
                    # Clear other data to keep alignment and skip this face
                    existing_face["rect"].clear()
                    existing_face["keypoints"].clear()
                    continue

            # Get mouth RoI kpts
            mouth_roi_kpts = np.array([
                # x0, y0, x1, y1
                face["kpts_split"]["kpts_xy"]["mouth"][0, 0],
                face["kpts_split"]["kpts_xy"]["nose"][0, 1],
                face["kpts_split"]["kpts_xy"]["mouth"][1, 0],
                face["kpts_split"]["kpts_xy"]["chin"][0, 1],
            ]).round().astype(np.int32)
            
            # Extract mouth RoI image
            mouth_roi_image = image[
                mouth_roi_kpts[1]:mouth_roi_kpts[3],
                mouth_roi_kpts[0]:mouth_roi_kpts[2],
            ]

            # FIXME: What to do if image too small?
            # if mouth_roi_image.shape[0] < 5 or mouth_roi_image.shape[1] < 5:
            #     # mouth area too small, skip this one
            #     continue
            face_img = image[
                int(face["xyxy"][1]):int(face["xyxy"][3]),
                int(face["xyxy"][0]):int(face["xyxy"][2]),
            ]
            scale = 3.0
            face_img = save_thermo_image(face_img, '', True, scale, False)
            face_img = draw_keypoints(face_img, face['kpts_rxy'], scale)
            face_img = save_thermo_image(face_img, 'debug/face.jpg', False, 1.0, True)

            

            # Dilate to remove noise and resize
            face_img = save_thermo_image(mouth_roi_image, 'debug/mouth_roi_raw.jpg', True, scale, True)
            mouth_roi_image = cv2.dilate(mouth_roi_image, self.dil_kernel, iterations=1)
            face_img = save_thermo_image(mouth_roi_image, 'debug/mouth_roi_dilated.jpg', True, scale, True)
            mouth_roi_image = cv2.resize(mouth_roi_image, tuple(self.conf.get("feature_image_size")))
            face_img = save_thermo_image(mouth_roi_image, 'debug/mouth_roi_resized.jpg', True, scale, True)

            # Compute feature descriptor
            feature_descriptor, hog_image = hog(mouth_roi_image, orientations=9, pixels_per_cell=(8, 8),
                                        cells_per_block=(2, 2), visualize=True)
            
            hog_image = exposure.rescale_intensity(hog_image, out_range=(0, 255))
            hog_image = cv2.resize(hog_image, dsize=None, fx=scale, fy=scale)
            cv2.imwrite('debug/hog.jpg', hog_image)
            
            
            # Store data
            existing_face['rect'].append(face["xyxy"])
            existing_face['keypoints'].append(face["kpts_split"]["kpts_xy"])
            existing_face['features'].append(feature_descriptor)


            if len(existing_face['features']) >= self.conf['model']['seq_len']:
                x = np.array([existing_face['features']])
                x = torch.Tensor(x).to('cuda')

                preds = self.model.predict(x)
                preds = np.argmax(preds)
                existing_face['yawns'][idx] = {
                    'start': idx - self.conf['model']['seq_len'],
                    'end': idx,
                    'state': preds,
                }

                face['yawning'] = preds == 1
            else:
                face['yawning'] = 0


        # remove old faces
        to_remove = [id for id in self.yawn_data.keys() if id not in faces]
        for id in to_remove:
            self.yawn_data.pop(id, None)

        # data['yawning'] = {
        #     k: v['yawns'][idx]['state'] for (k, v) in self.yawn_data.items()
        #     if idx in v['yawns']
        # }

        return data
