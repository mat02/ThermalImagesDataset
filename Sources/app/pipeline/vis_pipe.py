from utils.vis import visualize_face_locations, visualize_image_info

import cv2
from copy import deepcopy

from modules.head_pose_estimator.pose_estimator import PoseEstimator


class VisPipe:

    def __init__(self, in_key, out_key, conf, pose_estimator=None):
        self.in_key = in_key
        self.out_key = out_key
        self.conf = conf
        self.pe = deepcopy(pose_estimator)

    def __call__(self, data):
        
        image = data[self.in_key]

        if self.conf.get('resize', False):
            image = cv2.resize(image, dsize=None, fx=self.conf['resize']['fx'], fy=self.conf['resize']['fy'])
            scale = (self.conf['resize']['fx'], self.conf['resize']['fy'])
        else:
            image = image.copy()
            scale = (1.0, 1.0)
        
        if self.conf.get("colormap", False):
            image = cv2.applyColorMap(image, cv2.COLORMAP_INFERNO)

        # visualize_image_info(image, data["filename"])

        faces = data.get("tracked_faces", {})

        visualize_face_locations(image, faces, scale)

        face_num = 0
        for id, face in faces.items():
            metrics = face['metrics']

            pose = face.get('pose', None)
            if pose is not None:
                # draw box
                self.pe.set_camera(image.shape[1], image.shape[0])
                self.pe.draw_annotation_box(image, pose['rotation'], pose['translation'], color=(0, 255, 0))

                # cv2.putText(image, f"State: {pose['current_state']}", (int(10 + 5 * scale[0]), int(100 + 40  * scale[1] * face_num)), 0, 1, (0, 255, 0))
                # cv2.putText(image, f"{pose['current_state']}: {pose['yaw']:.1f}, {pose['pitch']:.1f}, {pose['roll']:.1f}", (int(10 + 5 * scale[0]), int(100 + 40  * scale[1] * face_num)), 0, 1, (0, 255, 0))
                cv2.putText(image, f"Perlook: {metrics['perlook']:.2f}, Yf: {metrics['yawn_freq']:.2f}/min, HDf: {metrics['head_drop_freq']:.2f}/min, ", (int(10 + 5 * scale[0]), int(25 + 15  * scale[1] * face_num)), 0, 0.5, (0, 255, 0))
                # cv2.putText(image, f"Perlook: {metrics['perlook']:.2f}, Perstatic: {metrics['perstatic']:.2f}, Yf: {metrics['yawn_freq']:.2f}/min, HDf: {metrics['head_drop_freq']:.2f}/min, ", (int(10 + 5 * scale[0]), int(25 + 20  * scale[1] * face_num)), 0, 0.5, (0, 255, 0))
                cv2.putText(image, f"State: {pose['current_state']}", (int(10 + 5 * scale[0]), int(25 + 15  * scale[1] * (face_num + 1))), 0, 0.5, (0, 255, 0))
                cv2.putText(image, f"Distraction score: {face['distraction_score']:.2f}, Fatigue score: {face['fatigue_score']:.2f}", (int(10 + 5 * scale[0]), int(25 + 15  * scale[1] * (face_num + 2))), 0, 0.5, (0, 255, 0))
            
            yawning = face.get('yawning', False)
            if yawning:
                cv2.putText(image, f"Yawning", (int(10 + 5 * scale[0]), int(140 + 40  * scale[1] * face_num)), 0, 1, (0, 0, 255))
            if face.get('distraction_alert', False):
                cv2.putText(image, f"Distraction alert!", (int(10 + 5 * scale[0]), int(140 + 40  * scale[1] * (face_num + 1))), 0, 1, (0, 0, 255), 2)

            if face.get('fatigue_alert', False):
                cv2.putText(image, f"Fatigue alert!", (int(10 + 5 * scale[0]), int(140 + 40  * scale[1] * (face_num + 2))), 0, 1, (0, 0, 255), 2)

            face_num += 1
        
        data[self.out_key] = image
        return data
