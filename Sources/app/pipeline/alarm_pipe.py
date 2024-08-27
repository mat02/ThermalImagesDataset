from modules.face_detector import FaceDetector
from collections import deque
from copy import deepcopy


class AlarmPipe:

    def __init__(self, conf, faces_key):
        self.faces_key = faces_key
        self.metrics = {}

        self.conf = conf
        print(self.conf)

    def __call__(self, data):
        if self.faces_key not in data:
            return data
        
        timestamp = data["idx"]
        
        faces = data[self.faces_key]

        for id, face in faces.items():

            distraction_score = 0.0

            distraction_score = \
                    face['metrics']['perlook'] * self.conf['distraction']['perlook'] + \
                    face['metrics']['yawn_freq'] * self.conf['distraction']['yawn_freq'] + \
                    face['metrics']['head_drop_freq'] * self.conf['distraction']['head_drop_freq']
            

            face['distraction_alert'] = distraction_score >= self.conf['distraction']['threshold']
            face['distraction_score'] = distraction_score

            fatigue_score = \
                    face['metrics']['perlook'] * self.conf['fatigue']['perlook'] + \
                    face['metrics']['yawn_freq'] * self.conf['fatigue']['yawn_freq'] + \
                    face['metrics']['head_drop_freq'] * self.conf['fatigue']['head_drop_freq']

            face['fatigue_alert'] = fatigue_score >= self.conf['fatigue']['threshold']
            face['fatigue_score'] = fatigue_score

        return data
