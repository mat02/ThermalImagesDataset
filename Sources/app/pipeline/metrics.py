from modules.face_detector import FaceDetector
from collections import deque
from copy import deepcopy


class MetricsPipe:

    def __init__(self, conf, faces_key):
        self.faces_key = faces_key
        self.metrics = {}

        self.conf = conf

    def __call__(self, data):
        if self.faces_key not in data:
            return data
        
        timestamp = data["idx"]
        
        faces = data[self.faces_key]

        for id, face in faces.items():
            if id not in self.metrics:
                self.metrics[id] = {
                    'perlook': deque(maxlen=self.conf['perlook']['history']),
                    'perstatic': deque(maxlen=self.conf['perstatic']['history']),
                    'yawn_freq': deque(maxlen=self.conf['yawn_freq']['history']),
                    'head_drops': deque(maxlen=self.conf['head_drop_freq']['history']),
                    'last_pose': None
                }
            self.metrics[id]['perlook'].append(
                0 if face['pose']['current_state'] == 'normal' else 1
            )
            if self.metrics[id]['last_pose'] is not None:
                dyaw, dpitch, droll = \
                    face['pose']['yaw'] - self.metrics[id]['last_pose']['yaw'], \
                    face['pose']['pitch'] - self.metrics[id]['last_pose']['pitch'], \
                    face['pose']['roll'] - self.metrics[id]['last_pose']['roll']
                self.metrics[id]['perstatic'].append(
                    1 if (dyaw <= self.conf['perstatic']['diff']['yaw']
                          and
                          dpitch <= self.conf['perstatic']['diff']['pitch']
                          and
                          droll <= self.conf['perstatic']['diff']['roll']
                         )
                    else 0
                )
            self.metrics[id]['yawn_freq'].append(
                1 if face['yawning'] else 0
            )

            head_drop_result = 0
            if len(face['pose']["state_events"]) > 0:
                for evt in face['pose']["state_events"]:
                    if evt.event == "head_drop":
                        head_drop_result = 1
            self.metrics[id]['head_drops'].append(head_drop_result)
            
            self.metrics[id]['last_pose'] = deepcopy(face['pose'])

            face['metrics'] = {
                'perlook': sum(self.metrics[id]['perlook']) / self.metrics[id]['perlook'].maxlen,
                # 'perstatic': sum(self.metrics[id]['perstatic']) / self.metrics[id]['perstatic'].maxlen,
                'yawn_freq': sum(self.metrics[id]['yawn_freq']) / self.conf['yawn_freq']['yawn_len'] / self.conf['yawn_freq']['window'],
                'head_drops': sum(self.metrics[id]['head_drops']),
                'head_drop_freq': sum(self.metrics[id]['head_drops']) / self.conf['head_drop_freq']['window'],
            }

        return data
