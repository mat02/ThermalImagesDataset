from modules.head_pose_estimator.pose_estimator import PoseEstimator
from scipy.spatial.transform import Rotation as R
from collections import deque, Counter
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional
import copy

@dataclass()
class Event:
    event: str
    state: str
    previous_state: str
    previous_state_len: int
    timestamp: int
    message: Optional[str] = ""

class HeadPoseStateMachine:
    states = ["normal", "looking_left", "looking_right", "head_up", "head_down", "unknown"]
    allowed_angles = {
        "normal": [[170, 190], [-10, 8], [-15, 15]], # yaw, pitch, roll (min, max)
        "looking_left": [[190, 270], [-10, 15], [-15, 15]],
        "looking_right": [[90, 170], [-10, 15], [-15, 15]],
        "head_up": [[-360, 360], [7, 60], [-180, 180]],
        "head_down": [[-360, 360], [-60, -10], [-180, 180]],
    }

    def __init__(self, 
                 stability_threshold=0.5, 
                 stability_len=10, 
                 max_history=100, 
                 head_offset=[0.0, 0.0, 0.0],
                 head_drop_threshold=3,
                 distraction_threshold=10):
        # Init history
        self.state_history = deque(maxlen=max_history)
        self.head_pose_history = deque(maxlen=max_history)

        # Set initial state to 'unknown'
        self.state = "unknown"
        self.last_stable_state = None
        self.head_offset = head_offset
        self.last_pose = np.array(self.head_offset)

        # Stable state params
        self.stability_threshold = stability_threshold
        self.stability_len = stability_len
        self.updates_since_stable = 0

        # Other parameters
        self.head_drop_threshold = head_drop_threshold
        self.distraction_threshold = distraction_threshold

    def copy(self, deep=True):
        return copy.deepcopy(self)

    def update(self, yaw, pitch, roll, timestamp=None):
        head_pose = np.array([yaw, pitch, roll])
        # apply offset
        head_pose += self.head_offset
        # check possible states
        possible_states = []
        closest_state = None
        closest_state_dist = np.inf
        for state in HeadPoseStateMachine.states:
            if state not in HeadPoseStateMachine.allowed_angles:
                continue
            if (
                HeadPoseStateMachine.allowed_angles[state][0][0] <= head_pose[0] <= HeadPoseStateMachine.allowed_angles[state][0][1]
            and
                HeadPoseStateMachine.allowed_angles[state][1][0] <= head_pose[1] <= HeadPoseStateMachine.allowed_angles[state][1][1]
            and
                HeadPoseStateMachine.allowed_angles[state][2][0] <= head_pose[2] <= HeadPoseStateMachine.allowed_angles[state][2][1]
            ):
                possible_states.append(state)
                state_min, state_max = \
                    np.array([HeadPoseStateMachine.allowed_angles[state][n][0] for n in range(3)]), \
                    np.array([HeadPoseStateMachine.allowed_angles[state][n][1] for n in range(3)])
                dist = np.min([np.linalg.norm([head_pose - state_min]), np.linalg.norm([head_pose - state_max])])
                if dist < closest_state_dist:
                    closest_state = state
                    closest_state_dist = dist

        # Compute current short term state
        if len(possible_states) == 0:
            self.state = "unknown"
        elif self.state in possible_states:
            # hysteresis to prevent hopping between states (prefer old state)
            self.state = self.state
        else:
            self.state = closest_state

        # Check if long term state has changed and handle events
        if self.last_stable_state is None:
            self.last_stable_state = (self.state, 0, 1.0)
        stable_state = self.get_stable_state(max_history=self.stability_len, threshold=self.stability_threshold)
        events = []

        # Stable state change event
        # if self.last_stable_state is not None and self.last_stable_state[0] != stable_state[0]:
        #     events.append(
        #         Event(
        #             event="to_" + stable_state[0],
        #             state=stable_state[0],
        #             previous_state=self.last_stable_state[0],
        #             previous_state_len=self.updates_since_stable,
        #             timestamp=timestamp
        #         )
        #     )
        #     self.updates_since_stable = 0

        # State-specific events
        if stable_state:
            if stable_state[0] != "normal" and self.updates_since_stable > self.distraction_threshold:
                events.append(
                    Event(
                        event="distracted",
                        state=stable_state[0],
                        previous_state=self.last_stable_state[0],
                        previous_state_len=self.updates_since_stable,
                        timestamp=timestamp
                    )
                )
            if self.state == "head_down" and all(s in ["head_down", "unknown"] for s in list(self.state_history)[-self.head_drop_threshold-1:-1]):
                events.append(
                    Event(
                        event="head_drop",
                        state=stable_state[0],
                        previous_state=self.last_stable_state[0],
                        previous_state_len=self.updates_since_stable,
                        timestamp=timestamp
                    )
                )


        # Update variables
        self.last_pose = head_pose
        self.last_stable_state = stable_state
        self.updates_since_stable += 1

        return events

    def _get_state_percentage(self, state, max_history=5):
        if state not in HeadPoseStateMachine.states:
            raise ValueError(f"'{state}' is not a valid state. Valid states: {HeadPoseStateMachine.states}")

        if max_history > self.state_history.maxlen:
            max_history = self.state_history.maxlen
        c = Counter(list(self.state_history)[-max_history:])

        return c[state] / max_history

    def get_stable_state(self, max_history=None, threshold=None):
        if max_history is None:
            max_history = self.stability_len
        if threshold is None:
            threshold = self.stability_threshold

        if max_history > self.state_history.maxlen:
            max_history = self.state_history.maxlen

        c = Counter(list(self.state_history)[-max_history:])
        most_common = c.most_common(1)[0]
        percentage = most_common[1] / max_history

        return (most_common[0], most_common[1], percentage) if percentage >= threshold else self.last_stable_state

    @property
    def last_pose(self):
        return self.head_pose_history[0] if len(self.head_pose_history) > 0 else None

    @last_pose.setter
    def last_pose(self, pose):
        self.head_pose_history.append(pose)

    @property
    def state(self):
        return self.state_history[-1] if len(self.state_history) > 0 else None

    @state.setter
    def state(self, new_state):
        if new_state not in HeadPoseStateMachine.states:
            raise ValueError(f"'{new_state}' is not a valid state. Valid states: {HeadPoseStateMachine.states}")
        self.state_history.append(new_state)


class HeadPosePipe:

    def __init__(self, conf):
        self.conf = conf
        self.pe = PoseEstimator(**conf["estimator"])
        self.state_machines = {}

    def __call__(self, data):
        return self.run(data)

    def run(self, data):
        timestamp = data["idx"]
        image = data["image"]
        faces = data.get("tracked_faces", {})
        objects = data.get("tracked_objects", {})

        img_sz = image.shape[:2] # height, width
        self.pe.set_camera(width=img_sz[1], height=img_sz[0])

        face_num = 1
        for id, face in faces.items():
            if objects[id]['disappeared'] == 0:
                rotVec, tranVec = self.pe.solve_pose(face["kpts_xy"].astype('float64'))

                rotationMtx = R.from_rotvec(rotVec.flatten())
                pose = rotationMtx.as_euler('zxy', degrees=True)
                
                # pitch, roll, yaw = pose
                roll, pitch, yaw = pose
                yaw = yaw if yaw >= 0 else yaw + 360
            else:
                yaw = pitch = roll = -1e4 # this will cause state machine to report unknown pose
                tranVec = np.array([[0.0],[0.0],[0.0]])
                rotVec = np.array([[0.0],[0.0],[0.0]])
                
            # WARNING! This modifies original dict from input data!
            face["pose"] = {
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "translation": tranVec,
                "rotation": rotVec
            }

            # Register new state machines
            if id not in self.state_machines:
                self.state_machines[id] = HeadPoseStateMachine(**self.conf["machine"])

            # Update state machines
            events = self.state_machines[id].update(yaw, pitch, roll, timestamp=timestamp)
            face['pose']["current_state"] = self.state_machines[id].state
            face['pose']["current_stable_state"] = self.state_machines[id].get_stable_state()[0]
            face['pose']["state_events"] = events
                # print(f'timestamps:', events)

            face_num += 1

        # Deregister old state machines
        to_deregister = [id for id in self.state_machines.keys() if id not in faces.keys()]
        for id in to_deregister:
            self.state_machines.pop(id, None)

        return data
