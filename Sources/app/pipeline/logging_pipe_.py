import os
from datetime import datetime

class LoggingPipe:
    def __init__(self, faces_key, output_path, *args, **kwargs):
        self.faces_key = faces_key
        self.output_path = output_path

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)
        
        self.output_files = {}

        self.prefix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.headers = ['timestamp', 'face id', 'yaw', 'pitch', 'roll', 'pose_state', 'pose_stable_state',
                        'head_drop_event', 'distracted_event', 'is_yawning']
        


    def close(self):
        for h in self.output_files.values():
            h.close()  

    def __call__(self, data):
        return self.save(data)
    
    def save_to_file(self, id, features):
        if id not in self.output_files:
            filename = f"{self.prefix}_face_{id}_yawning.txt" 
            filepath = os.path.join(self.output_path, filename)
            self.output_files[id] = open(filepath, "w")
            self.output_files[id].write("{}\n".format(";".join(self.headers)))
        
        output = "{}\n".format(";".join([f"{f:.6f}" if isinstance(f, float) else f"{f}" for f in features]))
        # print(output)
        self.output_files[id].write(output)

    def save(self, data):
        if self.faces_key not in data:
            return data
        
        timestamp = data["idx"]
        
        faces = data[self.faces_key]

        for id, face in faces.items():
            head_drop_event = False
            distracted_event = False
            for e in face['pose']['state_events']:
                if e.event == 'head_drop':
                    head_drop_event = True
                elif e.event == 'distracted':
                    distracted_event = True

            # timestamp, face id, yaw, pitch, roll, pose_state, pose_stable_state, head_drop_event, distracted_event, is_yawning
            features = [
                timestamp,
                id,
                face['pose']['yaw'],
                face['pose']['pitch'],
                face['pose']['roll'],
                face['pose']['current_state'],
                face['pose']['current_stable_state'],
                1 if head_drop_event else 0,
                1 if distracted_event else 0,
                1 if face['yawning'] else 0,
            ]
            self.save_to_file(id, features)

        return data
