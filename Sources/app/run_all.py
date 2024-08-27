SEQUENCES = [
# '../datasets/FLIR_A35/Martyna1',
# '../datasets/FLIR_A35/Person2',
# '../datasets/FLIR_A35/Mateusz2',
# '../datasets/FLIR_A35/Cinek1',
# '../datasets/FLIR_A35/Zuza',
# '../datasets/FLIR_A35/Person1',
# '../datasets/FLIR_A35/Mateusz1',
# '../datasets/FLIR_A35/Anita1',
'../datasets/FLIR_E6_Yawning/1_Man_NoGlasses_Yawning3',
'../datasets/FLIR_E6_Yawning/2_Man_Glasses1_Normal',
'../datasets/FLIR_E6_Yawning/3_Man_NoGlasses_Normal',
'../datasets/FLIR_E6_Yawning/3_Man_Glasses2_Yawning',
'../datasets/FLIR_E6_Yawning/1_Woman_NoGlasses_Yawning',
'../datasets/FLIR_E6_Yawning/1_Man_NoGlasses_Yawning2',
'../datasets/FLIR_E6_Yawning/1_Woman_Glasses2_Yawning',
'../datasets/FLIR_E6_Yawning/3_Man_Glasses1_Normal',
'../datasets/FLIR_E6_Yawning/3_Man_Glasses1_Yawning',
'../datasets/FLIR_E6_Yawning/1_Man_Glasses2_Yawning',
'../datasets/FLIR_E6_Yawning/1_Man_Glasses1_Yawning',
'../datasets/FLIR_E6_Yawning/2_Man_Glasses2_Normal',
'../datasets/FLIR_E6_Yawning/2_Woman_NoGlasses_Normal',
'../datasets/FLIR_E6_Yawning/2_Man_NoGlasses_Normal',
'../datasets/FLIR_E6_Yawning/3_Man_NoGlasses_Yawning',
'../datasets/FLIR_E6_Yawning/1_Woman_Glasses1_Yawning',
'../datasets/FLIR_E6_Yawning/1_Man_NoGlasses_Normal',
'../datasets/FLIR_E6_Yawning/2_Man_Glasses2_Yawning',
'../datasets/FLIR_E6_Yawning/2_Woman_Glasses2_Normal',
'../datasets/FLIR_E6_Yawning/3_Woman_Glasses1_Normal',
'../datasets/FLIR_E6_Yawning/1_Man_Glasses2_Normal',
'../datasets/FLIR_E6_Yawning/1_Woman_NoGlasses_Normal',
'../datasets/FLIR_E6_Yawning/1_Man_Glasses1_Normal',
'../datasets/FLIR_E6_Yawning/1_Man_NoGlasses_Yawning5',
'../datasets/FLIR_E6_Yawning/3_Man_Glasses2_Normal',
'../datasets/FLIR_E6_Yawning/3_Woman_NoGlasses_Normal',
'../datasets/FLIR_E6_Yawning/1_Man_NoGlasses_Yawning4',
'../datasets/FLIR_E6_Yawning/2_Woman_Glasses1_Normal',
]

CONFIG_FILE = "config/app_config.yml"

import app
import os
import copy
import cv2
import matplotlib.pyplot as plt

args = {
    "conf": CONFIG_FILE,
    "output": "output/export",
    "output_csv": "output/export_csv",
}

bad = []
if __name__ == "__main__":
    for ds in SEQUENCES:
        seq_args = copy.deepcopy(args)

        cfo = seq_args["conf_overwrites"] = []
        cfo.append(f"imageCapture.path={os.path.join(ds, 'Jpg')}")
        if "FLIR_E6" not in ds:
            if "Person2" in ds:
                cfo.append(f"headPoseEstimator.machine.head_offset=[-19, 0, 0]")
            elif "Person1" in ds:
                cfo.append(f"headPoseEstimator.machine.head_offset=[20, -10, 0]")
            elif "Cinek1" in ds:
                cfo.append(f"headPoseEstimator.machine.head_offset=[0, 10, 0]")
            elif "Mateusz2" in ds:
                cfo.append(f"headPoseEstimator.machine.head_offset=[0, 8, 0]")
            else:
                cfo.append(f"headPoseEstimator.machine.head_offset=[0, 0, 0]")
        # cfo.append(f"yawnDetector.csv_output_path={args['output']}")

        # seq_args["display"] = False
        seq_args["display"] = True
        seq_args["progress"] = True
        seq_args["delay"] = 1
        seq_args["output"] = os.path.join(seq_args["output"], os.path.basename(ds) + ".jpg")
        # del seq_args['output']
        seq_args["output_csv"] = os.path.join(seq_args["output_csv"], os.path.basename(ds))

        print(seq_args)
        # input()

        app.run(seq_args)
        cv2.destroyAllWindows()
                
print(bad)