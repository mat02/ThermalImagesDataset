import os
os.environ["OMP_NUM_THREADS"]='2'

from ultralytics import YOLO
# Load a model
models = [
     'yolov8-lite-t-pose.yaml',
     'yolov8-lite-t-pose-1,5.yaml',
     'yolov8-lite-s-pose.yaml',
     'yolov8-lite-s-pose-1,05.yaml',
     'yolov8-tiny-pose.yaml',
     'yolov8n-pose.yaml',
     'yolov8s-pose.yaml',
     'ablation_yolov8-lite-t-pose-bifpn.yaml',
     'ablation_yolov8-lite-t-pose-stematt.yaml',
     'best_yolov8-lite-t-pose-stematt-bifpn-t.yaml',
     'best_yolov8-lite-t-pose-stematt-bifpn-s.yaml',
     'best_yolov8-lite-t-pose-stematt-bifpn-l.yaml',
]

datasets = {
    'TFW': './data/TFW-outdoor.yaml',
    # 'SF-TL54': './data/SF-TL54.yaml',
}


# Specify GPUs
devices = [0]

# Num of epochs
epochs = 300

# Img size
imgsz = 320

# Batch
batch = 64

# Others
patience = False
seed = 0

# Train the model
for model_name in models:
    for (ds_name, ds_path) in datasets.items():
        exp_name = f'{model_name[:-5]}-{ds_name}-seed{seed}'

        model = YOLO(model_name)
        model.train(
            data=ds_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=devices,
            patience=patience,
            seed=seed,
            name=exp_name,
            deterministic=True
        )
