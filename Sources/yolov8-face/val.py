import os
os.environ["OMP_NUM_THREADS"]='2'

from ultralytics import YOLO
import pandas as pd
import glob
# Load a model

models_list = glob.iglob('runs/pose/*/**/weights/best.pt', recursive=True)

# Specify GPUs
devices = [0]

# Num of epochs
epochs = 300

# Img size
imgsz = 320

# Batch
batch = 64

# Others

# Val models
df = None
for model_path in models_list:
    print(model_path)
    model = YOLO(model_path)
    n_l, n_p, n_g, flops = model.info(True, True)
    model_info = {
        'num_of_layers': n_l,
        'num_of_parameters': n_p,
        'GFLOPS': flops
    }
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    if df is None:
        rdict = metrics.results_dict
        rdict = rdict | metrics.speed | model_info
        df = pd.DataFrame(rdict, index=[model.ckpt_path])
    else:
        rdict = metrics.results_dict
        rdict = rdict | metrics.speed | model_info
        new_metrics = pd.DataFrame(rdict, index=[model.ckpt_path])
        df = pd.concat([df, new_metrics], ignore_index=False)
    
if df is not None:
    df.to_csv("val.csv", sep=';')