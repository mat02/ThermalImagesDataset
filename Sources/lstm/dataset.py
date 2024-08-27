from time import time
import math

import numpy as np
import pandas as pd
import cv2


# Sklearn tools
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import skimage

# Neural Networks
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import torchmetrics
import pytorch_lightning as pl

from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore', module='ultralytics.yolo.engine.results.Boxes')

# Plotting
import matplotlib.pyplot as plt

# Input data files are available in the read-only '../input/' directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from collections import OrderedDict
from glob import glob as iglob
from random import shuffle
import pylru
import re

import albumentations as A

class SingleSequenceDataset(Dataset):   
    '''
    Single sequence
    '''
    def __init__(self, sequences: list, model_name: str, name=None, seq_len: int=48, step: int=8, yawn_thr: float=0.5, augment: bool=False):
        self.sequences = sequences
        self.model_name = model_name

        
        self.model = YOLO(self.model_name)
        self.model.overrides["verbose"] = False
        
        self.name = name
        self.seq_len = seq_len
        self.step = step
        self.yawn_thr = yawn_thr
        self.augment = augment

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(5, p=0.5),
        ])
        self.mouth_roi_size = (48, 48)

        self.cache = pylru.lrucache(1024 * 200)
        
        self.det_config = {
            'conf': 0.7,
            'iou': 0.6,
            'imgsz': [256, 320],
        }
        
        self.data = {
            'normal': [],
            'yawning': [],
        }

        cached = False
        try:
            self.load_cache(self.name)
            print(f"{self.name} loaded from cached file")
            # for seq in self.data['yawning'] + self.data['normal']:
            #     s, e = seq['start'], seq['end']
            #     if e-s != self.seq_len:
            #         print(seq['path'], s, e)
            #         raise
            cached = True
        except Exception as e:
            print(e)
            print("No valid cache found, rereading data")

        if not cached:
            for item in self.sequences:
                for seq in item['Yawning'] + item['Normal']:
                    sub_num = ((seq['num'] - self.seq_len) // self.step) + 1
                    for i in range(sub_num):
                        s, e = i * self.step, i * self.step + self.seq_len
                        y = seq['y'][s:e]
                        sample = {
                            'path': seq['path'],
                            'start': s + seq['offset'],
                            'end': e + seq['offset'],
                        }
    
                        out = self.get_features(sample)
    
                        if out is None:
                            continue
                        
                        if np.sum(y) >= y.size * self.yawn_thr:
                            self.data['yawning'].append(sample)
                        else:
                            self.data['normal'].append(sample)
            if name is not None:
                self.save_cache(self.name)

    def save_cache(self, name):
        import json

        with open(name, 'w') as fp:
            json.dump(self.data, fp, sort_keys=True, indent=2)
            print(f'{name} saved')

    def load_cache(self, name):
        import json

        with open(name, 'r') as fp:
            self.data = json.load(fp)
            print(f'{name} restored')

    def __len__(self):
        return sum([len(self.data[c]) for c in self.data.keys()])

    def len_by_class(self):
        return { k: len(self.data[k]) for k in self.data.keys() }

    def get_features(self, data):
        x = []

        filepaths = [os.path.join(data['path'], f"Imag_{s:05}.jpg") for s in range(data['start'], data['end'])]
        assert len(filepaths) == self.seq_len, f"Wrong sequence length in {data}: {len(filepaths)}"

        discont_cnt = 0

        fd = None
        seq_ok = True
        for (idx, fp) in enumerate(filepaths):
            if fp not in self.cache:
                # if more than 20% is non-continous then discard whole sequence
                if discont_cnt > int(math.ceil(0.05 * self.seq_len)):
                    seq_ok = False
                    print(f'discont ({discont_cnt}): ', fp)
                    break
                image = cv2.imread(fp)
                # TODO augmentation
                predictions = self.model.predict(image, **self.det_config)
                # HOW? extract data from layers 19, 24, 27
                boxes = predictions[0].boxes.cpu().numpy()
                keypoints = predictions[0].keypoints.cpu().numpy()
                if (keypoints.xy[0].size == 0 or boxes[0].xywh.size == 0):
                    # print(f'No keypoints or face in image: {fp}')
                    discont_cnt += 1
                    if fd is None:
                        # cannot transfer data from previous frame, discard
                        print('fd is None: ', fp)
                        seq_ok = False
                        break
                    x.append(fd)
                    self.cache[fp] = fd
                    continue
                
                # real image, reset discontinous counter
                discont_cnt = 0
                
                # take only first face
                mouth_roi_kpts = np.array([
                    # x0, y0, x1, y1
                    keypoints.xy[0][48, 0],
                    keypoints.xy[0][30, 1],
                    keypoints.xy[0][50, 0],
                    keypoints.xy[0][8, 1],
                ])
                
                mouth_roi_kpts_i = mouth_roi_kpts.astype('int32')
                mouth_roi_image = image[
                    mouth_roi_kpts_i[1]:mouth_roi_kpts_i[3],
                    mouth_roi_kpts_i[0]:mouth_roi_kpts_i[2],
                    0
                ]

                if (mouth_roi_image.size == 0):
                    # print(f'Mouth RoI empty in image: {fp}')
                    discont_cnt += 1
                    if fd is None:
                        # cannot transfer data from previous frame, discard
                        print('fd is None: ', fp)
                        seq_ok = False
                        break
                    x.append(fd)
                    self.cache[fp] = fd
                    continue
                
                mouth_roi_image = cv2.dilate(mouth_roi_image, np.ones((2, 2), np.uint8), iterations=1)
                mouth_roi_image = cv2.resize(mouth_roi_image, self.mouth_roi_size)
                self.cache[fp] = mouth_roi_image
            else:
                mouth_roi_image = self.cache[fp]

            if mouth_roi_image.shape[0] == self.mouth_roi_size[0] and mouth_roi_image.shape[1] == self.mouth_roi_size[1]:
                # real image, reset discontinous counter
                discont_cnt = 0

                if self.augment:
                    mouth_roi_image = self.transform(image=mouth_roi_image)["image"]

                fd = hog(mouth_roi_image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=False)

                # fd, hog_vis = hog(mouth_roi_image, orientations=9, pixels_per_cell=(8, 8),
                #         cells_per_block=(2, 2), visualize=True)

                # path = fp.split('/')
                # hog_vis = skimage.exposure.rescale_intensity(hog_vis, in_range=(0, 10), out_range=(0, 255))
                # cv2.imwrite("debug/" + f"{path[-3]}_{os.path.basename(fp)}", hog_vis)
            else:
                fd = mouth_roi_image
                mouth_roi_image = None

            
            # print(fd.shape)
            # plt.figure()
            # plt.imshow(hog_vis)
            x.append(fd)

        # if more than 25% is missing then discard whole sequence 
        if not seq_ok:
            print(f"Invalid sequence {data}")
            return None
        
        output = np.array(x, dtype=np.float32)
        return output

    def __getitem__(self, index):
        c = 'normal' if index < len(self.data['normal']) else 'yawning'
        c_idx = 0 if c == 'normal' else 1
        index = index - len(self.data['normal']) if c == 'yawning' else index
        data = self.data[c][index]
        # print('getitem: ', data)
        output = self.get_features(data), np.int64(c_idx)
        
        return output

    def get_sequence_meta(self, index):
        c = 'normal' if index < len(self.data['normal']) else 'yawning'
        c_idx = 0 if c == 'normal' else 1
        index = index - len(self.data['normal']) if c == 'yawning' else index
        data = self.data[c][index]
        data['class'] = c
        
        return data

class YawningDataModule(pl.LightningDataModule):
    '''
    PyTorch Lighting DataModule subclass:
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html

    Serves the purpose of aggregating all data loading 
      and processing work in one place.
    '''
    
    def __init__(self, annotations, model: str, seq_len: int = 48, step: int = 8,
                 yawn_thr: float = 0.5, batch_size: int = 128, num_workers: int = 0,
                 k: int = 1, num_folds: int = 1, split_seed: int = 91):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.annotations = annotations
        self.model = model
        self.seq_len = seq_len
        self.step = step
        self.yawn_thr = yawn_thr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.columns = None
        self.preprocessing = None

    def prepare_data(self):
        pass

    def setup(self, stage: str=None):
        if stage == 'fit' and self.X_train is not None:
            return 
        if stage == 'test' and self.X_test is not None:
            return
        if stage is None and self.X_train is not None and self.X_test is not None:  
            return
        
        # split sequences by person
        seq_keys = list(self.annotations.keys())
        kf = KFold(n_splits=self.hparams.num_folds, shuffle=True, random_state=self.hparams.split_seed)
        all_splits = [k for k in kf.split(seq_keys)]
        train, val = all_splits[self.hparams.k]
        train, val = [seq_keys[k] for k in train.tolist()], [seq_keys[k] for k in val.tolist()]
        
        self.X_train = [self.annotations[k] for k in train]
        self.X_val = [self.annotations[k] for k in val]
        # self.X_test = [self.annotations[k] for k in test]

    def train_dataloader(self):
        train_dataset = SingleSequenceDataset(self.X_train, self.model, f'train_{self.hparams.k}.json', self.seq_len, self.step, self.yawn_thr, augment=True)
        
        print(f"Sample weights per class in fold {self.hparams.k}: ", {k: 1/v for (k, v) in train_dataset.len_by_class().items()})
        sample_weights = [[1/v] * v for (k, v) in train_dataset.len_by_class().items()]
        sample_weights = [item for x in sample_weights for item in x]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights)
        )
        train_loader = DataLoader(train_dataset, 
                                  batch_size = self.batch_size,
                                  sampler=sampler,
                                  persistent_workers = True,
                                  num_workers = self.num_workers)
        
        return train_loader

    def val_dataloader(self):
        val_dataset = SingleSequenceDataset(self.X_val, self.model, f'val_{self.hparams.k}.json', self.seq_len, self.step, self.yawn_thr)
        
        val_loader = DataLoader(val_dataset, 
                                batch_size = self.batch_size, 
                                shuffle = False, 
                                persistent_workers = True,
                                num_workers = self.num_workers)

        return val_loader

    # def test_dataloader(self):
    #     test_dataset = SingleSequenceDataset(self.X_test, self.model, 'test.json', self.seq_len, self.step, self.yawn_thr)
        
    #     test_loader = DataLoader(test_dataset, 
    #                              batch_size = self.batch_size, 
    #                              shuffle = False,
    #                              persistent_workers = True,
    #                              num_workers = self.num_workers)

    #     return test_loader