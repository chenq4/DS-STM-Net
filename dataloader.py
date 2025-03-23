# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from sklearn.model_selection import train_test_split
from albumentations.augmentations import transforms
import albumentations as A
from albumentations import RandomRotate90,Resize

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, img_names, num_classes=0, transform=None): # 后缀都是.npy 并且input和mask名字一样
        
        self.img_names = img_names
        self.img_dir = os.path.join('./datasets',dataset,'images')
        self.mask_dir = os.path.join('./datasets',dataset,'masks')
        self.num_classes = num_classes
        self.transform = transform
        
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_id = self.img_names[idx]
        # 读出来是 (C,H,W)
        img,mask = np.load(os.path.join(self.img_dir,img_id)), np.load(os.path.join(self.mask_dir,img_id))
#         print(img.max(),img.min())
        # augmentations包接收的形式为(H,W,C) !!!
        if self.transform is not None:
            augmented = self.transform(image=img.transpose(1,2,0), mask=mask.transpose(1,2,0))
            img = augmented['image'].transpose(2,0,1)
            mask = augmented['mask'].transpose(2,0,1)
        return img,mask


class DatasetWithDB(torch.utils.data.Dataset):
    def __init__(self, dataset, img_names, num_classes=0, transform=None): # 后缀都是.npy 并且input和mask名字一样
        
        self.img_names = img_names
        self.img_dir = os.path.join('./datasets',dataset,'images')
        self.mask_dir = os.path.join('./datasets',dataset,'masks')
        self.shrinks_dir = os.path.join('./datasets',dataset,'shrinks')
        self.thresholds_dir = os.path.join('./datasets',dataset,'thresholds')
        self.num_classes = num_classes
        self.transform = transform
        
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_id = self.img_names[idx]
        # 读出来是 (C,H,W)
        img,mask = np.load(os.path.join(self.img_dir,img_id)), np.load(os.path.join(self.mask_dir,img_id))
        shrink,threshold = np.load(os.path.join(self.shrinks_dir,img_id)), np.load(os.path.join(self.thresholds_dir,img_id))
        label = np.concatenate((shrink,threshold,mask),axis=0)
#         print(img.max(),img.min())
        # augmentations包接收的形式为(H,W,C) !!!
        if self.transform is not None:
            augmented = self.transform(image=img.transpose(1,2,0), mask=label.transpose(1,2,0))
            img = augmented['image'].transpose(2,0,1)
            label = augmented['mask'].transpose(2,0,1)
        return img,label










