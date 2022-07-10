import pandas as pd
import numpy as np
import os
from PIL import Image

import torch
import torch.nn as nn
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision

from torch.utils.data import Dataset, DataLoader

class GetRepsDataset(Dataset):
    
    def __init__(self, df, dtype, transform=None):
        self.df = df
        self.dtype = dtype
        self.transform = transform
        self.typ_df = df[df['dtype']==dtype]
        
    def __len__(self):
        return len(self.typ_df)
    
    def __getitem__(self, idx):
        img_path = self.typ_df.patch_paths.iloc[idx]
        
        image = read_image(img_path, mode=ImageReadMode.RGB)
        
        if self.transform:
            image = self.transform(image)
        return image, img_path


class GetRepsDataset_old(Dataset):
    
    def __init__(self, dir_patch_dict, transform=None):
        self.dir_patch_dict = dir_patch_dict
        self.transform = transform
        self.patch_list = [x for xs in self.dir_patch_dict.values() for x in xs]
        
    def __len__(self):
        return len(self.patch_list)
    
    def __getitem__(self, idx):
        img_path = self.patch_list[idx]
        
        image = read_image(img_path, mode=ImageReadMode.RGB)
        
        if self.transform:
            image = self.transform(image)
            
        
        return image, img_path