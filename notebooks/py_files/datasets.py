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
    
    def __init__(self, df, pids, transform=None):
        self.df = df
        self.pids = pids
        self.transform = transform
        self.typ_df = df[df['pid'].isin(self.pids)]
        
    def __len__(self):
        return len(self.typ_df)
    
    def __getitem__(self, idx):
        img_path = self.typ_df.patch_paths.iloc[idx]
        
        image = read_image(img_path, mode=ImageReadMode.RGB)
        
        if self.transform:
            image = self.transform(image)
        return image, img_path
    
    
    
class WSIBatchedDataset(Dataset):
    
    def __init__(self, df, dtype, tokenizer, shuffle=True, pid_batch_size=8, img_per_pid=8, \
                 num_cluster=8, img_transform=None, text_transform=None):
        self.df = df
        self.dtype = dtype
        self.img_transform = img_transform
        self.text_transform = text_transform
        self.typ_df = df[df['dtype']==dtype]
        self.shuffle = shuffle
        self.pid_batch_size = pid_batch_size
        
        self.unique_pids = list(self.typ_df.pid.unique())
        
        if self.shuffle:
            random.shuffle(self.unique_pids)
        
        self.pid_batches = self.create_unique_pid_batches(self.unique_pids, self.pid_batch_size)
        
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.pid_batches)
    
    def __getitem__(self, idx):
        
        pids = self.pid_batches[idx]
        
        img_list, text_list, img_seps, pids = self.return_batch_img_list(self.typ_df, pids, num_cluster=8)
        img_tensor, text_tensor, attention_tensor, token_types = self.create_batch_tensors(img_list, text_list)
        
        
        return img_tensor, text_tensor,attention_tensor, token_types, img_seps, img_list
    
    def create_unique_pid_batches(self, pids, pid_batch_size=8):
        
        pid_batches = [list(pids[i:i+pid_batch_size]) for i in \
                            pid_batch_size*np.arange(0, len(pids)//pid_batch_size+1, 1)]
        
        pid_batches = [x + list(np.random.choice(pids, pid_batch_size-len(x))) \
           if (len(x)<pid_batch_size) else x for x in pid_batches ]
        
        return pid_batches
    
    def create_batch_tensors(self, img_list, text_list, img_transform=None, text_transform=None):
        
        img_tensor_list = []
        for img_path in img_list:
            image = read_image(img_path, mode=ImageReadMode.RGB)
            if self.img_transform:
                image = self.img_transform(image)
            image = image.unsqueeze(0)
            img_tensor_list.append(image)
            
        
        img_tensor = torch.cat(img_tensor_list)
        text_tensor, attention_tensor, token_types = self.create_caption_tensors(text_list, text_transform=text_transform)
        
        return img_tensor, text_tensor, attention_tensor, token_types
    
    def create_caption_tensors(self, text_list, max_length=80, text_transform=None):
        
        text_tensor_list = []
        attention_masks = []
        token_types = []
        for sent in text_list:
            encoded_dict = self.tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_length,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                        truncation=True,   # remove warnings from printing
                   )
            text_tensor_list.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            token_types.append(encoded_dict['token_type_ids'])
            
        
        return torch.cat(text_tensor_list, dim=0), torch.cat(attention_masks, dim=0), torch.cat(token_types, dim=0)
        
    def return_batch_img_list(self, df, pids, num_cluster=8):
        img_list=[]
        text_list=[]
        img_seps=[]
        for pid in pids:
            tdf = df[df['pid']==pid]
            if len(tdf) < num_cluster:
                img_per_cluster = num_cluster//len(tdf['cluster_assignment'].unique())
#                 print(img_per_cluster)
                for c in tdf['cluster_assignment'].unique():
                    img_list+=list(tdf[tdf['cluster_assignment']==c]['patch_paths'].sample(img_per_cluster, replace=True))
                    text_list+=list(tdf[tdf['cluster_assignment']==c]['notes'].sample(img_per_cluster, replace=True))
                extra_imgs = num_cluster-img_per_cluster*len(tdf['cluster_assignment'].unique())
                if (extra_imgs>0):
                    img_list+=list(tdf['patch_paths'].sample(extra_imgs))
                    text_list+=list(tdf['notes'].sample(extra_imgs))
            elif len(tdf['cluster_assignment'].unique())==num_cluster:
                for c in range(num_cluster):
                    img_list+=list(tdf[tdf['cluster_assignment']==c]['patch_paths'].sample(1))
                    text_list+=list(tdf[tdf['cluster_assignment']==c]['notes'].sample(1))
                        
            else:
                img_list+=list(tdf['patch_paths'].sample(num_cluster))
                text_list+=list(tdf['notes'].sample(num_cluster))
            img_seps.append(len(img_list))
            
        
        return img_list, text_list, img_seps, pids