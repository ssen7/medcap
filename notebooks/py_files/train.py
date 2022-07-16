import pandas as pd
import numpy as np
import pickle

import os
from PIL import Image

import torch
import torch.nn as nn
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from tqdm import trange, tqdm

import random

from transformers import BertTokenizer, AutoTokenizer, AutoModel, BertModel

import matplotlib.pyplot as plt

from py_files.models import BertEncoder, ResnetPreTrained, ImageEncoder
from py_files.datasets import WSIBatchedDataset, GetRepsDataset

import faiss

import gc
import torch.optim as optim


def train_global_cluster_model(encoder, dataloader, device, ncentroids=8):
    
    print("Getting patch representations...")
    
    rep_list = []
    path_list = []
    
    encoder.eval()
    
    for img, path in tqdm(dataloader):

        img.to(device)    

        in_batch_size = img.shape[0]
        
        with torch.no_grad():
            reps = encoder(img)
        rep_list.append(reps.detach().detach().cpu().numpy().reshape(in_batch_size, -1))
        path_list += path
        
        # clean up
        del img
        del reps
    
    print("\nTraining KMeans model...")
    
    X = np.concatenate(rep_list)
    X = np.ascontiguousarray(X)
    X = X.astype('float32')
    
    ncentroids = ncentroids
    niter = 300
    verbose = False
    d = X.shape[1]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, nredo=20)
    kmeans.train(X)
    
    print("\nFinished training KMeans model...")
    
    # clean up
    del encoder
    del dataloader
    torch.cuda.empty_cache()
    
    return kmeans


def cluster_all_patches(encoder, kmeans, dataloader, device, ncentroids=8):
    
    print("Clustering all patches...")
    
    encoder.eval()
    
    path_list, rep_list = [], []
    
    for img, path in tqdm(dataloader):
        img.to(device)
        
        in_batch_size = img.shape[0]
        
        with torch.no_grad():
            reps = image_encoder(img)
        rep_list.append(reps.detach().detach().cpu().numpy().reshape(in_batch_size, -1))
        path_list += path
        
        # clean up
        del img
        del reps
        
    X = np.concatenate(rep_list)
    X = np.ascontiguousarray(X)
    X = X.astype('float32')
    
    D, I = kmeans.index.search(X, 1)
    
    df = pd.DataFrame(path_list, columns=['patch_paths'])
    df['cluster_assignment'] = I
    
    print("\nFinished clustering all patches...")
    
    # clean up
    del encoder
    # del dataloader
    torch.cuda.empty_cache()
    
    return df


