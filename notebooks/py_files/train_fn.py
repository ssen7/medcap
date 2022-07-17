import pandas as pd
import numpy as np
import pickle

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

from tqdm.notebook import trange, tqdm

import random

from transformers import BertTokenizer, AutoTokenizer, AutoModel, BertModel

import matplotlib.pyplot as plt

from py_files.models import BertEncoder, ResnetPreTrained, ImageEncoder
from py_files.datasets import WSIBatchedDataset, GetRepsDataset

import faiss

import gc
import torch.optim as optim
import time

from .loss import global_loss, local_loss
from .utils import cosine_similarity, check_cuda


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


def cluster_all_patches(encoder, kmeans, dataloader, device, df, save_path='../data/', ncentroids=8):
    
    print("Clustering all patches...")
    
    encoder.eval()
    
    path_list, rep_list = [], []
    
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
        
    X = np.concatenate(rep_list)
    X = np.ascontiguousarray(X)
    X = X.astype('float32')
    
    D, I = kmeans.index.search(X, 1)
    
    df_c = pd.DataFrame(path_list, columns=['patch_paths'])
    df_c['cluster_assignment'] = I


    df = df.merge(df_c, on='patch_paths')

    df.to_csv(save_path+'df_cluster.csv', index=False)
    
    print("\nFinished clustering all patches...")
    
    # clean up
    del encoder
    # del dataloader
    torch.cuda.empty_cache()
    
    return df

def get_bert_params(text_encoder):
    freeze_modules = [text_encoder.module.model.embeddings, *text_encoder.module.model.encoder.layer[:-4]]
    non_freeze_modules = [*text_encoder.module.model.encoder.layer[-4:]]
    
    param_list = []
    for module in freeze_modules:
        for param in module.parameters():
            param.requires_grad = False
            param_list.append(param)

    for module in non_freeze_modules:
        for param in module.parameters():
            param_list.append(param)
            
    return param_list


def start_pretraining_warmup(img_encoder, text_encoder, train_loader, val_loader, device, pid_batch_size=8, num_cluster=8, epochs=50):

    if epochs > 1:
        print("\nWarming Up")
    
    params = list(img_encoder.parameters()) + get_bert_params(text_encoder)
    optimizer = optim.Adadelta([param for param in params \
                                if param.requires_grad == True],lr=1e-3,rho=0.95)
    print("Start training...\n")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Elapsed':^9}")
    print("-"*60)
    
    best_val_loss=10
    best_img_model=0
    best_text_model=0
    epochs_since_improvement = 0
    
    for epoch_i in range(epochs):
        
        
        if epochs_since_improvement == 20:
            break

        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            print("\nDECAYING learning rate.")
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.8
            print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
        
        img_encoder.train()
        text_encoder.train()
        
        # Tracking time and loss
        t0_epoch = time.time()
        total_loss = 0
        
        for i, (img, text, attention, token_typ, img_seps, pids) in enumerate(train_loader):

            img_seps = [0]+[i.numpy()[0] for i in img_seps]
            img_seps = [[img_seps[i],img_seps[i+1]] for i in range(len(img_seps)-1)]

            img, text = img.squeeze(0).to(device),text.squeeze(0).to(device)
            attention, token_typ = attention.squeeze(0).to(device), token_typ.squeeze(0).to(device)

            text_outputs = text_encoder(text, attention, token_typ)
            img_outputs = img_encoder(img)

            # clean up
            del img, text, attention, token_typ
            gc.collect()

            cap_lens = text_outputs[2]
            cap_lens = [cap_lens[i].item() for i in np.arange(0, len(cap_lens), num_cluster)]

            pid_word_embeddings = [text_outputs[0][x:y] for x, y in img_seps]
            pid_sent_embeddings = [text_outputs[1][x:y] for x, y in img_seps]
            pid_img_embeddings = [img_outputs[x:y] for x, y in img_seps]

            cnn_code = torch.stack([x.mean(dim=0) for x in pid_img_embeddings])
            rnn_code = torch.stack([x[0] for x in pid_sent_embeddings])

            img_features = [x.unsqueeze(2).unsqueeze(2) for x in pid_img_embeddings]
            img_features = [x.permute(2,1,0,3) for x in img_features]
            img_features = torch.stack(img_features, dim=0).squeeze(1)

            words_embs = [x[0] for x in pid_word_embeddings]
            words_embs = torch.stack(words_embs)

            gloss0, gloss1 = global_loss(cnn_code, rnn_code)
            loss0, loss1, att_maps=local_loss(img_features, words_embs, cap_lens)
            loss = loss0 + loss1 + 0.1*gloss0 + 0.1*gloss1
            
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_loader)
        
        # =======================================
        #               Evaluation
        # =======================================
        if val_loader is not None:
            
            val_loss = evaluate(img_encoder, text_encoder, val_loader, device, test_dataloader=None)
            
            is_best = val_loss < best_val_loss
            best_val_loss = min(val_loss, best_val_loss)
            
            if is_best:
                best_img_model = img_encoder
                best_text_model = text_encoder
            
            if not is_best:
                epochs_since_improvement += 1
            else:
                epochs_since_improvement = 0
            
            
            
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {time_elapsed:^9.2f}")

    
    return best_img_model, best_text_model

def evaluate(img_encoder, text_encoder, val_dataloader, device, pid_batch_size=8, num_cluster=8, test_dataloader=None,plot=False):
    img_encoder.eval()
    text_encoder.eval()
    
    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set..
    for img, text, attention, token_typ, img_seps, pids in val_dataloader:
        
        img_seps = [0]+[i.numpy()[0] for i in img_seps]
        img_seps = [[img_seps[i],img_seps[i+1]] for i in range(len(img_seps)-1)]

        img, text = img.squeeze(0).to(device),text.squeeze(0).to(device)
        attention, token_typ = attention.squeeze(0).to(device), token_typ.squeeze(0).to(device)
        
        with torch.no_grad():
            text_outputs = text_encoder(text, attention, token_typ)
            img_outputs = img_encoder(img)
        
        cap_lens = text_outputs[2]
        cap_lens = [cap_lens[i].item() for i in np.arange(0, len(cap_lens), num_cluster)]
        
        pid_word_embeddings = [text_outputs[0][x:y] for x, y in img_seps]
        pid_sent_embeddings = [text_outputs[1][x:y] for x, y in img_seps]
        pid_img_embeddings = [img_outputs[x:y] for x, y in img_seps]

        cnn_code = torch.stack([x.mean(dim=0) for x in pid_img_embeddings])
        rnn_code = torch.stack([x[0] for x in pid_sent_embeddings])

        img_features = [x.unsqueeze(2).unsqueeze(2) for x in pid_img_embeddings]
        img_features = [x.permute(2,1,0,3) for x in img_features]
        img_features = torch.stack(img_features, dim=0).squeeze(1)

        words_embs = [x[0] for x in pid_word_embeddings]
        words_embs = torch.stack(words_embs)

        gloss0, gloss1 = global_loss(cnn_code, rnn_code)
        loss0, loss1, att_maps=local_loss(img_features, words_embs, cap_lens)
        loss = loss0 + loss1 + gloss0 + gloss1

        val_loss.append(loss.item())
        
    val_loss = np.mean(val_loss)
    
    return val_loss
        