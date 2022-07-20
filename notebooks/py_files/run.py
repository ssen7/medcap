import pandas as pd
import numpy as np

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from PIL import Image

import torch
import torchvision.transforms as transforms
import torch.nn as nn

from .datasets import GetRepsDataset, WSIBatchedDataset, WSIBatchedDataset_NC
from .models import BertEncoder, ResnetPreTrained, ImageEncoder
from .utils import check_cuda

from .train_fn import train_global_cluster_model, cluster_all_patches, start_pretraining_warmup, start_pretraining_NC

from transformers import BertTokenizer, AutoTokenizer

import albumentations as A
from albumentations.pytorch import ToTensorV2

PATH = '/project/GutIntelligenceLab/ss4yd/gtex_data/'


def run_multiple_iterations_without_clustering(epochs, pid_batch_size=10, img_per_pid=10, experiment_id=2):

    model_save_path = os.path.join(PATH, f'experiment_{experiment_id}')

    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    device=check_cuda()

    df_path = "/home/ss4yd/image_captioning/medcapv4/df.csv"

    df = pd.read_csv(df_path)

    train_transform = A.Compose([
        A.Resize(224,224),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(),
        A.GaussianBlur(),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(224,224),
        A.Normalize(),
        ToTensorV2()
    ])

    print("\nLoading Models..")
    


    base_image_model = ResnetPreTrained()
    image_encoder = nn.DataParallel(ImageEncoder(base_image_model))
    image_encoder.to(device)
    base_image_model = nn.DataParallel(base_image_model)
    base_image_model.to(device)
    bert_model = nn.DataParallel(BertEncoder(device=device))
    bert_model.to(device)
    
    
    print("\nFinished Loading Models..")


    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    wsi_batch_dataset = WSIBatchedDataset_NC(df, dtype='train', tokenizer=tokenizer, \
                                        img_transform=train_transform, pid_batch_size=pid_batch_size, img_per_pid=img_per_pid)
    train_dloader = torch.utils.data.DataLoader(wsi_batch_dataset,batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    wsi_val_dataset = WSIBatchedDataset_NC(df, dtype='test', tokenizer=tokenizer, \
                                        img_transform=val_transform, pid_batch_size=pid_batch_size, img_per_pid=img_per_pid) # modify back to "val" later
    val_dloader = torch.utils.data.DataLoader(wsi_val_dataset,batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
    
    best_img_model, best_text_model = start_pretraining_NC(base_image_model, bert_model, train_dloader,\
                                                    val_loader=val_dloader, device=device, model_save_path=model_save_path, epochs=epochs, img_per_pid=img_per_pid)

    torch.save(best_img_model.module.state_dict(), os.path.join(model_save_path,'best_img_model_iter.pth'))
    torch.save(best_text_model.module.state_dict(), os.path.join(model_save_path,'best_text_model_iter.pth'))




def run_multiple_iterations_with_clustering(itr_idx,n_warmup=5, warmup_epochs=20, non_warmup_epochs=5):

    print("\nRunning iteration: {}".format(itr_idx))

    device=check_cuda()

    df_path = "/home/ss4yd/image_captioning/medcapv4/df.csv"

    df = pd.read_csv(df_path)
    pids = df.pid.unique()

    pid_percent_cluster=0.1
    n_pids_cluster = int(pid_percent_cluster*len(pids))
    num_cluster=8

    pids_cluster = np.random.choice(pids, size=n_pids_cluster)

    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    transform=transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ConvertImageDtype(torch.float),
        normalize,
            ])

    cluster_dataset = GetRepsDataset(df, pids_cluster, transform)
    cluster_loader = torch.utils.data.DataLoader(cluster_dataset,batch_size=64, shuffle=True, num_workers=1, pin_memory=True)

    if itr_idx==0:

        base_image_model = ResnetPreTrained()
        image_encoder = nn.DataParallel(ImageEncoder(base_image_model))
        image_encoder.to(device)
        base_image_model = nn.DataParallel(base_image_model)
        base_image_model.to(device)
        bert_model = nn.DataParallel(BertEncoder(device=device))
        bert_model.to(device)
    
    else:
        base_image_model = ResnetPreTrained()
        base_image_model.load_state_dict(torch.load(f'/project/GutIntelligenceLab/ss4yd/gtex_data/best_img_model_iter_{itr_idx-1}.pth'))
        image_encoder = nn.DataParallel(ImageEncoder(base_image_model))
        image_encoder.to(device)
        base_image_model = nn.DataParallel(base_image_model)
        base_image_model.to(device)
        bert_model = BertEncoder(device=device)
        bert_model.load_state_dict(torch.load(f'/project/GutIntelligenceLab/ss4yd/gtex_data/best_text_model_iter_{itr_idx-1}.pth'))
        bert_model = nn.DataParallel(bert_model)
        bert_model.to(device)

    kmeans = train_global_cluster_model(image_encoder, cluster_loader, device, ncentroids=num_cluster)

    gcluster_dataset = GetRepsDataset(df, pids, transform)
    gcluster_loader = torch.utils.data.DataLoader(gcluster_dataset,batch_size=128, shuffle=False, \
                                                num_workers=1, pin_memory=True)

    cluster_df = cluster_all_patches(image_encoder, kmeans, gcluster_loader, device, df, save_path='/home/ss4yd/image_captioning/medcapv4/', ncentroids=8)

    ## Checkpoint 1
    df = pd.read_csv('/home/ss4yd/image_captioning/medcapv4/df_cluster.csv')

    pid_batch_size = 10

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    wsi_batch_dataset = WSIBatchedDataset(df, dtype='train', tokenizer=tokenizer, \
                                        img_transform=transform, pid_batch_size=pid_batch_size)
    train_dloader = torch.utils.data.DataLoader(wsi_batch_dataset,batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    wsi_val_dataset = WSIBatchedDataset(df, dtype='val', tokenizer=tokenizer, \
                                        img_transform=transform, pid_batch_size=pid_batch_size)
    val_dloader = torch.utils.data.DataLoader(wsi_val_dataset,batch_size=1, shuffle=True, num_workers=1, pin_memory=True)


    if itr_idx < n_warmup:
        print("\nWarming Up...")
        best_img_model, best_text_model = start_pretraining_warmup(base_image_model, bert_model, train_dloader,\
                                                    val_loader=val_dloader, device=device, epochs=warmup_epochs)
    elif (itr_idx >= n_warmup):
        best_img_model, best_text_model = start_pretraining_warmup(base_image_model, bert_model, train_dloader,\
                                                    val_loader=val_dloader, device=device, epochs=non_warmup_epochs)

    torch.save(best_img_model.module.state_dict(), f'/project/GutIntelligenceLab/ss4yd/gtex_data/best_img_model_iter_{itr_idx}.pth')
    torch.save(best_text_model.module.state_dict(), f'/project/GutIntelligenceLab/ss4yd/gtex_data/best_text_model_iter_{itr_idx}.pth')
