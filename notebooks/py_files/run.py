import pandas as pd
import numpy as np

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from PIL import Image

import torch
import torchvision.transforms as transforms
import torch.nn as nn

from .datasets import GetRepsDataset, WSIBatchedDataset
from .models import BertEncoder, ResnetPreTrained, ImageEncoder
from .utils import check_cuda

from .train_fn import train_global_cluster_model, cluster_all_patches, start_pretraining_warmup

from transformers import BertTokenizer, AutoTokenizer

def run_multiple_iterations(itr_idx, warmup_epochs=20):

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


    if itr_idx ==0:
        best_img_model, best_text_model = start_pretraining_warmup(base_image_model, bert_model, train_dloader,\
                                                    val_loader=val_dloader, device=device, epochs=warmup_epochs)
    elif (itr_idx >= 2):
        best_img_model, best_text_model = start_pretraining_warmup(best_img_model, best_text_model, train_dloader,\
                                                    val_loader=val_dloader, device=device, epochs=1)

    torch.save(best_img_model.module.state_dict(), f'/project/GutIntelligenceLab/ss4yd/gtex_data/best_img_model_iter_{itr_idx}.pth')
    torch.save(best_text_model.module.state_dict(), f'/project/GutIntelligenceLab/ss4yd/gtex_data/best_text_model_iter_{itr_idx}.pth')
