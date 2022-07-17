import pandas as pd
import numpy as np

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from PIL import Image

import torch
import torchvision.transforms as transforms

from datasets import GetRepsDataset, WSIBatchedDataset
from models import BertEncoder, ResnetPreTrained, ImageEncoder
from utils import check_cuda

from train_fn import train_global_cluster_model, cluster_all_patches, start_pretraining

from transformers import BertTokenizer, AutoTokenizer

device=check_cuda()

df_path = "/home/ss4yd/image_captioning/medcapv4/df.csv"

df = pd.read_csv('../df.csv')
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

base_image_model = ResnetPreTrained()
image_encoder = ImageEncoder(base_image_model)
image_encoder.to(device)

kmeans = train_global_cluster_model(image_encoder, cluster_loader, device, ncentroids=num_cluster)

gcluster_dataset = GetRepsDataset(df, pids, transform)
gcluster_loader = torch.utils.data.DataLoader(gcluster_dataset,batch_size=128, shuffle=False, \
                                             num_workers=1, pin_memory=True)

cluster_df = cluster_all_patches(image_encoder, kmeans, gcluster_loader, device, df, save_path='/home/ss4yd/image_captioning/medcapv4/', ncentroids=8)

## Checkpoint 1
df = pd.read_csv('/home/ss4yd/image_captioning/medcapv4/df_clustered.csv')

pid_batch_size = 8

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
wsi_batch_dataset = WSIBatchedDataset(df, dtype='train', tokenizer=tokenizer, \
                                      img_transform=transform, pid_batch_size=pid_batch_size)
train_dloader = torch.utils.data.DataLoader(wsi_batch_dataset,batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

wsi_val_dataset = WSIBatchedDataset(df, dtype='val', tokenizer=tokenizer, \
                                      img_transform=transform, pid_batch_size=pid_batch_size)
val_dloader = torch.utils.data.DataLoader(wsi_val_dataset,batch_size=1, shuffle=True, num_workers=1, pin_memory=True)


base_image_model = ResnetPreTrained()
base_image_model.to(device)
bert_model = BertEncoder()
bert_model.to(device)

best_img_model, best_text_model = start_pretraining(base_image_model, bert_model, train_dloader,\
                                                    val_loader=val_dloader, device=device)

