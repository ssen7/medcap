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

from tqdm.notebook import trange, tqdm

import random

from transformers import BertTokenizer, AutoTokenizer, AutoModel, BertModel

import matplotlib.pyplot as plt

class BertEncoder(nn.Module):
    def __init__(self, bert_type="emilyalsentzer/Bio_ClinicalBERT", freeze_bert=False,\
                 agg_tokens=True, n_bert_layers=4, agg_method='sum', embedding_dim=768):
        super(BertEncoder, self).__init__()
    
        self.bert_type = bert_type
        self.freeze_bert = freeze_bert
        self.agg_tokens = agg_tokens
        self.n_bert_layers = n_bert_layers
        self.agg_method = agg_method
        self.embedding_dim = embedding_dim        
        
        self.model = BertModel.from_pretrained(self.bert_type, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        
        if self.freeze_bert is True:
            print("Freezing BERT model")
            for param in self.model.parameters():
                param.requires_grad = False
        
    def aggregate_tokens(self, embeddings, caption_ids):

        batch_size, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []

        # loop over batch
        for embs, caption_id in zip(embeddings, caption_ids):

            agg_embs = []
            token_bank = []
            words = []
            word_bank = []

            # loop over sentence
            for word_emb, word_id in zip(embs, caption_id):

                word = self.idxtoword[word_id.item()]

                if word == "[SEP]":
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))

                    agg_embs.append(word_emb)
                    words.append(word)
                    break

                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))

                        token_bank = [word_emb]
                        word_bank = [word]
                else:
                    if word.startswith("##"):
                        token_bank.append(word_emb)
                        word_bank.append(word[2:])

            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.to(agg_embs.device)
            words = words + ["[PAD]"] * padding_size

            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        return agg_embs_batch, sentences
    
    
    def forward(self, ids, attn_mask, token_type):
        
        outputs = self.model(ids, attn_mask, token_type)
        
        # aggregate intermetidate layers
        if self.n_bert_layers > 1:
            all_embeddings = outputs[2]
            embeddings = torch.stack(
                all_embeddings[-self.n_bert_layers :]
            )  # layers, batch, sent_len, embedding size

            embeddings = embeddings.permute(1, 0, 2, 3)
            
            
            if self.agg_tokens:
                embeddings, sents = self.aggregate_tokens(embeddings, ids)
            else:
                sents = [[self.idxtoword[w.item()] for w in sent] for sent in ids]

            sent_embeddings = embeddings.mean(axis=2)

            if self.agg_method == "sum":
                word_embeddings = embeddings.sum(axis=1)
                sent_embeddings = sent_embeddings.sum(axis=1)
            elif self.agg_method == "mean":
                word_embeddings = embeddings.mean(axis=1)
                sent_embeddings = sent_embeddings.mean(axis=1)
            else:
                print(self.agg_method)
                raise Exception("Aggregation method not implemented")

        # use last layer
        else:
            word_embeddings, sent_embeddings = outputs[0], outputs[1]

        batch_dim, num_words, feat_dim = word_embeddings.shape
        word_embeddings = word_embeddings.view(batch_dim * num_words, feat_dim)
        
        
        word_embeddings = word_embeddings.view(batch_dim, num_words, self.embedding_dim)
        word_embeddings = word_embeddings.permute(0, 2, 1)

        
        return word_embeddings, sent_embeddings, sents
    
    

    
class ImageEncoder(nn.Module):
    def __init__(self, base_model):
        super(ImageEncoder, self).__init__()
        
        self.resnet_head = base_model.resnet
        
    def forward(self, x):
        out = self.resnet_head(x)
        return out


class ResnetPreTrained(nn.Module):
    def __init__(self, freeze_cnn=False, embedding_dim=768, agg_method='mean'):
        super(ResnetPreTrained, self).__init__()
        
        
        self.freeze_cnn = freeze_cnn
        self.embedding_dim = embedding_dim
        self.agg_method = agg_method
        
        self.resnet = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])
        self.linear = nn.Sequential(nn.Linear(in_features=2048, out_features=768, bias=False),
                                   nn.ReLU())
        
        if self.freeze_cnn is True:
            print("Freezing ResNet model")
            for param in self.resnet.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        
        out = self.resnet(x)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        
#         if self.agg_method == "sum":
#             out = out.sum(axis=1)
#         elif self.agg_method == "mean":
#             out = out.mean(axis=1)
#         else:
#             print(self.agg_method)
#             raise Exception("Aggregation method not implemented")
        
        
        return out        