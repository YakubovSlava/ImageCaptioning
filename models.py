#!/usr/bin/env python
# coding: utf-8

# In[8]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision.models.inception import Inception3
import numpy as np


# In[9]:


word_index = np.load('word2index.npy',allow_pickle=True).item()
vacabulary = np.load('vocabulary.npy')

n_tokens = len(vacabulary)
eos = word_index['#END#']
unk = word_index['#UNK#']
pad = word_index['#PAD#']

def batch_of_captions_into_matrix(sequences, max_len=None):
    max_len = max_len or max(map(len, sequences))
    matrix = np.zeros((len(sequences), max_len), dtype='int32') + pad
    for i, seq in enumerate(sequences):
        numbered = [word_index.get(word, unk) for word in seq[:max_len]]
        matrix[i][:len(numbered)] = numbered
    return(matrix)


# In[4]:


class getCNN(Inception3):
    def forward(self,path):
        img = cv2.imread(path)
        img = torch.tensor(cv2.resize(img, (299, 299), interpolation=cv2.INTER_AREA))
        x = img.permute(2,0,1).unsqueeze(0)/255.0
        x = x.type('torch.DoubleTensor')
        x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x_for_attn = x = self.Mixed_7c(x)
        x = F.avg_pool2d(x, kernel_size=8)
        x_for_capt = x = x.view(x.size(0), -1)
        return x_for_capt
    def forward_img(self,img):
        img = torch.tensor(cv2.resize(img, (299, 299), interpolation=cv2.INTER_AREA))
        x = torch.tensor(img, dtype=torch.float32)
        x = x.permute(2,0,1).unsqueeze(0)/255.0
        x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x_for_attn = x = self.Mixed_7c(x)
        x = F.avg_pool2d(x, kernel_size=8)
        x_for_capt = x = x.view(x.size(0), -1)
        return x_for_capt


# In[6]:


class RNN(nn.Module):
    def __init__(self, n_tokens=n_tokens, emb_size=512, lstm_units=980, cnn_feature_size=2048):
        super(self.__class__, self).__init__()
        self.cnn_to_h0 = nn.Linear(cnn_feature_size, lstm_units)
        self.cnn_to_c0 = nn.Linear(cnn_feature_size, lstm_units)
        self.emb = nn.Embedding(n_tokens, emb_size, padding_idx=pad)
        self.lstm = nn.LSTM(emb_size, lstm_units, batch_first = True) 
        self.logits = nn.Linear(lstm_units, n_tokens)
    def forward(self, image_vectors, captions_ix):
        initial_cell = self.cnn_to_c0(image_vectors)
        initial_hid = self.cnn_to_h0(image_vectors)
        captions_emb = self.emb(captions_ix)
        state = (initial_cell[None], initial_hid[None])
        lstm_out, state = self.lstm(captions_emb, state)
        logits = self.logits(lstm_out)
        return logits


# In[ ]:




