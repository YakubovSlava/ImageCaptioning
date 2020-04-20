#!/usr/bin/env python
# coding: utf-8


from torch.utils.model_zoo import load_url

from models import getCNN
from models import RNN
import models
import torch
import torch.nn.functional as F



inception_url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'

cnn = getCNN()
cnn.load_state_dict(load_url(inception_url, map_location=torch.device('cpu')))
cnn = cnn.train(False)


rnn = RNN()
rnn.load_state_dict(torch.load('net_param.pt', torch.device('cpu')))
rnn = rnn.train(False)



vocabulary = models.vacabulary



batch_of_captions_into_matrix = models.batch_of_captions_into_matrix



def getCaption(path):
        img = cnn.forward(path=path)
        sentence = [['#START#']]
        for i in range(40):
            l = torch.tensor(batch_of_captions_into_matrix(sentence),dtype=torch.int64)
            res = rnn.forward(img, l)[0,-1]
            q = list(F.softmax(res, dim=-1).data)
            sentence[0].append(vocabulary[q.index(max(q))])
            if (vocabulary[q.index(max(q))]=='#END#'):break
        return(' '.join(sentence[0][1:-1]))


def getCaption_img(img):
    img = cnn.forward(img)
    sentence = [['#START#']]
    for i in range(40):
        l = torch.tensor(batch_of_captions_into_matrix(sentence), dtype=torch.int64)
        res = rnn.forward(img, l)[0, -1]
        q = list(F.softmax(res, dim=-1).data)
        sentence[0].append(vocabulary[q.index(max(q))])
        if (vocabulary[q.index(max(q))] == '#END#'): break
    return (' '.join(sentence[0][1:-1]))

getCaption('/Users/slava.yakubov/Library/Favorites/MyFiles/mlp2/coursework/poker.png')





cnn = getCNN()
inception_url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
cnn.load_state_dict(load_url(inception_url))
cnn = cnn.train(False)


print(getCaption_img('/Users/slava.yakubov/Downloads/Ub5-2hOJ6Y4.jpg'))

