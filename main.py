from flask import Flask, url_for
from flask import request
from flask import render_template
import numpy as np
import cv2
from models import getCNN
from models import RNN
import models
import torch
import torch.nn.functional as F
from torch.utils.model_zoo import load_url
from base64 import b64encode
inception_url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'

cnn = getCNN()
cnn.load_state_dict(load_url(inception_url, map_location=torch.device('cpu')))
cnn = cnn.train(False)


rnn = RNN()
rnn.load_state_dict(torch.load('net_param.pt', torch.device('cpu')))
rnn = rnn.train(False)



vocabulary = models.vacabulary



batch_of_captions_into_matrix = models.batch_of_captions_into_matrix
app = Flask(__name__)


def getCaption_img(img):
    img = cnn.forward_img(img)
    sentence = [['#START#']]
    for i in range(40):
        l = torch.tensor(batch_of_captions_into_matrix(sentence), dtype=torch.int64)
        res = rnn.forward(img, l)[0, -1]
        q = list(F.softmax(res, dim=-1).data)
        sentence[0].append(vocabulary[q.index(max(q))])
        if (vocabulary[q.index(max(q))] == '#END#'): break
    return (' '.join(sentence[0][1:-1]))

@app.route('/')
def hello_world():
    return 'Hello, World!'


from flask import request

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['uploaded_file'].read()
        data = f
        encoded = b64encode(data)
        mime = "image/jpeg"
        uri = "data:%s;base64,%s" % (mime, str(encoded)[2:-1])
        npimg = np.fromstring(f, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        return render_template('result.html', text=(getCaption_img(img)), url=uri)
    if request.method == 'GET':
        return render_template('form.html')