from flask import Flask, url_for
from flask import request
from flask import render_template
import numpy as np
import cv2
from models import getCNN
from models import RNN
from models import CNN_emotions
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



emotions = CNN_emotions()
emotions.load_state_dict(torch.load('emotions.pth', torch.device('cpu')))
rnn = rnn.train(False)

vocabulary = models.vacabulary



batch_of_captions_into_matrix = models.batch_of_captions_into_matrix
app = Flask(__name__)


tags = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}



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


def getEmotion_img(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image=img, minSize=(150, 150))

    cropped =[]
    images = []
    for i in range(len(faces)):
        x, y, h, w = faces[i]
        face = img[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
        images.append(face)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        cropped.append(face)
    cropped = np.array(cropped)
    res = emotions.forward(torch.tensor(cropped))
    # print(torch.argmax(res, dim=1))
    em = [tags[x.item()] for x in torch.argmax(res, dim=1)]
    return em, images

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

        emotions, images = getEmotion_img(img)
        mas = []
        for i in images:
            retval, buffer = cv2.imencode('.jpg', i)
            faceurl = "data:%s;base64,%s" % (mime, str(b64encode(buffer))[2:-1])
            mas.append(faceurl)
        return render_template('result.html', text=(getCaption_img(img)), url=uri, mas=mas, emotions=emotions, len=len(mas))
    if request.method == 'GET':
        return render_template('form.html')