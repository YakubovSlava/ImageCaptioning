# -*- coding: utf-8 -*-


from torchvision import transforms, models
from torch import nn
import cv2
import numpy as np
import torch

class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):

        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):

        x = self.features(x)

        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

def _make_divisible(v, divisor, min_value=None):

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class PredictPhoto:
  def __init__(self, model_path):
    self.val_transforms = transforms.Compose([
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    self.model = MobileNetV2()
    in_features = 1280#model.fc.in_features
    self.model.classifier = nn.Sequential(
        nn.Dropout(0.1),

        nn.Linear(in_features, in_features//2),
        nn.PReLU(),
        nn.Dropout(0.2),
        nn.BatchNorm1d(in_features//2),

        nn.Linear(in_features//2, in_features//4),
        nn.PReLU(),
        nn.Dropout(0.2),
        nn.BatchNorm1d(in_features//4),

        nn.Linear(in_features//4, 99)
    )
    self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    self.model.eval()

  def predict(self, faces):
    for i in range(len(faces)):
      faces[i] = self.val_transforms(faces[i])
    pred = self.model(faces.type(torch.float32))
    out = [self.class2age(x.item()) for x in torch.argmax(pred, dim=1)]

    return out

  def class2age(self, cl):
    cl2a = {93: 95.0,
        94: 96.0,
        95: 99.0,
        96: 100.0,
        97: 101.0,
        98: 110.0}
    if cl <= 92:
      return cl + 1
    elif cl in cl2a:
      return cl2a[cl]
    else:
      return -1

pred_photo = PredictPhoto('age_mobile_net.pt')

img = cv2.imread('/content/yg4H7PVOjUg.jpg')

def getAge(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image=img, minSize=(150, 150))

    cropped =[]
    images = []
    for i in range(len(faces)):
        x, y, h, w = faces[i]
        face = img[y:y + h, x:x + w]
        face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
        images.append(face)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        cropped.append(face)
    cropped = np.array(cropped)
    res = pred_photo.predict(torch.tensor(cropped/255).permute(0, 3, 1, 2))
    
    return res


