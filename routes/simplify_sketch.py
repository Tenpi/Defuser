from typing import Any
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from .functions import get_models_dir, get_device
import torch
import os

class SketchSimplificationModel():
    def __init__(self):
        self.model = nn.Sequential(
            nn.Conv2d(1, 48, (5, 5), (2, 2), (2, 2)),
            nn.ReLU(),
            nn.Conv2d(48, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, (4, 4), (2, 2), (1, 1), (0, 0)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, (4, 4), (2, 2), (1, 1), (0, 0)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 48, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 48, (4, 4), (2, 2), (1, 1), (0, 0)),
            nn.ReLU(),
            nn.Conv2d(48, 24, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(24, 1, (3, 3), (1, 1), (1, 1)),
            nn.Sigmoid(),
        )
        state_dict = torch.load(os.path.join(get_models_dir(), "misc/simplify.pth"), map_location=get_device())
        self.model.load_state_dict(state_dict)
        self.model.to(get_device())
        self.model.eval()
        self.immean = 0.9664114577640158
        self.imstd = 0.0858381272736797

    def __call__(self, image, output):
        data = image.convert("L")
        w, h = data.size[0], data.size[1]
        pw = 8 - (w % 8) if w % 8 != 0 else 0
        ph = 8 - (h % 8) if h % 8 != 0 else 0
        data = ((transforms.ToTensor()(data) - self.immean) / self.imstd).unsqueeze(0)
        if pw != 0 or ph != 0:
            data = torch.nn.ReplicationPad2d((0, pw, 0, ph))(data).data

        data = data.to(get_device())
        pred = self.model.forward(data)
        save_image(pred[0], output)