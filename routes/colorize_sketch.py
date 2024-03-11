import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from torchvision import transforms
from collections import namedtuple
import numpy as np
import array
from .functions import get_models_dir, get_normalized_dimensions

topk = 4
Norm = nn.BatchNorm2d
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

colorize_model = None

class AttentionBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 gate_channels,
                 inter_channels=None,
                 bias=True):
        super(AttentionBlock, self).__init__()
        """
        Implementation of Attention block in SegUnet
        """

        if inter_channels is None:
            inter_channels = in_channels // 2

        self.W = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias),
            nn.BatchNorm2d(in_channels),
        )

        # for skip-connection
        self.Wx = nn.Conv2d(
            in_channels,
            inter_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False)

        # for gating
        self.Wg = nn.Conv2d(
            gate_channels,
            inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias)

        # for internal feature
        self.psi = nn.Conv2d(
            inter_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias)

        # initialize
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.02)

    def test(self):
        # model = self(32, 16)
        skip = torch.randn(1, 32, 16, 16)
        g = torch.randn(1, 16, 8, 8)
        result = self(skip, g)
        print(result[0].shape)
        print(result[1].shape)
        pass

    def forward(self, x, g):
        """
        @param g: gated signal (queried)
        @param x: skip connected feature
        """

        # map info inter_channels
        g = self.Wg(g)
        down_x = self.Wx(x)

        g = F.interpolate(g, size=down_x.shape[2:])

        q = self.psi(torch.relu(g + down_x))
        q = torch.sigmoid(q)

        resampled = F.interpolate(q, size=x.shape[2:])
        result = resampled.expand_as(x) * x
        result = self.W(result)

        return result, resampled

class DeepUNetPaintGenerator(nn.Module):
    def __init__(self, bias=True):
        super(DeepUNetPaintGenerator, self).__init__()

        self.bias = bias
        self.dim = 64

        self.down_sampler = self._down_sample()
        self.up_sampler = self._up_sample()
        self.attentions = self._attention_blocks()

        self.first_layer = nn.Sequential(
            nn.Conv2d(15, self.dim, 3, 1, 1, bias=bias),
            Norm(self.dim),
        )
        self.gate_block = nn.Sequential(
            nn.Conv2d(
                self.dim * 8,
                self.dim * 8,
                kernel_size=1,
                stride=1,
                bias=self.bias), nn.BatchNorm2d(self.dim * 8))

        self.last_layer = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(self.dim, 3, 3, 1, 1, bias=bias),
            nn.Tanh(),
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.02)

    def test(self):
        x = torch.randn(1, 3, 512, 512)
        c = torch.randn(1, 12, 512, 512)
        print(self(x, c).shape)

    def forward(self, image, colors):
        cache = []
        image = torch.cat([image, colors], 1)

        image = self.first_layer(image)

        for i, layer in enumerate(self.down_sampler):
            image, connection, idx = layer(image)
            cache.append((connection, idx))

        cache = list(reversed(cache))
        gate = self.gate_block(image)
        attentions = []

        for i, (layer, attention, (connection, idx)) in enumerate(
                zip(self.up_sampler, self.attentions, cache)):
            connection, attr = attention(connection, gate)
            image = layer(image, connection, idx)
            attentions.append(attr)

        image = self.last_layer(image)

        return image, attentions

    def _attention_blocks(self):
        layers = nn.ModuleList()

        gate_channels = self.dim * 8

        layers.append(
            AttentionBlock(self.dim * 8, gate_channels, bias=self.bias))

        layers.append(
            AttentionBlock(self.dim * 8, gate_channels, bias=self.bias))

        layers.append(
            AttentionBlock(self.dim * 8, gate_channels, bias=self.bias))

        layers.append(
            AttentionBlock(self.dim * 8, gate_channels, bias=self.bias))

        layers.append(
            AttentionBlock(self.dim * 4, gate_channels, bias=self.bias))

        layers.append(
            AttentionBlock(self.dim * 2, gate_channels, bias=self.bias))

        return layers

    def _down_sample(self):
        layers = nn.ModuleList()

        # 256
        layers.append(DeepUNetDownSample(self.dim, self.dim * 2, self.bias))

        # 128
        layers.append(
            DeepUNetDownSample(self.dim * 2, self.dim * 4, self.bias))

        # 64
        layers.append(
            DeepUNetDownSample(self.dim * 4, self.dim * 8, self.bias))

        # 32
        layers.append(
            DeepUNetDownSample(self.dim * 8, self.dim * 8, self.bias))

        # 16
        layers.append(
            DeepUNetDownSample(self.dim * 8, self.dim * 8, self.bias))

        # 8
        layers.append(
            DeepUNetDownSample(self.dim * 8, self.dim * 8, self.bias))

        return layers

    def _up_sample(self):
        layers = nn.ModuleList()
        layers.append(
            DeepUNetUpSample(self.dim * 8 * 2, self.dim * 8, self.bias, True))
        layers.append(
            DeepUNetUpSample(self.dim * 8 * 2, self.dim * 8, self.bias, True))
        layers.append(
            DeepUNetUpSample(self.dim * 8 * 2, self.dim * 8, self.bias, True))
        layers.append(
            DeepUNetUpSample(self.dim * 8 * 2, self.dim * 4, self.bias))
        layers.append(
            DeepUNetUpSample(self.dim * 4 * 2, self.dim * 2, self.bias))
        layers.append(DeepUNetUpSample(self.dim * 2 * 2, self.dim, self.bias))
        return layers


class DeepUNetDownSample(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(DeepUNetDownSample, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm1 = Norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm2 = Norm(out_channels)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

        if in_channels == out_channels:
            self.channel_map = nn.Sequential()
        else:
            self.channel_map = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        feature = torch.relu(x)

        feature = self.conv1(feature)
        feature = self.norm1(feature)

        feature = torch.relu(feature)
        feature = self.conv2(feature)
        feature = self.norm2(feature)

        connection = feature + self.channel_map(x)
        feature, idx = self.pool(connection)
        return feature, connection, idx


class DeepUNetUpSample(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, dropout=False):
        super(DeepUNetUpSample, self).__init__()
        self.pool = nn.MaxUnpool2d(2, 2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm1 = Norm(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm2 = Norm(out_channels)

        self.dropout = nn.Dropout2d(0.5, True) if dropout else None
        if (in_channels // 2) == out_channels:
            self.channel_map = nn.Sequential()
        else:
            self.channel_map = nn.Conv2d((in_channels // 2),
                                         out_channels,
                                         1,
                                         bias=False)

    def forward(self, x, connection, idx):
        x = self.pool(x, idx)

        feature = torch.relu(x)
        feature = torch.cat([feature, connection], 1)
        feature = self.conv1(feature)
        feature = self.norm1(feature)

        feature = torch.relu(feature)
        feature = self.conv2(feature)
        feature = self.norm2(feature)

        feature = feature + self.channel_map(x)

        if self.dropout is not None:
            feature = self.dropout(feature)

        return feature

class Color(object):
    def __init__(self, r, g, b, proportion):
        Rgb = namedtuple("Rgb", ("r", "g", "b"))
        self.rgb = Rgb(r, g, b)
        self.proportion = proportion

    def __repr__(self):
        return "<colorgram.py Color: {}, {}%>".format(
            str(self.rgb), str(self.proportion * 100))

    @property
    def hsl(self):
        try:
            return self._hsl
        except AttributeError:
            Hsl = namedtuple("Hsl", ("h", "s", "l"))
            self._hsl = Hsl(*hsl(*self.rgb))
            return self._hsl
        
def sample(image):
    top_two_bits = 0b11000000

    sides = 1 << 2  # Left by the number of bits used.
    cubes = sides**7

    samples = array.array("l", (0 for _ in range(cubes)))
    width, height = image.size

    pixels = image.load()
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y][:3]
            h, s, l = hsl(r, g, b)
            Y = int(r * 0.2126 + g * 0.7152 + b * 0.0722)

            packed = (Y & top_two_bits) << 4
            packed |= (h & top_two_bits) << 2
            packed |= (l & top_two_bits) << 0
            packed *= 4
            samples[packed] += r
            samples[packed + 1] += g
            samples[packed + 2] += b
            samples[packed + 3] += 1
    return samples


def pick_used(samples):
    used = []
    for i in range(0, len(samples), 4):
        count = samples[i + 3]
        if count:
            used.append((count, i))
    return used


def get_colors(samples, used, number_of_colors):
    pixels = 0
    colors = []
    number_of_colors = min(number_of_colors, len(used))

    for count, index in used[:number_of_colors]:
        pixels += count

        color = Color(samples[index] // count, samples[index + 1] // count,
                      samples[index + 2] // count, count)

        colors.append(color)
    for color in colors:
        color.proportion /= pixels
    return colors

def extract(f, number_of_colors):
    if isinstance(f, str):
        image = Image.open(f)
    elif isinstance(f, torch.Tensor):
        image = transforms.ToPILImage()(f)
    image = f
    if image.mode not in ("RGB", "RGBA", "RGBa"):
        image = image.convert("RGB")

    samples = sample(image)
    used = pick_used(samples)
    used.sort(key=lambda x: x[0], reverse=True)
    return get_colors(samples, used, number_of_colors)

def hsl(r, g, b):
    if r > g:
        if b > r:
            most, least = b, g
        elif b > g:
            most, least = r, g
        else:
            most, least = r, b
    else:
        if b > g:
            most, least = b, r
        elif b > r:
            most, least = g, r
        else:
            most, least = g, b

    l = (most + least) >> 1
    if most == least:
        h = s = 0
    else:
        diff = most - least
        if l > 127:
            s = diff * 255 // (510 - most - least)
        else:
            s = diff * 255 // (most + least)
        if most == r:
            h = (g - b) * 255 // diff + (1530 if g < b else 0)
        elif most == g:
            h = (b - r) * 255 // diff + 510
        else:
            h = (r - g) * 255 // diff + 1020
        h //= 6
    return h, s, l

def make_colorgram_tensor(color_info, width=512, height=512):
    colors = list(color_info.values())
    topk = len(colors[0].keys())

    tensor = np.ones([topk * 3, height, width], dtype=np.float32)
    region = height // 4

    for i, color in enumerate(colors):
        idx = region * i
        for j in range(1, topk + 1):
            r, g, b = color[str(j)]

            red = (j - 1) * 3
            green = (j - 1) * 3 + 1
            blue = (j - 1) * 3 + 2

            tensor[red, idx:idx + region] *= r
            tensor[green, idx:idx + region] *= g
            tensor[blue, idx:idx + region] *= b

    tensor = torch.from_numpy(tensor.copy())
    return scale(tensor / 255.)

def scale(image):
    return (image * 2) - 1

def re_scale(image):
    return (image + 1) * 0.5

def get_rgb(colorgram_result):
    color = colorgram_result.rgb
    return (color.r, color.g, color.b)

def crop_region(image):
    width, height = image.size
    h1 = height // 4
    h2 = h1 + h1
    h3 = h2 + h1
    h4 = h3 + h1
    image1 = image.crop((0, 0, width, h1))
    image2 = image.crop((0, h1, width, h2))
    image3 = image.crop((0, h2, width, h3))
    image4 = image.crop((0, h3, width, h4))
    return (image1, image2, image3, image4)

def get_topk(color_info, k):
    colors = list(color_info.values())
    return list(map(lambda x: x[k], colors))

def colorize_sketch(sketch, style, output):
    global colorize_model
    if not colorize_model:
        checkpoint = torch.load(os.path.join(get_models_dir(), "misc/colorize.pth"), map_location=device)
        colorize_model = DeepUNetPaintGenerator()
        colorize_model.load_state_dict(checkpoint["model_state"])
        colorize_model = colorize_model.to(device)
    for param in colorize_model.parameters():
        param.requires_grad = False

    width = sketch.width
    height = sketch.height

    #style = Image.open(style).convert("RGB")
    style = transforms.Resize((512, 512))(style)
    style_pil = style

    #sketch = Image.open(sketch).convert("RGB")
    sketch_pil = transforms.Resize((512, 512))(sketch)

    transform = transforms.Compose(
        [transforms.Resize((512, 512)),
         transforms.ToTensor()])

    sketch = transform(sketch)
    sketch = scale(sketch)
    sketch = sketch.unsqueeze(0).to(device)

    to_pil = transforms.ToPILImage()

    images = list(crop_region(style))
    result = {}
    for i, img in enumerate(images, 1):
        colors = extract(img, topk + 1)
        result[str(i)] = {
            "%d" % i: get_rgb(colors[i])
            for i in range(1, topk + 1)
        }

    color_tensor = make_colorgram_tensor(result)
    color_tensor = color_tensor.unsqueeze(0).to(device)

    result, _ = colorize_model(sketch, color_tensor)
    result = result.squeeze(0)
    result = re_scale(result.detach().cpu())
    result = to_pil(result)
    result = result.resize((width, height))

    out_dir = os.path.dirname(output)
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    result.save(output)