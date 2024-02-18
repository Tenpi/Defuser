from unicodedata import normalize
import flask               
from __main__ import app, socketio
from io import BytesIO
from .functions import get_normalized_dimensions
import os
import torch
from .functions import next_index
from PIL import Image
import numpy as np
from controlnet_aux.processor import Processor
from controlnet_aux import CannyDetector, MidasDetector, LineartDetector, LineartAnimeDetector, HEDdetector
import pathlib
import subprocess
import PIL.ImageOps 
import cv2

dirname = os.path.dirname(__file__)
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

midas = None
lineart = None
lineart_anime = None
hed = None

def white_to_alpha(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    (row, col, channel) = image.shape
    for r in range(row):
        for c in range(col):
            px = image[r][c]
            if px[0] > 125 and px[1] > 125 and px[2] > 125:
                image[r][c][3] = 0
            else:
                image[r][c][0] = 0
                image[r][c][1] = 0
                image[r][c][2] = 0
                image[r][c][3] = 255
    cv2.imwrite(img_path, image)

def black_to_alpha(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    (row, col, channel) = image.shape
    for r in range(row):
        for c in range(col):
            px = image[r][c]
            if px[0] < 125 and px[1] < 125 and px[2] < 125:
                image[r][c][3] = 0
            else:
                image[r][c][0] = 255
                image[r][c][1] = 255
                image[r][c][2] = 255
                image[r][c][3] = 255
    cv2.imwrite(img_path, image)

def upscale_image(image: str, video: bool = False):
    program = os.path.join(dirname, "../models/upscaler/realesrgan-ncnn-vulkan")
    models = os.path.join(dirname, "../models/upscaler/models")
    network = "realesr-animevideov3" if video else "realesrgan-x4plus-anime"
    format = pathlib.Path(image).suffix.replace(".", "")
    subprocess.call([program, "-m", models, "-i", image, "-o", image, "-s", "4", "-f", format, "-n", network])

@socketio.on("load control models")
def load_control_models():
    global midas
    global lineart
    global lineart_anime
    global hed
    global device
    global dtype
    if not midas:
        midas = MidasDetector.from_pretrained(os.path.join(dirname, "../models/controlnet/annotator"), filename="midas.pt").to(device)
    if not lineart:
        lineart = LineartDetector.from_pretrained(os.path.join(dirname, "../models/controlnet/annotator"), filename="lineart.pt", coarse_filename="lineart2.pt").to(device)
    if not lineart_anime:
        lineart_anime = LineartAnimeDetector.from_pretrained(os.path.join(dirname, "../models/controlnet/annotator"), filename="lineart_anime.pt").to(device)
    if not hed:
        hed = HEDdetector.from_pretrained(os.path.join(dirname, "../models/controlnet/annotator"), filename="hed.pt").to(device)

@app.route("/control-image", methods=["POST"])
def control_image():
    global midas
    global lineart
    global lineart_anime
    global hed
    global device
    global dtype
    file = flask.request.files["image"]
    processor = flask.request.form.get("processor")
    upscale = flask.request.form.get("upscale")
    invert = flask.request.form.get("invert")
    alpha = flask.request.form.get("alpha")
    if not processor or processor == "none":
        return None

    if upscale == "true":
        socketio.emit("image starting")

    image = Image.open(file).convert("RGB")

    output_image = None
    if processor == "canny":
        canny = CannyDetector()
        output_image = canny(image)
    elif processor == "depth":
        if not midas:
            midas = MidasDetector.from_pretrained(os.path.join(dirname, "../models/controlnet/annotator"), filename="midas.pt").to(device)
        output_image = midas(image)
    elif processor == "lineart":
        if not lineart:
            lineart = LineartDetector.from_pretrained(os.path.join(dirname, "../models/controlnet/annotator"), filename="lineart.pt", coarse_filename="lineart2.pt").to(device)
        output_image = lineart(image, coarse=False)
    elif processor == "lineart anime":
        if not lineart_anime:
            lineart_anime = LineartAnimeDetector.from_pretrained(os.path.join(dirname, "../models/controlnet/annotator"), filename="lineart_anime.pt").to(device)
        output_image = lineart_anime(image)
    elif processor == "scribble":
        if not hed:
            hed = HEDdetector.from_pretrained(os.path.join(dirname, "../models/controlnet/annotator"), filename="hed.pt").to(device)
        output_image = hed(image, scribble=True)
    elif processor == "softedge":
        if not hed:
            hed = HEDdetector.from_pretrained(os.path.join(dirname, "../models/controlnet/annotator"), filename="hed.pt").to(device)
        output_image = hed(image, scribble=False)
    elif processor == "reference":
        output_image = image

    if upscale:
        dir_path = os.path.join(dirname, "../outputs/image")
        out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.png")
        output_image.save(out_path)
        socketio.emit("image upscaling")
        upscale_image(out_path)
        compressed = Image.open(out_path)
        if invert == "true":
            compressed = PIL.ImageOps.invert(compressed)
        compressed.save(out_path, quality=90, optimize=True)
        if alpha == "true":
            if invert == "true":
                white_to_alpha(out_path)
            else:
                black_to_alpha(out_path)
        socketio.emit("image complete", {"image": f"/outputs/image/{os.path.basename(out_path)}"})
        return "done"
    else:
        img_io = BytesIO()
        output_image.save(img_io, format="JPEG", quality=80)
        img_io.seek(0)
        return flask.send_file(img_io, mimetype="image/jpeg")