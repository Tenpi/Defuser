import flask               
from __main__ import app, socketio
from io import BytesIO
from .generate import upscale
import os
import torch
from .functions import next_index, get_models_dir, get_outputs_dir
from PIL import Image
from controlnet_aux import CannyDetector, MidasDetector, LineartDetector, LineartAnimeDetector, HEDdetector
from .lineart_manga import LineartMangaDetector
import PIL.ImageOps 
import cv2

dirname = os.path.dirname(__file__)
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

midas = None
lineart = None
lineart_anime = None
lineart_manga = None
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

@socketio.on("load control models")
def load_control_models(h):
    global midas
    global lineart
    global lineart_anime
    global lineart_manga
    global hed
    global device
    global dtype
    return
    if not lineart:
        lineart = LineartDetector.from_pretrained(os.path.join(get_models_dir(), "controlnet/annotator"), filename="lineart.pt", coarse_filename="lineart2.pt").to(device)
    if not lineart_anime:
        lineart_anime = LineartAnimeDetector.from_pretrained(os.path.join(get_models_dir(), "controlnet/annotator"), filename="lineart_anime.pt").to(device)
    if not lineart_manga:
        lineart_manga = LineartMangaDetector()
    if not hed:
        hed = HEDdetector.from_pretrained(os.path.join(get_models_dir(), "controlnet/annotator"), filename="hed.pt").to(device)
    if not midas:
        midas = MidasDetector.from_pretrained(os.path.join(get_models_dir(), "controlnet/annotator"), filename="midas.pt").to(device)

def unload_control_models():
    global midas
    global lineart
    global lineart_anime
    global lineart_manga
    global hed
    midas = None
    lineart = None
    lineart_anime = None
    lineart_manga = None
    hed = None

@app.route("/control-image", methods=["POST"])
def control_image():
    global midas
    global lineart
    global lineart_anime
    global lineart_manga
    global hed
    global device
    global dtype
    file = flask.request.files["image"]
    processor = flask.request.form.get("processor")
    upscale_image = flask.request.form.get("upscale")
    upscaler = flask.request.form.get("upscaler")
    invert = flask.request.form.get("invert")
    alpha = flask.request.form.get("alpha")
    if not processor or processor == "none":
        return None

    if upscale_image == "true":
        socketio.emit("image starting")

    image = Image.open(file).convert("RGB")

    output_image = None
    if processor == "canny":
        canny = CannyDetector()
        output_image = canny(image)
    elif processor == "depth":
        if not midas:
            midas = MidasDetector.from_pretrained(os.path.join(get_models_dir(), "controlnet/annotator"), filename="midas.pt").to(device)
        output_image = midas(image)
    elif processor == "lineart":
        if not lineart:
            lineart = LineartDetector.from_pretrained(os.path.join(get_models_dir(), "controlnet/annotator"), filename="lineart.pt", coarse_filename="lineart2.pt").to(device)
        output_image = lineart(image, coarse=False)
    elif processor == "lineart anime":
        if not lineart_anime:
            lineart_anime = LineartAnimeDetector.from_pretrained(os.path.join(get_models_dir(), "controlnet/annotator"), filename="lineart_anime.pt").to(device)
        output_image = lineart_anime(image)
    elif processor == "lineart manga":
        if not lineart_manga:
            lineart_manga = LineartMangaDetector()
        output_image = lineart_manga(image)
    elif processor == "scribble":
        if not hed:
            hed = HEDdetector.from_pretrained(os.path.join(get_models_dir(), "controlnet/annotator"), filename="hed.pt").to(device)
        output_image = hed(image, scribble=True)
    elif processor == "softedge":
        if not hed:
            hed = HEDdetector.from_pretrained(os.path.join(get_models_dir(), "controlnet/annotator"), filename="hed.pt").to(device)
        output_image = hed(image, scribble=False)
    elif processor == "reference":
        output_image = image

    if upscale_image:
        dir_path = os.path.join(get_outputs_dir(), "local/image")
        out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.png")
        output_image.save(out_path)
        socketio.emit("image upscaling")
        upscale(out_path, upscaler)
        compressed = Image.open(out_path)
        if invert == "true":
            compressed = PIL.ImageOps.invert(compressed)
        compressed.save(out_path, quality=90, optimize=True)
        if alpha == "true":
            if invert == "true":
                white_to_alpha(out_path)
            else:
                black_to_alpha(out_path)
        socketio.emit("image complete", {"image": f"/outputs/local/image/{os.path.basename(out_path)}"})
        return "done"
    else:
        img_io = BytesIO()
        output_image.save(img_io, format="JPEG", quality=80)
        img_io.seek(0)
        return flask.send_file(img_io, mimetype="image/jpeg")