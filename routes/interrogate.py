import flask               
from __main__ import app, socketio
import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import AutoProcessor, BlipForConditionalGeneration
from .deepbooru import DeepDanbooruModel
from .functions import get_models_dir
import pandas as pd
import torch
import os
import numpy as np
from PIL import Image

dirname = os.path.dirname(__file__)
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

deepbooru_model = None
wdtagger_model = None
blip_model = None
blip_processor = None

@socketio.on("load interrogate model")
def load_interrogate_model(model_name):
    global deepbooru_model
    global wdtagger_model
    return
    if model_name == "wdtagger":
        if not wdtagger_model:
            wdtagger_model = load_model(os.path.join(get_models_dir(), "interrogator/wdtagger/wdtagger"))
    elif model_name == "deepbooru":
        if not deepbooru_model:
            deepbooru_model = DeepDanbooruModel()
        deepbooru_model.load_state_dict(torch.load(os.path.join(get_models_dir(), "interrogator/deepbooru/deepbooru.pt"), map_location="cpu"))
        deepbooru_model.eval()
        deepbooru_model.to(device)

def process_deepbooru_image(img, dim = 512):
    img = img.resize((dim, dim), resample=Image.BICUBIC)
    img = np.array(img)
    img = img.astype(np.float32)
    img = np.expand_dims(img, 0) / 255
    return torch.from_numpy(img).to(device)

def predict_deepbooru(image):
    global deepbooru_model
    if not deepbooru_model:
        deepbooru_model = DeepDanbooruModel()
    deepbooru_model.load_state_dict(torch.load(os.path.join(get_models_dir(), "interrogator/deepbooru/deepbooru.pt"), map_location="cpu"))
    deepbooru_model.eval()
    deepbooru_model.to(device)
    tags = []
    with torch.no_grad():
        probs = deepbooru_model(image)[0]
    for i, p in enumerate(probs):
        if p >= 0.5:
            tags.append(deepbooru_model.tags[i])
    return ", ".join(tags)

def process_wdtagger_image(img, dim = 448):
    img = img.resize((dim, dim), resample=Image.BICUBIC)
    img = np.array(img)
    img = img.astype(np.float32)
    img = np.expand_dims(img, 0) / 255
    return tf.convert_to_tensor(img)

def predict_wdtagger(image, thresh = 0.3228):
    global wdtagger_model
    if not wdtagger_model:
        wdtagger_model = load_model(os.path.join(get_models_dir(), "interrogator/wdtagger/wdtagger"))
    label_names = pd.read_csv(os.path.join(get_models_dir(), "interrogator/wdtagger/selected_tags.csv"))
    probs = wdtagger_model.predict(image * 255)
    label_names["probs"] = probs[0]
    found_tags = label_names[label_names["probs"] > thresh]
    return ", ".join(found_tags["name"])

def predict_blip(image):
    global blip_model
    global blip_processor
    if not blip_model or not blip_processor:
        blip_model = BlipForConditionalGeneration.from_pretrained(os.path.join(get_models_dir(), "interrogator/blip"), local_files_only=True)
        blip_processor = AutoProcessor.from_pretrained(os.path.join(get_models_dir(), "interrogator/blip"), local_files_only=True)
    inputs = blip_processor(images=image, text="", return_tensors="pt")
    outputs = blip_model.generate(**inputs)
    result = blip_processor.decode(outputs[0], skip_special_tokens=True)
    return result

def interrogate(file, model_name):
    if not model_name:
        model_name = "wdtagger"

    image = Image.open(file).convert("RGB")

    result = ""
    if model_name == "wdtagger":
        image = process_wdtagger_image(image)
        result = predict_wdtagger(image)
    elif model_name == "deepbooru":
        image = process_deepbooru_image(image)
        result = predict_deepbooru(image)
    elif model_name == "blip":
        result = predict_blip(image)
    return result

@app.route("/interrogate", methods=["POST"])
def interrogate_route():
    file = flask.request.files["image"]
    model_name = flask.request.form.get("model_name")
    return interrogate(file, model_name)

