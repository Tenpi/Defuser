import flask               
from __main__ import app, socketio
from transformers import AutoProcessor, BlipForConditionalGeneration
from .deepbooru import DeepDanbooruModel
from .functions import get_models_dir, get_device
import pandas as pd
import torch
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
import threading
import inspect
import ctypes
import json
import timm

gen_thread = None
global_result = ""
deepbooru_model = None
wdtagger_model = None
blip_model = None
blip_processor = None

def load_model_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    if "pretrained_cfg" not in config:
        pretrained_cfg = config
        config = {}
        config["architecture"] = pretrained_cfg.pop("architecture")
        config["num_features"] = pretrained_cfg.pop("num_features", None)
        if "labels" in pretrained_cfg:
            pretrained_cfg["label_names"] = pretrained_cfg.pop("labels")
        config["pretrained_cfg"] = pretrained_cfg
    pretrained_cfg = config["pretrained_cfg"]
    if "num_classes" in config:
        pretrained_cfg["num_classes"] = config["num_classes"]

    if "label_names" in config:
        pretrained_cfg["label_names"] = config.pop("label_names")
    if "label_descriptions" in config:
        pretrained_cfg["label_descriptions"] = config.pop("label_descriptions")
    model_args = config.get("model_args", {})
    model_name = config["architecture"]
    kwargs = {}
    if model_args:
        for k, v in model_args.items():
            kwargs.setdefault(k, v)
    return pretrained_cfg, model_name, kwargs

def get_tags(probs, general_threshold = 0.35, character_threshold = 0.75):
    df = pd.read_csv(os.path.join(get_models_dir(), "interrogator/wdtagger/selected_tags.csv"), usecols=["name", "category"])
    labels = {"names": df["name"].tolist(), "rating": list(np.where(df["category"] == 9)[0]), 
              "general": list(np.where(df["category"] == 0)[0]), "character": list(np.where(df["category"] == 4)[0])}
    probs = list(zip(labels["names"], probs.numpy()))
    general_labels = [probs[i] for i in labels["general"]]
    general_labels = dict([x for x in general_labels if x[1] > general_threshold])
    general_labels = dict(sorted(general_labels.items(), key=lambda item: item[1], reverse=True))
    character_labels = [probs[i] for i in labels["character"]]
    character_labels = dict([x for x in character_labels if x[1] > character_threshold])
    character_labels = dict(sorted(character_labels.items(), key=lambda item: item[1], reverse=True))
    combined = [x for x in character_labels]
    combined.extend([x for x in general_labels])
    caption = ", ".join(combined).replace("_", " ")
    return caption

def pad_square(image):
    w, h = image.size
    px = max(image.size)
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas

def _async_raise(tid, exctype):
    '''Raises an exception in the threads with id tid'''
    if not inspect.isclass(exctype):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid),
                                                     ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

@app.route("/interrupt-interrogate", methods=["POST"])
def interrupt_interrogate():
    global gen_thread
    if gen_thread:
        try:
            _async_raise(gen_thread, ChildProcessError)
        except ChildProcessError:
            pass
        gen_thread = None
        return "done"

@socketio.on("load interrogate model")
def load_interrogate_model(model_name):
    global deepbooru_model
    global wdtagger_model
    return
    if model_name == "wdtagger":
        if not wdtagger_model:
            pretrained_cfg, model_name, kwargs = load_model_config(os.path.join(get_models_dir(), "interrogator/wdtagger/config.json"))
            wdtagger_model = timm.create_model(model_name, pretrained_cfg=pretrained_cfg, checkpoint_path=os.path.join(get_models_dir(), "interrogator/wdtagger/model.safetensors"), **kwargs).eval()
    elif model_name == "deepbooru":
        if not deepbooru_model:
            deepbooru_model = DeepDanbooruModel()
        deepbooru_model.load_state_dict(torch.load(os.path.join(get_models_dir(), "interrogator/deepbooru/deepbooru.pt"), map_location="cpu"))
        deepbooru_model.eval()
        deepbooru_model.to(get_device())

def unload_interrogate_models():
    global deepbooru_model
    global wdtagger_model
    global blip_model
    global blip_processor
    deepbooru_model = None
    wdtagger_model = None
    blip_model = None
    blip_processor = None

def process_deepbooru_image(img, dim = 512):
    img = img.resize((dim, dim), resample=Image.BICUBIC)
    img = np.array(img)
    img = img.astype(np.float32)
    img = np.expand_dims(img, 0) / 255
    return torch.from_numpy(img).to(get_device())

def predict_deepbooru(image):
    global deepbooru_model
    if not deepbooru_model:
        deepbooru_model = DeepDanbooruModel()
    deepbooru_model.load_state_dict(torch.load(os.path.join(get_models_dir(), "interrogator/deepbooru/deepbooru.pt"), map_location="cpu"))
    deepbooru_model.eval()
    deepbooru_model.to(get_device())
    tags = []
    with torch.no_grad():
        probs = deepbooru_model(image)[0]
    for i, p in enumerate(probs):
        if p >= 0.5:
            tags.append(deepbooru_model.tags[i])
    return ", ".join(tags)

def predict_wdtagger(image):
    global wdtagger_model
    if not wdtagger_model:
        pretrained_cfg, model_name, kwargs = load_model_config(os.path.join(get_models_dir(), "interrogator/wdtagger/config.json"))
        wdtagger_model = timm.create_model(model_name, pretrained_cfg=pretrained_cfg, checkpoint_path=os.path.join(get_models_dir(), "interrogator/wdtagger/model.safetensors"), **kwargs).eval()

    transform = timm.data.create_transform(**timm.data.resolve_data_config(wdtagger_model.pretrained_cfg, model=wdtagger_model))
    image = pad_square(image)
    input = transform(image).unsqueeze(0)
    input = input[:, [2, 1, 0]]
    wdtagger_model = wdtagger_model.to(get_device())
    input = input.to(get_device())

    with torch.no_grad():
        outputs = wdtagger_model.forward(input)
        outputs = F.sigmoid(outputs)
        outputs = outputs.squeeze(0).to("cpu")

    return get_tags(outputs)

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
    global gen_thread 
    global global_result
    gen_thread = threading.get_ident()
    if not model_name:
        model_name = "wdtagger"

    image = Image.open(file).convert("RGB")

    result = ""
    if model_name == "wdtagger":
        result = predict_wdtagger(image)
    elif model_name == "deepbooru":
        image = process_deepbooru_image(image)
        result = predict_deepbooru(image)
    elif model_name == "blip":
        result = predict_blip(image)
    global_result = result
    return result

@app.route("/interrogate", methods=["POST"])
def interrogate_route():
    global gen_thread
    global global_result
    global_result = ""
    file = flask.request.files["image"]
    model_name = flask.request.form.get("model_name")
    thread = threading.Thread(target=interrogate, args=(file, model_name))
    thread.start()
    thread.join()
    gen_thread = None
    return global_result

