import flask
from __main__ import app, socketio
from .classifier import train_classifier
from .info import open_folder, show_in_folder, select_file, select_folder
from .functions import clean_image, next_index, get_models_dir, get_outputs_dir
from .simplify_sketch import SketchSimplificationModel
from .shade_sketch import shade_sketch
from .colorize_sketch import colorize_sketch
from .layer_divide import layer_divide
from transformers import BeitFeatureExtractor, BeitForImageClassification
from PIL import Image
import os
import threading
import pathlib
import inspect
import ctypes
from io import BytesIO
import base64
import torch
import safetensors.torch

gen_thread = None
simplify_model = None
dirname = os.path.dirname(__file__)

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

@app.route("/interrupt-misc", methods=["POST"])
def interrupt_misc():
    global gen_thread
    if gen_thread:
        try:
            _async_raise(gen_thread, ChildProcessError)
        except ChildProcessError:
            pass
        gen_thread = None
        socketio.emit("train interrupt")
        return "done"

def image_classifier(train_dir, output_dir, num_train_epochs, learning_rate, gradient_accumulation_steps, lr_scheduler_type, save_steps, resolution):
    global gen_thread 
    gen_thread = threading.get_ident()
    socketio.emit("train starting")
    train_classifier(train_dir, output_dir, num_train_epochs, learning_rate, gradient_accumulation_steps, lr_scheduler_type, save_steps, resolution)
    socketio.emit("train complete")
    open_folder("", output_dir)
    return "done"

@app.route("/train-classifier", methods=["POST"])
def start_image_classifier():
    global gen_thread
    data = flask.request.json
    train_dir = data["train_dir"].strip()
    num_train_epochs = data["num_train_epochs"]
    learning_rate = data["learning_rate"]
    save_steps = data["save_steps"] 
    gradient_accumulation_steps = data["gradient_accumulation_steps"]
    lr_scheduler_type = data["learning_function"]
    resolution = data["resolution"] 
    output_dir = os.path.join(get_outputs_dir(), f"models/classifier/{pathlib.Path(train_dir).stem}")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    thread = threading.Thread(target=image_classifier, args=(train_dir, output_dir, num_train_epochs, learning_rate, 
    gradient_accumulation_steps, lr_scheduler_type, save_steps, resolution))
    thread.start()
    thread.join()
    gen_thread = None
    return "done"

def ai_detector(image):
    global gen_thread 
    gen_thread = threading.get_ident()
    socketio.emit("train starting")
    img = Image.open(BytesIO(base64.b64decode(image + "=="))).convert("RGB")
    img = clean_image(img)
    feature_extractor = BeitFeatureExtractor.from_pretrained(os.path.join(get_models_dir(), "detector"))
    model = BeitForImageClassification.from_pretrained(os.path.join(get_models_dir(), "detector"))
    inputs = feature_extractor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probs = logits.softmax(-1)
    predicted_class_idx = probs.argmax().item()
    label = model.config.id2label[predicted_class_idx]
    probability = probs[0][predicted_class_idx].item()
    socketio.emit("train complete", {"label": label, "probability": round(probability * 100, 2)})

@app.route("/ai-detector", methods=["POST"])
def start_ai_detector():
    global gen_thread
    data = flask.request.json
    image = data["image"]
    thread = threading.Thread(target=ai_detector, args=(image,))
    thread.start()
    thread.join()
    gen_thread = None
    return "done"

def simplify_sketch(image, format):
    global simplify_model
    global gen_thread 
    gen_thread = threading.get_ident()
    socketio.emit("train starting")
    img = Image.open(BytesIO(base64.b64decode(image + "=="))).convert("RGB")
    if not simplify_model:
        simplify_model = SketchSimplificationModel()

    dir_path = os.path.join(get_outputs_dir(), f"local/image")
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.{format}")
    simplify_model(img, out_path)
    compressed = Image.open(out_path)
    compressed.save(out_path, quality=90, optimize=True)
    socketio.emit("train complete", {"image": f"/outputs/local/image/{os.path.basename(out_path)}"})
    return "done"

@app.route("/simplify-sketch", methods=["POST"])
def start_simplify_sketch():
    global gen_thread
    data = flask.request.json
    image = data["image"]
    format = data["format"]
    if format == "gif": format = "jpg"
    thread = threading.Thread(target=simplify_sketch, args=(image, format))
    thread.start()
    thread.join()
    gen_thread = None
    return "done"

def run_shade_sketch(image, format, direction):
    global gen_thread 
    gen_thread = threading.get_ident()
    socketio.emit("train starting")
    img = Image.open(BytesIO(base64.b64decode(image + "=="))).convert("RGB")
    dir_path = os.path.join(get_outputs_dir(), f"local/image")
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.{format}")
    shade_sketch(img, out_path, direction)
    compressed = Image.open(out_path)
    compressed.save(out_path, quality=90, optimize=True)
    socketio.emit("train complete", {"image": f"/outputs/local/image/{os.path.basename(out_path)}"})
    return "done"

@app.route("/shade-sketch", methods=["POST"])
def start_shade_sketch():
    global gen_thread
    data = flask.request.json
    image = data["image"]
    format = data["format"]
    direction = data["direction"]
    if format == "gif": format = "jpg"
    thread = threading.Thread(target=run_shade_sketch, args=(image, format, direction))
    thread.start()
    thread.join()
    gen_thread = None
    return "done"

def run_colorize_sketch(sketch, style, format):
    global gen_thread 
    gen_thread = threading.get_ident()
    socketio.emit("train starting")
    sketch_img = Image.open(BytesIO(base64.b64decode(sketch + "=="))).convert("RGB")
    style_img = Image.open(BytesIO(base64.b64decode(style + "=="))).convert("RGB")
    dir_path = os.path.join(get_outputs_dir(), f"local/image")
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.{format}")

    colorize_sketch(sketch_img, style_img, out_path)
    compressed = Image.open(out_path)
    compressed.save(out_path, quality=90, optimize=True)
    socketio.emit("train complete", {"image": f"/outputs/local/image/{os.path.basename(out_path)}"})
    return "done"

@app.route("/colorize-sketch", methods=["POST"])
def start_colorize_sketch():
    global gen_thread
    data = flask.request.json
    sketch = data["sketch"]
    style = data["style"]
    format = data["format"]
    if format == "gif": format = "jpg"
    thread = threading.Thread(target=run_colorize_sketch, args=(sketch, style, format))
    thread.start()
    thread.join()
    gen_thread = None
    return "done"

def run_layer_divide(input_image, divide_mode, loops, clusters, cluster_threshold, blur_size, layer_mode, area, downscale):
    global gen_thread 
    gen_thread = threading.get_ident()
    socketio.emit("train starting")
    input_image = Image.open(BytesIO(base64.b64decode(input_image + "=="))).convert("RGB")
    output_dir = os.path.join(get_outputs_dir(), f"local/psd")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if downscale and int(downscale) > 0:
        input_image.thumbnail((int(downscale), int(downscale)))
    output = layer_divide(input_image, output_dir, divide_mode, loops, clusters, cluster_threshold, blur_size, layer_mode, area)
    socketio.emit("train complete")
    show_in_folder("", output)
    return "done"

@app.route("/layer-divide", methods=["POST"])
def start_layer_divide():
    global gen_thread
    data = flask.request.json
    input_image = data["image"]
    divide_mode = data["divide_mode"]
    layer_mode = data["layer_mode"]
    loops = int(data["loops"])
    clusters = int(data["clusters"])
    cluster_threshold = int(data["cluster_threshold"])
    blur_size = int(data["blur_size"])
    area = int(data["area"])
    downscale = data["downscale"]
    thread = threading.Thread(target=run_layer_divide, args=(input_image, divide_mode, loops, clusters, cluster_threshold, blur_size, layer_mode, area, downscale))
    thread.start()
    thread.join()
    gen_thread = None
    return "done"

def convert_embedding(input_path, output_path):
    file_extension = pathlib.Path(input_path).suffix
    sd15_tensor = None
    if file_extension == ".pt":
        sd15_embedding = torch.load(input_path, map_location=torch.device("cpu"), weights_only=True)
        sd15_tensor = sd15_embedding["string_to_param"]["*"]
    elif file_extension == ".bin":
        sd15_embedding = torch.load(input_path, map_location=torch.device("cpu"), weights_only=True)
        sd15_tensor = sd15_embedding["*"]
    elif file_extension == ".safetensors":
        loaded_tensors = safetensors.torch.load_file(input_path)
        sd15_tensor = loaded_tensors["emb_params"]
    else:
        raise ValueError("Unsupported file format")
    num_vectors = sd15_tensor.shape[0]
    clip_g_shape = (num_vectors, 1280)
    clip_l_shape = (num_vectors, 768)
    clip_g = torch.zeros(clip_g_shape, dtype=torch.float16)
    clip_l = torch.zeros(clip_l_shape, dtype=torch.float16)
    clip_l[:sd15_tensor.shape[0], :sd15_tensor.shape[1]] = sd15_tensor.to(dtype=torch.float16)
    safetensors.torch.save_file({"clip_g": clip_g, "clip_l": clip_l}, output_path)

@app.route("/convert-embedding", methods=["POST"])
def start_convert_embedding():
    data = flask.request.json
    mode = data["mode"]
    if mode == "folder":
        folder = select_folder()
        output_folder = os.path.join(os.path.dirname(folder), f"{pathlib.Path(folder).stem}XL")
        os.makedirs(output_folder, exist_ok=True)
        files = os.listdir(folder)
        for file in files:
            input_path = os.path.join(folder, file)
            output_path = os.path.join(output_folder, f"{pathlib.Path(file).stem}XL{pathlib.Path(file).suffix}")
            try:
                convert_embedding(input_path, output_path)
            except ValueError:
                continue
        show_in_folder("", output_folder)
    else:
        file = select_file()
        output_path = os.path.join(os.path.dirname(file), f"{pathlib.Path(file).stem}XL{pathlib.Path(file).suffix}")
        convert_embedding(file, output_path)
        show_in_folder("", output_path)
    return "done"