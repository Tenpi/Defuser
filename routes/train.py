import flask
from __main__ import app, socketio
from .functions import resize_box, get_models_dir, get_outputs_dir
from .interrogate import interrogate
from .textual_inversion import train_textual_inversion
from .hypernetwork import train_hypernetwork
from .lora import train_lora
from .dreambooth import train_dreambooth
from .dreamartist import train_dreamartist
from .checkpoint_merger import merge
from .info import show_in_folder
from pysaucenao import SauceNao
import asyncio
import time
import os
import subprocess
import platform
import threading
import inspect
import ctypes
import pathlib
import re
from PIL import Image
import cv2

gen_thread = None

@app.route("/show-text", methods=["POST"])
def show_text():
    data = flask.request.json
    image = data["image"]
    name, ext = os.path.splitext(image)
    dest = f"{name}.txt"
    if os.path.exists(dest):
        if platform.system() == "Windows":
            subprocess.call(["notepad.exe", dest])
        elif platform.system() == "Darwin":
            subprocess.call(["open", "-a", "TextEdit", dest])
        else:
            subprocess.call(["xdg-open", dest])
    return "done"

@app.route("/show-source", methods=["POST"])
def show_source():
    data = flask.request.json
    image = data["image"]
    name, ext = os.path.splitext(image)
    dest = f"{name}.source.txt"
    if os.path.exists(dest):
        if platform.system() == "Windows":
            subprocess.call(["notepad.exe", dest])
        elif platform.system() == "Darwin":
            subprocess.call(["open", "-a", "TextEdit", dest])
        else:
            subprocess.call(["xdg-open", dest])
    return "done"

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

@app.route("/interrupt-train", methods=["POST"])
def interrupt_train():
    global gen_thread
    if gen_thread:
        try:
            _async_raise(gen_thread, ChildProcessError)
        except ChildProcessError:
            pass
        gen_thread = None
        socketio.emit("train interrupt")
        return "done"
    
def tag(images, model_name, append):
    global gen_thread 
    gen_thread = threading.get_ident()
    socketio.emit("train starting")
    for i, image in enumerate(images):
        result = interrogate(image, model_name)
        tag_arr = result.split(",")
        new_tag_arr = []
        for tag in tag_arr:
            if len(tag) > 3:
                new_tag_arr.append(tag.replace("_", " ").strip())
            else:
                new_tag_arr.append(tag.strip())
        if append:
            new_tag_arr.insert(0, append)
        result = ", ".join(new_tag_arr)
        name, ext = os.path.splitext(image)
        dest = f"{name}.txt"
        f = open(os.path.normpath(dest), "w")
        f.write(result)
        f.close()
        socketio.emit("train progress", {"step": i+1, "total_step": len(images)})
    socketio.emit("train complete")
    return "done"

@app.route("/tag", methods=["POST"])
def start_tag():
    global gen_thread
    data = flask.request.json
    images = data["images"]
    model_name = data["model"]
    append = data["append"].strip()
    thread = threading.Thread(target=tag, args=(images, model_name, append))
    thread.start()
    thread.join()
    gen_thread = None
    return "done"

@app.route("/delete-tags", methods=["POST"])
def delete_tags():
    data = flask.request.json
    images = data["images"]
    for image in images:
        name, ext = os.path.splitext(image)
        dest = f"{name}.txt"
        if os.path.exists(dest):
            os.remove(dest)
    return "done"

async def source(images, saucenao_key):
    global gen_thread 
    gen_thread = threading.get_ident()
    socketio.emit("train starting")
    sn = SauceNao(api_key=saucenao_key, priority=[9, 5])
    for i, image in enumerate(images):
        result = ""
        try:
            pixiv_shortcut = re.search(r"\d{5,}", image)
            if pixiv_shortcut:
                result = f"https://www.pixiv.net/en/artworks/{pixiv_shortcut.group()}"
            else:
                results = await sn.from_file(image)
                result = results[0].source_url
        except:
            time.sleep(31)
            results = await sn.from_file(image)
            result = results[0].source_url
        if result:
            if "pixiv" in result or "pximg" in result:
                match = re.search(r"\d{5,}", result)
                if match:
                    result = f"https://www.pixiv.net/en/artworks/{match.group()}"
            name, ext = os.path.splitext(image)
            dest = f"{name}.source.txt"
            f = open(os.path.normpath(dest), "w")
            f.write(result)
            f.close()
        socketio.emit("train progress", {"step": i+1, "total_step": len(images)})
    socketio.emit("train complete")
    return "done"

@app.route("/source", methods=["POST"])
def start_source():
    global gen_thread
    data = flask.request.json
    images = data["images"]
    saucenao_key = data["saucenao_key"]
    thread = threading.Thread(target=asyncio.run, args=(source(images, saucenao_key),))
    thread.start()
    thread.join()
    gen_thread = None
    return "done"

@app.route("/delete-sources", methods=["POST"])
def delete_sources():
    data = flask.request.json
    images = data["images"]
    for image in images:
        name, ext = os.path.splitext(image)
        dest = f"{name}.source.txt"
        if os.path.exists(dest):
            os.remove(dest)
    return "done"

def textual_inversion(images, model_name, train_data, token, output, num_train_epochs, learning_rate, resolution, save_epochs, num_vectors, 
    gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler):
    global gen_thread 
    gen_thread = threading.get_ident()
    socketio.emit("train starting")
    train_textual_inversion(images, model_name, train_data, token, output, num_train_epochs, learning_rate, resolution, save_epochs, num_vectors, 
    gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler)
    socketio.emit("train complete")
    show_in_folder("", f"{output}/{token}.bin")
    return "done"

@app.route("/train-textual-inversion", methods=["POST"])
def start_textual_inversion():
    global gen_thread
    data = flask.request.json
    images = data["images"]
    model_name = data["model_name"]
    train_data = data["train_data"].strip()
    token = data["token"]
    num_train_epochs = data["num_train_epochs"]
    learning_rate = data["learning_rate"]
    resolution = data["resolution"]
    save_epochs = data["save_epochs"] 
    num_vectors = data["num_vectors"]
    gradient_accumulation_steps = data["gradient_accumulation_steps"]
    validation_prompt = data["validation_prompt"]
    validation_epochs = data["validation_epochs"]
    lr_scheduler = data["learning_function"]
    output = os.path.join(get_outputs_dir(), f"models/textual inversion/{token}")
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
    model_name = os.path.join(get_models_dir(), f"diffusion/{model_name}")

    thread = threading.Thread(target=textual_inversion, args=(images, model_name, train_data, token, output, num_train_epochs, learning_rate, resolution, save_epochs, num_vectors, 
    gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler))
    thread.start()
    thread.join()
    gen_thread = None
    return "done"

def hypernetwork(images, model_name, train_data, instance_prompt, output, num_train_epochs, learning_rate, text_encoder_lr, resolution, save_epochs, 
    gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler, sizes):
    global gen_thread 
    gen_thread = threading.get_ident()
    socketio.emit("train starting")
    train_hypernetwork(images, model_name, train_data, instance_prompt, output, num_train_epochs, learning_rate, text_encoder_lr, resolution, save_epochs, 
    gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler, sizes)
    socketio.emit("train complete")
    show_in_folder("", f"{output}/{instance_prompt}.pt")
    return "done"

@app.route("/train-hypernetwork", methods=["POST"])
def start_hypernetwork():
    global gen_thread
    data = flask.request.json
    images = data["images"]
    model_name = data["model_name"]
    train_data = data["train_data"].strip()
    instance_prompt = data["instance_prompt"]
    num_train_epochs = data["num_train_epochs"]
    learning_rate = data["learning_rate"]
    text_encoder_lr = data["text_encoder_lr"]
    resolution = data["resolution"]
    save_epochs = data["save_epochs"] 
    gradient_accumulation_steps = data["gradient_accumulation_steps"]
    validation_prompt = data["validation_prompt"]
    validation_epochs = data["validation_epochs"]
    lr_scheduler = data["learning_function"]
    sizes = data["sizes"]
    print(sizes)
    output = os.path.join(get_outputs_dir(), f"models/hypernetwork/{instance_prompt}")
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
    model_name = os.path.join(get_models_dir(), f"diffusion/{model_name}")

    thread = threading.Thread(target=hypernetwork, args=(images, model_name, train_data, instance_prompt, output, num_train_epochs, learning_rate, text_encoder_lr, resolution, save_epochs, 
    gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler, sizes))
    thread.start()
    thread.join()
    gen_thread = None
    return "done"

def lora(images, model_name, train_data, instance_prompt, output, num_train_epochs, learning_rate, text_encoder_lr, resolution, save_epochs, 
    gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler, rank):
    global gen_thread 
    gen_thread = threading.get_ident()
    socketio.emit("train starting")
    train_lora(images, model_name, train_data, instance_prompt, output, num_train_epochs, learning_rate, text_encoder_lr, resolution, save_epochs, 
    gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler, rank)
    socketio.emit("train complete")
    show_in_folder("", f"{output}/{instance_prompt}.safetensors")
    return "done"

@app.route("/train-lora", methods=["POST"])
def start_lora():
    global gen_thread
    data = flask.request.json
    images = data["images"]
    model_name = data["model_name"]
    train_data = data["train_data"].strip()
    instance_prompt = data["instance_prompt"]
    num_train_epochs = data["num_train_epochs"]
    learning_rate = data["learning_rate"]
    text_encoder_lr = data["text_encoder_lr"]
    resolution = data["resolution"]
    save_epochs = data["save_epochs"] 
    gradient_accumulation_steps = data["gradient_accumulation_steps"]
    validation_prompt = data["validation_prompt"]
    validation_epochs = data["validation_epochs"]
    lr_scheduler = data["learning_function"]
    rank = data["rank"]
    output = os.path.join(get_outputs_dir(), f"models/lora/{instance_prompt}")
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
    model_name = os.path.join(get_models_dir(), f"diffusion/{model_name}")

    thread = threading.Thread(target=lora, args=(images, model_name, train_data, instance_prompt, output, num_train_epochs, learning_rate, text_encoder_lr, resolution, save_epochs, 
    gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler, rank))
    thread.start()
    thread.join()
    gen_thread = None
    return "done"

def dreambooth(images, model_name, train_data, instance_prompt, output, num_train_epochs, learning_rate, text_encoder_lr, resolution, save_steps, 
    gradient_accumulation_steps, validation_prompt, validation_steps, lr_scheduler, new_unet, new_text_encoder):
    global gen_thread 
    gen_thread = threading.get_ident()
    socketio.emit("train starting")
    train_dreambooth(images, model_name, train_data, instance_prompt, output, num_train_epochs, learning_rate, text_encoder_lr, resolution, save_steps, 
    gradient_accumulation_steps, validation_prompt, validation_steps, lr_scheduler, new_unet, new_text_encoder)
    socketio.emit("train complete")
    show_in_folder("", f"{output}/{instance_prompt}.ckpt")
    return "done"

@app.route("/train-dreambooth", methods=["POST"])
def start_dreambooth():
    global gen_thread
    data = flask.request.json
    images = data["images"]
    model_name = data["model_name"]
    train_data = data["train_data"].strip()
    instance_prompt = data["instance_prompt"]
    num_train_epochs = data["num_train_epochs"]
    learning_rate = data["learning_rate"]
    text_encoder_lr = data["text_encoder_lr"]
    resolution = data["resolution"]
    save_steps = data["save_steps"] 
    gradient_accumulation_steps = data["gradient_accumulation_steps"]
    validation_prompt = data["validation_prompt"]
    validation_steps = data["validation_steps"]
    lr_scheduler = data["learning_function"]
    new_unet = data["new_unet"]
    new_text_encoder = data["new_text_encoder"]
    output = os.path.join(get_outputs_dir(), f"models/dreambooth/{instance_prompt}")
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
    model_name = os.path.join(get_models_dir(), f"diffusion/{model_name}")

    thread = threading.Thread(target=dreambooth, args=(images, model_name, train_data, instance_prompt, output, num_train_epochs, learning_rate, text_encoder_lr, resolution, save_steps, 
    gradient_accumulation_steps, validation_prompt, validation_steps, lr_scheduler, new_unet, new_text_encoder))
    thread.start()
    thread.join()
    gen_thread = None
    return "done"

def dreamartist(images, model_name, train_data, token, output, num_train_epochs, learning_rate, resolution, save_epochs, num_vectors, 
    num_neg_vectors, gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler, cfg_scale):
    global gen_thread 
    gen_thread = threading.get_ident()
    socketio.emit("train starting")
    train_dreamartist(images, model_name, train_data, token, output, num_train_epochs, learning_rate, resolution, save_epochs, num_vectors, 
    num_neg_vectors, gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler, cfg_scale)
    socketio.emit("train complete")
    show_in_folder("", f"{output}/{token}.bin")
    return "done"

@app.route("/train-dreamartist", methods=["POST"])
def start_dreamartist():
    global gen_thread
    data = flask.request.json
    images = data["images"]
    model_name = data["model_name"]
    train_data = data["train_data"].strip()
    token = data["token"]
    num_train_epochs = data["num_train_epochs"]
    learning_rate = data["learning_rate"]
    resolution = data["resolution"]
    save_epochs = data["save_epochs"] 
    num_vectors = data["num_vectors"]
    num_neg_vectors = data["num_neg_vectors"]
    gradient_accumulation_steps = data["gradient_accumulation_steps"]
    validation_prompt = data["validation_prompt"]
    validation_epochs = data["validation_epochs"]
    lr_scheduler = data["learning_function"]
    cfg_scale = data["cfg_scale"]
    output = os.path.join(get_outputs_dir(), f"models/dreamartist/{token}")
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
    model_name = os.path.join(get_models_dir(), f"diffusion/{model_name}")

    thread = threading.Thread(target=dreamartist, args=(images, model_name, train_data, token, output, num_train_epochs, learning_rate, resolution, save_epochs, num_vectors, 
    num_neg_vectors, gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler, cfg_scale))
    thread.start()
    thread.join()
    gen_thread = None
    return "done"

def merge_checkpoints(models, alpha, interpolation):
    global gen_thread 
    gen_thread = threading.get_ident()
    socketio.emit("train starting")
    models = list(map(lambda model: os.path.join(get_models_dir(), f"diffusion/{model}"), models))
    name = ""
    for i, model in enumerate(models):
        if i != 0: name += "+"
        name += pathlib.Path(model).stem
    output_dir = os.path.join(get_outputs_dir(), f"models/merged")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    output = os.path.join(output_dir, f"{name}.ckpt")
    merge(models, output, alpha=alpha, interp=interpolation)
    socketio.emit("train complete")
    show_in_folder(f"outputs/models/merged/{name}.ckpt", "")
    return "done"

@app.route("/merge", methods=["POST"])
def merge_route():
    global gen_thread
    data = flask.request.json
    models = data["models"]
    alpha = data["alpha"]
    interpolation = data["interpolation"]
    thread = threading.Thread(target=merge_checkpoints, args=(models, alpha, interpolation))
    thread.start()
    thread.join()
    gen_thread = None
    return "done"

def crop_anime_face_fallback(images, size=512):
    cascade_path = os.path.join(get_models_dir(), "misc/lbpcascade_animeface.xml")
    face_detector = cv2.CascadeClassifier(cascade_path)

    for i, img_path in enumerate(images):
        if (".DS_Store" in img_path): return
        img_path = os.path.normpath(img_path)
        img = cv2.imread(img_path)
        img_h, img_w, img_c = img.shape
        faces = face_detector.detectMultiScale(img)
        for x, y, w, h in faces:
            x, x1, y, y1 = resize_box(x, x + w, y, y + h, img_w, img_h, size)
            face_img = img[int(y):int(y1), int(x):int(x1)]
            cv2.imwrite(img_path, face_img)
        socketio.emit("train progress", {"step": i+1, "total_step": len(images)})

def crop_anime_face(images, size=512):
    global gen_thread 
    gen_thread = threading.get_ident()
    socketio.emit("train starting")
    try:
        import animeface
        for i, img_path in enumerate(images):
            if (".DS_Store" in img_path): return
            img_path = os.path.normpath(img_path)
            img = cv2.imread(img_path)
            img_h, img_w, img_c = img.shape
            faces = animeface.detect(Image.open(img_path))
            if len(faces):
                pos = faces[0].face.pos
                x, x1, y, y1 = resize_box(pos.x, pos.x + pos.width, pos.y, pos.y + pos.height, img_w, img_h, size)
                face_img = img[int(y):int(y1), int(x):int(x1)]
                cv2.imwrite(img_path, face_img)
            socketio.emit("train progress", {"step": i+1, "total_step": len(images)})
    except ImportError:
        crop_anime_face_fallback(images, size)
    socketio.emit("train complete")
    return "done"

@app.route("/crop", methods=["POST"])
def crop_anime_route():
    global gen_thread
    data = flask.request.json
    images = data["images"]
    size = data["resolution"]
    thread = threading.Thread(target=crop_anime_face, args=(images, int(size)))
    thread.start()
    thread.join()
    gen_thread = None
    return "done"