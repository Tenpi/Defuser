import flask
from __main__ import app, socketio
from .interrogate import interrogate
from .textual_inversion import train_textual_inversion
from .lora import train_lora
from .dreambooth import train_dreambooth
from .checkpoint import train_checkpoint
from .info import show_in_folder
import os
import subprocess
import platform
import threading
import inspect
import ctypes
import pathlib

gen_thread = None
dirname = os.path.dirname(__file__)

@app.route("/show-text", methods=["POST"])
def show_text():
    data = flask.request.json
    image = data["image"]
    name, ext = os.path.splitext(image)
    dest = f"{name}.txt"
    if os.path.exists(dest):
        if platform.system() == "Windows":
            subprocess.Popen(["notepad.exe", dest])
        elif platform.system() == "Darwin":
            subprocess.call(["open", "-a", "TextEdit", dest])
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
    
def tag(images, model_name):
    global gen_thread 
    gen_thread = threading.get_ident()
    socketio.emit("train starting")
    for i, image in enumerate(images):
        result = interrogate(image, model_name)
        name, ext = os.path.splitext(image)
        dest = f"{name}.txt"
        f = open(dest, "w")
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
    thread = threading.Thread(target=tag, args=(images, model_name))
    thread.start()
    thread.join()
    gen_thread = None
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
    output = os.path.join(dirname, f"../outputs/textual inversion/{token}")
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
    model_name = os.path.join(dirname, f"../models/diffusion/{model_name}")

    thread = threading.Thread(target=textual_inversion, args=(images, model_name, train_data, token, output, num_train_epochs, learning_rate, resolution, save_epochs, num_vectors, 
    gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler))
    thread.start()
    thread.join()
    gen_thread = None
    return "done"

def lora(images, model_name, train_data, instance_prompt, output, num_train_epochs, learning_rate, text_encoder_lr, resolution, save_epochs, 
    gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler):
    global gen_thread 
    gen_thread = threading.get_ident()
    socketio.emit("train starting")
    train_lora(images, model_name, train_data, instance_prompt, output, num_train_epochs, learning_rate, text_encoder_lr, resolution, save_epochs, 
    gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler)
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
    output = os.path.join(dirname, f"../outputs/lora/{instance_prompt}")
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
    model_name = os.path.join(dirname, f"../models/diffusion/{model_name}")

    thread = threading.Thread(target=lora, args=(images, model_name, train_data, instance_prompt, output, num_train_epochs, learning_rate, text_encoder_lr, resolution, save_epochs, 
    gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler))
    thread.start()
    thread.join()
    gen_thread = None
    return "done"

def dreambooth(images, model_name, train_data, instance_prompt, output, num_train_epochs, learning_rate, text_encoder_lr, resolution, save_epochs, 
    gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler):
    global gen_thread 
    gen_thread = threading.get_ident()
    socketio.emit("train starting")
    train_dreambooth(images, model_name, train_data, instance_prompt, output, num_train_epochs, learning_rate, text_encoder_lr, resolution, save_epochs, 
    gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler)
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
    save_epochs = data["save_epochs"] 
    gradient_accumulation_steps = data["gradient_accumulation_steps"]
    validation_prompt = data["validation_prompt"]
    validation_epochs = data["validation_epochs"]
    lr_scheduler = data["learning_function"]
    output = os.path.join(dirname, f"../outputs/dreambooth/{instance_prompt}")
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
    model_name = os.path.join(dirname, f"../models/diffusion/{model_name}")

    thread = threading.Thread(target=dreambooth, args=(images, model_name, train_data, instance_prompt, output, num_train_epochs, learning_rate, text_encoder_lr, resolution, save_epochs, 
    gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler))
    thread.start()
    thread.join()
    gen_thread = None
    return "done"

def checkpoint(images, model_name, train_data, instance_prompt, output, num_train_epochs, learning_rate, text_encoder_lr, resolution, save_epochs, 
    gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler):
    global gen_thread 
    gen_thread = threading.get_ident()
    socketio.emit("train starting")
    train_checkpoint(images, model_name, train_data, instance_prompt, output, num_train_epochs, learning_rate, text_encoder_lr, resolution, save_epochs, 
    gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler)
    socketio.emit("train complete")
    show_in_folder("", f"{output}/{instance_prompt}.ckpt")
    return "done"

@app.route("/train-checkpoint", methods=["POST"])
def start_checkpoint():
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
    output = os.path.join(dirname, f"../outputs/checkpoint/{instance_prompt}")
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
    model_name = os.path.join(dirname, f"../models/diffusion/{model_name}")

    thread = threading.Thread(target=checkpoint, args=(images, model_name, train_data, instance_prompt, output, num_train_epochs, learning_rate, text_encoder_lr, resolution, save_epochs, 
    gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler))
    thread.start()
    thread.join()
    gen_thread = None
    return "done"