import flask
from __main__ import app, socketio
from .classifier import train_classifier
from .info import open_folder
from .functions import clean_image
from transformers import BeitFeatureExtractor, BeitForImageClassification
from PIL import Image
import os
import threading
import pathlib
import inspect
import ctypes
from io import BytesIO
import base64

gen_thread = None
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

@app.route("/interrupt-classify", methods=["POST"])
def interrupt_classify():
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
    output_dir = os.path.join(dirname, f"../outputs/classifier/{pathlib.Path(train_dir).stem}")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    thread = threading.Thread(target=image_classifier, args=(train_dir, output_dir, num_train_epochs, learning_rate, 
    gradient_accumulation_steps, lr_scheduler_type, save_steps, resolution))
    thread.start()
    thread.join()
    gen_thread = None
    return "done"

@app.route("/ai-detector", methods=["POST"])
def ai_detector():
    data = flask.request.json
    image = data["image"]
    img = Image.open(BytesIO(base64.b64decode(image + "=="))).convert("RGB")
    img = clean_image(img)
    feature_extractor = BeitFeatureExtractor.from_pretrained(os.path.join(dirname, "../models/detector"))
    model = BeitForImageClassification.from_pretrained(os.path.join(dirname, "../models/detector"))
    inputs = feature_extractor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probs = logits.softmax(-1)
    predicted_class_idx = probs.argmax().item()
    label = model.config.id2label[predicted_class_idx]
    probability = probs[0][predicted_class_idx].item()
    return {"label": label, "probability": round(probability * 100, 2)}