import flask
from __main__ import app
from .functions import get_number_from_filename, is_image, is_unwanted, is_dir, is_file
from .invisiblewatermark import decode_watermark, encode_watermark
import os
import platform
import subprocess
from send2trash import send2trash
from PIL import Image, ExifTags, PngImagePlugin
import pathlib
import string
import piexif.helper
import piexif
import re

dirname = os.path.dirname(__file__)

@app.route("/diffusion-models")
def get_diffusion_models():
    files = os.listdir(os.path.join(dirname, "../models/diffusion"))
    return list(filter(lambda file: not is_unwanted(file) and not is_image(file), files))

@app.route("/vae-models")
def get_vae_models():
    files = os.listdir(os.path.join(dirname, "../models/vae"))
    return list(filter(lambda file: not is_unwanted(file) and not is_image(file), files))

@app.route("/clip-model")
def get_clip_model():
    return os.path.join(dirname, "../models/clip")

def crawl_model_folder(model_type: str, folder: str):
    files = os.listdir(os.path.join(dirname, f"../models/{model_type}", folder))
    model_map = []
    dirs = list(filter(lambda file: not is_unwanted(file) and is_dir(f"../models/{model_type}", file), files))
    for dir in dirs:
        stem = pathlib.Path(dir).stem
        dir_files = crawl_model_folder(model_type, os.path.join(folder, dir))
        model_map.append({"name": stem, "files": dir_files, "directory": True})
    models = list(filter(lambda file: not is_unwanted(file) and is_file(f"../models/{model_type}", file) and not is_image(file), files))
    images = list(filter(lambda file: not is_unwanted(file) and is_file(f"../models/{model_type}", file) and is_image(file), files))
    for i in range(len(models)):
        stem = pathlib.Path(models[i]).stem
        model = os.path.join(f"models/{model_type}", folder, models[i])
        image = ""
        for j in range(len(images)):
            image_stem = pathlib.Path(images[j]).stem
            if stem == image_stem:
                image = images[j]
                break
        if image:
            image = os.path.join(f"models/{model_type}", folder, image)
        model_map.append({"name": stem, "model": model, "image": image, "directory": False})
    return model_map

def get_model_files(model_type: str):
    files = os.listdir(os.path.join(dirname, f"../models/{model_type}"))
    model_map = []
    dirs = list(filter(lambda file: not is_unwanted(file) and is_dir(f"../models/{model_type}", file), files))
    for dir in dirs:
        stem = pathlib.Path(dir).stem
        dir_files = crawl_model_folder(model_type, dir)
        model_map.append({"name": stem, "files": dir_files, "directory": True})
    models = list(filter(lambda file: not is_unwanted(file) and is_file(f"../models/{model_type}", file) and not is_image(file), files))
    images = list(filter(lambda file: not is_unwanted(file) and is_file(f"../models/{model_type}", file) and is_image(file), files))
    for i in range(len(models)):
        stem = pathlib.Path(models[i]).stem
        model = os.path.join(f"models/{model_type}", models[i])
        image = ""
        for j in range(len(images)):
            image_stem = pathlib.Path(images[j]).stem
            if stem == image_stem:
                image = images[j]
                break
        if image:
            image = os.path.join(f"models/{model_type}", image)
        model_map.append({"name": stem, "model": model, "image": image, "directory": False})
    return model_map

@app.route("/textual-inversions")
def get_textual_inversions():
    return get_model_files("textual inversion")
    
@app.route("/hypernetworks")
def get_hypernetworks():
    return get_model_files("hypernetworks")

@app.route("/lora-models")
def get_lora_models():
    return get_model_files("lora")

def get_outputs(folder: str):
    files = os.listdir(os.path.join(dirname, f"../outputs/{folder}"))
    files = list(filter(lambda file: not is_unwanted(file), files))
    files = sorted(files, key=lambda x: get_number_from_filename(x), reverse=True)
    return list(map(lambda file: f"outputs/{folder}/{file}", files))

@app.route("/all-outputs")
def get_all_outputs():
    return get_outputs("text")

@app.route("/all-nsfw-outputs")
def get_all_nsfw_outputs():
    return get_outputs("text nsfw")

@app.route("/all-image-outputs")
def get_all_image_outputs():
    image_outputs = get_outputs("image")
    image_nsfw_outputs = get_outputs("image nsfw")
    all_outputs = image_outputs + image_nsfw_outputs
    return sorted(all_outputs, key=lambda x: os.path.getmtime(x), reverse=True)

@app.route("/show-in-folder", methods=["POST"])
def show_in_folder():
    data = flask.request.json
    path = data["path"]
    absolute = os.path.join(dirname, f"../{path}")
    if platform.system() == "Windows":
        subprocess.Popen(f'explorer /select, "{absolute}"')
    elif platform.system() == "Darwin":
        subprocess.call(["open", "-R", absolute])
    else:
        subprocess.Popen(["xdg-open", path])
    return "done"

@app.route("/delete-file", methods=["POST"])
def delete_file():
    data = flask.request.json
    path = data["path"]
    deletion = data["deletion"]
    if deletion == "trash":
        send2trash(path)
    else:
        os.remove(path)
    return "done"

def exif_data(image: str):
    image = Image.open(image)
    if not image._getexif():
        return ""
    exif = {
        ExifTags.TAGS[k]: v
        for k, v in image._getexif().items()
        if k in ExifTags.TAGS
    }
    if "UserComment" in exif:
        comment = exif["UserComment"]
        try:
            comment = comment.decode("utf-8")
        except:
            pass
        return comment.replace("UNICODE", "")
    return ""

@app.route("/get-exif", methods=["POST"])
def get_exif():
    return exif_data(flask.request.files["image"])

@app.route("/apply-watermark", methods=["POST"])
def apply_watermark():
    data = flask.request.json
    image = data["image"]
    image = os.path.join(dirname, f"../{image}")
    encode_watermark(image, image, "SDV2")
    return "done"

def get_prompt(image: str):
    ext = pathlib.Path(image).suffix
    prompt = ""
    if ext == ".png":
        img = Image.open(image)
        img.load()
        prompt = "Prompt: " + img.info["Prompt"]
    else:
        metadata = exif_data(image)
        metadata = "".join(filter(lambda c: c in string.printable, metadata))
        prompt = metadata.split("\n")[0]
    return prompt.strip().lower()

@app.route("/similar-images", methods=["POST"])
def similar_images():
    data = flask.request.json
    image = data["image"]
    image = os.path.join(dirname, f"../{image}")
    prompt = get_prompt(image)
    all_images = get_outputs("text") + get_outputs("text nsfw")
    images = []
    for img in all_images:
        prompt_check = get_prompt(img)
        if prompt == prompt_check:
            images.append(img)
    return images

@app.route("/save-watermark", methods=["POST"])
def save_watermark():
    file = flask.request.files["image"]
    path = flask.request.form.get("path")
    invisible_watermark = flask.request.form.get("invisible_watermark")
    image = Image.open(file)
    image.load()
    ext = pathlib.Path(path).suffix.replace(".", "")
    if path.startswith("/"):
        path = path[1:]
    if ext == "png":
        pnginfo = PngImagePlugin.PngInfo()
        for key, value in image.info.items():
            if key == "srgb":
                continue
            pnginfo.add_text(key, str(value))
        image.save(path, ext, optimize=True, pnginfo=pnginfo)
    else:
        if ext == "jpg":
            ext = "jpeg"
            image = image.convert("RGB")
        user_comment = image.info["UserComment"] if "UserComment" in image.info else ""
        if not user_comment:
            user_comment = exif_data(path)
        image.save(path, ext, optimize=True)
        exif = piexif.dump({
            "Exif": {piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(user_comment, encoding="unicode")}
        })
        piexif.insert(exif, path)
    if invisible_watermark: encode_watermark(path, path, "SDV2")
    return "done"