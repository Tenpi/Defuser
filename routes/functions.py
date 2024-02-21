import os
import re
import pathlib
from PIL import Image
import math
import base64
import struct
import json

dirname = os.path.dirname(__file__)

def get_number_from_filename(filename):
    num = re.search(r"\d+", filename)
    if num:
        return int(num.group(0)) if num else -1
    else:
        return -1

def next_index(dirname):
    files = os.listdir(dirname)
    files = list(filter(lambda file: file != ".DS_Store", files))
    if not len(files):
        return 1
    files = sorted(files, key=lambda x: get_number_from_filename(x), reverse=True)
    highest = get_number_from_filename(files[0])
    return highest + 1

def is_text(filename):
    if filename == ".DS_Store": return False
    if ".source.txt" in filename: return False
    ext = pathlib.Path(filename).suffix.lower().replace(".", "")
    text_exts = ["txt", "md"]
    if ext in text_exts:
        return True
    else:
        return False

def is_image(filename):
    if filename == ".DS_Store": return False
    ext = pathlib.Path(filename).suffix.lower().replace(".", "")
    image_exts = ["jpg", "jpeg", "png", "webp", "gif", "apng", "avif", "bmp"]
    if ext in image_exts:
        return True
    else:
        return False
    
def is_unwanted(filename):
    if filename == ".DS_Store": return True
    ext = pathlib.Path(filename).suffix.lower().replace(".", "")
    image_exts = ["txt"]
    if ext in image_exts:
        return True
    else:
        return False

def is_dir(base, filename):
    path = os.path.normpath(os.path.join(dirname, base, filename))
    return os.path.isdir(path)

def is_file(base, filename):
    path = os.path.normpath(os.path.join(dirname, base, filename))
    return not os.path.isdir(path)

def is_nsfw(prompt):
    bad_words = ["bnNmdw==", "Y3Vt", "c2V4", "cGVuaXM=", "dmFnaW5h", "cHVzc3k=", 
    "b3JhbA==", "ZmVsbGF0aW8=", "Ymxvd2pvYg==", "c2VtZW4=", "YW5hbA==", "dmFnaW5hbA==", 
    "ZWphY3VsYXRpb24=", "Y29jaw==", "Z2FuZyBiYW5n", "b3JnYXNt", "cGVuZXRyYXRpb24=", "aW50ZXJjb3Vyc2U=", 
    "cmFwZQ==", "bWFzdHVyYmF0aW9u", "Y29wdWxhdGlvbg==", "Zm9ybmljYXRpb24="]
    for raw in bad_words:
        bad = base64.b64decode(raw).decode("utf-8")
        if bad in str(prompt).lower():
            return True
    return False

def get_normalized_dimensions(img):
    greaterValue = img.width if img.width > img.height else img.height
    heightBigger = True if img.height > img.width else False
    ratio = greaterValue / 512
    width = math.floor(img.width / ratio)
    height = math.floor(img.height / ratio)
    if heightBigger:
        while width % 8 != 0:
            width -= 1
    else:
        while height % 8 != 0:
            height -= 1
    return {"width": width, "height": height}

def get_safetensors_metadata(filename):
    with open(filename, "rb") as f:
        safe_bytes = f.read()
    metadata_size = struct.unpack("<Q", safe_bytes[0:8])[0]
    metadata_as_bytes = safe_bytes[8:8+metadata_size]
    metadata_as_dict = json.loads(metadata_as_bytes.decode(errors="ignore"))
    return metadata_as_dict.get("__metadata__", {})