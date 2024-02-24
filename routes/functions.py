import os
import re
import pathlib
from PIL import Image
import math
import base64
import struct
import json
import torchvision
import itertools

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
    if ext == "txt":
        return True
    else:
        return False
    
def is_source_text(filename):
    if filename == ".DS_Store": return False
    if not ".source.txt" in filename: return False
    ext = pathlib.Path(filename).suffix.lower().replace(".", "")
    if ext == "txt":
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

def get_images(folder, resolution=512, center_crop=True, repeats=1):
    files = os.listdir(folder.strip())
    image_files = list(filter(lambda file: is_image(file), files))
    image_files = sorted(image_files, key=lambda x: get_number_from_filename(x), reverse=False)
    image_files = list(map(lambda file: Image.open(os.path.join(folder, file)).convert("RGB"), image_files))
    images = []
    for image in image_files:
        images.extend(itertools.repeat(image, repeats))

    tensors = []
    image_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(resolution, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                torchvision.transforms.CenterCrop(resolution) if center_crop else torchvision.transforms.RandomCrop(resolution),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5], [0.5]),
            ])
    for image in images:
        tensor = image_transforms(image).unsqueeze(0)
        tensors.append(tensor)
    return tensors

def get_captions(folder, default="", repeats=1):
    files = os.listdir(folder.strip())
    text_files = list(filter(lambda file: is_text(file), files))
    text_files = sorted(text_files, key=lambda x: get_number_from_filename(x), reverse=False)
    if len(text_files) == 0:
        image_files = list(filter(lambda file: is_image(file), files))
        for i in image_files:
            text_files.append(default)
    texts = []
    for text in text_files:
        texts.extend(itertools.repeat(text, repeats))

    captions = []
    for text in texts:
        f = open(os.path.join(folder, text))
        captions.append(f.read())
        f.close()
    return captions

def get_sources(folder):
    files = os.listdir(folder.strip())
    source_files = list(filter(lambda file: is_source_text(file), files))
    source_files = sorted(source_files, key=lambda x: get_number_from_filename(x), reverse=False)

    sources = []
    for source in source_files:
        f = open(os.path.join(folder, source))
        sources.append(f.read())
        f.close()
    return sources

def resize_box(x, x1, y, y1, img_w, img_h, size=512):
    if size == 0:
        return {"x": x, "x1": x1, "y": y, "y1": y1}
    center_x = math.floor((x + x1) / 2)
    center_y = math.floor((y + y1) / 2)
    new_x = center_x - math.floor(size / 2)
    corr_x1 = 0
    if new_x < 0:
        while new_x < 0:
            new_x += 1
            corr_x1 += 1
    new_x1 = center_x + math.floor(size / 2) + corr_x1
    if new_x1 > img_w:
        while new_x1 > img_w:
            new_x1 -= 1
            new_x -= 1
        if new_x < 0:
           new_x = 0

    new_y = center_y - math.floor(size / 2)
    corr_y1 = 0
    if new_y < 0:
        while new_y < 0:
            new_y += 1
            corr_y1 += 1
    new_y1 = center_y + math.floor(size / 2) + corr_y1
    if new_y1 > img_h:
        while new_y1 > img_h:
            new_y1 -= 1
            new_y -= 1
        if new_y < 0:
           new_y = 0
    return new_x, new_x1, new_y, new_y1