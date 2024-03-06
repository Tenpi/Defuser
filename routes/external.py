import flask      
from __main__ import app, socketio
from .functions import is_nsfw, next_index, get_seed, upscale, append_info, get_normalized_dimensions
from .invisiblewatermark import encode_watermark
from PIL import Image
import pathlib
import requests
import random
import os
import base64
import zipfile
from io import BytesIO

dirname = os.path.dirname(__file__)
infinite = False
upscaling = True

def update_ext_infinite(value):
    global infinite
    infinite = value

def update_ext_upscaling(value):
    global upscaling
    upscaling = value

def pil_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def novelai_sampler(sampler):
    if sampler == "euler a":
        return "k_euler_ancestral"
    elif sampler == "euler":
        return "k_euler"
    elif sampler == "dpm++":
        return "k_dpmpp_2m"
    elif sampler == "ddim":
        return "ddim_v3"

def request_novelai(action="generate", prompt="", negative_prompt="", model="nai-diffusion", width=640, height=640, cfg=5, sampler="euler a", steps=20, style_fidelity=0.5,
                    denoise=0.7, image=None, mask=None, control_scale=1, control_processor="hed", controlnet_condition=None, reference_image=None, novelai_token=""):
    seed = get_seed(-1)
    payload = {
        "input": prompt,
        "model": model,
        "action": action,
        "parameters": {
            "params_version": 1,
            "width": width,
            "height": height,
            "scale": cfg,
            "sampler": novelai_sampler(sampler),
            "noise_schedule": "exponential",
            "steps": steps,
            "n_samples": 1,
            "strength": denoise,
            "noise": 0,
            "ucPreset": 0,
            "qualityToggle": True,
            "sm": True,
            "sm_dyn": True,
            "dynamic_thresholding": True,
            "controlnet_strength": control_scale,
            "legacy": False,
            "legacy_v3_extend": False,
            "add_original_image": True,
            "uncond_scale":	1,
            "cfg_rescale": 0,
            "reference_information_extracted":1,
            "reference_strength": style_fidelity,
            "seed": seed,
            "extra_noise_seed": seed,
            "negative_prompt": negative_prompt
        }
    }

    if image is not None:
        payload["parameters"]["image"] = pil_to_base64(image)
    if mask is not None:
        payload["parameters"]["mask"] = pil_to_base64(mask)
    if controlnet_condition is not None:
        payload["parameters"]["controlnet_condition"] = pil_to_base64(controlnet_condition)
        payload["parameters"]["controlnet_model"] = control_processor
    if reference_image is not None:
        payload["parameters"]["reference_image"] = pil_to_base64(reference_image)

    headers = {
        "authorization": f"Bearer {novelai_token}",
        "content-type": "application/json",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36 OPR/93.0.0.0",
        "referer": "https://novelai.net/",
        "origin": "https://novelai.net",
        "host": "image.novelai.net"
    }
    bytes_zip = requests.post("https://api.novelai.net/ai/generate-image", json=payload, headers=headers).content
    zip = zipfile.ZipFile(BytesIO(bytes_zip))
    image_bytes = zip.read(zip.infolist()[0])
    zip.close()
    return Image.open(BytesIO(image_bytes)).convert("RGB")

def generate_novelai(data, request_files):
    global infinite
    global upscaling

    mode = "text"
    action = "generate"
    seed = get_seed(data["seed"]) if "seed" in data else get_seed(-1)
    amount = int(data["amount"]) if "amount" in data else 1
    steps = int(data["steps"]) if "steps" in data else 20
    cfg = int(data["cfg"]) if "cfg" in data else 7
    width = int(data["width"]) if "width" in data else 640
    height = int(data["height"]) if "height" in data else 640
    denoise = float(data["denoise"]) if "denoise" in data else 1
    prompt = data["prompt"] if "prompt" in data else ""
    sampler = data["sampler"] if "sampler" in data else "euler a"
    format = data["format"] if "format" in data else "png"
    model_name = data["model_name"] if "model_name" in data else "nai-diffusion"
    vae_name = data["vae_name"] if "vae_name" in data else "None"
    control_processor = data["control_processor"] if "control_processor" in data else "hed"
    control_reference_image = data["control_reference_image"] if "control_reference_image" in data else False
    control_scale = float(data["control_scale"]) if "control_scale" in data else 1.0
    style_fidelity = float(data["style_fidelity"]) if "style_fidelity" in data else 0.5
    upscaler = data["upscaler"] if "upscaler" in data else ""
    watermark = data["watermark"] if "watermark" in data else False
    invisible_watermark = data["invisible_watermark"] if "invisible_watermark" in data else True
    nsfw_enabled = data["nsfw_tab"] if "nsfw_tab" in data else False
    novelai_token = data["novelai_token"] if "novelai_token" in data else ""

    if "best quality" not in prompt or "amazing quality" not in prompt or "very aesthetic" not in prompt or "absurdres" not in prompt:
        prompt += ", best quality, amazing quality, very aesthetic, absurdres"
    negative_prompt = "lowres, {bad}, error, fewer, extra, missing, worst quality, jpeg artifacts, bad quality, watermark, unfinished, displeasing, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"
    
    input_image = None
    input_mask = None
    control_image = None
    reference_image = None
    if "image" in request_files:
        mode = "image"
        input_image = Image.open(request_files["image"]).convert("RGB")
        action = "img2img"
        if "mask" in request_files:
            input_mask =  Image.open(request_files["mask"]).convert("RGB")
            action = "infill"
            if "control_image" in request_files:
                control_image =  Image.open(request_files["control_image"]).convert("RGB")
        else:
            if "control_image" in request_files:
                control_image =  Image.open(request_files["control_image"]).convert("RGB")
                action = "generate"
                input_image = None
                if control_processor == "reference":
                    reference_image = Image.open(request_files["image"]).convert("RGB")
                    
    if control_processor == "depth":
        control_processor = "depth"
    elif control_processor == "scribble":
        control_processor = "scribble"
    else:
        control_processor = "hed"

    if action == "infill":
        model_name += "-inpainting"
    
    socketio.emit("image starting")

    if input_image:
        normalized = get_normalized_dimensions(input_image, dim=768)
        input_image = input_image.resize((normalized["width"], normalized["height"]), resample=Image.BICUBIC)
    if input_mask:
        normalized = get_normalized_dimensions(input_mask, dim=768)
        input_mask = input_mask.resize((normalized["width"], normalized["height"]), resample=Image.BICUBIC)
    if control_image:
        normalized = get_normalized_dimensions(control_image, dim=768)
        control_image = control_image.resize((normalized["width"], normalized["height"]), resample=Image.BICUBIC)
    if reference_image:
        normalized = get_normalized_dimensions(reference_image, dim=768)
        reference_image = reference_image.resize((normalized["width"], normalized["height"]), resample=Image.BICUBIC)

    images = []
    for n in range(amount):
        image = None
        if mode == "text":
            folder = "text"
            if is_nsfw(prompt):
                folder = "text nsfw"
                if not nsfw_enabled:
                    socketio.emit("image complete", {"image": "", "needs_watermark": False})
                    return images
            image = request_novelai(action=action, prompt=prompt, negative_prompt=negative_prompt, 
                                    model=model_name, width=width, height=height, cfg=cfg, style_fidelity=style_fidelity,
                                    sampler=sampler, steps=steps, denoise=denoise, image=input_image,
                                    mask=input_mask, control_scale=control_scale, control_processor=control_processor,
                                    controlnet_condition=control_image, reference_image=reference_image, novelai_token=novelai_token)
            dir_path = os.path.join(dirname, f"../outputs/novel ai/{folder}")
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.{format}")
            extra_info = {}
            for key, value in image.info.items():
                extra_info[key] = value
            image.save(out_path)
            if upscaling:
                socketio.emit("image upscaling")
                upscale(out_path, upscaler)
            compressed = Image.open(out_path)
            compressed.save(out_path, quality=90, optimize=True)
            if invisible_watermark: encode_watermark(out_path, out_path, "SDV2")
            info = {"Prompt": prompt, "Negative Prompt": negative_prompt, "Size": f"{width}x{height}", "Model": model_name, 
                     "VAE": model_name, "Steps": steps, "CFG": cfg, "Sampler": sampler, "Clip Skip": 2, "Seed": seed}
            for key, value in extra_info.items():
                info[key] = value
            append_info(out_path, info)
            socketio.emit("image complete", {"image": f"/outputs/novel ai/{folder}/{os.path.basename(out_path)}", "needs_watermark": watermark})
            images.append(out_path)
            seed += 1
        else:
            folder = "image"
            if is_nsfw(prompt):
                folder = "image nsfw"
                if not nsfw_enabled:
                    socketio.emit("image complete", {"image": "", "needs_watermark": False})
                    return images
            image = request_novelai(action=action, prompt=prompt, negative_prompt=negative_prompt, 
                                    model=model_name, width=width, height=height, cfg=cfg, style_fidelity=style_fidelity,
                                    sampler=sampler, steps=steps, denoise=denoise, image=input_image,
                                    mask=input_mask, control_scale=control_scale, control_processor=control_processor,
                                    controlnet_condition=control_image, reference_image=reference_image, novelai_token=novelai_token)
            dir_path = os.path.join(dirname, f"../outputs/novel ai/{folder}")
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.{format}")
            extra_info = {}
            for key, value in image.info.items():
                extra_info[key] = value
            image.save(out_path)
            if upscaling:
                socketio.emit("image upscaling")
                upscale(out_path, upscaler)
            compressed = Image.open(out_path)
            compressed.save(out_path, quality=90, optimize=True)
            if invisible_watermark: encode_watermark(out_path, out_path, "SDV2")
            info = {"Prompt": prompt, "Negative Prompt": negative_prompt, "Size": f"{width}x{height}", "Model": model_name, 
                     "VAE": model_name, "Steps": steps, "CFG": cfg, "Sampler": sampler, "Clip Skip": 2, "Seed": seed}
            for key, value in extra_info.items():
                info[key] = value
            append_info(out_path, info)
            socketio.emit("image complete", {"image": f"/outputs/novel ai/{folder}/{os.path.basename(out_path)}", "needs_watermark": watermark})
            images.append(out_path)
            seed += 1
    if infinite:
        socketio.emit("repeat generation")
    return images
