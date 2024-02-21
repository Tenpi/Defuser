from urllib import request
import flask      
from __main__ import app, socketio
import random
import os
import torch
from torchvision.transforms.functional import pil_to_tensor
from .functions import next_index, is_nsfw, get_normalized_dimensions
from .invisiblewatermark import encode_watermark
from .info import get_diffusion_models, get_vae_models, get_clip_model
import numpy as np
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionControlNetImg2ImgPipeline, \
StableDiffusionControlNetInpaintPipeline, StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline, StableDiffusionXLControlNetImg2ImgPipeline, StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, UNet2DConditionModel, \
EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, DDPMScheduler, DDIMScheduler, UniPCMultistepScheduler, DEISMultistepScheduler, DPMSolverMultistepScheduler, \
HeunDiscreteScheduler, AutoencoderKL, MotionAdapter, AnimateDiffPipeline
from diffusers.utils import export_to_gif
from .stable_diffusion_controlnet_reference import StableDiffusionControlNetReferencePipeline
from .stable_diffusion_xl_reference import StableDiffusionXLReferencePipeline
from .hypernet import Hypernetwork, add_hypernet, clear_hypernets
from transformers import CLIPTextModel
from compel import Compel, ReturnedEmbeddingsType, DiffusersTextualInversionManager
from PIL import Image, PngImagePlugin
import subprocess
import pathlib
import piexif
import piexif.helper
import json
from io import BytesIO
from itertools import chain
import inspect
import ctypes
import threading
import platform
import gc

dirname = os.path.dirname(__file__)
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

gen_thread = None
generator = None
generator_name = None
generator_mode = "text"
generator_clip_skip = 2
vae_name = None
infinite = False
upscaling = True
safety_checker = None
dtype = torch.float32
controlnet = None
control_processor = "none"
motion_adapter = None

def append_info(image: str, info: dict):
    ext = pathlib.Path(image).suffix
    img = Image.open(image)
    if ext == ".png":
        pnginfo = PngImagePlugin.PngInfo()
        for key, value in (info).items():
            pnginfo.add_text(key, str(value))
        img.save(image, pnginfo=pnginfo)
    else:
        info_list = list()
        for key, value in (info).items():
            info_list.append(f"{key}: {str(value)}")
        exif = piexif.dump({
            "Exif": {piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump("\n".join(info_list), encoding="unicode")}
        })
        piexif.insert(exif, image)

def upscale(image: str, upscaler: str, video: bool = False):
    if upscaler == "waifu2x":
        program = os.path.join(dirname, "../models/upscaler/waifu2x-ncnn-vulkan")
        if platform.system() == "Windows":
            program = os.path.join(dirname, "../models/upscaler/waifu2x-ncnn-vulkan.exe")
        if platform.system() == "Darwin":
            program = os.path.join(dirname, "../models/upscaler/waifu2x-ncnn-vulkan.app")
        format = pathlib.Path(image).suffix.replace(".", "")
        subprocess.call([program, "-i", image, "-o", image, "-s", "4", "-f", format])
    elif upscaler == "real-esrgan":
        program = os.path.join(dirname, "../models/upscaler/realesrgan-ncnn-vulkan")
        if platform.system() == "Windows":
            program = os.path.join(dirname, "../models/upscaler/realesrgan-ncnn-vulkan.exe")
        if platform.system() == "Darwin":
            program = os.path.join(dirname, "../models/upscaler/realesrgan-ncnn-vulkan.app")
        models = os.path.join(dirname, "../models/upscaler/models")
        network = "realesr-animevideov3" if video else "realesrgan-x4plus-anime"
        format = pathlib.Path(image).suffix.replace(".", "")
        subprocess.call([program, "-i", image, "-o", image, "-s", "4", "-f", format, "-m", models, "-n", network])
    elif upscaler == "real-cugan":
        program = os.path.join(dirname, "../models/upscaler/realcugan-ncnn-vulkan")
        if platform.system() == "Windows":
            program = os.path.join(dirname, "../models/upscaler/realcugan-ncnn-vulkan.exe")
        if platform.system() == "Darwin":
            program = os.path.join(dirname, "../models/upscaler/realcugan-ncnn-vulkan.app")
        format = pathlib.Path(image).suffix.replace(".", "")
        subprocess.call([program, "-i", image, "-o", image, "-s", "4", "-f", format])

def get_seed(seed):
    if not seed or seed == -1:
        return int(random.randrange(4294967294))
    return int(seed)

def load_hypernet(path, multiplier=None):
        hyper_model = torch.load(path, map_location=device)
        hypernetwork = Hypernetwork()
        hypernetwork.load_state_dict(hyper_model)
        hypernetwork.set_multiplier(multiplier if multiplier else 1.0)
        hypernetwork.to(device=device)
        hypernetwork.eval()
        return hypernetwork

def get_motion_adapter():
    global motion_adapter
    motion_model = os.path.join(dirname, "../models/animatediff")
    motion_adapter = MotionAdapter.from_pretrained(motion_model, local_files_only=True)
    return motion_adapter

def get_controlnet(processor: str = "none"):
    global controlnet
    global control_processor
    if not controlnet or control_processor != processor:
        if processor == "canny":
            control_model = os.path.join(dirname, "../models/controlnet/canny")
            controlnet = ControlNetModel.from_pretrained(control_model, local_files_only=True)
        elif processor == "depth":
            control_model = os.path.join(dirname, "../models/controlnet/depth")
            controlnet = ControlNetModel.from_pretrained(control_model, local_files_only=True)
        if processor == "lineart":
            control_model = os.path.join(dirname, "../models/controlnet/lineart")
            controlnet = ControlNetModel.from_pretrained(control_model, local_files_only=True)
        if processor == "lineart anime":
            control_model = os.path.join(dirname, "../models/controlnet/lineart anime")
            controlnet = ControlNetModel.from_pretrained(control_model, local_files_only=True)
        if processor == "scribble":
            control_model = os.path.join(dirname, "../models/controlnet/scribble")
            controlnet = ControlNetModel.from_pretrained(control_model, local_files_only=True)
        if processor == "softedge":
            control_model = os.path.join(dirname, "../models/controlnet/softedge")
            controlnet = ControlNetModel.from_pretrained(control_model, local_files_only=True)
        if processor == "reference":
            control_model = os.path.join(dirname, "../models/controlnet/canny")
            controlnet = ControlNetModel.from_pretrained(control_model, local_files_only=True)
        control_processor = processor
    return controlnet

def get_generator(model_name: str = "", vae: str = "", mode: str = "text", clip_skip: int = 2, cpu: bool = False, control_processor: str = "none"):
    global generator
    global generator_name
    global generator_mode
    global generator_clip_skip
    global vae_name
    global safety_checker
    global dtype
    global device
    processor = "cpu" if cpu else device
    update_model = False
    if not model_name:
        model_name = get_diffusion_models()[0]
    if not generator or generator_name != model_name or generator_mode != mode:
        model = os.path.join(dirname, "../models/diffusion", model_name)
        if generator_name != model_name:
            update_model = True

        if mode == "text":
            if "xl" in model.lower():
                if update_model:
                    if os.path.isdir(model):
                        generator = StableDiffusionXLPipeline.from_pretrained(model, local_files_only=True)  
                    else:
                        generator = StableDiffusionXLPipeline.from_single_file(model)
                else:
                    generator = StableDiffusionXLPipeline(vae=generator.vae, text_encoder=generator.text_encoder, 
                                                        tokenizer=generator.tokenizer, unet=generator.unet, 
                                                        scheduler=generator.scheduler, safety_checker=generator.safety_checker, 
                                                        text_encoder_2=generator.text_encoder_2, tokenizer_2=generator.tokenizer_2)
            else:
                if update_model:
                    if os.path.isdir(model):
                        generator = StableDiffusionPipeline.from_pretrained(model, local_files_only=True) 
                    else:
                        generator = StableDiffusionPipeline.from_single_file(model)
                else:
                    generator = StableDiffusionPipeline(vae=generator.vae, text_encoder=generator.text_encoder, 
                                                        tokenizer=generator.tokenizer, unet=generator.unet, 
                                                        scheduler=generator.scheduler, safety_checker=generator.safety_checker, 
                                                        feature_extractor=generator.feature_extractor)
        elif mode == "image":
            if "xl" in model.lower():
                if update_model:
                    if os.path.isdir(model):
                        generator = StableDiffusionXLImg2ImgPipeline.from_pretrained(model, local_files_only=True)
                    else:
                        generator = StableDiffusionXLImg2ImgPipeline.from_single_file(model)
                else:
                    generator = StableDiffusionXLImg2ImgPipeline(vae=generator.vae, text_encoder=generator.text_encoder, 
                                                        tokenizer=generator.tokenizer, unet=generator.unet, 
                                                        scheduler=generator.scheduler, safety_checker=generator.safety_checker, 
                                                        text_encoder_2=generator.text_encoder_2, tokenizer_2=generator.tokenizer_2)
            else:
                if update_model:
                    if os.path.isdir(model):
                        generator = StableDiffusionImg2ImgPipeline.from_pretrained(model, local_files_only=True)
                    else:
                        generator = StableDiffusionImg2ImgPipeline.from_single_file(model)
                else:
                    generator = StableDiffusionImg2ImgPipeline(vae=generator.vae, text_encoder=generator.text_encoder, 
                                                        tokenizer=generator.tokenizer, unet=generator.unet, 
                                                        scheduler=generator.scheduler, safety_checker=generator.safety_checker, 
                                                        feature_extractor=generator.feature_extractor)
        elif mode == "inpaint":
            if "xl" in model.lower():
                if update_model:
                    if os.path.isdir(model):
                        generator = StableDiffusionXLInpaintPipeline.from_pretrained(model, num_in_channels=4, local_files_only=True)
                    else:
                        generator = StableDiffusionXLInpaintPipeline.from_single_file(model, num_in_channels=4)
                else:
                    generator = StableDiffusionXLInpaintPipeline(vae=generator.vae, text_encoder=generator.text_encoder, 
                                                        tokenizer=generator.tokenizer, unet=generator.unet, 
                                                        scheduler=generator.scheduler, safety_checker=generator.safety_checker, 
                                                        text_encoder_2=generator.text_encoder_2, tokenizer_2=generator.tokenizer_2)
            else:
                if update_model:
                    if os.path.isdir(model):
                        generator = StableDiffusionInpaintPipeline.from_pretrained(model, num_in_channels=4, local_files_only=True)
                    else:
                        generator = StableDiffusionInpaintPipeline.from_single_file(model, num_in_channels=4)
                else:
                    generator = StableDiffusionInpaintPipeline(vae=generator.vae, text_encoder=generator.text_encoder, 
                                                        tokenizer=generator.tokenizer, unet=generator.unet, 
                                                        scheduler=generator.scheduler, safety_checker=generator.safety_checker, 
                                                        feature_extractor=generator.feature_extractor)
        elif mode == "controlnet":
            controlnet = get_controlnet(control_processor).to(device=processor, dtype=dtype)
            if "xl" in model.lower():
                if update_model:
                    if os.path.isdir(model):
                        generator = StableDiffusionXLControlNetPipeline.from_pretrained(model, controlnet=controlnet, local_files_only=True)
                    else:
                        generator = StableDiffusionXLControlNetPipeline.from_single_file(model, controlnet=controlnet)
                else:
                    generator = StableDiffusionXLControlNetPipeline(controlnet=controlnet, vae=generator.vae, text_encoder=generator.text_encoder, 
                                                        tokenizer=generator.tokenizer, unet=generator.unet, 
                                                        scheduler=generator.scheduler, safety_checker=generator.safety_checker, 
                                                        text_encoder_2=generator.text_encoder_2, tokenizer_2=generator.tokenizer_2)
            else:
                if update_model:
                    if os.path.isdir(model):
                        generator = StableDiffusionControlNetPipeline.from_pretrained(model, controlnet=controlnet, local_files_only=True)
                    else:
                        generator = StableDiffusionControlNetPipeline.from_single_file(model, controlnet=controlnet)
                else:
                    generator = StableDiffusionControlNetPipeline(controlnet=controlnet, vae=generator.vae, text_encoder=generator.text_encoder, 
                                                        tokenizer=generator.tokenizer, unet=generator.unet, 
                                                        scheduler=generator.scheduler, safety_checker=generator.safety_checker, 
                                                        feature_extractor=generator.feature_extractor)
        elif mode == "controlnet image":
            controlnet = get_controlnet(control_processor).to(device=processor, dtype=dtype)
            if "xl" in model.lower():
                if update_model:
                    if os.path.isdir(model):
                        generator = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(model, controlnet=controlnet, local_files_only=True)
                    else:
                        generator = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(model, controlnet=controlnet)
                else:
                    generator = StableDiffusionXLControlNetImg2ImgPipeline(controlnet=controlnet, vae=generator.vae, text_encoder=generator.text_encoder, 
                                                        tokenizer=generator.tokenizer, unet=generator.unet, 
                                                        scheduler=generator.scheduler, safety_checker=generator.safety_checker, 
                                                        text_encoder_2=generator.text_encoder_2, tokenizer_2=generator.tokenizer_2)
            else:
                if update_model:
                    if os.path.isdir(model):
                        generator = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(model, controlnet=controlnet, local_files_only=True)
                    else:
                        generator = StableDiffusionControlNetImg2ImgPipeline.from_single_file(model, controlnet=controlnet)
                else:
                    generator = StableDiffusionControlNetImg2ImgPipeline(controlnet=controlnet, vae=generator.vae, text_encoder=generator.text_encoder, 
                                                        tokenizer=generator.tokenizer, unet=generator.unet, 
                                                        scheduler=generator.scheduler, safety_checker=generator.safety_checker, 
                                                        feature_extractor=generator.feature_extractor)
        elif mode == "controlnet inpaint":
            controlnet = get_controlnet(control_processor).to(device=processor, dtype=dtype)
            if "xl" in model.lower():
                if update_model:
                    if os.path.isdir(model):
                        generator = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(model, controlnet=controlnet, num_in_channels=4, local_files_only=True)
                    else:
                        generator = StableDiffusionXLControlNetInpaintPipeline.from_single_file(model, num_in_channels=4, controlnet=controlnet)
                else:
                    generator = StableDiffusionXLControlNetInpaintPipeline(controlnet=controlnet, vae=generator.vae, text_encoder=generator.text_encoder, 
                                                        tokenizer=generator.tokenizer, unet=generator.unet, 
                                                        scheduler=generator.scheduler, safety_checker=generator.safety_checker, 
                                                        text_encoder_2=generator.text_encoder_2, tokenizer_2=generator.tokenizer_2)
            else:
                if update_model:
                    if os.path.isdir(model):
                        generator = StableDiffusionControlNetInpaintPipeline.from_pretrained(model, num_in_channels=4, controlnet=controlnet, local_files_only=True)
                    else:
                        generator = StableDiffusionControlNetInpaintPipeline.from_single_file(model, num_in_channels=4, controlnet=controlnet)
                else:
                    generator = StableDiffusionControlNetInpaintPipeline(controlnet=controlnet, vae=generator.vae, text_encoder=generator.text_encoder, 
                                                        tokenizer=generator.tokenizer, unet=generator.unet, 
                                                        scheduler=generator.scheduler, safety_checker=generator.safety_checker, 
                                                        feature_extractor=generator.feature_extractor)
        elif mode == "controlnet reference":
            controlnet = get_controlnet(control_processor).to(device=processor, dtype=dtype)
            if "xl" in model.lower():
                if update_model:
                    if os.path.isdir(model):
                        generator = StableDiffusionXLReferencePipeline.from_pretrained(model, local_files_only=True)
                    else:
                        generator = StableDiffusionXLReferencePipeline.from_single_file(model)
                else:
                    generator = StableDiffusionXLReferencePipeline(vae=generator.vae, text_encoder=generator.text_encoder, 
                                                        tokenizer=generator.tokenizer, unet=generator.unet, 
                                                        scheduler=generator.scheduler, safety_checker=generator.safety_checker, 
                                                        text_encoder_2=generator.text_encoder_2, tokenizer_2=generator.tokenizer_2)
            else:
                if update_model:
                    if os.path.isdir(model):
                        generator = StableDiffusionControlNetReferencePipeline.from_pretrained(model, controlnet=controlnet, local_files_only=True)
                    else:
                        generator = StableDiffusionControlNetReferencePipeline.from_single_file(model, controlnet=controlnet)
                else:
                    generator = StableDiffusionControlNetReferencePipeline(controlnet=controlnet, vae=generator.vae, text_encoder=generator.text_encoder, 
                                                        tokenizer=generator.tokenizer, unet=generator.unet, 
                                                        scheduler=generator.scheduler, safety_checker=generator.safety_checker, 
                                                        feature_extractor=generator.feature_extractor)
        elif mode == "animatediff":
            motion_adapter = get_motion_adapter().to(device=processor, dtype=dtype)
            if update_model:
                if os.path.isdir(model):
                    generator = StableDiffusionPipeline.from_pretrained(model, local_files_only=True)
                else:
                    generator = StableDiffusionPipeline.from_single_file(model)
                generator = AnimateDiffPipeline(motion_adapter=motion_adapter, vae=generator.vae, 
                                                text_encoder=generator.text_encoder, tokenizer=generator.tokenizer, 
                                                unet=generator.unet, scheduler=generator.scheduler)
            else:
                generator = AnimateDiffPipeline(motion_adapter=motion_adapter, vae=generator.vae, 
                                                text_encoder=generator.text_encoder, tokenizer=generator.tokenizer, 
                                                unet=generator.unet, scheduler=generator.scheduler)
        generator_name = model_name
        generator_mode = mode
    if not vae_name or vae_name != vae or update_model:
        if not vae:
            vae = get_vae_models()[0]
        vae_model = os.path.join(dirname, "../models/vae", vae)
        if os.path.isdir(vae_model):
            generator.vae = AutoencoderKL.from_pretrained(vae_model, local_files_only=True)
        else:
            generator.vae = AutoencoderKL.from_single_file(vae_model)
        generator.vae = generator.vae.to(device=processor, dtype=dtype)
        generator.vae.enable_slicing()
        generator.vae.enable_tiling()
        vae_name = vae
    generator = generator.to(device=processor, dtype=dtype)
    generator.enable_attention_slicing()
    generator.safety_checker = None
    return generator

@socketio.on("load diffusion model")
def load_diffusion_model(model_name, vae_name, clip_skip, processing):
    global generator
    #generator = get_generator(model_name, vae_name, "text", int(clip_skip), processing == "cpu")
    return "done"

@app.route("/update-infinite", methods=["POST"])
def update_infinite():
    global infinite
    data = flask.request.json
    infinite = data["infinite"]
    return "done"

@app.route("/update-upscaling", methods=["POST"])
def update_upscaling():
    global upscaling
    data = flask.request.json
    upscaling = data["upscaling"]
    return "done"

@app.route("/update-precision", methods=["POST"])
def update_precision():
    global dtype
    data = flask.request.json
    precision = data["precision"]
    if precision == "full":
        dtype = torch.float32
    elif precision == "half":
        dtype = torch.float16
    return "done"

def generate(request_data, request_files):
    global gen_thread 
    gen_thread = threading.get_ident()
    mode = "text"
    data = json.loads(request_data)
                
    seed = get_seed(data["seed"]) if "seed" in data else get_seed(-1)
    amount = int(data["amount"]) if "amount" in data else 1
    steps = int(data["steps"]) if "steps" in data else 20
    cfg = int(data["cfg"]) if "cfg" in data else 7
    clip_skip = int(data["clip_skip"]) if "clip_skip" in data else 2
    width = int(data["width"]) if "width" in data else 512
    height = int(data["height"]) if "height" in data else 512
    denoise = float(data["denoise"]) if "denoise" in data else 1
    prompt = data["prompt"] if "prompt" in data else ""
    negative_prompt = data["negative_prompt"] if "negative_prompt" in data else ""
    sampler = data["sampler"] if "sampler" in data else "euler a"
    processing = data["processing"] if "processing" in data else "gpu"
    format = data["format"] if "format" in data else "png"
    model_name = data["model_name"] if "model_name" in data else get_diffusion_models()[0]
    vae_name = data["vae_name"] if "vae_name" in data else get_vae_models()[0]
    textual_inversions = data["textual_inversions"] if "textual_inversions" in data else []
    hypernetworks = data["hypernetworks"] if "hypernetworks" in data else []
    loras = data["loras"] if "loras" in data else []
    control_processor = data["control_processor"] if "control_processor" in data else "none"
    control_reference_image = data["control_reference_image"] if "control_reference_image" in data else False
    control_scale = float(data["control_scale"]) if "control_scale" in data else 1.0
    control_start = float(data["control_start"]) if "control_start" in data else 0.0
    control_end = float(data["control_end"]) if "control_end" in data else 1.0
    style_fidelity = float(data["style_fidelity"]) if "style_fidelity" in data else 0.5
    guess_mode = data["guess_mode"] if "guess_mode" in data else False
    upscaler = data["upscaler"] if "upscaler" in data else ""
    watermark = data["watermark"] if "watermark" in data else False
    invisible_watermark = data["invisible_watermark"] if "invisible_watermark" in data else True
    nsfw_enabled = data["nsfwTab"] if "nsfwTab" in data else False

    xl = True if "xl" in model_name.lower() else False

    input_image = None
    input_mask = None
    control_image = None
    cross_attention_kwargs = {}
    if "image" in request_files:
        input_image = Image.open(request_files["image"]).convert("RGB")
        mode = "image"
        if "mask" in request_files:
            input_mask =  Image.open(request_files["mask"]).convert("RGB")
            mode = "inpaint"
            if "control_image" in request_files:
                control_image =  Image.open(request_files["control_image"]).convert("RGB")
                mode = "controlnet inpaint"
        else:
            if "control_image" in request_files:
                control_image =  Image.open(request_files["control_image"]).convert("RGB")
                mode = "controlnet"
                if control_reference_image:
                    mode = "controlnet image"
                if control_processor == "reference":
                    mode = "controlnet reference"
    if format == "gif":
        mode = "animatediff"

    socketio.emit("image starting")

    generator = get_generator(model_name, vae_name, mode, clip_skip, processing == "cpu", control_processor)

    if sampler == "euler a":
        generator.scheduler = EulerAncestralDiscreteScheduler.from_config(generator.scheduler.config)
    elif sampler == "euler":
        generator.scheduler = EulerDiscreteScheduler.from_config(generator.scheduler.config)
    elif sampler == "dpm++":
        generator.scheduler = DPMSolverMultistepScheduler.from_config(generator.scheduler.config)
    elif sampler == "ddim":
        generator.scheduler = DDIMScheduler.from_config(generator.scheduler.config)
    elif sampler == "ddpm":
        generator.scheduler = DDPMScheduler.from_config(generator.scheduler.config)
    elif sampler == "unipc":
        generator.scheduler = UniPCMultistepScheduler.from_config(generator.scheduler.config)
    elif sampler == "deis":
        generator.scheduler = DEISMultistepScheduler.from_config(generator.scheduler.config)
    elif sampler == "heun":
        generator.scheduler = HeunDiscreteScheduler.from_config(generator.scheduler.config)

    if mode == "animatediff":
        generator.scheduler.beta_schedule = "linear"
        generator.scheduler.clip_sample = False

    for textual_inversion in textual_inversions:
        textual_inversion_name = textual_inversion["name"]
        textual_inversion_path = os.path.join(dirname, f'../{textual_inversion["model"]}')
        try:
            generator.load_textual_inversion(textual_inversion_path, token=textual_inversion_name)
        except ValueError:
            continue
    
    has_hypernet = False
    for hypernetwork in hypernetworks:
        hypernet_scale = float(hypernetwork["weight"])
        hypernet_path = os.path.join(dirname, f'../{hypernetwork["model"]}')
        has_hypernet = True
        try:
            hypernet = load_hypernet(hypernet_path, hypernet_scale)
            add_hypernet(generator.unet, hypernet)
        except ValueError:
            continue
    if not has_hypernet:
        clear_hypernets(generator.unet)

    has_lora = False
    generator.unfuse_lora()
    adapters = []
    adapter_weights = []
    for lora in loras:
        lora_scale = float(lora["weight"])
        weight_name = os.path.basename(lora["model"])
        lora_name = lora["name"]
        #cross_attention_kwargs["scale"] = lora_scale
        lora = os.path.join(dirname, f'../{lora["model"]}')
        has_lora = True
        try:
            adapters.append(lora_name)
            adapter_weights.append(lora_scale)
            generator.load_lora_weights(lora, weight_name=weight_name, adapter_name=lora_name)
        except ValueError:
            continue
    generator.set_adapters(adapters, adapter_weights=adapter_weights)
    generator.fuse_lora()
    generator.enable_lora()
    if not has_lora:
        generator.unload_lora_weights()
        generator.disable_lora()
        cross_attention_kwargs.pop("scale", None)
    
    conditioning = None
    negative_conditioning = None
    pooled = None
    negative_pooled = None

    textual_inversion_manager = DiffusersTextualInversionManager(generator)

    returned_embeddings_type = ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED
    if clip_skip > 1:
        returned_embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED

    if xl:
        compel = Compel(tokenizer=[generator.tokenizer, generator.tokenizer_2] , text_encoder=[generator.text_encoder, generator.text_encoder_2], requires_pooled=[False, True],
                        returned_embeddings_type=returned_embeddings_type, textual_inversion_manager=textual_inversion_manager, truncate_long_prompts=True)
        conditioning, pooled = compel.build_conditioning_tensor(prompt)
        negative_conditioning, negative_pooled = compel.build_conditioning_tensor(negative_prompt)
        [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])
    else:
        compel = Compel(tokenizer=generator.tokenizer, text_encoder=generator.text_encoder, returned_embeddings_type=returned_embeddings_type,
                        textual_inversion_manager=textual_inversion_manager, truncate_long_prompts=False)
        conditioning = compel.build_conditioning_tensor(prompt)
        negative_conditioning = compel.build_conditioning_tensor(negative_prompt)
        [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])

    def step_progress(self, step: int, timestep: int, call_dict: dict):
        latent = None
        if type(call_dict) is torch.Tensor:
            latent = call_dict
        else:
            latent = call_dict.get("latents")
        with torch.no_grad():
            latent = 1 / 0.18215 * latent
            image = generator.vae.decode(latent).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = generator.numpy_to_pil(image)[0]
            w, h = image.size
            pixels = list(image.convert("RGBA").getdata())
            pixels = list(chain(*pixels))
            total_steps = steps 
            if mode == "image" or mode == "inpaint" or mode == "controlnet image" or mode == "controlnet inpaint":
                total_steps = int(total_steps / 2)
            if sampler == "heun":
                total_steps = int(total_steps * 2)
            socketio.emit("step progress", {"step": step, "total_step": total_steps, "width": w, "height": h, "image": pixels})
        return call_dict

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
            if xl:
                image = generator(
                    prompt_embeds=conditioning,
                    pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=negative_conditioning,
                    negative_pooled_prompt_embeds=negative_pooled,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    num_images_per_prompt=1,
                    generator=torch.manual_seed(seed),
                    clip_skip=clip_skip,
                    callback_on_step_end=step_progress,
                    cross_attention_kwargs=cross_attention_kwargs
                ).images[0]
            else:
                image = generator(
                    prompt_embeds=conditioning,
                    negative_prompt_embeds=negative_conditioning,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    num_images_per_prompt=1,
                    generator=torch.manual_seed(seed),
                    clip_skip=clip_skip,
                    callback_on_step_end=step_progress,
                    cross_attention_kwargs=cross_attention_kwargs
                ).images[0]
            dir_path = os.path.join(dirname, f"../outputs/{folder}")
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.{format}")
            image.save(out_path)
            if upscaling:
                socketio.emit("image upscaling")
                upscale(out_path, upscaler)
            compressed = Image.open(out_path)
            compressed.save(out_path, quality=90, optimize=True)
            if invisible_watermark: encode_watermark(out_path, out_path, "SDV2")
            info = {"Prompt": prompt, "Negative Prompt": negative_prompt, "Size": f"{width}x{height}", "Model": model_name, 
                     "VAE": vae_name, "Steps": steps, "CFG": cfg, "Sampler": sampler, "Clip Skip": clip_skip, "Seed": seed}
            append_info(out_path, info)
            socketio.emit("image complete", {"image": f"/outputs/{folder}/{os.path.basename(out_path)}", "needs_watermark": watermark})
            images.append(out_path)
            seed += 1
        elif mode == "image":
            normalized = get_normalized_dimensions(input_image)
            input_image = input_image.resize((normalized["width"], normalized["height"]), resample=Image.BICUBIC)
            folder = "image"
            if is_nsfw(prompt):
                folder = "image nsfw"
                if not nsfw_enabled:
                    socketio.emit("image complete", {"image": "", "needs_watermark": False})
                    return images
            if xl:
                image = generator(
                    image=input_image,
                    strength=denoise,
                    prompt_embeds=conditioning,
                    pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=negative_conditioning,
                    negative_pooled_prompt_embeds=negative_pooled,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=torch.manual_seed(seed),
                    clip_skip=clip_skip,
                    callback_on_step_end=step_progress,
                    cross_attention_kwargs=cross_attention_kwargs
                ).images[0]
            else:
                image = generator(
                    image=input_image,
                    strength=denoise,
                    prompt_embeds=conditioning,
                    negative_prompt_embeds=negative_conditioning,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=torch.manual_seed(seed),
                    clip_skip=clip_skip,
                    callback_on_step_end=step_progress,
                    cross_attention_kwargs=cross_attention_kwargs
                ).images[0]
            dir_path = os.path.join(dirname, f"../outputs/{folder}")
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.{format}")
            image.save(out_path)
            if upscaling:
                socketio.emit("image upscaling")
                upscale(out_path, upscaler)
            compressed = Image.open(out_path)
            compressed.save(out_path, quality=90, optimize=True)
            if invisible_watermark: encode_watermark(out_path, out_path, "SDV2")
            info = {"Prompt": prompt, "Negative Prompt": negative_prompt, "Size": f"{width}x{height}", "Denoise": denoise,
                    "Model": model_name, "VAE": vae_name, "Steps": steps, "CFG": cfg, "Sampler": sampler, "Clip Skip": clip_skip, 
                    "Seed": seed}
            append_info(out_path, info)
            socketio.emit("image complete", {"image": f"/outputs/{folder}/{os.path.basename(out_path)}", "needs_watermark": watermark})
            images.append(out_path)
            seed += 1
        elif mode == "inpaint":
            normalized = get_normalized_dimensions(input_image)
            input_image = input_image.resize((normalized["width"], normalized["height"]), resample=Image.BICUBIC)
            input_mask = input_mask.resize((normalized["width"], normalized["height"]), resample=Image.BICUBIC)
            folder = "image"
            if is_nsfw(prompt):
                folder = "image nsfw"
                if not nsfw_enabled:
                    socketio.emit("image complete", {"image": "", "needs_watermark": False})
                    return images
            if xl:
                image = generator(
                    image=input_image,
                    mask_image=input_mask,
                    width=normalized["width"],
                    height=normalized["height"],
                    strength=denoise,
                    prompt_embeds=conditioning,
                    pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=negative_conditioning,
                    negative_pooled_prompt_embeds=negative_pooled,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=torch.manual_seed(seed),
                    clip_skip=clip_skip,
                    callback_on_step_end=step_progress,
                    cross_attention_kwargs=cross_attention_kwargs
                ).images[0]
            else:
                image = generator(
                    image=input_image,
                    mask_image=input_mask,
                    width=normalized["width"],
                    height=normalized["height"],
                    strength=denoise,
                    prompt_embeds=conditioning,
                    negative_prompt_embeds=negative_conditioning,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=torch.manual_seed(seed),
                    clip_skip=clip_skip,
                    callback_on_step_end=step_progress,
                    cross_attention_kwargs=cross_attention_kwargs
                ).images[0]
            dir_path = os.path.join(dirname, f"../outputs/{folder}")
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.{format}")
            image.save(out_path)
            if upscaling:
                socketio.emit("image upscaling")
                upscale(out_path, upscaler)
            compressed = Image.open(out_path)
            compressed.save(out_path, quality=90, optimize=True)
            if invisible_watermark: encode_watermark(out_path, out_path, "SDV2")
            info = {"Prompt": prompt, "Negative Prompt": negative_prompt, "Size": f"{width}x{height}", "Denoise": denoise,
                    "Model": model_name, "VAE": vae_name, "Steps": steps, "CFG": cfg, "Sampler": sampler, "Clip Skip": clip_skip, 
                    "Seed": seed}
            append_info(out_path, info)
            socketio.emit("image complete", {"image": f"/outputs/{folder}/{os.path.basename(out_path)}", "needs_watermark": watermark})
            images.append(out_path)
            seed += 1
        elif mode == "controlnet":
            normalized = get_normalized_dimensions(input_image)
            control_image = control_image.resize((normalized["width"], normalized["height"]), resample=Image.BICUBIC)
            folder = "image"
            if is_nsfw(prompt):
                folder = "image nsfw"
                if not nsfw_enabled:
                    socketio.emit("image complete", {"image": "", "needs_watermark": False})
                    return images
            if xl:
                image = generator(
                    image=control_image,
                    prompt_embeds=conditioning,
                    pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=negative_conditioning,
                    negative_pooled_prompt_embeds=negative_pooled,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=torch.manual_seed(seed),
                    clip_skip=clip_skip,
                    callback_on_step_end=step_progress,
                    cross_attention_kwargs=cross_attention_kwargs,
                    controlnet_conditioning_scale=control_scale,
                    guess_mode=guess_mode,
                    control_guidance_start=control_start,
                    control_guidance_end=control_end
                ).images[0]
            else:
                image = generator(
                    image=control_image,
                    prompt_embeds=conditioning,
                    negative_prompt_embeds=negative_conditioning,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=torch.manual_seed(seed),
                    clip_skip=clip_skip,
                    callback_on_step_end=step_progress,
                    cross_attention_kwargs=cross_attention_kwargs,
                    controlnet_conditioning_scale=control_scale,
                    guess_mode=guess_mode,
                    control_guidance_start=control_start,
                    control_guidance_end=control_end
                ).images[0]
            dir_path = os.path.join(dirname, f"../outputs/{folder}")
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.{format}")
            image.save(out_path)
            if upscaling:
                socketio.emit("image upscaling")
                upscale(out_path, upscaler)
            compressed = Image.open(out_path)
            compressed.save(out_path, quality=90, optimize=True)
            if invisible_watermark: encode_watermark(out_path, out_path, "SDV2")
            info = {"Prompt": prompt, "Negative Prompt": negative_prompt, "Size": f"{width}x{height}", "Denoise": denoise,
                    "Model": model_name, "VAE": vae_name, "Steps": steps, "CFG": cfg, "Sampler": sampler, "Clip Skip": clip_skip, 
                    "Seed": seed}
            append_info(out_path, info)
            socketio.emit("image complete", {"image": f"/outputs/{folder}/{os.path.basename(out_path)}", "needs_watermark": watermark})
            images.append(out_path)
            seed += 1
        elif mode == "controlnet image":
            normalized = get_normalized_dimensions(input_image)
            input_image = input_image.resize((normalized["width"], normalized["height"]), resample=Image.BICUBIC)
            control_image = control_image.resize((normalized["width"], normalized["height"]), resample=Image.BICUBIC)
            if is_nsfw(prompt):
                folder = "image nsfw"
                if not nsfw_enabled:
                    socketio.emit("image complete", {"image": "", "needs_watermark": False})
                    return images
            if xl:
                image = generator(
                    image=input_image,
                    control_image=control_image,
                    strength=denoise,
                    prompt_embeds=conditioning,
                    pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=negative_conditioning,
                    negative_pooled_prompt_embeds=negative_pooled,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=torch.manual_seed(seed),
                    clip_skip=clip_skip,
                    callback_on_step_end=step_progress,
                    cross_attention_kwargs=cross_attention_kwargs,
                    controlnet_conditioning_scale=control_scale,
                    guess_mode=guess_mode,
                    control_guidance_start=control_start,
                    control_guidance_end=control_end
                ).images[0]
            else:
                image = generator(
                    image=input_image,
                    control_image=control_image,
                    strength=denoise,
                    prompt_embeds=conditioning,
                    negative_prompt_embeds=negative_conditioning,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=torch.manual_seed(seed),
                    clip_skip=clip_skip,
                    callback_on_step_end=step_progress,
                    cross_attention_kwargs=cross_attention_kwargs,
                    controlnet_conditioning_scale=control_scale,
                    guess_mode=guess_mode,
                    control_guidance_start=control_start,
                    control_guidance_end=control_end
                ).images[0]
            folder = "image"
            dir_path = os.path.join(dirname, f"../outputs/{folder}")
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.{format}")
            image.save(out_path)
            if upscaling:
                socketio.emit("image upscaling")
                upscale(out_path, upscaler)
            compressed = Image.open(out_path)
            compressed.save(out_path, quality=90, optimize=True)
            if invisible_watermark: encode_watermark(out_path, out_path, "SDV2")
            info = {"Prompt": prompt, "Negative Prompt": negative_prompt, "Size": f"{width}x{height}", "Denoise": denoise,
                    "Model": model_name, "VAE": vae_name, "Steps": steps, "CFG": cfg, "Sampler": sampler, "Clip Skip": clip_skip, 
                    "Seed": seed}
            append_info(out_path, info)
            socketio.emit("image complete", {"image": f"/outputs/{folder}/{os.path.basename(out_path)}", "needs_watermark": watermark})
            images.append(out_path)
            seed += 1
        elif mode == "controlnet inpaint":
            normalized = get_normalized_dimensions(input_image)
            input_image = input_image.resize((normalized["width"], normalized["height"]), resample=Image.BICUBIC)
            input_mask = input_mask.resize((normalized["width"], normalized["height"]), resample=Image.BICUBIC)
            control_image = control_image.resize((normalized["width"], normalized["height"]), resample=Image.BICUBIC)
            if is_nsfw(prompt):
                folder = "image nsfw"
                if not nsfw_enabled:
                    socketio.emit("image complete", {"image": "", "needs_watermark": False})
                    return images
            if xl:
                image = generator(
                    image=input_image,
                    mask_image=input_mask,
                    control_image=control_image,
                    strength=denoise,
                    prompt_embeds=conditioning,
                    pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=negative_conditioning,
                    negative_pooled_prompt_embeds=negative_pooled,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=torch.manual_seed(seed),
                    clip_skip=clip_skip,
                    callback_on_step_end=step_progress,
                    cross_attention_kwargs=cross_attention_kwargs,
                    controlnet_conditioning_scale=control_scale,
                    guess_mode=guess_mode,
                    control_guidance_start=control_start,
                    control_guidance_end=control_end
                ).images[0]
            else:
                image = generator(
                    image=input_image,
                    mask_image=input_mask,
                    control_image=control_image,
                    strength=denoise,
                    prompt_embeds=conditioning,
                    negative_prompt_embeds=negative_conditioning,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=torch.manual_seed(seed),
                    clip_skip=clip_skip,
                    callback_on_step_end=step_progress,
                    cross_attention_kwargs=cross_attention_kwargs,
                    controlnet_conditioning_scale=control_scale,
                    guess_mode=guess_mode,
                    control_guidance_start=control_start,
                    control_guidance_end=control_end
                ).images[0]
            folder = "image"
            dir_path = os.path.join(dirname, f"../outputs/{folder}")
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.{format}")
            image.save(out_path)
            if upscaling:
                socketio.emit("image upscaling")
                upscale(out_path, upscaler)
            compressed = Image.open(out_path)
            compressed.save(out_path, quality=90, optimize=True)
            if invisible_watermark: encode_watermark(out_path, out_path, "SDV2")
            info = {"Prompt": prompt, "Negative Prompt": negative_prompt, "Size": f"{width}x{height}", "Denoise": denoise,
                    "Model": model_name, "VAE": vae_name, "Steps": steps, "CFG": cfg, "Sampler": sampler, "Clip Skip": clip_skip, 
                    "Seed": seed}
            append_info(out_path, info)
            socketio.emit("image complete", {"image": f"/outputs/{folder}/{os.path.basename(out_path)}", "needs_watermark": watermark})
            images.append(out_path)
            seed += 1
        elif mode == "controlnet reference":
            normalized = get_normalized_dimensions(input_image)
            input_image = input_image.resize((normalized["width"], normalized["height"]), resample=Image.BICUBIC)
            control_image = control_image.resize((normalized["width"], normalized["height"]), resample=Image.BICUBIC)
            folder = "image"
            if is_nsfw(prompt):
                folder = "image nsfw"
                if not nsfw_enabled:
                    socketio.emit("image complete", {"image": "", "needs_watermark": False})
                    return images
            if xl:
                image = generator(
                    ref_image=input_image,
                    prompt_embeds=conditioning,
                    pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=negative_conditioning,
                    negative_pooled_prompt_embeds=negative_pooled,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=torch.manual_seed(seed),
                    callback=step_progress,
                    cross_attention_kwargs=cross_attention_kwargs,
                    style_fidelity=style_fidelity,
                    reference_attn=True,
                    reference_adain=True
                ).images[0]
            else:
                image = generator(
                    image=control_image,
                    ref_image=input_image,
                    prompt_embeds=conditioning,
                    negative_prompt_embeds=negative_conditioning,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=torch.manual_seed(seed),
                    callback=step_progress,
                    cross_attention_kwargs=cross_attention_kwargs,
                    controlnet_conditioning_scale=control_scale,
                    guess_mode=guess_mode,
                    style_fidelity=style_fidelity,
                    reference_attn=True,
                    reference_adain=True
                ).images[0]
            dir_path = os.path.join(dirname, f"../outputs/{folder}")
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.{format}")
            image.save(out_path)
            if upscaling:
                socketio.emit("image upscaling")
                upscale(out_path, upscaler)
            compressed = Image.open(out_path)
            compressed.save(out_path, quality=90, optimize=True)
            if invisible_watermark: encode_watermark(out_path, out_path, "SDV2")
            info = {"Prompt": prompt, "Negative Prompt": negative_prompt, "Size": f"{width}x{height}", "Denoise": denoise,
                    "Model": model_name, "VAE": vae_name, "Steps": steps, "CFG": cfg, "Sampler": sampler, "Clip Skip": clip_skip, 
                    "Seed": seed}
            append_info(out_path, info)
            socketio.emit("image complete", {"image": f"/outputs/{folder}/{os.path.basename(out_path)}", "needs_watermark": watermark})
            images.append(out_path)
            seed += 1
        elif mode == "animatediff":
            folder = "text"
            if is_nsfw(prompt):
                folder = "text nsfw"
                if not nsfw_enabled:
                    socketio.emit("image complete", {"image": "", "needs_watermark": False})
                    return images
            # num_frames
            frames = generator(
                prompt_embeds=conditioning,
                negative_prompt_embeds=negative_conditioning,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=torch.manual_seed(seed),
                callback_on_step_end=step_progress,
                cross_attention_kwargs=cross_attention_kwargs
            ).frames[0]
            dir_path = os.path.join(dirname, f"../outputs/{folder}")
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.{format}")
            export_to_gif(frames, out_path)
            socketio.emit("image complete", {"image": f"/outputs/{folder}/{os.path.basename(out_path)}", "needs_watermark": watermark})
            images.append(out_path)
            seed += 1
    gc.collect()
    torch.mps.empty_cache()
    torch.cuda.empty_cache()
    if infinite:
        socketio.emit("repeat generation")
    return images

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

@app.route("/interrupt", methods=["POST"])
def interrupt_generate():
    global gen_thread
    if gen_thread:
        try:
            _async_raise(gen_thread, ChildProcessError)
        except ChildProcessError:
            pass
        gen_thread = None
        socketio.emit("image interrupt")
        return "done"

@app.route("/generate", methods=["POST"])
def start_generate():
    global gen_thread
    request_data = flask.request.form.get("data")
    request_files = flask.request.files
    thread = threading.Thread(target=generate, args=(request_data, request_files))
    thread.start()
    thread.join()
    gen_thread = None
    return "done"