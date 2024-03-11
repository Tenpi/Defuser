import flask      
from __main__ import app, socketio
import os
import torch
from .functions import next_index, is_nsfw, get_normalized_dimensions, is_image, get_number_from_filename, \
get_seed, append_info, upscale, get_models_dir, get_outputs_dir, analyze_checkpoint, check_for_updates
from .invisiblewatermark import encode_watermark
from .info import get_diffusion_models, get_vae_models, get_clip_model
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, StableDiffusionControlNetImg2ImgPipeline, \
StableDiffusionControlNetInpaintPipeline, StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LCMScheduler, \
DDPMScheduler, DDIMScheduler, UniPCMultistepScheduler, DEISMultistepScheduler, DPMSolverMultistepScheduler, HeunDiscreteScheduler, AutoencoderKL, MotionAdapter, AnimateDiffPipeline
from diffusers.utils import export_to_gif
from diffusers.image_processor import IPAdapterMaskProcessor
from .stable_diffusion_controlnet_reference import StableDiffusionControlNetReferencePipeline
from .hypernet import load_hypernet, add_hypernet, clear_hypernets
from .external import generate_novelai, generate_holara, update_ext_upscaling, update_ext_infinite
from .generate_xl import generate_xl, unload_models_xl, update_upscaling_xl, update_infinite_xl, update_precision_xl
from .generate_cascade import generate_cascade, unload_models_cascade, update_upscaling_cascade, update_infinite_cascade, update_precision_cascade
from .controlnet import unload_control_models
from .interrogate import unload_interrogate_models
from compel import Compel, ReturnedEmbeddingsType, DiffusersTextualInversionManager
from PIL import Image
import pathlib
import json
from itertools import chain
import inspect
import ctypes
import threading
import asyncio
import gc

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

update_dismissed = False
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

def get_motion_adapter():
    global motion_adapter
    motion_model = os.path.join(get_models_dir(), "animatediff")
    motion_adapter = MotionAdapter.from_pretrained(motion_model, local_files_only=True)
    return motion_adapter

def get_controlnet(processor: str = "none"):
    global controlnet
    global control_processor
    if not controlnet or control_processor != processor:
        if processor == "canny":
            control_model = os.path.join(get_models_dir(), "controlnet/canny")
            controlnet = ControlNetModel.from_pretrained(control_model, local_files_only=True)
        elif processor == "depth":
            control_model = os.path.join(get_models_dir(), "controlnet/depth")
            controlnet = ControlNetModel.from_pretrained(control_model, local_files_only=True)
        if processor == "lineart":
            control_model = os.path.join(get_models_dir(), "controlnet/lineart")
            controlnet = ControlNetModel.from_pretrained(control_model, local_files_only=True)
        if processor == "lineart anime":
            control_model = os.path.join(get_models_dir(), "controlnet/lineart anime")
            controlnet = ControlNetModel.from_pretrained(control_model, local_files_only=True)
        if processor == "lineart manga":
            control_model = os.path.join(get_models_dir(), "controlnet/lineart anime")
            controlnet = ControlNetModel.from_pretrained(control_model, local_files_only=True)
        if processor == "scribble":
            control_model = os.path.join(get_models_dir(), "controlnet/scribble")
            controlnet = ControlNetModel.from_pretrained(control_model, local_files_only=True)
        if processor == "softedge":
            control_model = os.path.join(get_models_dir(), "controlnet/softedge")
            controlnet = ControlNetModel.from_pretrained(control_model, local_files_only=True)
        if processor == "reference":
            control_model = os.path.join(get_models_dir(), "controlnet/canny")
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
        model = os.path.join(get_models_dir(), "diffusion", model_name)
        if generator_name != model_name:
            update_model = True
        if generator is not None and not generator.vae:
            update_model = True

        if mode == "text":
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
        vae_model = os.path.join(get_models_dir(), "vae", vae)
        if os.path.isdir(vae_model):
            generator.vae = AutoencoderKL.from_pretrained(vae_model, local_files_only=True)
        else:
            generator.vae = AutoencoderKL.from_single_file(vae_model)
        generator.vae = generator.vae.to(device=processor, dtype=dtype)
        vae_name = vae
    generator = generator.to(device=device, dtype=dtype)
    generator.safety_checker = None
    return generator

@socketio.on("check update")
def check_update():
    global update_dismissed
    if update_dismissed: return
    update_available, new_version = check_for_updates()
    if update_available:
        socketio.emit("update available", {"version": new_version})
    else:
        update_dismissed = True
    return "done"

@app.route("/dismiss-update", methods=["POST"])
def dismiss_update():
    global update_dismissed
    update_dismissed = True
    return "done"

@socketio.on("load diffusion model")
def load_diffusion_model(model_name, vae_name, clip_skip, processing, generator_type):
    global generator
    #if generator_type == "local":
        #generator = get_generator(model_name, vae_name, "text", int(clip_skip), processing == "cpu")
    #load_diffusion_model_xl(model_name, vae_name, clip_skip, processing, generator_type)
    return "done"

@app.route("/unload-models", methods=["POST"])
def unload_models():
    global generator
    global generator_name
    global vae_name
    global safety_checker
    global controlnet
    global control_processor
    global motion_adapter
    generator = None
    generator_name = None
    vae_name = None
    safety_checker = None
    controlnet = None
    control_processor = "none"
    motion_adapter = None
    unload_control_models()
    unload_interrogate_models()
    unload_models_xl()
    unload_models_cascade()
    gc.collect()
    torch.mps.empty_cache()
    torch.cuda.empty_cache()
    return "done"

@app.route("/update-infinite", methods=["POST"])
def update_infinite():
    global infinite
    data = flask.request.json
    infinite = data["infinite"]
    update_ext_infinite(infinite)
    update_infinite_xl(infinite)
    update_infinite_cascade(infinite)
    return "done"

@app.route("/update-upscaling", methods=["POST"])
def update_upscaling():
    global upscaling
    data = flask.request.json
    upscaling = data["upscaling"]
    update_ext_upscaling(upscaling)
    update_upscaling_xl(upscaling)
    update_upscaling_cascade(upscaling)
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
    update_precision_xl(precision)
    update_precision_cascade(precision)
    return "done"

async def clear_step_frames():
    step_dir = os.path.join(get_outputs_dir(), f"local/steps")
    pathlib.Path(step_dir).mkdir(parents=True, exist_ok=True)
    images = os.listdir(step_dir)
    images = list(filter(lambda file: is_image(file, False), images))
    images = sorted(images, key=lambda x: get_number_from_filename(x), reverse=False)
    images = list(map(lambda image: os.path.join(step_dir, image), images))
    for image in images:
        os.remove(image)

async def generate_step_animation():
    step_dir = os.path.join(get_outputs_dir(), f"local/steps")
    pathlib.Path(step_dir).mkdir(parents=True, exist_ok=True)
    images = os.listdir(step_dir)
    images = list(filter(lambda file: is_image(file, False), images))
    images = sorted(images, key=lambda x: get_number_from_filename(x), reverse=False)
    images = list(map(lambda image: os.path.join(step_dir, image), images))
    frames = list(map(lambda image: Image.open(image).convert("RGB"), images))
    gif_path = os.path.join(step_dir, "step.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    socketio.emit("step animation complete", {"path": f"/outputs/local/steps/step.gif"})

def generate(request_data, request_files):
    global gen_thread
    global device
    global infinite
    global upscaling
    gen_thread = threading.get_ident()
    mode = "text"
    data = json.loads(request_data)

    asyncio.run(clear_step_frames())
    generator_type = data["generator"] if "generator" in data else "local"
    if generator_type == "novel ai":
        return generate_novelai(data, request_files)
    elif generator_type == "holara ai":
        return generate_holara(data, request_files)
                
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
    nsfw_enabled = data["nsfw_tab"] if "nsfw_tab" in data else False
    freeu = data["freeu"] if "freeu" in data else False
    ip_adapter = data["ip_adapter"] if "ip_adapter" in data else "None"
    ip_processor = data["ip_processor"] if "ip_processor" in data else "off"
    ip_weight = float(data["ip_weight"]) if "ip_weight" in data else 0.5
    frames = int(data["frames"]) if "frames" in data else 8

    xl, cascade = analyze_checkpoint(model_name, device)
    if xl:
        return generate_xl(data, request_files, get_controlnet, clear_step_frames, generate_step_animation)
    elif cascade:
        return generate_cascade(data, request_files, clear_step_frames, generate_step_animation)

    input_image = None
    input_mask = None
    control_image = None
    ip_adapter_image = None
    ip_adapter_mask = None
    cross_attention_kwargs = {}
    if "ip_image" in request_files and ip_processor == "on":
        ip_adapter_image = Image.open(request_files["ip_image"]).convert("RGB")
    if "ip_mask" in request_files and ip_processor == "on":
        ip_adapter_mask = Image.open(request_files["ip_mask"]).convert("RGB")
        ip_mask_processor = IPAdapterMaskProcessor()
        converted_masks = ip_mask_processor.preprocess([ip_adapter_mask], height=height, width=width)
        cross_attention_kwargs["ip_adapter_masks"] = converted_masks
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

    if freeu:
        generator.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)

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
    elif sampler == "lcm":
        generator.scheduler = LCMScheduler.from_config(generator.scheduler.config)

    if mode == "animatediff":
        generator.scheduler.beta_schedule = "linear"
        generator.scheduler.clip_sample = False

    generator.unload_ip_adapter()
    if ip_processor == "on":
        ip_adapter_path = os.path.join(get_models_dir(), "ipadapter")
        generator.load_ip_adapter(ip_adapter_path, subfolder="models", weight_name=ip_adapter, local_files_only=True)
        generator.set_ip_adapter_scale(ip_weight)

    for textual_inversion in textual_inversions:
        textual_inversion_name = textual_inversion["name"]
        textual_inversion_path = os.path.join(get_models_dir(), textual_inversion["model"].replace("models/", ""))
        try:
            generator.load_textual_inversion(textual_inversion_path, token=textual_inversion_name)
        except ValueError:
            continue

    has_hypernet = False
    for hypernetwork in hypernetworks:
        hypernet_scale = float(hypernetwork["weight"])
        hypernet_path = os.path.join(get_models_dir(), hypernetwork["model"].replace("models/", ""))
        has_hypernet = True
        try:
            hypernet = load_hypernet(hypernet_path, hypernet_scale, device)
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
        lora_path = os.path.join(get_models_dir(), lora["model"].replace("models/", ""))
        has_lora = True
        try:
            adapters.append(lora_name)
            adapter_weights.append(lora_scale)
            generator.load_lora_weights(lora_path, weight_name=weight_name, adapter_name=lora_name)
        except ValueError:
            continue
    generator.set_adapters(adapters, adapter_weights=adapter_weights)
    generator.fuse_lora()
    generator.enable_lora()
    if not has_lora:
        generator.unload_lora_weights()
        generator.disable_lora()
        #cross_attention_kwargs.pop("scale", None)

    generator.vae.enable_slicing()
    generator.vae.enable_tiling()
    generator.unet.fuse_qkv_projections()
    if ip_processor == "off" and not has_hypernet:
        generator.enable_attention_slicing()
    
    conditioning = None
    negative_conditioning = None
    pooled = None
    negative_pooled = None

    textual_inversion_manager = DiffusersTextualInversionManager(generator)

    returned_embeddings_type = ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED
    if clip_skip > 1:
        returned_embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED

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
            step_dir = os.path.join(get_outputs_dir(), f"local/steps")
            pathlib.Path(step_dir).mkdir(parents=True, exist_ok=True)
            img_path = os.path.join(step_dir, f"step{step}.png")
            image.save(img_path)
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
            image = generator(
                prompt_embeds=conditioning,
                negative_prompt_embeds=negative_conditioning,
                ip_adapter_image=ip_adapter_image,
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
            dir_path = os.path.join(get_outputs_dir(), f"local/{folder}")
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.{format}")
            image.save(out_path)
            asyncio.run(generate_step_animation())
            if upscaling:
                socketio.emit("image upscaling")
                upscale(out_path, upscaler)
            compressed = Image.open(out_path)
            compressed.save(out_path, quality=90, optimize=True)
            if invisible_watermark: encode_watermark(out_path, out_path, "SDV2")
            info = {"Prompt": prompt, "Negative Prompt": negative_prompt, "Size": f"{width}x{height}", "Model": model_name, 
                     "VAE": vae_name, "Steps": steps, "CFG": cfg, "Sampler": sampler, "Clip Skip": clip_skip, "Seed": seed}
            append_info(out_path, info)
            socketio.emit("image complete", {"image": f"/outputs/local/{folder}/{os.path.basename(out_path)}", "needs_watermark": watermark})
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
            image = generator(
                image=input_image,
                strength=denoise,
                ip_adapter_image=ip_adapter_image,
                prompt_embeds=conditioning,
                negative_prompt_embeds=negative_conditioning,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=torch.manual_seed(seed),
                clip_skip=clip_skip,
                callback_on_step_end=step_progress,
                cross_attention_kwargs=cross_attention_kwargs
            ).images[0]
            dir_path = os.path.join(get_outputs_dir(), f"local/{folder}")
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.{format}")
            image.save(out_path)
            asyncio.run(generate_step_animation())
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
            socketio.emit("image complete", {"image": f"/outputs/local/{folder}/{os.path.basename(out_path)}", "needs_watermark": watermark})
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
            image = generator(
                image=input_image,
                mask_image=input_mask,
                ip_adapter_image=ip_adapter_image,
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
            dir_path = os.path.join(get_outputs_dir(), f"local/{folder}")
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.{format}")
            image.save(out_path)
            asyncio.run(generate_step_animation())
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
            socketio.emit("image complete", {"image": f"/outputs/local/{folder}/{os.path.basename(out_path)}", "needs_watermark": watermark})
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
            image = generator(
                image=control_image,
                prompt_embeds=conditioning,
                negative_prompt_embeds=negative_conditioning,
                ip_adapter_image=ip_adapter_image,
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
            dir_path = os.path.join(get_outputs_dir(), f"local/{folder}")
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.{format}")
            image.save(out_path)
            asyncio.run(generate_step_animation())
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
            socketio.emit("image complete", {"image": f"/outputs/local/{folder}/{os.path.basename(out_path)}", "needs_watermark": watermark})
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
            image = generator(
                image=input_image,
                control_image=control_image,
                strength=denoise,
                ip_adapter_image=ip_adapter_image,
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
            dir_path = os.path.join(get_outputs_dir(), f"local/{folder}")
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.{format}")
            image.save(out_path)
            asyncio.run(generate_step_animation())
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
            socketio.emit("image complete", {"image": f"/outputs/local/{folder}/{os.path.basename(out_path)}", "needs_watermark": watermark})
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
                    return image
            image = generator(
                image=input_image,
                mask_image=input_mask,
                control_image=control_image,
                ip_adapter_image=ip_adapter_image,
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
            dir_path = os.path.join(get_outputs_dir(), f"local/{folder}")
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.{format}")
            image.save(out_path)
            asyncio.run(generate_step_animation())
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
            socketio.emit("image complete", {"image": f"/outputs/local/{folder}/{os.path.basename(out_path)}", "needs_watermark": watermark})
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
            dir_path = os.path.join(get_outputs_dir(), f"local/{folder}")
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.{format}")
            image.save(out_path)
            asyncio.run(generate_step_animation())
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
            socketio.emit("image complete", {"image": f"/outputs/local/{folder}/{os.path.basename(out_path)}", "needs_watermark": watermark})
            images.append(out_path)
            seed += 1
        elif mode == "animatediff":
            folder = "text"
            if is_nsfw(prompt):
                folder = "text nsfw"
                if not nsfw_enabled:
                    socketio.emit("image complete", {"image": "", "needs_watermark": False})
                    return images
            frames = generator(
                prompt_embeds=conditioning,
                negative_prompt_embeds=negative_conditioning,
                num_inference_steps=steps,
                num_frames=frames,
                guidance_scale=cfg,
                generator=torch.manual_seed(seed),
                callback_on_step_end=step_progress,
                cross_attention_kwargs=cross_attention_kwargs
            ).frames[0]
            dir_path = os.path.join(get_outputs_dir(), f"local/{folder}")
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.{format}")
            export_to_gif(frames, out_path)
            socketio.emit("image complete", {"image": f"/outputs/local/{folder}/{os.path.basename(out_path)}", "needs_watermark": watermark})
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