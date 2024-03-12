from __main__ import app, socketio
import os
import torch
from .functions import next_index, is_nsfw, get_normalized_dimensions, get_seed, get_seed_generator, append_info, upscale, get_models_dir, get_outputs_dir
from .invisiblewatermark import encode_watermark
from .info import get_diffusion_models, get_vae_models
from diffusers import StableCascadePriorPipeline, StableCascadeDecoderPipeline, DDPMWuerstchenScheduler
from .hypernet import load_hypernet, add_hypernet, clear_hypernets
from compel import Compel, ReturnedEmbeddingsType, DiffusersTextualInversionManager
from PIL import Image
from itertools import chain
import pathlib
import asyncio
import gc

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

gen_thread = None
prior = None
prior_name = None
decoder = None
infinite = False
upscaling = True
dtype = torch.float32

def get_cascade_generators(model_name: str = "", cpu: bool = False):
    global prior
    global prior_name
    global decoder
    global dtype
    global device
    processor = "cpu" if cpu else device
    update_model = False
    if not model_name:
        model_name = get_diffusion_models()[0]
    if not prior or prior_name != model_name:
        model = os.path.join(get_models_dir(), "diffusion", model_name)
        if prior_name != model_name:
            update_model = True
        if prior is not None and not prior.prior:
            update_model = True

        if update_model:
            if os.path.isdir(model):
                prior = StableCascadePriorPipeline.from_pretrained(model, local_files_only=True)  
            else:
                prior = StableCascadePriorPipeline.from_single_file(model)
        else:
            prior = StableCascadePriorPipeline(prior=prior.prior, text_encoder=prior.text_encoder, feature_extractor=prior.feature_extractor,
                                                image_encoder=prior.image_encoder, tokenizer=prior.tokenizer, scheduler=prior.scheduler)
        prior_name = model_name
    if not decoder:
        decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade")
    prior = prior.to(device=processor, dtype=dtype)
    decoder = decoder.to(device=processor, dtype=dtype)
    return prior, decoder

def unload_models_cascade():
    global prior
    global prior_name
    global decoder
    prior = None
    prior_name = None
    decoder = None
    return "done"

def update_infinite_cascade(value):
    global infinite
    infinite = value
    return "done"

def update_upscaling_cascade(value):
    global upscaling
    upscaling = value
    return "done"

def update_precision_cascade(value):
    global dtype
    precision = value
    if precision == "full":
        dtype = torch.float32
    elif precision == "half":
        dtype = torch.bfloat16
    return "done"

def generate_cascade(data, request_files, clear_step_frames=None, generate_step_animation=None):
    global device
    global infinite
    global upscaling
    mode = "text"

    if clear_step_frames is not None:
        asyncio.run(clear_step_frames())

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
    upscaler = data["upscaler"] if "upscaler" in data else ""
    watermark = data["watermark"] if "watermark" in data else False
    invisible_watermark = data["invisible_watermark"] if "invisible_watermark" in data else True
    nsfw_enabled = data["nsfw_tab"] if "nsfw_tab" in data else False

    input_image = None
    if "image" in request_files:
        input_image = Image.open(request_files["image"]).convert("RGB")
        mode = "image"

    socketio.emit("image starting")

    prior, decoder = get_cascade_generators(model_name, processing == "cpu")

    prior.scheduler = DDPMWuerstchenScheduler.from_config(prior.scheduler.config)

    prior.enable_attention_slicing()
    decoder.enable_attention_slicing()

    def step_progress_prior(self, step: int, timestep: int, call_dict: dict):
        socketio.emit("step progress", {"step": step, "total_step": steps-1})
        return call_dict

    def step_progress(self, step: int, timestep: int, call_dict: dict):
        latent = None
        if type(call_dict) is torch.Tensor:
            latent = call_dict
        else:
            latent = call_dict.get("latents")
        with torch.no_grad():
            latent = 1 / 0.3764 * latent
            image = decoder.vqgan.decode(latent).sample.clamp(0, 1)
            image = image.permute(0, 2, 3, 1).cpu().float().numpy()
            image = decoder.numpy_to_pil(image)[0]
            w, h = image.size
            pixels = list(image.convert("RGBA").getdata())
            pixels = list(chain(*pixels))
            total_steps = 10 
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
        folder = "text"
        if is_nsfw(prompt):
            folder = "text nsfw"
            if not nsfw_enabled:
                socketio.emit("image complete", {"image": "", "needs_watermark": False})
                return images
        if input_image is not None:
            folder = "image"
            if is_nsfw(prompt):
                folder = "image nsfw"
        prior_output = prior(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            width=width,
            height=height,
            guidance_scale=cfg,
            num_inference_steps=steps,
            num_images_per_prompt=1,
            generator=get_seed_generator(seed, device),
            callback_on_step_end=step_progress_prior
        )
        image = decoder(
            image_embeddings=prior_output.image_embeddings,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=0.0,
            num_inference_steps=10,
            num_images_per_prompt=1,
            generator=get_seed_generator(seed, device),
            callback_on_step_end=step_progress
        ).images[0]
        dir_path = os.path.join(get_outputs_dir(), f"local/{folder}")
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(dir_path, f"image{next_index(dir_path)}.{format}")
        image.save(out_path)
        if generate_step_animation is not None:
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
    gc.collect()
    torch.mps.empty_cache()
    torch.cuda.empty_cache()
    if infinite:
        socketio.emit("repeat generation")
    return images