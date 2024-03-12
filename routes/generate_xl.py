from __main__ import app, socketio
import os
import torch
from .functions import next_index, is_nsfw, get_normalized_dimensions, get_seed, get_seed_generator, append_info, upscale, get_models_dir, get_outputs_dir
from .invisiblewatermark import encode_watermark
from .info import get_diffusion_models, get_vae_models
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, \
StableDiffusionXLControlNetPipeline, StableDiffusionXLControlNetImg2ImgPipeline, StableDiffusionXLControlNetInpaintPipeline, LCMScheduler, \
EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, DDPMScheduler, DDIMScheduler, UniPCMultistepScheduler, DEISMultistepScheduler, DPMSolverMultistepScheduler, \
HeunDiscreteScheduler, AutoencoderKL
from diffusers.image_processor import IPAdapterMaskProcessor
from .stable_diffusion_xl_reference import StableDiffusionXLReferencePipeline
from .hypernet import load_hypernet, add_hypernet, clear_hypernets
from .x_adapter import load_adapter_lora, unload_adapter_loras, Adapter_XL, UNet2DConditionAdapterModel, StableDiffusionXLAdapterPipeline, StableDiffusionXLAdapterControlnetPipeline, StableDiffusionXLAdapterControlnetI2IPipeline
from compel import Compel, ReturnedEmbeddingsType, DiffusersTextualInversionManager
from PIL import Image
from itertools import chain
import pathlib
import asyncio
import gc

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

gen_thread = None
generator = None
generator_name = None
generator_sd1 = None
generator_sd1_name = None
generator_mode = "text"
generator_clip_skip = 2
vae_name = None
infinite = False
upscaling = True
safety_checker = None
dtype = torch.float32
controlnet = None
control_processor = "none"
x_adapter = None

def get_controlnet_xl(processor: str = "none", get_controlnet=None):
    global controlnet
    global control_processor
    if get_controlnet is None: return controlnet
    if not controlnet or control_processor != processor:
        controlnet = get_controlnet(processor)
        control_processor = processor
    return controlnet

def get_generator_xl(model_name: str = "", vae: str = "", mode: str = "text", clip_skip: int = 2, 
                     cpu: bool = False, control_processor: str = "none", get_controlnet=None, x_adapt_model="None"):
    global generator
    global generator_name
    global generator_sd1
    global generator_sd1_name
    global generator_mode
    global generator_clip_skip
    global vae_name
    global safety_checker
    global x_adapter
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
        if generator is not None and not generator.text_encoder_2:
            update_model = True

        if mode == "text":
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
        elif mode == "image":
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
        elif mode == "inpaint":
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
        elif mode == "controlnet":
            controlnet = get_controlnet_xl(control_processor, get_controlnet).to(device=processor, dtype=dtype)
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
        elif mode == "controlnet image":
            controlnet = get_controlnet_xl(control_processor, get_controlnet).to(device=processor, dtype=dtype)
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
        elif mode == "controlnet inpaint":
            controlnet = get_controlnet_xl(control_processor, get_controlnet).to(device=processor, dtype=dtype)
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
        elif mode == "controlnet reference":
            controlnet = get_controlnet_xl(control_processor, get_controlnet).to(device=processor, dtype=dtype)
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
        generator_name = model_name
        generator_mode = mode
    if x_adapt_model and x_adapt_model != "None":
        if not generator_sd1 or generator_sd1_name != x_adapt_model:
            model_sd1 = os.path.join(get_models_dir(), "diffusion", x_adapt_model)
            if os.path.isdir(model_sd1):
                generator_sd1 = StableDiffusionPipeline.from_pretrained(model_sd1, local_files_only=True)  
            else:
                generator_sd1 = StableDiffusionPipeline.from_single_file(model_sd1)

        if not x_adapter:
            x_adapter = Adapter_XL()
            adapter_weights = torch.load(os.path.join(get_models_dir(), "misc/X_Adapter.bin"))
            x_adapter.load_state_dict(adapter_weights)

        unet_sd1_5 = UNet2DConditionAdapterModel.from_config(generator_sd1.unet.config)
        unet_sd1_5.load_state_dict(generator_sd1.unet.state_dict().copy())
        generator_sd1.unet = unet_sd1_5
        unet = UNet2DConditionAdapterModel.from_config(generator.unet.config)
        unet.load_state_dict(generator.unet.state_dict().copy())
        generator.unet = unet
        if mode == "text":
            generator = StableDiffusionXLAdapterPipeline(
                vae=generator.vae,
                text_encoder=generator.text_encoder,
                text_encoder_2=generator.text_encoder_2,
                tokenizer=generator.tokenizer,
                tokenizer_2=generator.tokenizer_2,
                unet=generator.unet,
                scheduler=generator.scheduler,
                vae_sd1_5=generator_sd1.vae,
                text_encoder_sd1_5=generator_sd1.text_encoder,
                tokenizer_sd1_5=generator_sd1.tokenizer,
                unet_sd1_5=generator_sd1.unet,
                scheduler_sd1_5=generator_sd1.scheduler,
                adapter=x_adapter
            )
        elif mode == "controlnet":
            controlnet = get_controlnet_xl(control_processor, get_controlnet).to(device=processor, dtype=dtype)
            generator = StableDiffusionXLAdapterControlnetPipeline(
                vae=generator.vae,
                text_encoder=generator.text_encoder,
                text_encoder_2=generator.text_encoder_2,
                tokenizer=generator.tokenizer,
                tokenizer_2=generator.tokenizer_2,
                unet=generator.unet,
                scheduler=generator.scheduler,
                vae_sd1_5=generator_sd1.vae,
                text_encoder_sd1_5=generator_sd1.text_encoder,
                tokenizer_sd1_5=generator_sd1.tokenizer,
                unet_sd1_5=generator_sd1.unet,
                scheduler_sd1_5=generator_sd1.scheduler,
                adapter=x_adapter,
                controlnet=controlnet
            )
        elif mode == "controlnet image":
            controlnet = get_controlnet_xl(control_processor, get_controlnet).to(device=processor, dtype=dtype)
            generator = StableDiffusionXLAdapterControlnetI2IPipeline(
                vae=generator.vae,
                text_encoder=generator.text_encoder,
                text_encoder_2=generator.text_encoder_2,
                tokenizer=generator.tokenizer,
                tokenizer_2=generator.tokenizer_2,
                unet=generator.unet,
                scheduler=generator.scheduler,
                vae_sd1_5=generator_sd1.vae,
                text_encoder_sd1_5=generator_sd1.text_encoder,
                tokenizer_sd1_5=generator_sd1.tokenizer,
                unet_sd1_5=generator_sd1.unet,
                scheduler_sd1_5=generator_sd1.scheduler,
                adapter=x_adapter,
                controlnet=controlnet
            )
        generator_sd1_name = x_adapt_model
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

def load_diffusion_model_xl(model_name, vae_name, clip_skip, processing, generator_type):
    global generator
    #if generator_type == "local":
        #generator = get_generator(model_name, vae_name, "text", int(clip_skip), processing == "cpu")
    return "done"

def unload_models_xl():
    global generator
    global generator_name
    global generator_sd1
    global generator_sd1_name
    global vae_name
    global safety_checker
    global controlnet
    global control_processor
    global x_adapter
    generator = None
    generator_name = None
    generator_sd1 = None
    generator_sd1_name = None
    vae_name = None
    safety_checker = None
    controlnet = None
    control_processor = "none"
    x_adapter = None
    return "done"

def update_infinite_xl(value):
    global infinite
    infinite = value
    return "done"

def update_upscaling_xl(value):
    global upscaling
    upscaling = value
    return "done"

def update_precision_xl(value):
    global dtype
    precision = value
    if precision == "full":
        dtype = torch.float32
    elif precision == "half":
        dtype = torch.bfloat16
    return "done"

def generate_xl(data, request_files, get_controlnet=None, clear_step_frames=None, generate_step_animation=None):
    global device
    global infinite
    global upscaling
    global generator_sd1
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
    x_adapt_model = data["x_adapt_model"] if "x_adapt_model" in data else "None"

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

    socketio.emit("image starting")

    use_x_adapter = False
    if x_adapt_model and x_adapt_model != "None":
        use_x_adapter = True
    generator = get_generator_xl(model_name, vae_name, mode, clip_skip, processing == "cpu", control_processor, get_controlnet, x_adapt_model)

    if freeu:
        generator.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)

    if sampler == "euler a":
        generator.scheduler = EulerAncestralDiscreteScheduler.from_config(generator.scheduler.config)
        if use_x_adapter: generator.scheduler_sd1_5 = EulerAncestralDiscreteScheduler.from_config(generator.scheduler_sd1_5.config)
    elif sampler == "euler":
        generator.scheduler = EulerDiscreteScheduler.from_config(generator.scheduler.config)
        if use_x_adapter: generator.scheduler_sd1_5 = EulerDiscreteScheduler.from_config(generator.scheduler_sd1_5.config)
    elif sampler == "dpm++":
        generator.scheduler = DPMSolverMultistepScheduler.from_config(generator.scheduler.config)
        if use_x_adapter: generator.scheduler_sd1_5 = DPMSolverMultistepScheduler.from_config(generator.scheduler_sd1_5.config)
    elif sampler == "ddim":
        generator.scheduler = DDIMScheduler.from_config(generator.scheduler.config)
        if use_x_adapter: generator.scheduler_sd1_5 = DDIMScheduler.from_config(generator.scheduler_sd1_5.config)
    elif sampler == "ddpm":
        generator.scheduler = DDPMScheduler.from_config(generator.scheduler.config)
        if use_x_adapter: generator.scheduler_sd1_5 = DDPMScheduler.from_config(generator.scheduler_sd1_5.config)
    elif sampler == "unipc":
        generator.scheduler = UniPCMultistepScheduler.from_config(generator.scheduler.config)
        if use_x_adapter: generator.scheduler_sd1_5 = UniPCMultistepScheduler.from_config(generator.scheduler_sd1_5.config)
    elif sampler == "deis":
        generator.scheduler = DEISMultistepScheduler.from_config(generator.scheduler.config)
        if use_x_adapter: generator.scheduler_sd1_5 = DEISMultistepScheduler.from_config(generator.scheduler_sd1_5.config)
    elif sampler == "heun":
        generator.scheduler = HeunDiscreteScheduler.from_config(generator.scheduler.config)
        if use_x_adapter: generator.scheduler_sd1_5 = HeunDiscreteScheduler.from_config(generator.scheduler_sd1_5.config)
    elif sampler == "lcm":
        generator.scheduler = LCMScheduler.from_config(generator.scheduler.config)
        if use_x_adapter: generator.scheduler_sd1_5 = LCMScheduler.from_config(generator.scheduler_sd1_5.config)
    
    if use_x_adapter:
        generator.scheduler = DDPMScheduler.from_config(generator.scheduler.config)
        generator.scheduler_sd1_5 = DDPMScheduler.from_config(generator.scheduler_sd1_5.config)
        generator.scheduler_sd1_5.config.timestep_spacing = "leading"

    if mode == "animatediff":
        generator.scheduler.beta_schedule = "linear"
        generator.scheduler.clip_sample = False

    if not use_x_adapter:
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
            if use_x_adapter:
                add_hypernet(generator.unet_sd1_5, hypernet)
            else:
                add_hypernet(generator.unet, hypernet)
        except ValueError:
            continue
    if not has_hypernet:
        if use_x_adapter:
            clear_hypernets(generator.unet_sd1_5)
        else:
            clear_hypernets(generator.unet)

    if use_x_adapter:
        unload_adapter_loras(generator)
        for lora in loras:
            lora_scale = float(lora["weight"])
            lora_path = os.path.join(get_models_dir(), lora["model"].replace("models/", ""))
            load_adapter_lora(generator, lora_path, lora_scale, device)
    else:
        has_lora = False
        generator.unfuse_lora()
        adapters = []
        adapter_weights = []
        for lora in loras:
            lora_scale = float(lora["weight"])
            weight_name = os.path.basename(lora["model"])
            lora_name = lora["name"]
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
            cross_attention_kwargs.pop("scale", None)

    generator.vae.enable_slicing()
    generator.vae.enable_tiling()
    if ip_processor == "off" and not has_hypernet:
        generator.enable_attention_slicing()
    
    conditioning = None
    negative_conditioning = None
    pooled = None
    negative_pooled = None
    conditioning_sd1 = None
    negative_conditioning_sd1 = None

    textual_inversion_manager = DiffusersTextualInversionManager(generator)

    returned_embeddings_type = ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED
    if clip_skip > 1:
        returned_embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED

    compel = Compel(tokenizer=[generator.tokenizer, generator.tokenizer_2] , text_encoder=[generator.text_encoder, generator.text_encoder_2], requires_pooled=[False, True],
                    returned_embeddings_type=returned_embeddings_type, textual_inversion_manager=textual_inversion_manager, truncate_long_prompts=True)
    conditioning, pooled = compel.build_conditioning_tensor(prompt)
    negative_conditioning, negative_pooled = compel.build_conditioning_tensor(negative_prompt)
    [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])

    if use_x_adapter:
        compel_sd1 = Compel(tokenizer=generator.tokenizer_sd1_5, text_encoder=generator.text_encoder_sd1_5, returned_embeddings_type=returned_embeddings_type,
                    textual_inversion_manager=textual_inversion_manager, truncate_long_prompts=False)
        conditioning_sd1 = compel_sd1.build_conditioning_tensor(prompt)
        negative_conditioning_sd1 = compel_sd1.build_conditioning_tensor(negative_prompt)
        [conditioning_sd1, negative_conditioning_sd1] = compel_sd1.pad_conditioning_tensors_to_same_length([conditioning_sd1, negative_conditioning_sd1])

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
            if use_x_adapter:
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
                    generator=get_seed_generator(seed, device),
                    callback=step_progress,
                    cross_attention_kwargs=cross_attention_kwargs,
                    prompt_embeds_sd1_5=conditioning_sd1, 
                    negative_prompt_embeds_sd1_5=negative_conditioning_sd1,
                    width_sd1_5=round(width/2),
                    height_sd1_5=round(height/2),
                    adapter_guidance_start=0.7,
                    adapter_condition_scale=1.0,
                    adapter_type="de"
                ).images[0]
            else:
                image = generator(
                    prompt_embeds=conditioning,
                    pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=negative_conditioning,
                    negative_pooled_prompt_embeds=negative_pooled,
                    ip_adapter_image=ip_adapter_image,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    num_images_per_prompt=1,
                    generator=get_seed_generator(seed, device),
                    clip_skip=clip_skip,
                    callback_on_step_end=step_progress,
                    cross_attention_kwargs=cross_attention_kwargs
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
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=negative_conditioning,
                negative_pooled_prompt_embeds=negative_pooled,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=get_seed_generator(seed, device),
                clip_skip=clip_skip,
                callback_on_step_end=step_progress,
                cross_attention_kwargs=cross_attention_kwargs
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
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=negative_conditioning,
                negative_pooled_prompt_embeds=negative_pooled,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=get_seed_generator(seed, device),
                clip_skip=clip_skip,
                callback_on_step_end=step_progress,
                cross_attention_kwargs=cross_attention_kwargs
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
            if use_x_adapter:
                image = generator(
                    image=control_image,
                    prompt_embeds=conditioning,
                    pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=negative_conditioning,
                    negative_pooled_prompt_embeds=negative_pooled,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=get_seed_generator(seed, device),
                    callback=step_progress,
                    cross_attention_kwargs=cross_attention_kwargs,
                    controlnet_conditioning_scale=control_scale,
                    guess_mode=guess_mode,
                    control_guidance_start=control_start,
                    control_guidance_end=control_end,
                    prompt_embeds_sd1_5=conditioning_sd1, 
                    negative_prompt_embeds_sd1_5=negative_conditioning_sd1,
                    width_sd1_5=round(width/2),
                    height_sd1_5=round(height/2),
                    adapter_guidance_start=0.7,
                    adapter_condition_scale=1.0,
                    adapter_type="de"
                ).images[0]
            else:
                image = generator(
                    image=control_image,
                    prompt_embeds=conditioning,
                    pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=negative_conditioning,
                    negative_pooled_prompt_embeds=negative_pooled,
                    ip_adapter_image=ip_adapter_image,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=get_seed_generator(seed, device),
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
            if generate_step_animation is not None:
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
            if use_x_adapter:
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
                    generator=get_seed_generator(seed, device),
                    callback=step_progress,
                    cross_attention_kwargs=cross_attention_kwargs,
                    controlnet_conditioning_scale=control_scale,
                    guess_mode=guess_mode,
                    control_guidance_start=control_start,
                    control_guidance_end=control_end,
                    prompt_embeds_sd1_5=conditioning_sd1, 
                    negative_prompt_embeds_sd1_5=negative_conditioning_sd1,
                    width_sd1_5=round(width/2),
                    height_sd1_5=round(height/2),
                    adapter_guidance_start=0.7,
                    adapter_condition_scale=1.0,
                    adapter_type="de"
                ).images[0]
            else:
                image = generator(
                    image=input_image,
                    control_image=control_image,
                    strength=denoise,
                    ip_adapter_image=ip_adapter_image,
                    prompt_embeds=conditioning,
                    pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=negative_conditioning,
                    negative_pooled_prompt_embeds=negative_pooled,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=get_seed_generator(seed, device),
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
            if generate_step_animation is not None:
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
                    return images
            image = generator(
                image=input_image,
                mask_image=input_mask,
                control_image=control_image,
                strength=denoise,
                ip_adapter_image=ip_adapter_image,
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=negative_conditioning,
                negative_pooled_prompt_embeds=negative_pooled,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=get_seed_generator(seed, device),
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
            if generate_step_animation is not None:
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
                ref_image=input_image,
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=negative_conditioning,
                negative_pooled_prompt_embeds=negative_pooled,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=get_seed_generator(seed, device),
                callback=step_progress,
                cross_attention_kwargs=cross_attention_kwargs,
                style_fidelity=style_fidelity,
                reference_attn=True,
                reference_adain=True
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
            info = {"Prompt": prompt, "Negative Prompt": negative_prompt, "Size": f"{width}x{height}", "Denoise": denoise,
                    "Model": model_name, "VAE": vae_name, "Steps": steps, "CFG": cfg, "Sampler": sampler, "Clip Skip": clip_skip, 
                    "Seed": seed}
            append_info(out_path, info)
            socketio.emit("image complete", {"image": f"/outputs/local/{folder}/{os.path.basename(out_path)}", "needs_watermark": watermark})
            images.append(out_path)
            seed += 1
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.mps.empty_cache()
    except:
        pass
    if infinite:
        socketio.emit("repeat generation")
    return images