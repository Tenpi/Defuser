from __main__ import app, socketio
from .functions import is_image, is_text, get_number_from_filename, get_sources
from .convert_to_ckpt import convert_to_ckpt
import functools
import gc
import itertools
import math
import os
import random
import shutil
from pathlib import Path
from torch.utils.data import Dataset
import safetensors.torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from torchvision import transforms
from tqdm.auto import tqdm
from itertools import chain
from diffusers import (
    DDPMScheduler,
    LCMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import (
    check_min_version,
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_kohya,
    is_wandb_available,
)
from diffusers.optimization import get_scheduler

MAX_SEQ_LENGTH = 77

logger = get_logger(__name__)
pipeline = None

def autocast_decorator(autocast_instance, func):
  @functools.wraps(func)
  def decorate_autocast(*args, **kwargs):
    with autocast_instance:
      return func(*args, **kwargs)
  decorate_autocast.__script_unsupported = '@autocast() decorator is not supported in script mode'
  return decorate_autocast

class totally_legit_autocast:
  def __init__(
    self,
    device_type : str,
    dtype = None,
    enabled : bool = True,
    cache_enabled = None,
  ): pass
  def __enter__(self): pass
  def __exit__(self, exc_type, exc_val, exc_tb): pass
  def __call__(self, func):
    if torch._jit_internal.is_scripting():
      return func
    return autocast_decorator(self, func)
  
def step_progress(self, step: int, timestep: int, call_dict: dict):
    global pipeline
    latent = None
    if type(call_dict) is torch.Tensor:
        latent = call_dict
    else:
        latent = call_dict.get("latents")
    with torch.no_grad():
        latent = 1 / 0.18215 * latent
        image = pipeline.vae.decode(latent).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = pipeline.numpy_to_pil(image)[0]
        w, h = image.size
        pixels = list(image.convert("RGBA").getdata())
        pixels = list(chain(*pixels))
        total_steps = 10
        socketio.emit("train image progress", {"step": step, "total_step": total_steps, "width": w, "height": h, "image": pixels})
    return call_dict

def get_module_kohya_state_dict(module, prefix: str, dtype: torch.dtype, adapter_name: str = "default"):
    kohya_ss_state_dict = {}
    for peft_key, weight in get_peft_model_state_dict(module, adapter_name=adapter_name).items():
        kohya_key = peft_key.replace("base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)

        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(module.peft_config[adapter_name].lora_alpha).to(dtype)

    return kohya_ss_state_dict

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        class_prompt,
        dataset_name,
        dataset_config_name,
        cache_dir,
        image_column,
        caption_column,
        train_text_encoder_ti,
        class_data_root=None,
        class_num=None,
        token_abstraction_dict=None,  # token mapping for textual inversion
        size=1024,
        repeats=1,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop

        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt
        self.token_abstraction_dict = token_abstraction_dict
        self.train_text_encoder_ti = train_text_encoder_ti
        # if --dataset_name is provided or a metadata jsonl file is provided in the local --instance_data directory,
        # we load the training data using load_dataset
        if dataset_name is not None:
            files = os.listdir(dataset_name)
            image_files = list(filter(lambda file: is_image(file), files))
            image_files = sorted(image_files, key=lambda x: get_number_from_filename(x), reverse=False)
            instance_images = list(map(lambda file: Image.open(os.path.join(dataset_name, file)), image_files))

            text_files = list(filter(lambda file: is_text(file), files))
            text_files = sorted(text_files, key=lambda x: get_number_from_filename(x), reverse=False)
            custom_instance_prompts = []
            for text_file in text_files:
                f = open(os.path.normpath(os.path.join(dataset_name, text_file)))
                caption = f.read()
                custom_instance_prompts.append(caption)
                f.close()

            if len(custom_instance_prompts) == 0:
                logger.info(
                    "No captions provided, defaulting to instance_prompt for all images."
                )
                self.custom_instance_prompts = None
            else:
                self.custom_instance_prompts = []
                for caption in custom_instance_prompts:
                    self.custom_instance_prompts.extend(itertools.repeat(caption, repeats))
        else:
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError("Instance images root doesn't exists.")
            instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir())]
            self.custom_instance_prompts = None

        self.instance_images = []
        for img in instance_images:
            self.instance_images.extend(itertools.repeat(img, repeats))
        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.instance_images[index % self.num_instance_images]
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            if caption:
                if self.train_text_encoder_ti:
                    # replace instances of --token_abstraction in caption with the new tokens: "<si><si+1>" etc.
                    for token_abs, token_replacement in self.token_abstraction_dict.items():
                        caption = caption.replace(token_abs, "".join(token_replacement))
                example["instance_prompt"] = caption
            else:
                example["instance_prompt"] = self.instance_prompt

        else:  # costum prompts were provided, but length does not match size of image dataset
            example["instance_prompt"] = self.instance_prompt

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt"] = self.class_prompt

        return example

def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts}
    return batch

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example
    
def log_validation(vae, unet, args, accelerator, weight_dtype, step, name="target"):
    global pipeline
    logger.info("Running validation... ")

    xl = False
    if "XL" in args.pretrained_teacher_model:
        xl = True

    unet = accelerator.unwrap_model(unet)
    if xl:
        pipeline = StableDiffusionXLPipeline.from_single_file(
            args.pretrained_teacher_model,
            vae=vae,
            unet=unet,
            torch_dtype=weight_dtype,
        )
    else:
        pipeline = StableDiffusionPipeline.from_single_file(
            args.pretrained_teacher_model,
            vae=vae,
            unet=unet,
            torch_dtype=weight_dtype,
        )
    pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
    #pipeline.set_progress_bar_config(disable=True)

    if args.save_lora:
        lora_state_dict = get_module_kohya_state_dict(unet, "lora_unet", weight_dtype)
        pipeline.load_lora_weights(lora_state_dict)
        pipeline.fuse_lora()

    pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)
    pipeline.enable_attention_slicing()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images = pipeline(
        prompt=args.validation_prompt,
        num_inference_steps=4,
        num_images_per_prompt=1,
        generator=generator,
        callback_on_step_end=step_progress
    ).images
    gc.collect()
    torch.cuda.empty_cache()
    torch.mps.empty_cache()
    return images

# From LatentConsistencyModel.get_guidance_scale_embedding
def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`torch.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


# From LCMScheduler.get_scalings_for_boundary_condition_discrete
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    scaled_timestep = timestep_scaling * timestep
    c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
    c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5
    return c_skip, c_out

# Compare LCMScheduler.step, Step 4
def get_predicted_original_sample(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_x_0

# Based on step 4 in DDIMScheduler.step
def get_predicted_noise(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_epsilon = model_output
    elif prediction_type == "sample":
        pred_epsilon = (sample - alphas * model_output) / sigmas
    elif prediction_type == "v_prediction":
        pred_epsilon = alphas * model_output + sigmas * sample
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_epsilon

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t.type(torch.int64))
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device, dtype):
        self.ddim_timesteps = self.ddim_timesteps.to(device, dtype=dtype)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device, dtype=dtype)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device, dtype=dtype)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev

@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device))[0]

    return prompt_embeds

def encode_prompt_xl(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

def main(args):
    global pipeline
    name = args.name

    xl = False
    if "XL" in args.pretrained_teacher_model:
        xl = True

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        split_batches=True
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)

    # DDPMScheduler calculates the alpha and sigma noise schedules (based on the alpha bars) for us
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    # Initialize the DDIM ODE solver for distillation.
    solver = DDIMSolver(
        noise_scheduler.alphas_cumprod.numpy(),
        timesteps=noise_scheduler.config.num_train_timesteps,
        ddim_timesteps=args.num_ddim_timesteps,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if xl:
        pipeline = StableDiffusionXLPipeline.from_single_file(args.pretrained_teacher_model, torch_dtype=weight_dtype)
    else:
        pipeline = StableDiffusionPipeline.from_single_file(args.pretrained_teacher_model, torch_dtype=weight_dtype)

    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    if xl:
        tokenizer_2 = pipeline.tokenizer_2
        text_encoder_2 = pipeline.text_encoder_2

    vae = pipeline.vae
    teacher_unet = pipeline.unet

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    teacher_unet.requires_grad_(False)

    # 7. Create online student U-Net. This will be updated by the optimizer (e.g. via backpropagation.)
    # Add `time_cond_proj_dim` to the student U-Net if `teacher_unet.config.time_cond_proj_dim` is None
    time_cond_proj_dim = (
        teacher_unet.config.time_cond_proj_dim
        if teacher_unet.config.time_cond_proj_dim is not None
        else args.unet_time_cond_proj_dim
    )
    unet = UNet2DConditionModel.from_config(teacher_unet.config, time_cond_proj_dim=time_cond_proj_dim)
    unet.load_state_dict(teacher_unet.state_dict(), strict=False)
    unet.train()

    # 8. Create target student U-Net. This will be updated via EMA updates (polyak averaging).
    # Initialize from (online) unet
    target_unet = UNet2DConditionModel.from_config(unet.config)
    target_unet.load_state_dict(unet.state_dict())
    target_unet.train()
    target_unet.requires_grad_(False)

    if args.save_lora:
        if args.lora_target_modules is not None:
            lora_target_modules = [module_key.strip() for module_key in args.lora_target_modules.split(",")]
        else:
            lora_target_modules = [
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "ff.net.0.proj",
                "ff.net.2",
                "conv1",
                "conv2",
                "conv_shortcut",
                "downsamplers.0.conv",
                "upsamplers.0.conv",
                "time_emb_proj",
            ]
        lora_config = LoraConfig(
            r=args.lora_rank,
            target_modules=lora_target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        target_unet = get_peft_model(target_unet, lora_config)

    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )
    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device)
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Move teacher_unet to device, optionally cast to weight_dtype
    target_unet.to(accelerator.device)
    teacher_unet.to(accelerator.device)
    if args.cast_teacher_unet:
        teacher_unet.to(dtype=weight_dtype)

    # Also move the alpha and sigma noise schedules to accelerator.device.
    alpha_schedule = alpha_schedule.to(accelerator.device)
    sigma_schedule = sigma_schedule.to(accelerator.device)
    solver = solver.to(accelerator.device, dtype=weight_dtype)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 13. Dataset creation and data processing
    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.
    def compute_embeddings(prompt_batch, proportion_empty_prompts, text_encoder, tokenizer, is_train=True):
        prompt_embeds = encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train)
        return {"prompt_embeds": prompt_embeds}
    
    def compute_embeddings_xl(
        prompt_batch, original_sizes, crop_coords, proportion_empty_prompts, text_encoders, tokenizers, is_train=True
    ):
        target_size = (args.resolution, args.resolution)
        original_sizes = list(map(list, zip(*original_sizes)))
        crops_coords_top_left = list(map(list, zip(*crop_coords)))

        original_sizes = torch.tensor(original_sizes, dtype=torch.long)
        crops_coords_top_left = torch.tensor(crops_coords_top_left, dtype=torch.long)

        prompt_embeds, pooled_prompt_embeds = encode_prompt_xl(
            prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train
        )
        add_text_embeds = pooled_prompt_embeds

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
        add_time_ids = torch.cat([original_sizes, crops_coords_top_left, add_time_ids], dim=-1)
        add_time_ids = add_time_ids.to(accelerator.device, dtype=prompt_embeds.dtype)

        prompt_embeds = prompt_embeds.to(accelerator.device)
        add_text_embeds = add_text_embeds.to(accelerator.device)
        unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=None,
        instance_prompt=None,
        class_prompt=None,
        dataset_name=args.dataset_name,
        dataset_config_name=None,
        cache_dir=None,
        image_column=None,
        train_text_encoder_ti=None,
        caption_column=None,
        class_data_root=None,
        token_abstraction_dict=None,
        class_num=None,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    text_encoders = [text_encoder]
    tokenizers = [tokenizer]
    if xl:
        text_encoders = [text_encoder, text_encoder_2]
        tokenizers = [tokenizer, tokenizer_2]

    compute_embeddings_fn = None
    if xl:
        compute_embeddings_fn = functools.partial(
            compute_embeddings_xl,
            proportion_empty_prompts=0,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
        )
    else:
        compute_embeddings_fn = functools.partial(
            compute_embeddings,
            proportion_empty_prompts=0,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    learning_function = args.lr_scheduler
    power = 1.0
    if args.lr_scheduler == "quadratic":
        args.lr_scheduler = "polynomial"
        power = 2.0
    elif args.lr_scheduler == "cubic":
        args.lr_scheduler = "polynomial"
        power = 3.0
    elif args.lr_scheduler == "quartic":
        args.lr_scheduler = "polynomial"
        power = 4.0

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=power
    )

    unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    uncond_input_ids = tokenizer(
        [""] * args.train_batch_size, return_tensors="pt", padding="max_length", max_length=77
    ).input_ids.to(accelerator.device)
    uncond_prompt_embeds = text_encoder(uncond_input_ids)[0]

    # 16. Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = list(filter(lambda d: os.path.isdir(os.path.join(args.output_dir, d)), dirs))
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            socketio.emit("train progress", {"step": global_step + 1, "total_step": args.max_train_steps, "epoch": epoch + 1, "total_epoch": args.num_train_epochs})
            with accelerator.accumulate(unet):
                text = batch["prompts"]
                image = batch["pixel_values"]

                orig_size=[(args.resolution, args.resolution)]
                crop_coords=[(0,0)]

                image = image.to(accelerator.device, non_blocking=True)
                if xl:
                    encoded_text = compute_embeddings_fn(text, orig_size, crop_coords)
                else:
                    encoded_text = compute_embeddings_fn(text)

                pixel_values = image.to(dtype=weight_dtype)
                if vae.dtype != weight_dtype:
                    vae.to(dtype=weight_dtype)

                # encode pixel values with batch size of at most args.vae_encode_batch_size
                latents = []
                for i in range(0, pixel_values.shape[0], args.vae_encode_batch_size):
                    latents.append(vae.encode(pixel_values[i : i + args.vae_encode_batch_size]).latent_dist.sample())
                latents = torch.cat(latents, dim=0)

                latents = latents * vae.config.scaling_factor
                latents = latents.to(weight_dtype)
                bsz = latents.shape[0]

                # 2. Sample a random timestep for each image t_n from the ODE solver timesteps without bias.
                # For the DDIM solver, the timestep schedule is [T - 1, T - k - 1, T - 2 * k - 1, ...]
                topk = noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps
                index = torch.randint(0, args.num_ddim_timesteps, (bsz,), device=latents.device).long()
                start_timesteps = solver.ddim_timesteps[index]
                timesteps = start_timesteps - topk
                timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

                # 3. Get boundary scalings for start_timesteps and (end) timesteps.
                c_skip_start, c_out_start = scalings_for_boundary_conditions(
                    start_timesteps, timestep_scaling=args.timestep_scaling_factor
                )
                c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
                c_skip, c_out = scalings_for_boundary_conditions(
                    timesteps, timestep_scaling=args.timestep_scaling_factor
                )
                c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

                # 4. Sample noise from the prior and add it to the latents according to the noise magnitude at each
                # timestep (this is the forward diffusion process) [z_{t_{n + k}} in Algorithm 1]
                noise = torch.randn_like(latents)
                noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps.long())

                # 5. Sample a random guidance scale w from U[w_min, w_max] and embed it
                w = (args.w_max - args.w_min) * torch.rand((bsz,)) + args.w_min
                w_embedding = guidance_scale_embedding(w, embedding_dim=time_cond_proj_dim)
                w = w.reshape(bsz, 1, 1, 1)
                # Move to U-Net device and dtype
                w = w.to(device=latents.device, dtype=latents.dtype)
                w_embedding = w_embedding.to(device=latents.device, dtype=latents.dtype)

                # 6. Prepare prompt embeds and unet_added_conditions
                prompt_embeds = encoded_text.pop("prompt_embeds")

                # 7. Get online LCM prediction on z_{t_{n + k}} (noisy_model_input), w, c, t_{n + k} (start_timesteps)
                noise_pred = unet(
                    noisy_model_input,
                    start_timesteps,
                    timestep_cond=w_embedding,
                    encoder_hidden_states=prompt_embeds.float(),
                    added_cond_kwargs=encoded_text,
                ).sample

                pred_x_0 = get_predicted_original_sample(
                    noise_pred,
                    start_timesteps,
                    noisy_model_input,
                    noise_scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )

                model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0

                # 8. Compute the conditional and unconditional teacher model predictions to get CFG estimates of the
                # predicted noise eps_0 and predicted original sample x_0, then run the ODE solver using these
                # estimates to predict the data point in the augmented PF-ODE trajectory corresponding to the next ODE
                # solver timestep.
                with torch.no_grad():
                    # 1. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and conditional embedding c
                    cond_teacher_output = teacher_unet(
                        noisy_model_input.to(weight_dtype),
                        start_timesteps,
                        encoder_hidden_states=prompt_embeds.to(weight_dtype),
                    ).sample
                    cond_pred_x0 = get_predicted_original_sample(
                        cond_teacher_output,
                        start_timesteps,
                        noisy_model_input,
                        noise_scheduler.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )
                    cond_pred_noise = get_predicted_noise(
                        cond_teacher_output,
                        start_timesteps,
                        noisy_model_input,
                        noise_scheduler.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )

                    # 2. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and unconditional embedding 0
                    uncond_teacher_output = teacher_unet(
                        noisy_model_input.to(weight_dtype),
                        start_timesteps,
                        encoder_hidden_states=uncond_prompt_embeds.to(weight_dtype),
                    ).sample
                    uncond_pred_x0 = get_predicted_original_sample(
                        uncond_teacher_output,
                        start_timesteps,
                        noisy_model_input,
                        noise_scheduler.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )
                    uncond_pred_noise = get_predicted_noise(
                        uncond_teacher_output,
                        start_timesteps,
                        noisy_model_input,
                        noise_scheduler.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )

                    # 3. Calculate the CFG estimate of x_0 (pred_x0) and eps_0 (pred_noise)
                    # Note that this uses the LCM paper's CFG formulation rather than the Imagen CFG formulation
                    pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                    pred_noise = cond_pred_noise + w * (cond_pred_noise - uncond_pred_noise)
                    # 4. Run one step of the ODE solver to estimate the next point x_prev on the
                    # augmented PF-ODE trajectory (solving backward in time)
                    # Note that the DDIM step depends on both the predicted x_0 and source noise eps_0.
                    x_prev = solver.ddim_step(pred_x0, pred_noise, index)

                # 9. Get target LCM prediction on x_prev, w, c, t_n (timesteps)
                with torch.no_grad():
                    target_noise_pred = target_unet(
                        x_prev.float(),
                        timesteps,
                        timestep_cond=w_embedding,
                        encoder_hidden_states=prompt_embeds.float(),
                    ).sample
                    pred_x_0 = get_predicted_original_sample(
                        target_noise_pred,
                        timesteps,
                        x_prev,
                        noise_scheduler.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )
                    target = c_skip * x_prev + c_out * pred_x_0

                if args.loss_type == "l2":
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                elif args.loss_type == "huber":
                    loss = torch.mean(
                        torch.sqrt((model_pred.float() - target.float()) ** 2 + args.huber_c**2) - args.huber_c
                    )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                # 12. Make EMA update to target student model parameters (`target_unet`)
                update_ema(target_unet.parameters(), unet.parameters(), args.ema_decay)
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"{name}-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                        if args.save_lora:
                            unet = accelerator.unwrap_model(unet)
                            unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
                            if xl:
                                StableDiffusionXLPipeline.save_lora_weights(
                                    save_directory=args.output_dir,
                                    unet_lora_layers=unet_lora_layers
                                )
                            else:
                                StableDiffusionPipeline.save_lora_weights(
                                    save_directory=args.output_dir,
                                    unet_lora_layers=unet_lora_layers
                                )
                            temp_file = f"{args.output_dir}/pytorch_lora_weights.safetensors"
                            lora_state_dict = safetensors.torch.load_file(temp_file)
                            peft_state_dict = convert_all_state_dict_to_peft(lora_state_dict)
                            kohya_state_dict = convert_state_dict_to_kohya(peft_state_dict)
                            metadata = {
                                "name": name,
                                "steps": str(global_step),
                                "epochs": str(epoch),
                                "checkpoint": os.path.basename(args.pretrained_teacher_model),
                                "images": str(len(train_dataloader)),
                                "learning_rate": str(args.learning_rate),
                                "gradient_accumulation_steps": str(args.gradient_accumulation_steps),
                                "learning_function": learning_function,
                                "sources": "\n".join(args.sources)
                            }
                            safetensors.torch.save_file(kohya_state_dict, f"{args.output_dir}/{name}-{global_step}.safetensors", metadata=metadata)
                            os.remove(temp_file)
                        else:
                            unet = accelerator.unwrap_model(unet)
                            if xl:
                                pipeline = StableDiffusionXLPipeline.from_single_file(
                                    args.pretrained_teacher_model,
                                    unet=unet,
                                    torch_dtype=weight_dtype
                                )
                            else:
                                pipeline = StableDiffusionPipeline.from_single_file(
                                    args.pretrained_teacher_model,
                                    unet=unet,
                                    torch_dtype=weight_dtype
                                )
                            scheduler_args = {}
                            if "variance_type" in pipeline.scheduler.config:
                                variance_type = pipeline.scheduler.config.variance_type
                                if variance_type in ["learned", "learned_range"]:
                                    variance_type = "fixed_small"
                                scheduler_args["variance_type"] = variance_type
                            pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)

                            temp = f"{args.output_dir}/temp"
                            pipeline.save_pretrained(temp)
                            metadata = {
                                "name": name,
                                "steps": str(global_step),
                                "epochs": str(epoch),
                                "checkpoint": os.path.basename(args.pretrained_teacher_model),
                                "images": str(len(train_dataloader)),
                                "learning_rate": str(args.learning_rate),
                                "gradient_accumulation_steps": str(args.gradient_accumulation_steps),
                                "learning_function": learning_function,
                                "sources": args.sources
                            }
                            convert_to_ckpt(temp, f"{args.output_dir}/{name}-{global_step}.ckpt", metadata=metadata)
                            shutil.rmtree(temp)

                    if global_step % args.validation_steps == 0:
                        images = log_validation(vae, target_unet, args, accelerator, weight_dtype, global_step, "target")
                        save_img_path = os.path.join(args.output_dir, f"{name}-{global_step}.png")
                        images[0].save(save_img_path)
                        socketio.emit("train image complete", {"image": save_img_path})
                        

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.save_lora:
            unet = accelerator.unwrap_model(unet)
            unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
            if xl:
                StableDiffusionXLPipeline.save_lora_weights(
                    save_directory=args.output_dir,
                    unet_lora_layers=unet_lora_layers
                )
            else:
                StableDiffusionPipeline.save_lora_weights(
                    save_directory=args.output_dir,
                    unet_lora_layers=unet_lora_layers
                )
            temp_file = f"{args.output_dir}/pytorch_lora_weights.safetensors"
            lora_state_dict = safetensors.torch.load_file(temp_file)
            peft_state_dict = convert_all_state_dict_to_peft(lora_state_dict)
            kohya_state_dict = convert_state_dict_to_kohya(peft_state_dict)
            metadata = {
                "name": name,
                "steps": str(global_step),
                "epochs": str(epoch),
                "checkpoint": os.path.basename(args.pretrained_teacher_model),
                "images": str(len(train_dataloader)),
                "learning_rate": str(args.learning_rate),
                "gradient_accumulation_steps": str(args.gradient_accumulation_steps),
                "learning_function": learning_function,
                "sources": "\n".join(args.sources)
            }
            safetensors.torch.save_file(kohya_state_dict, f"{args.output_dir}/{name}.safetensors", metadata=metadata)
            os.remove(temp_file)
        else:
            unet = accelerator.unwrap_model(unet)
            if xl:
                pipeline = StableDiffusionXLPipeline.from_single_file(
                    args.pretrained_teacher_model,
                    unet=unet,
                    torch_dtype=weight_dtype
                )
            else:
                pipeline = StableDiffusionPipeline.from_single_file(
                    args.pretrained_teacher_model,
                    unet=unet,
                    torch_dtype=weight_dtype
                )
            scheduler_args = {}
            if "variance_type" in pipeline.scheduler.config:
                variance_type = pipeline.scheduler.config.variance_type
                if variance_type in ["learned", "learned_range"]:
                    variance_type = "fixed_small"
                scheduler_args["variance_type"] = variance_type
            pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)

            temp = f"{args.output_dir}/temp"
            pipeline.save_pretrained(temp)
            metadata = {
                "name": name,
                "steps": str(global_step),
                "epochs": str(epoch),
                "checkpoint": os.path.basename(args.pretrained_teacher_model),
                "images": str(len(train_dataloader)),
                "learning_rate": str(args.learning_rate),
                "gradient_accumulation_steps": str(args.gradient_accumulation_steps),
                "learning_function": learning_function,
                "sources": args.sources
            }
            convert_to_ckpt(temp, f"{args.output_dir}/{name}.ckpt", metadata=metadata)
            shutil.rmtree(temp)
    accelerator.end_training()

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_options(name, model_name, train_data, output, max_train_steps, learning_rate, resolution, save_steps, 
    gradient_accumulation_steps, validation_prompt, validation_steps, lr_scheduler, save_lora, rank):
    args = {}
    args["save_lora"] = save_lora
    args["lora_rank"] = rank
    args["lora_alpha"] = round(rank/2)
    args["lora_dropout"] = 0.0
    args["lora_target_modules"] = None
    args["name"] = name
    args["pretrained_teacher_model"] = model_name
    args["pretrained_vae_model_name_or_path"] = None
    args["output_dir"] = output
    args["cache_dir"] = None
    args["dataset_name"] = train_data
    args["seed"] = None
    args["checkpointing_steps"] = save_steps
    args["checkpoints_total_limit"] = None
    args["resume_from_checkpoint"] = "latest"
    args["resolution"] = resolution
    args["interpolation_type"] = "bilinear"
    args["center_crop"] = True
    args["random_flip"] = True
    args["dataloader_num_workers"] = 0
    args["train_batch_size"] = 1
    args["num_train_epochs"] = 1
    args["max_train_steps"] = max_train_steps
    args["max_train_samples"] = None
    args["learning_rate"] = learning_rate
    args["scale_lr"] = False
    args["lr_scheduler"] = lr_scheduler
    args["lr_warmup_steps"] = round(max_train_steps * 0.05)
    args["gradient_accumulation_steps"] = gradient_accumulation_steps
    args["use_8bit_adam"] = False
    args["adam_beta1"] = 0.9
    args["adam_beta2"] = 0.999
    args["adam_weight_decay"] = 1e-2
    args["adam_epsilon"] = 1e-08
    args["max_grad_norm"] = 1.0
    args["proportion_empty_prompts"] = 0
    args["w_min"] = 5.0
    args["w_max"] = 15.0
    args["num_ddim_timesteps"] = 50
    args["loss_type"] = "l2"
    args["huber_c"] = 0.001
    args["unet_time_cond_proj_dim"] = 256
    args["vae_encode_batch_size"] = 32
    args["timestep_scaling_factor"] = 10.0
    args["ema_decay"] = 0.95
    args["mixed_precision"] = "no"
    args["allow_tf32"] = True
    args["cast_teacher_unet"] = True
    args["gradient_checkpointing"] = True
    args["local_rank"] = 1
    args["validation_steps"] = validation_steps
    args["validation_prompt"] = validation_prompt
    args["ddpm_num_steps"] = 1000
    args["ddpm_beta_schedule"] = "linear"
    args["lr_num_cycles"] = 3
    args["repeats"] = 1
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args["local_rank"]:
        args["local_rank"] = env_local_rank
    return DotDict(args)

def train_lcm(images, name, model_name, train_data, output, num_train_epochs, learning_rate, resolution, save_steps, 
    gradient_accumulation_steps, validation_prompt, validation_steps, lr_scheduler,  save_lora, rank):

    if not model_name: model_name = ""
    if not train_data: train_data = ""
    if not name: name = ""
    if not output: output = ""
    if not num_train_epochs: num_train_epochs = 20
    if not validation_steps: validation_steps = 500
    if not save_steps: save_steps = 500
    if not learning_rate: learning_rate = 1e-4
    if not resolution: resolution = 256
    if not validation_prompt: validation_prompt = ""
    if not lr_scheduler: lr_scheduler = "constant"
    if not gradient_accumulation_steps: gradient_accumulation_steps = 1
    if save_lora is None: save_lora = True
    if not rank: rank = 32

    steps_per_epoch = math.ceil(len(images) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * steps_per_epoch
    #save_steps = save_epochs * steps_per_epoch
    #validation_steps = validation_epochs * steps_per_epoch

    options = get_options(name, model_name, train_data, output, max_train_steps, learning_rate, resolution, save_steps, 
    gradient_accumulation_steps, validation_prompt, validation_steps, lr_scheduler, save_lora, rank)

    options.sources = get_sources(train_data)

    main(options)