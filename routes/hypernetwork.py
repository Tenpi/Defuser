from __main__ import app, socketio
from .functions import is_image, is_text, get_number_from_filename, get_sources
from .hypernet import create_hypernetwork, add_hypernet, clear_hypernets
import argparse
import gc
import hashlib
import itertools
import logging
import math
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import List, Optional
from itertools import chain
from transformers import CLIPTextModel

import numpy as np
import torch
import torch.nn.functional as F

import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import (
    check_min_version,
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_kohya,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_xformers_available
import functools
import copy

dirname = os.path.dirname(__file__)
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

"""
def import_model_class_from_model_name_or_path(pipeline):
    text_encoder_config = pipeline.text_encoder.config

    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")
"""

class TokenEmbeddingsHandler:
    def __init__(self, text_encoders, tokenizers):
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers

        self.train_ids: Optional[torch.Tensor] = None
        self.inserting_toks: Optional[List[str]] = None
        self.embeddings_settings = {}

    def initialize_new_tokens(self, inserting_toks: List[str]):
        idx = 0
        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            assert isinstance(inserting_toks, list), "inserting_toks should be a list of strings."
            assert all(
                isinstance(tok, str) for tok in inserting_toks
            ), "All elements in inserting_toks should be strings."

            self.inserting_toks = inserting_toks
            special_tokens_dict = {"additional_special_tokens": self.inserting_toks}
            tokenizer.add_special_tokens(special_tokens_dict)
            text_encoder.resize_token_embeddings(len(tokenizer))

            self.train_ids = tokenizer.convert_tokens_to_ids(self.inserting_toks)

            # random initialization of new tokens
            std_token_embedding = text_encoder.text_model.embeddings.token_embedding.weight.data.std()

            print(f"{idx} text encodedr's std_token_embedding: {std_token_embedding}")

            text_encoder.text_model.embeddings.token_embedding.weight.data[self.train_ids] = (
                torch.randn(len(self.train_ids), text_encoder.text_model.config.hidden_size)
                .to(device=self.device)
                .to(dtype=self.dtype)
                * std_token_embedding
            )
            self.embeddings_settings[
                f"original_embeddings_{idx}"
            ] = text_encoder.text_model.embeddings.token_embedding.weight.data.clone()
            self.embeddings_settings[f"std_token_embedding_{idx}"] = std_token_embedding

            inu = torch.ones((len(tokenizer),), dtype=torch.bool)
            inu[self.train_ids] = False

            self.embeddings_settings[f"index_no_updates_{idx}"] = inu

            print(self.embeddings_settings[f"index_no_updates_{idx}"].shape)

            idx += 1

    def save_embeddings(self, file_path: str):
        assert self.train_ids is not None, "Initialize new tokens before saving embeddings."
        tensors = {}
        # text_encoder_0 - CLIP ViT-L/14, text_encoder_1 -  CLIP ViT-G/14 - TODO - change for sd
        idx_to_text_encoder_name = {0: "clip_l", 1: "clip_g"}
        for idx, text_encoder in enumerate(self.text_encoders):
            assert text_encoder.text_model.embeddings.token_embedding.weight.data.shape[0] == len(
                self.tokenizers[0]
            ), "Tokenizers should be the same."
            new_token_embeddings = text_encoder.text_model.embeddings.token_embedding.weight.data[self.train_ids]

            # New tokens for each text encoder are saved under "clip_l" (for text_encoder 0), "clip_g" (for
            # text_encoder 1) to keep compatible with the ecosystem.
            # Note: When loading with diffusers, any name can work - simply specify in inference
            tensors[idx_to_text_encoder_name[idx]] = new_token_embeddings
            # tensors[f"text_encoders_{idx}"] = new_token_embeddings

        save_file(tensors, file_path)

    @property
    def dtype(self):
        return self.text_encoders[0].dtype

    @property
    def device(self):
        return self.text_encoders[0].device

    @torch.no_grad()
    def retract_embeddings(self):
        for idx, text_encoder in enumerate(self.text_encoders):
            index_no_updates = self.embeddings_settings[f"index_no_updates_{idx}"]
            text_encoder.text_model.embeddings.token_embedding.weight.data[index_no_updates] = (
                self.embeddings_settings[f"original_embeddings_{idx}"][index_no_updates]
                .to(device=text_encoder.device)
                .to(dtype=text_encoder.dtype)
            )

            # for the parts that were updated, we need to normalize them
            # to have the same std as before
            std_token_embedding = self.embeddings_settings[f"std_token_embedding_{idx}"]

            index_updates = ~index_no_updates
            new_embeddings = text_encoder.text_model.embeddings.token_embedding.weight.data[index_updates]
            off_ratio = std_token_embedding / new_embeddings.std()

            new_embeddings = new_embeddings * (off_ratio**0.1)
            text_encoder.text_model.embeddings.token_embedding.weight.data[index_updates] = new_embeddings


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
                f = open(os.path.join(dataset_name, text_file))
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


def tokenize_prompt(tokenizer, prompt, add_special_tokens=False):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

    return prompt_embeds[0]

def encode_prompt_xl(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

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
    name = args.instance_prompt

    xl = False
    if "XL" in args.pretrained_model_name_or_path:
        xl = True

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if torch.backends.mps.is_available():
        try:
            torch.autocast(enabled=False, device_type='mps')
        except:
            torch.autocast = totally_legit_autocast

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if xl:
        pipeline = StableDiffusionXLPipeline.from_single_file(args.pretrained_model_name_or_path, torch_dtype=weight_dtype)
    else:
        pipeline = StableDiffusionPipeline.from_single_file(args.pretrained_model_name_or_path, torch_dtype=weight_dtype)

    tokenizer = pipeline.tokenizer
    tokenizer_two = None
    if xl:
        tokenizer_two = pipeline.tokenizer_2
    noise_scheduler = pipeline.scheduler #DDPMScheduler.from_config(pipeline.scheduler.config)
    text_encoder = pipeline.text_encoder
    text_encoder_two = None
    if xl:
        text_encoder_two = pipeline.text_encoder_2

    vae = pipeline.vae
    vae_scaling_factor = vae.config.scaling_factor

    unet = pipeline.unet

    # We only train the hypernetwork layers
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    if xl:
        text_encoder_two.requires_grad_(False)

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)

    # The VAE is always in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=torch.float32)

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    if xl:
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Add Hypernetwork
    hypernetwork = create_hypernetwork(name, enable_sizes=args.sizes)
    hypernetwork.to(accelerator.device)
    add_hypernet(unet, hypernetwork)

    weights = hypernetwork.weights()
    hypernetwork.train()

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [unet]
        for model in models:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(torch.float32)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # If neither --train_text_encoder nor --train_text_encoder_ti, text_encoders remain frozen during training
    freeze_text_encoder = not (args.train_text_encoder or args.train_text_encoder_ti)

    params_to_optimize = weights

    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warn(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warn(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
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
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warn(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warn(
                f"Learning rates were provided both for the unet and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one to be
            # --learning_rate
            params_to_optimize[1]["lr"] = args.learning_rate

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_prompt=args.class_prompt,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        cache_dir=args.cache_dir,
        image_column=args.image_column,
        train_text_encoder_ti=args.train_text_encoder_ti,
        caption_column=args.caption_column,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_num=args.num_class_images,
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
    
    def compute_time_ids(crops_coords_top_left, original_size=None):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        if original_size is None:
            original_size = (args.resolution, args.resolution)
        target_size = (args.resolution, args.resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids
    

    if not args.train_text_encoder:
        tokenizers = [tokenizer]
        text_encoders = [text_encoder]
        if xl:
            tokenizers = [tokenizer, tokenizer_two]
            text_encoders = [text_encoder, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
                prompt_embeds = prompt_embeds.to(accelerator.device)
            return prompt_embeds
        
        def compute_text_embeddings_xl(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = encode_prompt_xl(text_encoders, tokenizers, prompt)
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    if freeze_text_encoder and not train_dataset.custom_instance_prompts:
        if xl:
            instance_prompt_hidden_states, instance_pooled_prompt_embeds = compute_text_embeddings_xl(
                args.instance_prompt, text_encoders, tokenizers
            )
        else:
            instance_prompt_hidden_states = compute_text_embeddings(args.instance_prompt, text_encoders, tokenizers)

    # Clear the memory here
    if freeze_text_encoder and not train_dataset.custom_instance_prompts:
        del tokenizers, text_encoders
        gc.collect()
        torch.mps.empty_cache()
        torch.cuda.empty_cache()

    # if --train_text_encoder_ti we need add_special_tokens to be True for textual inversion
    add_special_tokens = True if args.train_text_encoder_ti else False

    if not train_dataset.custom_instance_prompts:
        if xl:
            if freeze_text_encoder:
                prompt_embeds = instance_prompt_hidden_states
                unet_add_text_embeds = instance_pooled_prompt_embeds
            # if we're optmizing the text encoder (both if instance prompt is used for all images or custom prompts) we need to tokenize and encode the
            # batch prompts on all training steps
            else:
                tokens_one = tokenize_prompt(tokenizer, args.instance_prompt, add_special_tokens)
                tokens_two = tokenize_prompt(tokenizer_two, args.instance_prompt, add_special_tokens)
        else:
            if freeze_text_encoder:
                prompt_embeds = instance_prompt_hidden_states

            # if we're optimizing the text encoder (both if instance prompt is used for all images or custom prompts) we need to tokenize and encode the
            # batch prompts on all training steps
            else:
                tokens_one = tokenize_prompt(tokenizer, args.instance_prompt, add_special_tokens)
                if args.with_prior_preservation:
                    class_tokens_one = tokenize_prompt(tokenizer, args.class_prompt, add_special_tokens)
                    tokens_one = torch.cat([tokens_one, class_tokens_one], dim=0)

    if args.train_text_encoder_ti and args.validation_prompt:
        # replace instances of --token_abstraction in validation prompt with the new tokens: "<si><si+1>" etc.
        for token_abs, token_replacement in train_dataset.token_abstraction_dict.items():
            args.validation_prompt = args.validation_prompt.replace(token_abs, "".join(token_replacement))
    print("validation prompt:", args.validation_prompt)

    if args.cache_latents:
        latents_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(
                    accelerator.device, non_blocking=True, dtype=torch.float32
                )
                latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)

        if args.validation_prompt is None:
            del vae
            torch.mps.empty_cache()
            torch.cuda.empty_cache()

    # Scheduler and math around the number of training steps.
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
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=power
    )

    hypernetwork, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            hypernetwork,  optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
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
            clear_hypernets(unet)
            accelerator.load_state(os.path.join(args.output_dir, path))
            add_hypernet(unet, hypernetwork)
            epoch = int(path.split("-")[1])

            initial_global_step = epoch * num_update_steps_per_epoch
            first_epoch = epoch
            global_step = initial_global_step

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    if args.train_text_encoder:
        num_train_epochs_text_encoder = int(args.train_text_encoder_frac * args.num_train_epochs)
    elif args.train_text_encoder_ti:  # args.train_text_encoder_ti
        num_train_epochs_text_encoder = int(args.train_text_encoder_ti_frac * args.num_train_epochs)

    for epoch in range(first_epoch, args.num_train_epochs):
        # if performing any kind of optimization of text_encoder params
        if args.train_text_encoder or args.train_text_encoder_ti:
            if epoch == num_train_epochs_text_encoder:
                print("PIVOT HALFWAY", epoch)
                # stopping optimization of text_encoder params
                # re setting the optimizer to optimize only on unet params
                optimizer.param_groups[1]["lr"] = 0.0

            else:
                # still optimizng the text encoder
                text_encoder.train()
                if xl:
                    text_encoder_two.train()
                # set top parameter requires_grad = True for gradient checkpointing works
                if args.train_text_encoder:
                    text_encoder.text_model.embeddings.requires_grad_(True)
                    if xl:
                        text_encoder_two.text_model.embeddings.requires_grad_(True)

        unet.train()
        for step, batch in enumerate(train_dataloader):
            socketio.emit("train progress", {"step": global_step + 1, "total_step": args.max_train_steps, "epoch": epoch + 1, "total_epoch": args.num_train_epochs})
            with accelerator.accumulate(unet):
                prompts = batch["prompts"]
                # encode batch prompts when custom prompts are provided for each image -
                if train_dataset.custom_instance_prompts:
                    if freeze_text_encoder:
                        prompt_embeds = compute_text_embeddings(prompts, text_encoders, tokenizers)

                    else:
                        tokens_one = tokenize_prompt(tokenizer, prompts, add_special_tokens)
                        if xl:
                            tokens_two = tokenize_prompt(tokenizer_two, prompts, add_special_tokens)

                if args.cache_latents:
                    model_input = latents_cache[step].sample()
                else:
                    pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                    model_input = vae.encode(pixel_values).latent_dist.sample()

                model_input = model_input * vae_scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    model_input = model_input.to(weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                if args.noise_offset:
                    noise += args.noise_offset * torch.randn(
                        (model_input.shape[0], model_input.shape[1], 1, 1), device=model_input.device
                    )
                bsz = model_input.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                )
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                if xl:
                    add_time_ids = torch.cat(
                        [
                            compute_time_ids(original_size=s, crops_coords_top_left=c)
                            for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])
                        ]
                    )

                # Calculate the elements to repeat depending on the use of prior-preservation and custom captions.
                if not train_dataset.custom_instance_prompts:
                    elems_to_repeat_text_embeds = bsz // 2 if args.with_prior_preservation else bsz

                else:
                    elems_to_repeat_text_embeds = 1

                # Predict the noise residual
                if freeze_text_encoder:
                    prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                    model_pred = unet(noisy_model_input, timesteps, prompt_embeds_input).sample
                else:
                    if xl:
                        unet_added_conditions = {"time_ids": add_time_ids}
                        prompt_embeds, pooled_prompt_embeds = encode_prompt_xl(
                            text_encoders=[text_encoder, text_encoder_two],
                            tokenizers=None,
                            prompt=None,
                            text_input_ids_list=[tokens_one, tokens_two],
                        )
                        unet_added_conditions.update(
                            {"text_embeds": pooled_prompt_embeds.repeat(elems_to_repeat_text_embeds, 1)}
                        )
                        prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                        model_pred = unet(
                            noisy_model_input, timesteps, prompt_embeds_input, added_cond_kwargs=unet_added_conditions
                        ).sample
                    else:
                        prompt_embeds = encode_prompt(
                            text_encoders=[text_encoder],
                            tokenizers=None,
                            prompt=None,
                            text_input_ids_list=[tokens_one],
                        )
                        prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                        model_pred = unet(noisy_model_input, timesteps, prompt_embeds_input).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.

                    if args.with_prior_preservation:
                        # if we're using prior preservation, we calc snr for instance loss only -
                        # and hence only need timesteps corresponding to instance images
                        snr_timesteps, _ = torch.chunk(timesteps, 2, dim=0)
                    else:
                        snr_timesteps = timesteps

                    snr = compute_snr(noise_scheduler, snr_timesteps)
                    base_weight = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(snr_timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective needs to be floored to an SNR weight of one.
                        mse_loss_weights = base_weight + 1
                    else:
                        # Epsilon and sample both use the same loss weights.
                        mse_loss_weights = base_weight

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()


                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        weights
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
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

                        save_path = os.path.join(args.output_dir, f"{name}-{epoch + 1}")
                        clear_hypernets(unet)
                        accelerator.save_state(save_path)
                        add_hypernet(unet, hypernetwork)
                        logger.info(f"Saved state to {save_path}")

                        hypernetwork = accelerator.unwrap_model(hypernetwork)
                        
                        metadata = {
                            "name": name,
                            "steps": str(global_step),
                            "epochs": str(epoch),
                            "checkpoint": os.path.basename(args.pretrained_model_name_or_path),
                            "images": str(len(train_dataloader)),
                            "learning_rate": str(args.learning_rate),
                            "gradient_accumulation_steps": str(args.gradient_accumulation_steps),
                            "learning_function": learning_function,
                            "sources": "\n".join(args.sources)
                        }
                        save_path = os.path.join(args.output_dir, f"{name}-{epoch + 1}.pt")
                        hypernetwork.save(save_path, metadata=metadata)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and (epoch+1) % args.validation_epochs == 0:
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompt}."
                )
                # create pipeline
                v_unet = copy.deepcopy(accelerator.unwrap_model(unet))
                hypernetwork = accelerator.unwrap_model(hypernetwork)
                if xl:
                    pipeline = StableDiffusionXLPipeline.from_single_file(
                        args.pretrained_model_name_or_path,
                        unet=v_unet,
                        torch_dtype=weight_dtype,
                    )
                else:
                    pipeline = StableDiffusionPipeline.from_single_file(
                        args.pretrained_model_name_or_path,
                        unet=v_unet,
                        torch_dtype=weight_dtype,
                    )
                add_hypernet(v_unet, hypernetwork)


                # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
                scheduler_args = {}

                if "variance_type" in pipeline.scheduler.config:
                    variance_type = pipeline.scheduler.config.variance_type

                    if variance_type in ["learned", "learned_range"]:
                        variance_type = "fixed_small"

                    scheduler_args["variance_type"] = variance_type

                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipeline.scheduler.config, **scheduler_args
                )

                pipeline.vae.enable_slicing()
                pipeline.vae.enable_tiling()
                pipeline = pipeline.to(device=accelerator.device)
                pipeline.enable_attention_slicing()
                #pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
                pipeline_args = {"prompt": args.validation_prompt}

                
                images = [
                    pipeline(**pipeline_args, num_inference_steps=10, generator=generator, callback_on_step_end=step_progress).images[0]
                    for _ in range(args.num_validation_images)
                ]
                if len(images) > 1:
                    for i in range(len(images)):
                        save_path = os.path.join(args.output_dir, f"{name}-{epoch + 1}-{i}.png")
                        images[i].save(save_path)
                else:
                    save_path = os.path.join(args.output_dir, f"{name}-{epoch + 1}.png")
                    images[0].save(save_path)
                    socketio.emit("train image complete", {"image": save_path})

                del pipeline
                del v_unet
                torch.mps.empty_cache()
                torch.cuda.empty_cache()
                
    # Save the hypernetwork
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        hypernetwork = accelerator.unwrap_model(hypernetwork)
                        
        metadata = {
            "name": name,
            "steps": str(global_step),
            "epochs": str(epoch),
            "checkpoint": os.path.basename(args.pretrained_model_name_or_path),
            "images": str(len(train_dataloader)),
            "learning_rate": str(args.learning_rate),
            "gradient_accumulation_steps": str(args.gradient_accumulation_steps),
            "learning_function": learning_function,
            "sources": "\n".join(args.sources)
        }
        save_path = os.path.join(args.output_dir, f"{name}.pt")
        hypernetwork.save(save_path, metadata=metadata)
    accelerator.end_training()

def get_options(model_name, train_data, instance_prompt, output, max_train_steps, learning_rate, text_encoder_lr, resolution, save_steps, 
    gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler):
    options = {}
    options["pretrained_model_name_or_path"] = model_name
    options["pretrained_vae_model_name_or_path"] = None
    options["revision"] = None
    options["variant"] = None
    options["dataset_name"] = train_data
    options["dataset_config_name"] = None
    options["instance_data_dir"] = None
    options["cache_dir"] = None
    options["image_column"] = "image"
    options["caption_column"] = "text"
    options["repeats"] = 1
    options["class_data_dir"] = None
    options["instance_prompt"] = instance_prompt
    options["token_abstraction"] = "token"
    options["num_new_tokens_per_abstraction"] = 2
    options["class_prompt"] = None
    options["validation_prompt"] = validation_prompt
    options["num_validation_images"] = 1
    options["validation_epochs"] = validation_epochs
    options["with_prior_preservation"] = False
    options["prior_loss_weight"] = 1.0
    options["num_class_images"] = 100
    options["output_dir"] = output
    options["seed"] = None
    options["resolution"] = resolution
    options["center_crop"] = True
    options["train_text_encoder"] = False
    options["train_batch_size"] = 1
    options["sample_batch_size"] = 1
    options["num_train_epochs"] = 1
    options["max_train_steps"] = max_train_steps
    options["checkpointing_steps"] = save_steps
    options["checkpoints_total_limit"] = None
    options["resume_from_checkpoint"] = "latest"
    options["gradient_accumulation_steps"] = gradient_accumulation_steps
    options["gradient_checkpointing"] = True
    options["learning_rate"] = learning_rate
    options["text_encoder_lr"] = text_encoder_lr
    options["scale_lr"] = False
    options["lr_scheduler"] = lr_scheduler
    options["snr_gamma"] = 5
    options["lr_warmup_steps"] = round(max_train_steps * 0.05)
    options["lr_num_cycles"] = 3
    options["lr_power"] = 1
    options["dataloader_num_workers"] = 0
    options["train_text_encoder_ti"] = False
    options["train_text_encoder_ti_frac"] = 0.5
    options["train_text_encoder_frac"] = 1.0
    options["optimizer"] = "adamW"
    options["use_8bit_adam"] = False
    options["adam_beta1"] = 0.9
    options["adam_beta2"] = 0.999
    options["prodigy_beta3"] = None
    options["prodigy_decouple"] = True
    options["adam_weight_decay"] = 1e-04
    options["adam_weight_decay_text_encoder"] = None
    options["adam_epsilon"] = 1e-08
    options["prodigy_use_bias_correction"] = True
    options["prodigy_safeguard_warmup"] = True
    options["max_grad_norm"] = 1.0
    options["logging_dir"] = None
    options["allow_tf32"] = True
    options["report_to"] = "tensorboard"
    options["mixed_precision"] = None
    options["prior_generation_precision"] = None
    options["local_rank"] = 1
    options["enable_xformers_memory_efficient_attention"] = False
    options["noise_offse"] = 0
    options["rank"] = 4
    options["cache_latents"] = True
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != options["local_rank"]:
        options["local_rank"] = env_local_rank
    return DotDict(options)

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def train_hypernetwork(images, model_name, train_data, instance_prompt, output, num_train_epochs, learning_rate, text_encoder_lr, resolution, save_epochs, 
    gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler, sizes):

    if not model_name: model_name = ""
    if not train_data: train_data = ""
    if not instance_prompt: instance_prompt = ""
    if not output: output = ""
    if not num_train_epochs: num_train_epochs = 20
    if not validation_epochs: validation_epochs = 5
    if not save_epochs: save_epochs = 5
    if not learning_rate: learning_rate = 1e-4
    if not text_encoder_lr: text_encoder_lr = 5e-6
    if not resolution: resolution = 256
    if not validation_prompt: validation_prompt = ""
    if not lr_scheduler: lr_scheduler = "constant"
    if not gradient_accumulation_steps: gradient_accumulation_steps = 1
    if not sizes: sizes = ["320", "640", "768", "1024", "1280"]

    steps_per_epoch = math.ceil(len(images) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * steps_per_epoch
    save_steps = save_epochs * steps_per_epoch
    validation_steps = validation_epochs * steps_per_epoch

    options = get_options(model_name, train_data, instance_prompt, output, max_train_steps, learning_rate, text_encoder_lr, resolution, save_steps, 
    gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler)

    options.sources = get_sources(train_data)
    options.sizes = sizes

    main(options)