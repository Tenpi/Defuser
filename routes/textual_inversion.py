from __main__ import app, socketio
from .functions import get_sources
import argparse
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path

import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder

from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
import functools
import pathlib
from itertools import chain

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

def log_validation(text_encoder, tokenizer, unet, vae, options, accelerator, weight_dtype, epoch):
    global pipeline
    logger.info(
        f"Running validation... \n Generating {options.num_validation_images} images with prompt:"
        f" {options.validation_prompt}."
    )
    pipeline = StableDiffusionPipeline.from_single_file(options.model, torch_dtype=weight_dtype)
    pipeline.tokenizer = tokenizer
    pipeline.text_encoder = accelerator.unwrap_model(text_encoder)
    pipeline.unet = unet
    pipeline.vae = vae
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    pipeline = pipeline.to(device=accelerator.device)
    pipeline.enable_attention_slicing()
    pipeline.safety_checker = None
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    #pipeline.set_progress_bar_config(disable=True)
    generator = None if options.seed is None else torch.Generator(device=accelerator.device).manual_seed(options.seed)
    images = []
    for i in range(options.num_validation_images):
        image = pipeline(options.validation_prompt, num_inference_steps=10, generator=generator, callback_on_step_end=step_progress).images[0]
        images.append(image)
    del pipeline
    torch.mps.empty_cache()
    torch.cuda.empty_cache()
    return images

def save_progress(text_encoder, placeholder_token_ids, accelerator, options, save_path, save_data):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {
        options.placeholder_token: learned_embeds.detach().cpu(),
        "name": save_data.name,
        "vectors": save_data.vectors,
        "steps": save_data.steps,
        "epochs": save_data.epochs,
        "checkpoint": save_data.checkpoint,
        "images": save_data.images,
        "learning_rate": save_data.learning_rate,
        "gradient_accumulation_steps": save_data.gradient_accumulation_steps,
        "learning_function": save_data.learning_function,
        "sources": save_data.sources
    }
    torch.save(learned_embeds_dict, save_path)

def is_image(filename):
    ext = pathlib.Path(filename).suffix.lower().replace(".", "")
    image_exts = ["jpg", "jpeg", "png", "webp", "gif", "apng", "avif", "bmp"]
    if ext in image_exts:
        return True
    else:
        return False

class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        repeats=100,
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        im_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
        self.image_paths = list(filter(lambda file: is_image(file), im_paths))

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = Image.Resampling.BICUBIC

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        text = self.placeholder_token

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


def main(options):
    global pipeline
    name = options.initializer_token
    accelerator_project_config = ProjectConfiguration(project_dir=options.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=options.gradient_accumulation_steps,
        mixed_precision=options.mixed_precision,
        log_with=options.report_to,
        project_config=accelerator_project_config,
    )

    if torch.backends.mps.is_available():
        try:
            torch.autocast(enabled=False, device_type='mps')
        except:
            torch.autocast = totally_legit_autocast

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

    if options.seed is not None:
        set_seed(options.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if options.output_dir is not None:
            os.makedirs(options.output_dir, exist_ok=True)

    generator = StableDiffusionPipeline.from_single_file(options.model, local_files_only=True)

    tokenizer = generator.tokenizer

    noise_scheduler = generator.scheduler
    text_encoder = generator.text_encoder
    vae = generator.vae
    unet = generator.unet

    placeholder_tokens = [options.placeholder_token]

    if options.num_vectors < 1:
        raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {options.num_vectors}")

    additional_tokens = []
    for i in range(1, options.num_vectors):
        additional_tokens.append(f"{options.placeholder_token}_{i}")
    placeholder_tokens += additional_tokens

    token_ids = tokenizer.encode(options.initializer_token, add_special_tokens=False)

    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    text_encoder.resize_token_embeddings(len(tokenizer))

    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    if options.gradient_checkpointing:
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if options.scale_lr:
        options.learning_rate = (
            options.learning_rate * options.gradient_accumulation_steps * options.train_batch_size * accelerator.num_processes
        )

    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),
        lr=options.learning_rate,
        betas=(options.adam_beta1, options.adam_beta2),
        weight_decay=options.adam_weight_decay,
        eps=options.adam_epsilon,
    )

    train_dataset = TextualInversionDataset(
        data_root=options.train_data,
        tokenizer=tokenizer,
        size=options.resolution,
        placeholder_token=options.placeholder_token,
        repeats=options.repeats,
        center_crop=options.center_crop,
        set="train",
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=options.train_batch_size, shuffle=True, num_workers=options.dataloader_num_workers
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / options.gradient_accumulation_steps)
    if options.max_train_steps is None:
        options.max_train_steps = options.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    learning_function = options.lr_scheduler
    power = 1.0
    if options.lr_scheduler == "quadratic":
        options.lr_scheduler = "polynomial"
        power = 2.0
    elif options.lr_scheduler == "cubic":
        options.lr_scheduler = "polynomial"
        power = 3.0
    elif options.lr_scheduler == "quartic":
        options.lr_scheduler = "polynomial"
        power = 4.0

    lr_scheduler = get_scheduler(
        options.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=options.lr_warmup_steps * options.gradient_accumulation_steps,
        num_training_steps=options.max_train_steps * options.gradient_accumulation_steps,
        num_cycles=options.lr_num_cycles * options.gradient_accumulation_steps,
        power=power
    )

    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / options.gradient_accumulation_steps)
    if overrode_max_train_steps:
        options.max_train_steps = options.num_train_epochs * num_update_steps_per_epoch
    options.num_train_epochs = math.ceil(options.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
            accelerator.init_trackers("textual_inversion", config=vars(options))

    total_batch_size = options.train_batch_size * accelerator.num_processes * options.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Images = {len(train_dataset)}")
    logger.info(f"  Epochs = {options.num_train_epochs}")
    logger.info(f"  Batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation Steps = {options.gradient_accumulation_steps}")
    logger.info(f"  Total Steps = {options.max_train_steps}")
    global_step = 0
    first_epoch = 0
    if options.resume_from_checkpoint:
        if options.resume_from_checkpoint != "latest":
            path = os.path.basename(options.resume_from_checkpoint)
        else:
            dirs = os.listdir(options.output_dir)
            dirs = list(filter(lambda d: os.path.isdir(os.path.join(options.output_dir, d)), dirs))
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{options.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            options.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(options.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * options.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * options.gradient_accumulation_steps)

    progress_bar = tqdm(range(global_step, options.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()

    for epoch in range(first_epoch, options.num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            socketio.emit("train progress", {"step": global_step + 1, "total_step": options.max_train_steps, "epoch": epoch + 1, "total_epoch": options.num_train_epochs})
            if options.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % options.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(text_encoder):
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]

            if accelerator.sync_gradients:
                images = []
                progress_bar.update(1)
                global_step += 1
                if global_step % options.save_steps == 0:
                    save_path = os.path.join(options.output_dir, f"{name}-{epoch + 1}.bin")
                    save_data = {
                        "name": name,
                        "vectors": options.num_vectors,
                        "steps": global_step,
                        "epochs": epoch,
                        "checkpoint": os.path.basename(options.model),
                        "images": len(train_dataloader),
                        "learning_rate": options.learning_rate,
                        "gradient_accumulation_steps": options.gradient_accumulation_steps,
                        "learning_function": learning_function,
                        "sources": options.sources
                    }
                    options.name = name
                    options.steps = global_step
                    options.epochs = epoch
                    save_progress(text_encoder, placeholder_token_ids, accelerator, options, save_path, DotDict(save_data))

                if accelerator.is_main_process:
                    if global_step % options.checkpointing_steps == 0:
                        if options.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(options.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= options.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - options.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(options.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(options.output_dir, f"{name}-{epoch + 1}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if options.validation_prompt is not None and global_step % options.validation_steps == 0:
                        images = log_validation(
                            text_encoder, tokenizer, unet, vae, options, accelerator, weight_dtype, epoch
                        )
                        if len(images) > 1:
                            for i in range(len(images)):
                                save_path = os.path.join(options.output_dir, f"{name}-{epoch + 1}-{i}.png")
                                images[i].save(save_path)
                        else:
                            save_path = os.path.join(options.output_dir, f"{name}-{epoch + 1}.png")
                            images[0].save(save_path)
                            socketio.emit("train image complete", {"image": save_path})

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= options.max_train_steps:
                break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if options.push_to_hub and not options.save_as_full_pipeline:
            logger.warn("Enabling full model saving because --push_to_hub=True was specified.")
            save_full_model = True
        else:
            save_full_model = options.save_as_full_pipeline
        if save_full_model:
            pipeline = StableDiffusionPipeline.from_single_file(options.model)
            pipeline.text_encoder = accelerator.unwrap_model(text_encoder)
            pipeline.vae = vae
            pipeline.unet = unet
            pipeline.tokenizer = tokenizer
            pipeline.save_pretrained(options.output_dir)
        save_path = os.path.join(options.output_dir, f"{name}.bin")
        save_data = {
            "name": name,
            "vectors": options.num_vectors,
            "steps": global_step,
            "epochs": epoch,
            "checkpoint": os.path.basename(options.model),
            "images": len(train_dataloader),
            "learning_rate": options.learning_rate,
            "gradient_accumulation_steps": options.gradient_accumulation_steps,
            "learning_function": learning_function,
            "sources": options.sources
        }
        save_progress(text_encoder, placeholder_token_ids, accelerator, options, save_path, DotDict(save_data))
    accelerator.end_training()

def get_options(model_name, train_data, initializer_token, output_dir, max_train_steps, learning_rate, resolution,
    save_steps, num_vectors, gradient_accumulation_steps, validation_prompt, validation_steps, lr_scheduler):
    options = {}
    options["save_steps"] = save_steps
    options["save_as_full_pipeline"] = False
    options["num_vectors"] = num_vectors
    options["model"] = model_name
    options["train_data"] = train_data
    options["placeholder_token"] = "*"
    options["initializer_token"] = initializer_token
    options["repeats"] = 1
    options["output_dir"] = output_dir
    options["seed"] = None
    options["resolution"] = resolution
    options["center_crop"] = True
    options["train_batch_size"] = 1
    options["num_train_epochs"] = None
    options["max_train_steps"] = max_train_steps
    options["gradient_accumulation_steps"] = gradient_accumulation_steps
    options["gradient_checkpointing"] = True
    options["learning_rate"] = learning_rate
    options["scale_lr"] =  False
    options["lr_scheduler"] = lr_scheduler
    options["lr_warmup_steps"] = 0
    options["lr_num_cycles"] = 1
    options["dataloader_num_workers"] = 0
    options["adam_beta1"] = 0.9
    options["adam_beta2"] = 0.999
    options["adam_weight_decay"] = 1e-2
    options["adam_epsilon"] = 1e-08
    options["validation_prompt"] = validation_prompt
    options["num_validation_images"] = 1
    options["validation_steps"] = validation_steps
    options["local_rank"] = -1
    options["checkpointing_steps"] = save_steps
    options["checkpoints_total_limit"] = None
    options["resume_from_checkpoint"] = "latest"
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != options["local_rank"]:
        options["local_rank"] = env_local_rank
    return DotDict(options)

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def train_textual_inversion(images, model_name, train_data, token, output, num_train_epochs, learning_rate, resolution, save_epochs, num_vectors, 
    gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler):

    if not model_name: model_name = ""
    if not train_data: train_data = ""
    if not token: token = ""
    if not output: output = ""
    if not num_train_epochs: num_train_epochs = 20
    if not validation_epochs: validation_epochs = 5
    if not save_epochs: save_epochs = 5
    if not learning_rate: learning_rate = 1e-4
    if not resolution: resolution = 256
    if not validation_prompt: validation_prompt = ""
    if not lr_scheduler: lr_scheduler = "constant"
    if not num_vectors: num_vectors = 1
    if not gradient_accumulation_steps: gradient_accumulation_steps = 1

    steps_per_epoch = math.ceil(len(images) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * steps_per_epoch
    save_steps = save_epochs * steps_per_epoch
    validation_steps = validation_epochs * steps_per_epoch

    options = get_options(model_name, train_data, token, output, max_train_steps, learning_rate, resolution, save_steps, num_vectors, 
    gradient_accumulation_steps, validation_prompt, validation_steps, lr_scheduler)

    options.sources = get_sources(train_data)

    main(options)