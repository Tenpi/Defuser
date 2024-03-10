from __main__ import app, socketio
from .functions import get_sources
import pathlib
import itertools
import math
import os
import random
from typing import Optional, Iterable
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from PIL import Image, ImageOps
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import functools
from itertools import chain
import shutil

logger = get_logger(__name__)
pipeline = None

def autocast_decorator(autocast_instance, func):
  @functools.wraps(func)
  def decorate_autocast(*args, **kwargs):
    with autocast_instance:
      return func(*args, **kwargs)
  decorate_autocast.__script_unsupported = "@autocast() decorator is not supported in script mode"
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

def save_progress(text_encoder, placeholder_token_ids, accelerator, options, save_path, save_data):
    logger.info(f"Saving embedding {save_data.name}")
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

def log_validation(text_encoder, tokenizer, unet, vae, options, accelerator, weight_dtype, negative_prompt=None, text_encoder_2=None, tokenizer_2=None):
    global pipeline
    logger.info(
        f"Running validation... \n Generating {options.num_validation_images} images with prompt:"
        f" {options.validation_prompt}."
    )
    xl = False
    if "XL" in options.model:
        xl = True

    if xl:
        pipeline = StableDiffusionXLPipeline.from_single_file(options.model, torch_dtype=weight_dtype)
        pipeline.text_encoder_2 = text_encoder_2
        pipeline.tokenizer_2 = tokenizer_2
    else:
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
        image = pipeline(options.validation_prompt, negative_prompt=negative_prompt, num_inference_steps=10, generator=generator, callback_on_step_end=step_progress).images[0]
        images.append(image)
    del pipeline
    torch.mps.empty_cache()
    torch.cuda.empty_cache()
    return images

def add_tokens_and_get_placeholder_token(args, token_ids, tokenizer, text_encoder, num_vec_per_token, original_placeholder_token, is_random=False, is_negative=False):
    assert num_vec_per_token >= len(token_ids)
    placeholder_tokens = [f"{original_placeholder_token}_{i}" for i in range(num_vec_per_token)]

    for placeholder_token in placeholder_tokens:
        num_added_tokens = tokenizer.add_tokens(placeholder_token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )
    placeholder_token = " ".join(placeholder_tokens)
    placeholder_token_ids = tokenizer.encode(placeholder_token, add_special_tokens=False)
    print(f"The placeholder tokens are {placeholder_token} while the ids are {placeholder_token_ids}")
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    if is_random:
        # Initialize them to be random
        for i, placeholder_token_id in enumerate(placeholder_token_ids):
            token_embeds[placeholder_token_id] = torch.randn_like(token_embeds[placeholder_token_id])
    elif args.initialize_rest_random:
        # The idea is that the placeholder tokens form adjectives as in x x x white dog.
        for i, placeholder_token_id in enumerate(placeholder_token_ids):
            if len(placeholder_token_ids) - i < len(token_ids):
                token_embeds[placeholder_token_id] = token_embeds[token_ids[i % len(token_ids)]]
            else:
                token_embeds[placeholder_token_id] = torch.randn_like(token_embeds[placeholder_token_id])
    else:
        for i, placeholder_token_id in enumerate(placeholder_token_ids):
            token_embeds[placeholder_token_id] = token_embeds[token_ids[i * len(token_ids) // num_vec_per_token]]
        if is_negative:
            token_embeds[placeholder_token_id] += torch.randn_like(token_embeds[placeholder_token_id])*1e-3
    return placeholder_token, placeholder_token_ids

class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                tmp = self.decay * (s_param - param)
                s_param.sub_(tmp)
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()
        torch.mps.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.
        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]

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
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        neg_placeholder_token=None,
        center_crop=False
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.neg_placeholder_token = neg_placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
        self.image_paths = list(filter(lambda file: is_image(file), self.image_paths))

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
        image = ImageOps.exif_transpose(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        text = self.placeholder_token
        neg_text = self.neg_placeholder_token

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        if self.neg_placeholder_token:
            example["neg_input_ids"] = self.tokenizer(
                neg_text,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
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

def freeze_params(params):
    for param in params:
        param.requires_grad = False

def get_original(scheduler, model_output, sample: torch.FloatTensor, timestep: int):
        t = timestep
        alpha_prod_t = scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        if scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )
        return pred_original_sample

def main(args):
    global pipeline
    name = args.initializer_token
    name_neg = args.initializer_token + "-neg"

    xl = False
    if "XL" in args.model:
        xl = True

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    print(f"accelerator device is {accelerator.device}")

    if torch.backends.mps.is_available():
        try:
            torch.autocast(enabled=False, device_type="mps")
        except:
            torch.autocast = totally_legit_autocast

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    generator = None
    if xl:
        generator = StableDiffusionXLPipeline.from_single_file(args.model, local_files_only=True)
    else:
        generator = StableDiffusionPipeline.from_single_file(args.model, local_files_only=True)

    tokenizer = generator.tokenizer

    noise_scheduler = generator.scheduler
    text_encoder = generator.text_encoder
    vae = generator.vae
    unet = generator.unet
    text_encoder_2 = None
    tokenizer_2 = None
    if xl:
        tokenizer_2 = generator.tokenizer_2
        text_encoder_2 = generator.text_encoder_2

    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)

    # Load models and create wrapper for stable diffusion
    # Add the placeholder token in tokenizer
    placeholder_token, placeholder_token_ids = add_tokens_and_get_placeholder_token(
        args, token_ids, tokenizer, text_encoder, args.num_vectors, args.placeholder_token
    )
    neg_placeholder_token, neg_placeholder_token_ids = None, None
    if args.use_neg:
        neg_placeholder_token, neg_placeholder_token_ids = add_tokens_and_get_placeholder_token(
            args, token_ids, tokenizer, text_encoder, args.num_neg_vectors, args.placeholder_token+"-neg", is_negative=True, is_random=False
        )
    # Freeze vae and unet
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    # Freeze all parameters except for the token embeddings in text encoder
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    freeze_params(params_to_freeze)

    if xl:
        text_encoder_2.requires_grad_(False)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    if args.use_ema:
        ema_embedding = EMAModel(text_encoder.get_input_embeddings().parameters())
        ema_embedding.to(device=accelerator.device)

    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=placeholder_token,
        neg_placeholder_token=neg_placeholder_token,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        set="train",
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

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
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles * args.gradient_accumulation_steps,
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
    if xl:
        text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    # Keep vae and unet in eval model as we don"t train these
    vae.eval()
    unet.eval()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
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
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            epoch = int(path.split("-")[1])
            resume_global_step = epoch * num_update_steps_per_epoch * args.gradient_accumulation_steps
            first_epoch = epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
            global_step = resume_global_step

    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    original_token_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    
    for epoch in range(args.num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            socketio.emit("train progress", {"step": global_step + 1, "total_step": args.max_train_steps, "epoch": epoch + 1, "total_epoch": args.num_train_epochs})
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(text_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we"ll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                #noisy_latents = torch.cat([noisy_latents] * 2)

                # Get the text embedding for conditioning
                cond_embedding = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)
                uncond_embedding = None
                if args.use_neg:
                    uncond_embedding = text_encoder(batch["neg_input_ids"])[0].to(dtype=weight_dtype)
                
                if xl:
                    encoder_output_2 = text_encoder_2(
                        batch["input_ids_2"].reshape(batch["input_ids_1"].shape[0], -1), output_hidden_states=True
                    )
                    encoder_hidden_states_2 = encoder_output_2.hidden_states[-2].to(dtype=weight_dtype)
                    original_size = [
                        (batch["original_size"][0][i].item(), batch["original_size"][1][i].item())
                        for i in range(args.train_batch_size)
                    ]
                    crop_top_left = [
                        (batch["crop_top_left"][0][i].item(), batch["crop_top_left"][1][i].item())
                        for i in range(args.train_batch_size)
                    ]
                    target_size = (args.resolution, args.resolution)
                    add_time_ids = torch.cat(
                        [
                            torch.tensor(original_size[i] + crop_top_left[i] + target_size)
                            for i in range(args.train_batch_size)
                        ]
                    ).to(accelerator.device, dtype=weight_dtype)
                    added_cond_kwargs = {"text_embeds": encoder_output_2[0], "time_ids": add_time_ids}
                    cond_embedding = torch.cat([cond_embedding, encoder_hidden_states_2], dim=-1)

                # Predict the noise residual
                # print(noisy_latents.shape, text_embeddings.shape)
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=cond_embedding).sample
                noise_pred_uncond = None
                if args.use_neg:
                    noise_pred_uncond = unet(noisy_latents, timesteps, encoder_hidden_states=uncond_embedding).sample
                    model_pred = noise_pred_uncond + args.guidance_scale * (model_pred - noise_pred_uncond)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred, target, reduction="none").mean([1, 2, 3]).mean()
                rec_latent = None
                rec = None
                if args.use_l1 or args.use_l1_pixel:
                    # noisy_latents is x_t, noise_pred is the predicted noise. getting x_0
                    rec_latent = get_original(noise_scheduler, model_pred, noisy_latents, timesteps)
                    if args.use_l1_pixel:
                        rec_latent /= 0.18215
                        rec = vae.decode(rec_latent).sample
                        loss += F.l1_loss(rec, batch["pixel_values"])*args.l1_weight
                    else:
                        loss += F.l1_loss(rec_latent, latents)*args.l1_weight
                accelerator.backward(loss)
                
                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if accelerator.num_processes > 1:
                    token_embeds = text_encoder.module.get_input_embeddings().weight
                else:
                    token_embeds = text_encoder.get_input_embeddings().weight
                # Get the index for tokens that we want to zero the grads for
                grad_mask = torch.arange(len(tokenizer)) != placeholder_token_ids[0]
                for i in range(1, len(placeholder_token_ids)):
                    grad_mask = grad_mask & (torch.arange(len(tokenizer)) != placeholder_token_ids[i])
                for i in range(1, len(neg_placeholder_token_ids)):
                    grad_mask = grad_mask & (torch.arange(len(tokenizer)) != neg_placeholder_token_ids[i])
                token_embeds.data[grad_mask, :] = original_token_embeds[grad_mask, :]

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                del noise
                del noisy_latents
                del timesteps
                del model_pred
                del noise_pred_uncond
                del uncond_embedding
                del cond_embedding


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # Adding back weight decay
                progress_bar.update(1)
                if args.use_ema:
                    ema_embedding.step(text_encoder.get_input_embeddings().parameters())
                    if accelerator.num_processes > 1:
                        token_embeds = text_encoder.module.get_input_embeddings().weight
                    else:
                        token_embeds = text_encoder.get_input_embeddings().weight
                    # Get the index for tokens that we want to zero the grads for
                    grad_mask = torch.arange(len(tokenizer)) != placeholder_token_ids[0]
                    for i in range(1, len(placeholder_token_ids)):
                        grad_mask = grad_mask & (torch.arange(len(tokenizer)) != placeholder_token_ids[i])
                    for i in range(1, len(neg_placeholder_token_ids)):
                        grad_mask = grad_mask & (torch.arange(len(tokenizer)) != neg_placeholder_token_ids[i])
                    token_embeds.data[grad_mask, :] = original_token_embeds[grad_mask, :]
                global_step += 1
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"{name}-{epoch + 1}.bin")
                    save_path_neg = os.path.join(args.output_dir, f"{name}-{epoch + 1}-neg.bin")
                    save_data = {
                        "name": name,
                        "vectors": args.num_vectors,
                        "steps": global_step,
                        "epochs": epoch,
                        "checkpoint": os.path.basename(args.model),
                        "images": len(train_dataloader),
                        "learning_rate": args.learning_rate,
                        "gradient_accumulation_steps": args.gradient_accumulation_steps,
                        "learning_function": learning_function,
                        "cfg": args.cfg_scale,
                        "sources": args.sources
                    }
                    args.name = name
                    args.steps = global_step
                    args.epochs = epoch
                    save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path, DotDict(save_data))
                    save_data["name"] = name + "-neg"
                    save_data["vectors"] = args.num_neg_vectors
                    save_progress(text_encoder, neg_placeholder_token_ids, accelerator, args, save_path_neg, DotDict(save_data))

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

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
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        images = log_validation(
                            text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, name_neg, text_encoder_2, tokenizer_2
                        )
                        if len(images) > 1:
                            for i in range(len(images)):
                                save_path = os.path.join(args.output_dir, f"{name}-{epoch + 1}-{i}.png")
                                images[i].save(save_path)
                        else:
                            save_path = os.path.join(args.output_dir, f"{name}-{epoch + 1}.png")
                            images[0].save(save_path)
                            socketio.emit("train image complete", {"image": save_path})

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        if args.use_ema:
            ema_embedding.copy_to(text_encoder.get_input_embeddings().parameters())
        if args.push_to_hub and not args.save_as_full_pipeline:
            logger.warn("Enabling full model saving because --push_to_hub=True was specified.")
            save_full_model = True
        else:
            save_full_model = args.save_as_full_pipeline
        if save_full_model:
            if xl:
                pipeline = StableDiffusionXLPipeline.from_single_file(args.model)
                pipeline.text_encoder_2 = text_encoder_2
                pipeline.tokenizer_2 = tokenizer_2
            else:
                pipeline = StableDiffusionPipeline.from_single_file(args.model)
            pipeline.text_encoder = accelerator.unwrap_model(text_encoder)
            pipeline.vae = vae
            pipeline.unet = unet
            pipeline.tokenizer = tokenizer
            pipeline.save_pretrained(args.output_dir)
        save_path = os.path.join(args.output_dir, f"{name}.bin")
        save_path_neg = os.path.join(args.output_dir, f"{name}-neg.bin")
        save_data = {
            "name": name,
            "vectors": args.num_vectors,
            "neg_vectors": args.num_neg_vectors,
            "steps": global_step,
            "epochs": epoch,
            "checkpoint": os.path.basename(args.model),
            "images": len(train_dataloader),
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_function": learning_function,
            "cfg": args.cfg_scale,
            "sources": args.sources
        }
        save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path, DotDict(save_data))
        save_data["name"] = name + "-neg"
        save_data["vectors"] = args.num_neg_vectors
        save_progress(text_encoder, neg_placeholder_token_ids, accelerator, args, save_path_neg, DotDict(save_data))
        # Also save the newly trained embeddings
    accelerator.end_training()

def get_options(model_name, train_data, token, output, max_train_steps, learning_rate, resolution, save_steps, num_vectors, 
    num_neg_vectors, gradient_accumulation_steps, validation_prompt, validation_steps, lr_scheduler, cfg_scale):
    args = {}
    args["num_vectors"] = num_vectors
    args["num_neg_vectors"] = num_neg_vectors
    args["use_ema"] = True
    args["l1_weight"] = 1
    args["use_l1_pixel"] = True
    args["use_l1"] = True
    args["use_neg"] = True
    args["guidance_scale"] = cfg_scale
    args["save_steps"] = save_steps
    args["model"] = model_name
    args["train_data_dir"] = train_data
    args["placeholder_token"] = "*"
    args["initializer_token"] = token
    args["learnable_property"] = "object"
    args["validation_prompt"] = validation_prompt
    args["validation_steps"] = validation_steps
    args["num_validation_images"] = 1
    args["repeats"] = 1
    args["output_dir"] = output
    args["seed"] = None
    args["resolution"] = resolution
    args["center_crop"] = True
    args["train_batch_size"] = 1
    args["num_train_epochs"] = None
    args["max_train_steps"] = max_train_steps
    args["gradient_accumulation_steps"] = gradient_accumulation_steps
    args["learning_rate"] = learning_rate
    args["scale_lr"] = False
    args["lr_scheduler"] = lr_scheduler
    args["lr_warmup_steps"] = round(max_train_steps * 0.05)
    args["adam_beta1"] = 0.9
    args["adam_beta2"] = 0.999
    args["adam_weight_decay"] = 1e-2
    args["adam_epsilon"] = 1e-08
    args["mixed_precision"] = "no"
    args["lr_num_cycles"] = 3
    args["local_rank"] = 1
    args["checkpointing_steps"] = save_steps
    args["checkpoints_total_limit"] = None
    args["resume_from_checkpoint"] = "latest"
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return DotDict(args)

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def train_dreamartist(images, model_name, train_data, token, output, num_train_epochs, learning_rate, resolution, save_epochs, num_vectors, 
    num_neg_vectors, gradient_accumulation_steps, validation_prompt, validation_epochs, lr_scheduler, cfg_scale):

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
    if not num_neg_vectors: num_neg_vectors = 1
    if not gradient_accumulation_steps: gradient_accumulation_steps = 1
    if not cfg_scale: cfg_scale = 3.0

    steps_per_epoch = math.ceil(len(images) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * steps_per_epoch
    save_steps = save_epochs * steps_per_epoch
    validation_steps = validation_epochs * steps_per_epoch

    options = get_options(model_name, train_data, token, output, max_train_steps, learning_rate, resolution, save_steps, num_vectors, 
    num_neg_vectors, gradient_accumulation_steps, validation_prompt, validation_steps, lr_scheduler, cfg_scale)

    options.sources = get_sources(train_data)

    main(options)