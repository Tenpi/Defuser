from __main__ import app, socketio
from .functions import is_image, is_text, get_number_from_filename
import os
from PIL import Image
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
import torch
import torch.nn.functional as F
from itertools import chain
from .convert_to_ckpt import convert_to_ckpt
import random
import shutil
import math
import itertools

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

def get_images(folder, resolution=512, center_crop=True, repeats=1):
    files = os.listdir(folder)
    image_files = list(filter(lambda file: is_image(file), files))
    image_files = sorted(image_files, key=lambda x: get_number_from_filename(x), reverse=False)
    image_files = list(map(lambda file: Image.open(os.path.join(folder, file)).convert("RGB"), image_files))
    images = []
    for image in image_files:
        images.extend(itertools.repeat(image, repeats))

    tensors = []
    image_transforms = transforms.Compose([
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
    for image in images:
        tensor = image_transforms(image).unsqueeze(0)
        tensors.append(tensor)
    return tensors

def get_captions(folder, default="", repeats=1):
    files = os.listdir(folder)
    text_files = list(filter(lambda file: is_text(file), files))
    text_files = sorted(text_files, key=lambda x: get_number_from_filename(x), reverse=False)
    texts = []
    for text in text_files:
        texts.extend(itertools.repeat(text, repeats))

    captions = []
    for text in texts:
        f = open(os.path.join(folder, text))
        captions.append(f.read())
        f.close()
    if len(captions) == 0:
        image_files = list(filter(lambda file: is_image(file), files))
        for i in image_files:
            for j in range(repeats):
                captions.append(default)
    return captions

def create_unet():
    return UNet2DConditionModel(
        sample_size = 64,
        in_channels = 4,
        out_channels = 4,
        center_input_sample = False,
        flip_sin_to_cos = True,
        freq_shift = 0,
        layers_per_block = 2,
        block_out_channels=(320, 640, 1280, 1280),
        act_fn = "silu",
        cross_attention_dim = 768,
        down_block_types=(
            "CrossAttnDownBlock2D", 
            "CrossAttnDownBlock2D", 
            "CrossAttnDownBlock2D", 
            "DownBlock2D"
        ),
        mid_block_type = "UNetMidBlock2DCrossAttn",
        up_block_types=(
            "UpBlock2D", 
            "CrossAttnUpBlock2D", 
            "CrossAttnUpBlock2D", 
            "CrossAttnUpBlock2D"
        ),
        only_cross_attention = False,
        downsample_padding = 1,
        mid_block_scale_factor = 1.0,
        dropout = 0.0,
        norm_num_groups = 32,
        norm_eps = 1e-5,
        transformer_layers_per_block = 1
    )

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


def main(config, unet, pipeline, train_images, train_captions):
    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)
    learning_function = config.lr_scheduler
    power = 1.0
    if config.lr_scheduler == "quadratic":
        config.lr_scheduler = "polynomial"
        power = 2.0
    elif config.lr_scheduler == "cubic":
        config.lr_scheduler = "polynomial"
        power = 3.0
    elif config.lr_scheduler == "quartic":
        config.lr_scheduler = "polynomial"
        power = 4.0
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
        num_cycles=config.lr_num_cycles,
        power=power
    )
    if config.seed is not None:
        set_seed(config.seed)
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
    
    unet, optimizer, train_images, lr_scheduler = accelerator.prepare(unet, optimizer, train_images, lr_scheduler)

    weight_dtype = torch.float32
    if config.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    pipeline.enable_attention_slicing()
    pipeline.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    
    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_images), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        unet.train()
        for step, image in enumerate(train_images):
            socketio.emit("train progress", {"step": global_step + 1, "total_step": config.max_train_steps, "epoch": epoch + 1, "total_epoch": config.num_epochs})
            image = image.to(accelerator.device)
            latent = pipeline.vae.encode(image).latent_dist.sample()
            noise = torch.randn_like(latent)
            bsz = latent.shape[0]

            timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device).long()
            noisy_latent = pipeline.scheduler.add_noise(latent, noise, timesteps)

            def compute_text_embeddings(prompt, text_encoders, tokenizers):
                with torch.no_grad():
                    prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
                    prompt_embeds = prompt_embeds.to(accelerator.device)
                return prompt_embeds
            
            with accelerator.accumulate(unet):
                prompt_embeds = compute_text_embeddings(train_captions[step], [pipeline.text_encoder], [pipeline.tokenizer])
                prompt_embeds_input = prompt_embeds.repeat(1, 1, 1)

                noise_pred = unet(noisy_latent, timesteps, prompt_embeds_input).sample
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1

        if accelerator.is_main_process:
            pipeline.unet = accelerator.unwrap_model(unet)
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                generator = torch.Generator(device=accelerator.device).manual_seed(config.seed) if config.seed else None
                image = pipeline(
                    prompt=config.save_image_prompt,
                    num_inference_steps=10, 
                    generator=generator,
                    callback_on_step_end=step_progress
                ).images[0]
                save_path = os.path.join(config.output_dir, f"{config.name}-{epoch + 1}.png")
                image.save(save_path)
                socketio.emit("train image complete", {"image": save_path})

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                temp = f"{config.output_dir}/temp"
                pipeline.save_pretrained(temp)
                metadata = {
                    "name": config.name,
                    "steps": str(global_step),
                    "epochs": str(epoch),
                    "checkpoint": os.path.basename(config.model),
                    "images": str(len(train_images)),
                    "learning_rate": str(config.learning_rate),
                    "gradient_accumulation_steps": str(config.gradient_accumulation_steps),
                    "learning_functions": learning_function
                }
                convert_to_ckpt(temp, f"{config.output_dir}/{config.name}.ckpt", metadata=metadata)
                shutil.rmtree(temp)
                pipeline.save_pretrained(config.output_dir)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline.unet = accelerator.unwrap_model(unet)

        temp = f"{config.output_dir}/temp"
        pipeline.save_pretrained(temp)
        metadata = {
            "name": config.name,
            "steps": str(global_step),
            "epochs": str(epoch),
            "checkpoint": os.path.basename(config.model),
            "images": str(len(train_images)),
            "learning_rate": str(config.learning_rate),
            "gradient_accumulation_steps": str(config.gradient_accumulation_steps),
            "learning_functions": learning_function
        }
        convert_to_ckpt(temp, f"{config.output_dir}/{config.name}.ckpt", metadata=metadata)
        shutil.rmtree(temp)
        pipeline.save_pretrained(config.output_dir)

    accelerator.end_training()

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_config(name, model_name, train_data, output, num_train_epochs, learning_rate, resolution,
    gradient_accumulation_steps, save_epochs, validation_prompt, validation_epochs, lr_scheduler):
    config = {}
    config["name"] = name
    config["model"] = model_name
    config["data_dir"] = train_data
    config["output_dir"] = output
    config["image_size"] = resolution
    config["learning_rate"] = learning_rate
    config["center_crop"] = True
    config["repeats"] = 1
    config["lr_scheduler"] = lr_scheduler
    config["lr_warmup_steps"] = 0
    config["max_train_steps"] = 0
    config["lr_num_cycles"] = 1
    config["mixed_precision"] = "no"
    config["gradient_accumulation_steps"] = gradient_accumulation_steps
    config["num_epochs"] = num_train_epochs
    config["save_image_prompt"] = validation_prompt
    config["save_image_epochs"] = validation_epochs
    config["save_model_epochs"] = save_epochs
    config["seed"] = None

    return DotDict(config)

def train_checkpoint(name, model_name, train_data, output, num_train_epochs, learning_rate, resolution,
    gradient_accumulation_steps, save_epochs, validation_prompt, validation_epochs, lr_scheduler):

    if not name: name = ""
    if not model_name: model_name = ""
    if not train_data: train_data = ""
    if not output: output = ""
    if not num_train_epochs: num_train_epochs = 20
    if not learning_rate: learning_rate = 1e-4
    if not resolution: resolution = 256
    if not gradient_accumulation_steps: gradient_accumulation_steps = 1
    if not save_epochs: save_epochs = 5
    if not validation_epochs: validation_epochs = 5
    if not lr_scheduler: lr_scheduler = "constant"

    config = get_config(name, model_name, train_data, output, num_train_epochs, learning_rate, resolution,
    gradient_accumulation_steps, save_epochs, validation_prompt, validation_epochs, lr_scheduler)

    images = get_images(config.data_dir, config.image_size, config.center_crop, config.repeats)
    captions = get_captions(config.data_dir, config.name, config.repeats)

    steps_per_epoch = math.ceil(len(images) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * steps_per_epoch

    config.max_train_steps = max_train_steps

    pipeline = StableDiffusionPipeline.from_single_file(config.model)
    unet = create_unet()

    main(config, unet, pipeline, images, captions)