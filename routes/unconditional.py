from __main__ import app, socketio
from .functions import is_image, get_number_from_filename
import os
import gc
from PIL import Image
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from diffusers import DDPMPipeline, UNet2DModel, DDPMScheduler
from diffusers.optimization import get_scheduler
import torch
import torch.nn.functional as F
import shutil
import math
import itertools

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

def create_unet(resolution):
    return UNet2DModel(
        sample_size=resolution,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

def main(config, unet, train_images):
    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=config.ddpm_num_steps, beta_schedule=config.ddpm_beta_schedule)
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

    unet.to(accelerator.device, dtype=weight_dtype)
    
    global_step = 0

    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint != "latest":
            path = os.path.basename(config.resume_from_checkpoint)
        else:
            dirs = os.listdir(config.output_dir)
            dirs = list(filter(lambda d: os.path.isdir(os.path.join(config.output_dir, d)), dirs))
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run.")
            config.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(config.output_dir, path))
            global_step = int(path.split("-")[1])

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=config.max_train_steps, initial=global_step, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Steps")

        unet.train()
        for step, image in enumerate(train_images):
            socketio.emit("train progress", {"step": global_step + 1, "total_step": config.max_train_steps, "epoch": epoch + 1, "total_epoch": config.num_epochs})
            image = image.to(accelerator.device)
            noise = torch.randn(image.shape, dtype=weight_dtype, device=image.device)
            bsz = image.shape[0]

            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device).long()
            noisy_images = noise_scheduler.add_noise(image, noise, timesteps)
            
            with accelerator.accumulate(unet):
                noise_pred = unet(noisy_images, timesteps).sample
                loss = F.mse_loss(noise_pred.float(), noise.float())
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1

            if accelerator.is_main_process:
                if global_step % config.checkpointing_steps == 0:
                    if config.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(config.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= config.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            accelerator.print(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                            accelerator.print(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(config.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(config.output_dir, f"{config.name}-{global_step}")
                    accelerator.save_state(save_path)
                    accelerator.print(f"Saved state to {save_path}")

                if global_step % config.save_image_steps == 0 or epoch == config.num_epochs - 1:
                    unet = accelerator.unwrap_model(unet)
                    pipeline = DDPMPipeline(
                        unet=unet,
                        scheduler=noise_scheduler
                    )
                    pipeline = pipeline.to(device=accelerator.device)
                    image = pipeline(
                        num_inference_steps=10
                    ).images[0]
                    save_path = os.path.join(config.output_dir, f"{config.name}-{global_step}.png")
                    image.save(save_path)
                    socketio.emit("train image complete", {"image": save_path})

                if global_step % config.checkpointing_steps == 0 or epoch == config.num_epochs - 1:
                    unet = accelerator.unwrap_model(unet)
                    pipeline = DDPMPipeline(
                        unet=unet,
                        scheduler=noise_scheduler
                    )
                    pipeline.save_pretrained(config.output_dir)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = DDPMPipeline(
            unet=unet,
            scheduler=noise_scheduler
        )
        pipeline.save_pretrained(config.output_dir)

    accelerator.end_training()

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_config(name, train_data, output, max_train_steps, learning_rate, resolution,
    gradient_accumulation_steps, save_steps, save_image_steps, lr_scheduler):
    config = {}
    config["name"] = name
    config["data_dir"] = train_data
    config["output_dir"] = output
    config["image_size"] = resolution
    config["learning_rate"] = learning_rate
    config["center_crop"] = True
    config["repeats"] = 1
    config["lr_scheduler"] = lr_scheduler
    config["lr_warmup_steps"] = round(max_train_steps * 0.05)
    config["max_train_steps"] = max_train_steps
    config["lr_num_cycles"] = 1
    config["mixed_precision"] = "no"
    config["gradient_accumulation_steps"] = gradient_accumulation_steps
    config["num_epochs"] = 1
    config["save_image_steps"] = save_image_steps
    config["checkpoints_total_limit"] = 20
    config["checkpointing_steps"] = save_steps
    config["ddpm_num_steps"] = 1000
    config["ddpm_beta_schedule"] = "linear"
    config["resume_from_checkpoint"] = "latest"
    config["seed"] = None
    return DotDict(config)

def train_unconditional(name, train_data, output, num_train_epochs, learning_rate, resolution,
    gradient_accumulation_steps, save_steps, save_image_steps, lr_scheduler):

    if not name: name = ""
    if not train_data: train_data = ""
    if not output: output = ""
    if not num_train_epochs: num_train_epochs = 20
    if not learning_rate: learning_rate = 1e-4
    if not resolution: resolution = 256
    if not gradient_accumulation_steps: gradient_accumulation_steps = 1
    if not save_steps: save_steps = 100
    if not save_image_steps: save_image_steps = 100
    if not lr_scheduler: lr_scheduler = "constant"

    images = get_images(train_data, resolution)

    steps_per_epoch = math.ceil(len(images) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * steps_per_epoch

    config = get_config(name, train_data, output, max_train_steps, learning_rate, resolution,
    gradient_accumulation_steps, save_steps, save_image_steps, lr_scheduler)

    config.num_epochs = num_train_epochs
    unet = create_unet(resolution)
    
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.mps.empty_cache()
    except:
        pass

    main(config, unet, images)