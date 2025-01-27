from __main__ import app, socketio
import argparse
import json
import logging
import math
import os
from pathlib import Path
import datasets
import evaluate
import torch
import gc
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ColorJitter,
    RandomRotation,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm

import transformers
from transformers import ResNetConfig, ResNetForImageClassification, ConvNextConfig, ConvNextImageProcessor, ConvNextForImageClassification, \
ConvNextV2Config, ConvNextV2ForImageClassification, ViTConfig, ViTImageProcessor, ViTForImageClassification, Swinv2Config, Swinv2ForImageClassification, \
Swin2SRImageProcessor, BeitConfig, BeitImageProcessor, BeitForImageClassification, AutoConfig, AutoImageProcessor, get_scheduler
from transformers.utils.versions import require_version

from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = get_logger(__name__)

def main(args):
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    logger.info(accelerator.state)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name)
    else:
        data_files = {}
        if args.train_dir is not None:
            data_files["train"] = os.path.join(args.train_dir, "**")
        if args.validation_dir is not None:
            data_files["validation"] = os.path.join(args.validation_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
        )
    dataset_column_names = dataset["train"].column_names if "train" in dataset else dataset["validation"].column_names
    if args.image_column_name not in dataset_column_names:
        raise ValueError(
            f"--image_column_name {args.image_column_name} not found in dataset '{args.dataset_name}'. "
            "Make sure to set `--image_column_name` to the correct audio column - one of "
            f"{', '.join(dataset_column_names)}."
        )
    if args.label_column_name not in dataset_column_names:
        raise ValueError(
            f"--label_column_name {args.label_column_name} not found in dataset '{args.dataset_name}'. "
            "Make sure to set `--label_column_name` to the correct text column - one of "
            f"{', '.join(dataset_column_names)}."
        )
    
    args.train_val_split = 0.1
    if dataset["train"].num_rows > 5000:
        args.train_val_split = 0.05
    if dataset["train"].num_rows > 10000:
        args.train_val_split = 0.01
    if dataset["train"].num_rows > 100000:
        args.train_val_split = 0.005
    if dataset["train"].num_rows > 1000000:
        args.train_val_split = 0.001

    args.train_val_split = None if "validation" in dataset.keys() else args.train_val_split
    if isinstance(args.train_val_split, float) and args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    labels = dataset["train"].features[args.label_column_name].names
    id2label = {str(i): label for i, label in enumerate(labels)}
    label2id = {label: str(i) for i, label in enumerate(labels)}

    if args.architecture == "resnet":
        model_name = "microsoft/resnet-50"
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )
        image_processor = AutoImageProcessor.from_pretrained(model_name)
        model = ResNetForImageClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )
    elif args.architecture == "convnext":
        model_name = "facebook/convnext-base-224-22k"
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )
        image_processor = AutoImageProcessor.from_pretrained(model_name)
        model = ConvNextForImageClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )
    elif args.architecture == "convnextv2":
        model_name = "facebook/convnextv2-base-22k-224"
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )
        image_processor = AutoImageProcessor.from_pretrained(model_name)
        model = ConvNextV2ForImageClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )
    elif args.architecture == "vit":
        model_name = "google/vit-base-patch16-224-in21k"
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )
        image_processor = AutoImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )
    elif args.architecture == "swinv2":
        model_name = "microsoft/swinv2-base-patch4-window8-256"
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )
        image_processor = AutoImageProcessor.from_pretrained(model_name)
        model = Swinv2ForImageClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )
    elif args.architecture == "beit":
        model_name = "microsoft/beit-base-patch16-224-pt22k-ft22k"
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )
        image_processor = AutoImageProcessor.from_pretrained(model_name)
        model = BeitForImageClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )

    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    normalize = (
        Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std")
        else Lambda(lambda x: x)
    )
    train_transforms = Compose([
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            RandomRotation(15),
            ToTensor(),
            normalize,
        ])
    val_transforms = Compose([
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ])

    def preprocess_train(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            train_transforms(image.convert("RGB")) for image in example_batch[args.image_column_name]
        ]
        return example_batch

    def preprocess_val(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [
            val_transforms(image.convert("RGB")) for image in example_batch[args.image_column_name]
        ]
        return example_batch

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset = dataset["train"].with_transform(preprocess_train)
        if args.max_eval_samples is not None:
            dataset["validation"] = dataset["validation"].shuffle(seed=args.seed).select(range(args.max_eval_samples))
        eval_dataset = dataset["validation"].with_transform(preprocess_val)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example[args.label_column_name] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    power = 1.0
    if args.lr_scheduler_type == "quadratic":
        args.lr_scheduler_type = "polynomial"
        power = 2.0
    elif args.lr_scheduler_type == "cubic":
        args.lr_scheduler_type = "polynomial"
        power = 3.0
    elif args.lr_scheduler_type == "quartic":
        args.lr_scheduler_type = "polynomial"
        power = 4.0

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and str(checkpointing_steps).isdigit():
        checkpointing_steps = int(checkpointing_steps)

    metric = evaluate.load("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    resume_step = 0
    completed_steps = 0
    starting_epoch = 0
    if args.resume_from_checkpoint:
        try:
            if args.resume_from_checkpoint is not None and args.resume_from_checkpoint != "latest":
                checkpoint_path = args.resume_from_checkpoint
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                dirs = [os.path.join(args.output_dir, f.name) for f in os.scandir(args.output_dir) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]
                checkpoint_path = path
                path = os.path.basename(checkpoint_path)

            accelerator.print(f"Resumed from checkpoint: {path}")
            accelerator.load_state(checkpoint_path)
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
                completed_steps = starting_epoch * num_update_steps_per_epoch
            else:
                resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
                starting_epoch = resume_step // len(train_dataloader)
                completed_steps = resume_step // args.gradient_accumulation_steps
                resume_step -= starting_epoch * len(train_dataloader)
        except IndexError:
            pass

    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            socketio.emit("train progress", {"step": completed_steps + 1, "total_step": args.max_train_steps, "epoch": epoch + 1, "total_epoch": args.num_train_epochs})
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        args.output_dir,
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                    )
                    if accelerator.is_main_process:
                        image_processor.save_pretrained(args.output_dir)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")

        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy": eval_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            image_processor.save_pretrained(args.output_dir)
            all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
            with open(os.path.normpath(os.path.join(args.output_dir, "all_results.json")), "w") as f:
                json.dump(all_results, f)

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_args(train_dir, output_dir, num_train_epochs, learning_rate, gradient_accumulation_steps, 
             lr_scheduler_type, save_steps, resolution, architecture):
    args = DotDict()
    args.dataset_name = None
    args.train_dir = train_dir
    args.validation_dir = None
    args.max_train_samples = None
    args.max_eval_samples = None
    args.train_val_split = 0.1
    args.model_name_or_path = ""
    args.per_device_train_batch_size = 16
    args.per_device_eval_batch_size = 16
    args.learning_rate = learning_rate
    args.weight_decay = 0.01
    args.num_train_epochs = num_train_epochs
    args.max_train_steps = None
    args.gradient_accumulation_steps = gradient_accumulation_steps
    args.lr_scheduler_type = lr_scheduler_type
    args.num_warmup_steps = 500
    args.output_dir = output_dir
    args.seed = 42
    args.checkpointing_steps = save_steps
    args.resume_from_checkpoint = "latest"
    args.image_column_name = "image"
    args.label_column_name = "label"
    args.resolution = resolution
    args.architecture = architecture
    args.with_tracking = False

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

def train_classifier(train_dir, output_dir, num_train_epochs, learning_rate, gradient_accumulation_steps, 
    lr_scheduler_type, save_steps, resolution, architecture):

    if not train_dir: train_dir = ""
    if not output_dir: output_dir = ""
    if not num_train_epochs: num_train_epochs = 20
    if not save_steps: save_steps = 500
    if not resolution: resolution = 224
    if not learning_rate: learning_rate = 5e-5
    if not lr_scheduler_type: lr_scheduler_type = "linear"
    if not gradient_accumulation_steps: gradient_accumulation_steps = 1
    if not architecture: architecture = "resnet"

    args = get_args(train_dir, output_dir, num_train_epochs, learning_rate, gradient_accumulation_steps, 
    lr_scheduler_type, save_steps, resolution, architecture)

    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.mps.empty_cache()
    except:
        pass

    main(args)