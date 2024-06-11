from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block
from tqdm import tqdm
from .functions import get_device

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class LatentTransparencyOffsetEncoder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            zero_module(torch.nn.Conv2d(256, 4, kernel_size=3, padding=1, stride=1)),
        )

    def __call__(self, x):
        return self.blocks(x)


# 1024 * 1024 * 3 -> 16 * 16 * 512 -> 1024 * 1024 * 3
class UNet1024(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = (
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types: Tuple[str] = (
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        block_out_channels: Tuple[int] = (32, 32, 64, 128, 256, 512, 512),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 4,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        # input
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1)
        )
        self.latent_conv_in = zero_module(
            nn.Conv2d(4, block_out_channels[2], kernel_size=1)
        )

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=None,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim
                if attention_head_dim is not None
                else output_channel,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift="default",
                downsample_type=downsample_type,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            dropout=dropout,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            attention_head_dim=attention_head_dim
            if attention_head_dim is not None
            else block_out_channels[-1],
            resnet_groups=norm_num_groups,
            attn_groups=None,
            add_attention=True,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=None,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim
                if attention_head_dim is not None
                else output_channel,
                resnet_time_scale_shift="default",
                upsample_type=upsample_type,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=3, padding=1
        )

    def forward(self, x, latent):
        sample_latent = self.latent_conv_in(latent)
        sample = self.conv_in(x)
        emb = None

        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):
            if i == 3:
                sample = sample + sample_latent

            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        sample = self.mid_block(sample, emb)

        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]
            sample = upsample_block(sample, res_samples, emb)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


def checkerboard(shape):
    return np.indices(shape).sum(axis=0) % 2


class TransparentVAEDecoder:
    def __init__(self, sd, mod_number=1, fp16=False):
        device = get_device()
        self.load_device = device
        self.offload_device = device
        self.dtype = torch.float16 if fp16 else torch.float32

        model = UNet1024(in_channels=3, out_channels=4)
        model.load_state_dict(sd, strict=True)
        model.to(device=self.offload_device, dtype=self.dtype)
        model.eval()

        self.model = model
        self.mod_number = mod_number
        return

    @torch.no_grad()
    def estimate_single_pass(self, pixel, latent):
        y = self.model(pixel, latent)
        return y

    @torch.no_grad()
    def estimate_augmented(self, pixel, latent):
        args = [
            [False, 0],
            [False, 1],
            [False, 2],
            [False, 3],
            [True, 0],
            [True, 1],
            [True, 2],
            [True, 3],
        ]

        result = []

        for flip, rok in tqdm(args):
            feed_pixel = pixel.clone()
            feed_latent = latent.clone()

            if flip:
                feed_pixel = torch.flip(feed_pixel, dims=(3,))
                feed_latent = torch.flip(feed_latent, dims=(3,))

            feed_pixel = torch.rot90(feed_pixel, k=rok, dims=(2, 3))
            feed_latent = torch.rot90(feed_latent, k=rok, dims=(2, 3))

            eps = self.estimate_single_pass(feed_pixel, feed_latent).clip(0, 1)
            eps = torch.rot90(eps, k=-rok, dims=(2, 3))

            if flip:
                eps = torch.flip(eps, dims=(3,))

            result += [eps]

        result = torch.stack(result, dim=0)
        if self.load_device == torch.device("mps"):
            median = torch.median(result.cpu(), dim=0).values
            median = median.to(device=self.load_device, dtype=self.dtype)
        else:
            median = torch.median(result, dim=0).values
        return median

    def decode_wrapper(self, p):
        @torch.no_grad()
        def wrapper(func, latent):
            pixel = func(latent).sample.to(device=self.load_device, dtype=self.dtype)

            latent = latent.to(device=self.load_device, dtype=self.dtype)
            self.model = self.model.to(self.load_device)
            vis_list = []

            for i in range(int(latent.shape[0])):
                if self.mod_number != 1 and i % self.mod_number != 0:
                    vis_list.append(pixel[i : i + 1].movedim(1, -1))
                    continue

                y = self.estimate_augmented(pixel[i : i + 1], latent[i : i + 1])

                y = y.clip(0, 1).movedim(1, -1)
                alpha = y[..., :1]
                fg = y[..., 1:]

                B, H, W, C = fg.shape
                cb = checkerboard(shape=(H // 64, W // 64))
                cb = cv2.resize(cb, (W, H), interpolation=cv2.INTER_NEAREST)
                cb = (0.5 + (cb - 0.5) * 0.1)[None, ..., None]
                cb = torch.from_numpy(cb).to(fg)

                vis = fg * alpha + cb * (1 - alpha)
                vis_list.append(vis)

                png = torch.cat([fg, alpha], dim=3)[0]
                png = (
                    (png * 255.0)
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                p.extra_result_images.append(png)

            vis_list = torch.cat(vis_list, dim=0)
            return vis_list

        return wrapper

    def patch(self, p, vae_patcher, output_origin):
        @torch.no_grad()
        def wrapper(func, latent):
            pixel = (
                func(latent)
                .movedim(-1, 1)
                .to(device=self.load_device, dtype=self.dtype)
            )

            if output_origin:
                origin_outputs = (
                    (pixel.movedim(1, -1) * 255.0)
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                for png in origin_outputs:
                    p.extra_result_images.append(png)

            latent = latent.to(device=self.load_device, dtype=self.dtype)
            self.model = self.model.to(self.load_device)
            vis_list = []

            for i in range(int(latent.shape[0])):
                if self.mod_number != 1 and i % self.mod_number != 0:
                    vis_list.append(pixel[i : i + 1].movedim(1, -1))
                    continue

                y = self.estimate_augmented(pixel[i : i + 1], latent[i : i + 1])

                y = y.clip(0, 1).movedim(1, -1)
                alpha = y[..., :1]
                fg = y[..., 1:]

                B, H, W, C = fg.shape
                cb = checkerboard(shape=(H // 64, W // 64))
                cb = cv2.resize(cb, (W, H), interpolation=cv2.INTER_NEAREST)
                cb = (0.5 + (cb - 0.5) * 0.1)[None, ..., None]
                cb = torch.from_numpy(cb).to(fg)

                vis = fg * alpha + cb * (1 - alpha)
                vis_list.append(vis)

                png = torch.cat([fg, alpha], dim=3)[0]
                png = (
                    (png * 255.0)
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                p.extra_result_images.append(png)

            vis_list = torch.cat(vis_list, dim=0)
            return vis_list

        vae_patcher.set_model_vae_decode_wrapper(wrapper)
        return wrapper
    
class TransparentVAEEncoder:
    def __init__(self, sd, fp16=False):
        device = get_device()
        self.load_device = device
        self.offload_device = device
        self.dtype = torch.float16 if fp16 else torch.float32

        model = LatentTransparencyOffsetEncoder()
        model.load_state_dict(sd, strict=True)
        model.to(device=self.offload_device, dtype=self.dtype)
        model.eval()

        self.model = model
        return

    def patch(self, p, vae_patcher):
        @torch.no_grad()
        def wrapper(func, latent):
            return func(latent)

        vae_patcher.set_model_vae_encode_wrapper(wrapper)
        return
    
def device_supports_non_blocking(device):
    if hasattr(device, "type") and device.type.startswith("mps"):
        return False
    return True

def cast_to_device(tensor, device, dtype, copy=False):
    device_supports_cast = False
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
        device_supports_cast = True
    elif tensor.dtype == torch.bfloat16:
        if hasattr(device, "type") and device.type.startswith("cuda"):
            device_supports_cast = True

    non_blocking = device_supports_non_blocking(device)

    if device_supports_cast:
        if copy:
            if tensor.device == device:
                return tensor.to(dtype, copy=copy, non_blocking=non_blocking)
            return tensor.to(device, copy=copy, non_blocking=non_blocking).to(
                dtype, non_blocking=non_blocking
            )
        else:
            return tensor.to(device, non_blocking=non_blocking).to(
                dtype, non_blocking=non_blocking
            )
    else:
        return tensor.to(device, dtype, copy=copy, non_blocking=non_blocking)
    
def set_attr(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    setattr(obj, attrs[-1], torch.nn.Parameter(value, requires_grad=False))
    del prev

def copy_to_param(obj, attr, value):
    # inplace update tensor instead of replacing it
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    prev.data.copy_(value)

class TransparentUnetPatcher:
    def __init__(self, model, offload_device):
        model_sd = model.state_dict()
        self.model = model
        self.model_keys = set(model_sd.keys())
        self.patches = {}
        self.backup = {}
        self.offload_device = offload_device

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        p = set()
        for k in patches:
            if k in self.model_keys:
                p.add(k)
                current_patches = self.patches.get(k, [])
                current_patches.append((strength_patch, patches[k], strength_model))
                self.patches[k] = current_patches

        return list(p)

    def load_frozen_patcher(self, state_dict, strength):
        patch_dict = {}
        for k, w in state_dict.items():
            if "::" not in k:
                if "down.weight" in k:
                    k += "::lora::1"
                elif "up.weight" in k:
                    k += "::lora::0"
                else:
                    continue
            model_key, patch_type, weight_index = k.split("::")
            if model_key not in patch_dict:
                patch_dict[model_key] = {}
            if patch_type not in patch_dict[model_key]:
                patch_dict[model_key][patch_type] = [None] * 16
            patch_dict[model_key][patch_type][int(weight_index)] = w

        patch_flat = {}
        for model_key, v in patch_dict.items():
            for patch_type, weight_list in v.items():
                patch_flat[model_key] = (patch_type, weight_list)

        self.add_patches(
            patches=patch_flat, strength_patch=float(strength), strength_model=1.0
        )
        return

    def model_state_dict(self, filter_prefix=None):
        sd = self.model.state_dict()
        keys = list(sd.keys())
        if filter_prefix is not None:
            for k in keys:
                if not k.startswith(filter_prefix):
                    sd.pop(k)
        return sd

    def patch_model(self, device_to=None, patch_weights=True):
        if patch_weights:
            model_sd = self.model_state_dict()
            for key in self.patches:
                if key not in model_sd:
                    print("could not patch. key doesn't exist in model:", key)
                    continue

                weight = model_sd[key]

                inplace_update = True  # condition? maybe

                if key not in self.backup:
                    self.backup[key] = weight.to(
                        device=self.offload_device, copy=inplace_update
                    )

                if device_to is not None:
                    temp_weight = cast_to_device(
                        weight, device_to, torch.float32, copy=True
                    )
                else:
                    temp_weight = weight.to(torch.float32, copy=True)
                out_weight = self.calculate_weight(
                    self.patches[key], temp_weight, key
                ).to(weight.dtype)
                if inplace_update:
                    copy_to_param(self.model, key, out_weight)
                else:
                    set_attr(self.model, key, out_weight)
                del temp_weight

            if device_to is not None:
                self.model.to(device_to)
                self.current_device = device_to

        return self.model

    def calculate_weight(self, patches, weight, key):
        for p in patches:
            alpha = p[0]
            v = p[1]
            strength_model = p[2]

            if strength_model != 1.0:
                weight *= strength_model

            if isinstance(v, list):
                v = (self.calculate_weight(v[1:], v[0].clone(), key),)

            if len(v) == 1:
                patch_type = "diff"
            elif len(v) == 2:
                patch_type = v[0]
                v = v[1]
            else:
                raise Exception("Could not detect patch_type")

            if patch_type == "diff":
                w1 = v[0]
                if alpha != 0.0:
                    if w1.shape != weight.shape:
                        if w1.ndim == weight.ndim == 4:
                            new_shape = [
                                max(n, m) for n, m in zip(weight.shape, w1.shape)
                            ]
                            print(f"Merged with {key} channel changed to {new_shape}")
                            new_diff = alpha * cast_to_device(
                                w1, weight.device, weight.dtype
                            )
                            new_weight = torch.zeros(size=new_shape).to(weight)
                            new_weight[
                                : weight.shape[0],
                                : weight.shape[1],
                                : weight.shape[2],
                                : weight.shape[3],
                            ] = weight
                            new_weight[
                                : new_diff.shape[0],
                                : new_diff.shape[1],
                                : new_diff.shape[2],
                                : new_diff.shape[3],
                            ] += new_diff
                            new_weight = new_weight.contiguous().clone()
                            weight = new_weight
                        else:
                            print(
                                "WARNING SHAPE MISMATCH {} WEIGHT NOT MERGED {} != {}".format(
                                    key, w1.shape, weight.shape
                                )
                            )
                    else:
                        weight += alpha * cast_to_device(
                            w1, weight.device, weight.dtype
                        )
            elif patch_type == "lora":  # lora/locon
                mat1 = cast_to_device(
                    v[0], weight.device, torch.float32
                )
                mat2 = cast_to_device(
                    v[1], weight.device, torch.float32
                )
                if v[2] is not None:
                    alpha *= v[2] / mat2.shape[0]
                if v[3] is not None:
                    # locon mid weights, hopefully the math is fine because I didn't properly test it
                    mat3 = cast_to_device(
                        v[3], weight.device, torch.float32
                    )
                    final_shape = [
                        mat2.shape[1],
                        mat2.shape[0],
                        mat3.shape[2],
                        mat3.shape[3],
                    ]
                    mat2 = (
                        torch.mm(
                            mat2.transpose(0, 1).flatten(start_dim=1),
                            mat3.transpose(0, 1).flatten(start_dim=1),
                        )
                        .reshape(final_shape)
                        .transpose(0, 1)
                    )
                try:
                    weight += (
                        (
                            alpha
                            * torch.mm(
                                mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)
                            )
                        )
                        .reshape(weight.shape)
                        .type(weight.dtype)
                    )
                except Exception as e:
                    print("ERROR", key, e)
            elif patch_type == "lokr":
                w1 = v[0]
                w2 = v[1]
                w1_a = v[3]
                w1_b = v[4]
                w2_a = v[5]
                w2_b = v[6]
                t2 = v[7]
                dim = None

                if w1 is None:
                    dim = w1_b.shape[0]
                    w1 = torch.mm(
                        cast_to_device(
                            w1_a, weight.device, torch.float32
                        ),
                        cast_to_device(
                            w1_b, weight.device, torch.float32
                        ),
                    )
                else:
                    w1 = cast_to_device(
                        w1, weight.device, torch.float32
                    )

                if w2 is None:
                    dim = w2_b.shape[0]
                    if t2 is None:
                        w2 = torch.mm(
                            cast_to_device(
                                w2_a, weight.device, torch.float32
                            ),
                            cast_to_device(
                                w2_b, weight.device, torch.float32
                            ),
                        )
                    else:
                        w2 = torch.einsum(
                            "i j k l, j r, i p -> p r k l",
                            cast_to_device(
                                t2, weight.device, torch.float32
                            ),
                            cast_to_device(
                                w2_b, weight.device, torch.float32
                            ),
                            cast_to_device(
                                w2_a, weight.device, torch.float32
                            ),
                        )
                else:
                    w2 = cast_to_device(
                        w2, weight.device, torch.float32
                    )

                if len(w2.shape) == 4:
                    w1 = w1.unsqueeze(2).unsqueeze(2)
                if v[2] is not None and dim is not None:
                    alpha *= v[2] / dim

                try:
                    weight += alpha * torch.kron(w1, w2).reshape(weight.shape).type(
                        weight.dtype
                    )
                except Exception as e:
                    print("ERROR", key, e)
            elif patch_type == "loha":
                w1a = v[0]
                w1b = v[1]
                if v[2] is not None:
                    alpha *= v[2] / w1b.shape[0]
                w2a = v[3]
                w2b = v[4]
                if v[5] is not None:  # cp decomposition
                    t1 = v[5]
                    t2 = v[6]
                    m1 = torch.einsum(
                        "i j k l, j r, i p -> p r k l",
                        cast_to_device(
                            t1, weight.device, torch.float32
                        ),
                        cast_to_device(
                            w1b, weight.device, torch.float32
                        ),
                        cast_to_device(
                            w1a, weight.device, torch.float32
                        ),
                    )

                    m2 = torch.einsum(
                        "i j k l, j r, i p -> p r k l",
                        cast_to_device(
                            t2, weight.device, torch.float32
                        ),
                        cast_to_device(
                            w2b, weight.device, torch.float32
                        ),
                        cast_to_device(
                            w2a, weight.device, torch.float32
                        ),
                    )
                else:
                    m1 = torch.mm(
                        cast_to_device(
                            w1a, weight.device, torch.float32
                        ),
                        cast_to_device(
                            w1b, weight.device, torch.float32
                        ),
                    )
                    m2 = torch.mm(
                        cast_to_device(
                            w2a, weight.device, torch.float32
                        ),
                        cast_to_device(
                            w2b, weight.device, torch.float32
                        ),
                    )

                try:
                    weight += (alpha * m1 * m2).reshape(weight.shape).type(weight.dtype)
                except Exception as e:
                    print("ERROR", key, e)
            elif patch_type == "glora":
                if v[4] is not None:
                    alpha *= v[4] / v[0].shape[0]

                a1 = cast_to_device(
                    v[0].flatten(start_dim=1), weight.device, torch.float32
                )
                a2 = cast_to_device(
                    v[1].flatten(start_dim=1), weight.device, torch.float32
                )
                b1 = cast_to_device(
                    v[2].flatten(start_dim=1), weight.device, torch.float32
                )
                b2 = cast_to_device(
                    v[3].flatten(start_dim=1), weight.device, torch.float32
                )

                weight += (
                    (
                        (
                            torch.mm(b2, b1)
                            + torch.mm(torch.mm(weight.flatten(start_dim=1), a2), a1)
                        )
                        * alpha
                    )
                    .reshape(weight.shape)
                    .type(weight.dtype)
                )
            # elif patch_type in extra_weight_calculators:
            #     weight = extra_weight_calculators[patch_type](weight, alpha, v)
            else:
                print("patch type not recognized", patch_type, key)

        return weight