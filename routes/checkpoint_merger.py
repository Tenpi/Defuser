import os
import safetensors.torch
import torch


@torch.no_grad()
def merge(models, name, **kwargs):
    dtype = kwargs.pop("dtype", None)
    device = kwargs.pop("device", None)

    alpha = kwargs.pop("alpha", 0.5)
    interp = kwargs.pop("interp", None)

    print("Received list", models)
    print(f"Combining with alpha={alpha}, interpolation mode={interp}")

    checkpoint_count = len(models)

    if checkpoint_count > 3 or checkpoint_count < 2:
        raise ValueError("Received incorrect number of checkpoints to merge. Ensure that either 2 or 3 checkpoints are being passed.")

    print("Received the right number of checkpoints")

    if interp == "sigmoid":
        theta_func = sigmoid
    elif interp == "inv_sigmoid":
        theta_func = inv_sigmoid
    elif interp == "add_diff":
        theta_func = add_difference
    else:
        theta_func = weighted_sum

    checkpoint_path_0 = os.path.join(models[0])
    checkpoint_path_1 = os.path.join(models[1])
    checkpoint_path_2 = None
    if len(models) > 2:
        checkpoint_path_2 = os.path.join(models[2])

    theta_0 = None
    if (checkpoint_path_0.endswith(".safetensors")):
        theta_0 = safetensors.torch.load_file(checkpoint_path_0, device=device)
    else:
        theta_0 = torch.load(checkpoint_path_0, map_location=device)["state_dict"]

    theta_1 = None
    if (checkpoint_path_1.endswith(".safetensors")):
        theta_1 = safetensors.torch.load_file(checkpoint_path_1, device=device)
    else:
        theta_1 = torch.load(checkpoint_path_1, map_location=device)["state_dict"]

    theta_2 = None
    if checkpoint_path_2:
        if (checkpoint_path_2.endswith(".safetensors")):
            theta_2 = safetensors.torch.load_file(checkpoint_path_2, device=device)
        else:
            theta_2 = torch.load(checkpoint_path_2, map_location=device)["state_dict"]
            
    for key in theta_0.keys():
        try:
            if theta_2:
                theta_0[key] = theta_func(theta_0[key], theta_1[key], theta_2[key], alpha)
            else:
                theta_0[key] = theta_func(theta_0[key], theta_1[key], None, alpha)
            if dtype is torch.float16:
                theta_0[key] = theta_0[key].half()
        except:
            print(f"Key error: {key}")

    safetensors.torch.save_file(theta_0, f"{name}.safetensors")

    del theta_0
    del theta_1
    del theta_2

def weighted_sum(theta0, theta1, theta2, alpha):
    return ((1 - alpha) * theta0) + (alpha * theta1)

def sigmoid(theta0, theta1, theta2, alpha):
    alpha = alpha * alpha * (3 - (2 * alpha))
    return theta0 + ((theta1 - theta0) * alpha)

def inv_sigmoid(theta0, theta1, theta2, alpha):
    import math
    alpha = 0.5 - math.sin(math.asin(1.0 - 2.0 * alpha) / 3.0)
    return theta0 + ((theta1 - theta0) * alpha)

def add_difference(theta0, theta1, theta2, alpha):
    return theta0 + (theta1 - theta2) * (1.0 - alpha)