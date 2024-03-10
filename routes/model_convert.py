import torch
import safetensors.torch
from collections import OrderedDict
import pathlib
import struct
import json

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def get_safetensors_metadata(filename):
    with open(filename, "rb") as f:
        safe_bytes = f.read()
    metadata_size = struct.unpack("<Q", safe_bytes[0:8])[0]
    metadata_as_bytes = safe_bytes[8:8+metadata_size]
    metadata_as_dict = json.loads(metadata_as_bytes.decode(errors="ignore"))
    return metadata_as_dict.get("__metadata__", {})

def stringify_metadata(metadata):
    metadata_str = {}
    for key, value in metadata.items():
        metadata_str[key] = str(value)
    return metadata_str

def get_torch_metadata(model):
    metadata = {}
    for key, value in model.items():
        cont = False
        if type(value) == dict:
            for k,v in model[key].items():
                if type(v) == torch.Tensor:
                    cont = True
                    break
        if cont: continue
        if type(value) == str:
            metadata[key] = value
        if type(value) == int:
            metadata[key] = value
        if type(value) == float:
            metadata[key] = value
        if type(value) == bool:
            metadata[key] = value
        if type(value) == dict:
            metadata[key] = value
        if type(value) == list:
            metadata[key] = value
        #metadata[key] = value
    return metadata

def is_textual_inversion(model):
    if "*" in model:
        return True
    if "emb_params" in model:
        return True
    if "string_to_param" in model:
        return True
    
def convert_textual_inversion(model, output_path, metadata={}):
    output_ext = pathlib.Path(output_path).suffix
    tensors = None
    if "*" in model:
        tensors = model["*"]
    if "emb_params" in model:
        tensors = model["emb_params"]
    if "string_to_param" in model:
        tensors = model["string_to_param"]["*"]

    if output_ext == ".safetensors":
        save_dict = {"emb_params": tensors}
        safetensors.torch.save_file(save_dict, output_path, metadata=stringify_metadata(metadata))
    elif output_ext == ".pt":
        save_dict = metadata
        save_dict["string_to_param"] = {}
        save_dict["string_to_param"]["*"] = tensors
        torch.save(save_dict, output_path)
    elif output_ext == ".bin":
        save_dict = metadata
        save_dict["*"] = tensors
        torch.save(save_dict, output_path)
    elif output_ext == ".ckpt":
        save_dict = metadata
        save_dict["string_to_param"] = {}
        save_dict["string_to_param"]["*"] = tensors
        torch.save(save_dict, output_path)

def get_tensor_keys(model):
    tensor_keys = {}
    for key, value in model.items():
        if type(value) == tuple:
            for x in value:
                if isinstance(x, OrderedDict):
                    for y in x.items():
                        if type(y) == tuple:
                            tensor_keys[f"{key}.{y[0]}"] = y[1]
        if type(value) == torch.Tensor:
            tensor_keys[key] = value
    return tensor_keys

def get_ordered_keys(model):
    if 320 in model or 640 in model or 768 in model or 1024 in model or 1280 in model:
        return model
    dict_320 = {}
    dict_640 = {}
    dict_768 = {}
    dict_1024 = {}
    dict_1280 = {}
    for key, value in model.items():
        split = key.split(".")
        key_num = int(split[0])
        key_tensor = key.replace(f"{str(key_num)}.", "")
        if key_num == 320:
            dict_320[key_tensor] = value
        if key_num == 640:
            dict_640[key_tensor] = value
        if key_num == 768:
            dict_768[key_tensor] = value
        if key_num == 1024:
            dict_1024[key_tensor] = value
        if key_num == 1280:
            dict_1280[key_tensor] = value
    ordered_keys = {}
    if dict_320:
        ordered_keys[320] = (OrderedDict(dict_320),)
    if dict_640:
        ordered_keys[640] = (OrderedDict(dict_640),)
    if dict_768:
        ordered_keys[768] = (OrderedDict(dict_768),)
    if dict_1024:
        ordered_keys[1024] = (OrderedDict(dict_1024),)
    if dict_1280:
        ordered_keys[1280] = (OrderedDict(dict_1280),)
    return ordered_keys

def is_hypernetwork(model):
    if 320 in model or 640 in model or 768 in model or 1024 in model or 1280 in model:
        return True
    for key in model:
        if "320" in key or "640" in key or "768" in key or "1024" in key or "1280" in key:
            return True
    
def convert_hypernetwork(model, output_path, metadata={}):
    output_ext = pathlib.Path(output_path).suffix
    if output_ext == ".safetensors":
        tensor_keys = get_tensor_keys(model)
        safetensors.torch.save_file(tensor_keys, output_path, metadata=stringify_metadata(metadata))
    else:
        ordered_keys = get_ordered_keys(model)
        if output_ext == ".pt":
            for key, value in metadata.items():
                ordered_keys[key] = value
            torch.save(ordered_keys, output_path)
        elif output_ext == ".bin":
            for key, value in metadata.items():
                ordered_keys[key] = value
            torch.save(ordered_keys, output_path)
        elif output_ext == ".ckpt":
            for key, value in metadata.items():
                ordered_keys[key] = value
            torch.save(ordered_keys, output_path)

def is_lora(model):
    for key, value in model.items():
        if "lora" in key:
            return True
        
def convert_lora(model, output_path, metadata={}):
    output_ext = pathlib.Path(output_path).suffix
    if output_ext == ".safetensors":
        safetensors.torch.save_file(model, output_path, metadata=stringify_metadata(metadata))
    elif output_ext == ".pt":
        for key, value in metadata.items():
            model[key] = value
        torch.save(model, output_path)
    elif output_ext == ".bin":
        for key, value in metadata.items():
            model[key] = value
        torch.save(model, output_path)
    elif output_ext == ".ckpt":
        for key, value in metadata.items():
            model[key] = value
        torch.save(model, output_path)

def is_dreambooth(model):
    if "state_dict" in model:
        return True
    for key, value in model.items():
        if "diffusion" in key:
            return True

def convert_dreambooth(model, output_path, metadata={}):
    output_ext = pathlib.Path(output_path).suffix
    state_dict = None
    if "state_dict" in model:
        state_dict = model["state_dict"]
    else:
        state_dict = model
    if output_ext == ".safetensors":
        safetensors.torch.save_file(state_dict, output_path, metadata=stringify_metadata(metadata))
    elif output_ext == ".pt":
        model_dict = {"state_dict": state_dict}
        for key, value in metadata.items():
            model_dict[key] = value
        torch.save(model_dict, output_path)
    elif output_ext == ".bin":
        model_dict = {"state_dict": state_dict}
        for key, value in metadata.items():
            model_dict[key] = value
        torch.save(model_dict, output_path)
    elif output_ext == ".ckpt":
        model_dict = {"state_dict": state_dict}
        for key, value in metadata.items():
            model_dict[key] = value
        torch.save(model_dict, output_path)

def convert_other(model, output_path, metadata={}):
    output_ext = pathlib.Path(output_path).suffix
    if output_ext == ".safetensors":
        safetensors.torch.save_file(model, output_path, metadata=stringify_metadata(metadata))
    elif output_ext == ".pt":
        for key, value in metadata.items():
            model[key] = value
        torch.save(model, output_path)
    elif output_ext == ".bin":
        for key, value in metadata.items():
            model[key] = value
        torch.save(model, output_path)
    elif output_ext == ".ckpt":
        for key, value in metadata.items():
            model[key] = value
        torch.save(model, output_path)

def inference(input_path):
    ext = pathlib.Path(input_path).suffix
    model = None
    metadata = None
    if ext == ".safetensors":
        model = safetensors.torch.load_file(input_path, device=device)
        metadata = get_safetensors_metadata(input_path)
    else:
        model = torch.load(input_path, map_location=device)
        metadata = get_torch_metadata(model)
    print(model)
    print(model.keys())
    print(metadata)

def model_convert(input_path, output_path):
    ext = pathlib.Path(input_path).suffix

    model = None
    metadata = None
    if ext == ".safetensors":
        model = safetensors.torch.load_file(input_path, device=device)
        metadata = get_safetensors_metadata(input_path)
    else:
        model = torch.load(input_path, map_location=device)
        metadata = get_torch_metadata(model)

    if is_textual_inversion(model):
        convert_textual_inversion(model, output_path, metadata)
    elif is_hypernetwork(model):
        convert_hypernetwork(model, output_path, metadata)
    elif is_lora(model):
        convert_lora(model, output_path, metadata)
    elif is_dreambooth(model):
        convert_dreambooth(model, output_path, metadata)
    else:
        convert_other(model, output_path, metadata)
    return output_path