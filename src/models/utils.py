import torch
from typing import Any

def move_tensor_obj_to_device(tensor_obj: Any, device: torch.device | str):
    if isinstance(device, str):
        device = torch.device(device)
    obj = tensor_obj
    if isinstance(tensor_obj, torch.Tensor):
        obj = tensor_obj.to(device)
    elif isinstance(tensor_obj, dict):
        obj = {k: move_tensor_obj_to_device(v, device) for k, v in tensor_obj.items()}
    elif isinstance(tensor_obj, tuple):
        obj = tuple([move_tensor_obj_to_device(v, device) for v in tensor_obj])
    elif isinstance(tensor_obj, list):
        obj = [move_tensor_obj_to_device(v, device) for v in tensor_obj]
    return obj