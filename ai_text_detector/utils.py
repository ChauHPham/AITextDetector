import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def device_info():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    capability = None
    if cuda:
        capability = torch.cuda.get_device_name(0)
    return {"cuda": cuda, "device": str(device), "name": capability}

def auto_fp16(requested_fp16: bool | None) -> bool:
    import torch
    if requested_fp16 is None:
        return torch.cuda.is_available()
    return requested_fp16
