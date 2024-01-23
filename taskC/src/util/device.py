import torch

PREFERRED_CUDA_DEVICE = 'cuda:0'

def set_preferred_device(device):
    global PREFERRED_CUDA_DEVICE
    PREFERRED_CUDA_DEVICE = device

def get_device():
    if torch.cuda.is_available():
        return torch.device(PREFERRED_CUDA_DEVICE)
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    else:
        return torch.device('cpu')
