import torch

def temporal_pool(x):
    return torch.mean(x, dim=1)

def sptial_pool(x):
    return torch.mean(x, dim=2)

def spa_tem_pool(x):
    """
    """