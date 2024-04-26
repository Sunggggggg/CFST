import torch

def temporal_pool(x):
    return torch.mean(x, dim=1)

def sptial_pool(x):
    return torch.mean(x, dim=2)

def spa_tem_pool(x, mid_idx, tem_stride, spa_stride):
    """
    x : [B, T, N, d]
    """
    x_local = x[:, mid_idx-tem_stride : mid_idx+tem_stride+1]

    return x_local
