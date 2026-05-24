import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.functional as F
import torchvision.transforms as transforms
from functools import reduce
from operator import mul


import os
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch.optim as optim

from mpl_toolkits.mplot3d import Axes3D 
# import umap
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import re
import math

def compute_filter_representations(model, layer, dataloader, device):
    reps = []

    def hook(module, input, output):
        # output: (B, C, H, W)
        B, C, H, W = output.shape
        gap = output.mean(dim=[2, 3])    # (B, C)
        reps.append(gap.detach())

    handle = layer.register_forward_hook(hook)

    model.eval()
    with torch.no_grad():
        for x, _ in dataloader:
            model(x.to(device))

    handle.remove()

    reps = torch.cat(reps, dim=0)  # (N, C)
    reps = reps - reps.mean(dim=0, keepdim=True)
    return reps  # N × C

def linear_cka(X, Y):
    # X, Y: (N, D)
    # X -= X.mean(0, keepdim=True)
    # Y -= Y.mean(0, keepdim=True)
    
    # K = X @ X.T
    # L = Y @ Y.T
    
    # hsic = (K * L).sum()
    # norm_x = (K * K).sum().sqrt()
    # norm_y = (L * L).sum().sqrt()
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)

    K = Xc @ Xc.T
    L = Yc @ Yc.T

    hsic = (K * L).sum()
    norm_x = (K * K).sum().sqrt()
    norm_y = (L * L).sum().sqrt()
    return hsic / (norm_x * norm_y)

def filter_similarity_matrix(reps):
    N, C = reps.shape
    sims = torch.zeros(C, C)

    for i in range(C):
        for j in range(i, C):
            sims[i, j] = sims[j, i] = linear_cka(
                reps[:, i:i+1], 
                reps[:, j:j+1]
            )
    return sims


import torch

@torch.no_grad()
def compute_filter_representations_2(model, layer, dataloader, device):
    """
    Collect global-average-pooled activations for a single conv layer.
    Returns reps of shape (N_samples, C_out) on CPU.
    """
    reps = []

    def hook(module, input, output):
        # output: (B, C, H, W)
        gap = output.mean(dim=[2, 3])    # (B, C)
        reps.append(gap.detach().cpu())

    handle = layer.register_forward_hook(hook)

    model.eval()
    for x, _ in dataloader:
        x = x.to(device)
        model(x)

    handle.remove()

    reps = torch.cat(reps, dim=0)  # (N, C)
    return reps  # not centered here on purpose


@torch.no_grad()
def filter_cka_matrix(reps):
    """
    Compute CKA-like similarity between filters using squared correlation.

    reps: (N, C) — N samples, C filters
    returns: sim (C, C), where sim[i,j] ∈ [0, 1], sim[i,i] = 1
    """
    # Center across samples (rows)
    Xc = reps - reps.mean(dim=0, keepdim=True)  # (N, C)

    # Gram matrix G = X^T X  (C, C)
    G = Xc.T @ Xc  # (C, C)

    # Variances for each filter (diagonal)
    diag = torch.diag(G)  # (C,)

    # Denominator for each pair (C, C): var_i * var_j
    denom = diag.unsqueeze(1) * diag.unsqueeze(0)  # outer product

    # Squared correlation = G^2 / (var_i * var_j)
    sim = (G ** 2) / (denom + 1e-8)

    # Clamp numeric noise
    sim = torch.clamp(sim, 0.0, 1.0)

    return sim  # (C, C)
