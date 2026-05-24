
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.cluster import KMeans
import time
import os
from util.kmeans import k_means_gpu, k_means_gpu_sparsity
import copy
import math

from models import *          # your custom ResNet, etc.
from cka_util import filter_cka_matrix, compute_filter_representations_2
from compr import ClassSubsetSieve, build_mask_from_classes
from torch.utils.data import Dataset
import csv
from pathlib import Path
from collections import Counter
from torch.utils.data import WeightedRandomSampler
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"



class RemappedSubset(Dataset):
    def __init__(self, base_dataset, indices, class_ids):
        self.dataset = base_dataset
        self.indices = indices
        self.class_ids = class_ids
        self.class_map = {c: i for i, c in enumerate(class_ids)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        return x, self.class_map[y]


############################
# 1. DATA & MODEL HELPERS  #
############################

def build_cifar100_subset_loaders(
    train_dataset,
    test_dataset,
    class_ids,
    batch_size=64,
    num_workers=2
):
    train_indices = [
        i for i, t in enumerate(train_dataset.targets)
        if t in class_ids
    ]

    test_indices = [
        i for i, t in enumerate(test_dataset.targets)
        if t in class_ids
    ]

    train_subset = RemappedSubset(train_dataset, train_indices, class_ids)
    test_subset  = RemappedSubset(test_dataset, test_indices, class_ids)

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader


def data_initialize(batch_size=64):
    workers = 2

    transform_train = transforms.Compose([
        # transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        # transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True, download=True, transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False, download=True, transform=transform_test)

    # transform_train = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761)
    # ),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761)
    # ),
    # ])

    # trainset = torchvision.datasets.CIFAR100(
    #     root='./data',
    #     train=True, download=True, transform=transform_train
    # )

    # testset = torchvision.datasets.CIFAR100(
    #     root='./data',
    #     train=False, download=True, transform=transform_test
    # )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=workers
    )

    return trainloader, testloader

def add_rotation_transform(dataset, degrees=15):
    """
    Safely add RandomRotation to CIFAR-style datasets,
    including wrapped datasets (e.g., RemappedSubset).
    """
    # unwrap if needed
    if hasattr(dataset, "dataset"):
        base_dataset = dataset.dataset
    else:
        base_dataset = dataset

    if not hasattr(base_dataset, "transform"):
        raise AttributeError("Base dataset has no 'transform' attribute")

    base_transform = base_dataset.transform

    base_dataset.transform = transforms.Compose([
        transforms.RandomRotation(degrees),
        base_transform
    ])




def model_initialize(model_type):
    if model_type == 'vgg16':
        path_vgg = './models/BaseVGGc10.pth'
        model = torch.load(path_vgg)
        print(f"VGG16 printed:")
        print(model)
    elif model_type == 'alexnet':
        path_anet = './models/Alexnet_C10SGD_BASE.pth'
        model = torch.load(path_anet)
        print(f"alexnet printed:")
        print(model)
    elif model_type== 'resnet50':
        path_resnet = './models/PretrainedResNet50.pth'
        model = torch.load(path_resnet)
        print(f"ResNet50 printed:")
        print(model)
    elif model_type== 'resnet50_c100':
        path_resnet = './models/PretrainedResNet50_c100.pth'
        model = torch.load(path_resnet)
        print(f"ResNet50 C100 printed:")
        print(model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.to(device)
    model.eval()
    print("Pretrained model loaded successfully.")
   
    criterion = nn.CrossEntropyLoss().to(device)
    return model

def get_layers(model):
    all_layers = []
    conv_layers = []
    linear_layers = []
    bn_layers = []
    clus_layers = []

    def _walk(module):
        for child in module.children():
            if isinstance(child, nn.Conv2d):
                conv_layers.append(child)
                all_layers.append(child)
            # elif isinstance(child, ClusteredConv2d):
            #     clus_layers.append(child)
            #     all_layers.append(child)
            elif isinstance(child, nn.Linear):
                linear_layers.append(child)
                all_layers.append(child)
            elif isinstance(child, nn.BatchNorm2d):
                bn_layers.append(child)
                all_layers.append(child)
            else:
                _walk(child)

    _walk(model)
    # return all_layers, conv_layers,clus_layers, linear_layers, bn_layers
    return all_layers, conv_layers, linear_layers, bn_layers

def get_conv_stream(model):
    conv_stream = []

    def _walk(module):
        for child in module.children():
            if isinstance(child, (nn.Conv2d, ClusteredConv2d)):
                conv_stream.append(child)
            else:
                _walk(child)

    _walk(model)
    return conv_stream



def print_conv_layers(model):
    _, conv_layers,clus_layers, _, _ = get_layers(model)
    print("=== Conv layers (index, out_channels, kernel, stride, name) ===")

    named_convs = []

    def _walk_named(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d):
                named_convs.append((full_name, child))
            else:
                _walk_named(child, full_name)

    _walk_named(model)

    for idx, (name, conv) in enumerate(named_convs):
        print(f"[{idx:2d}] {name:40s}  "
              f"out={conv.out_channels:4d}, k={conv.kernel_size}, stride={conv.stride}")


##############################
# 2. CKA-BASED CLUSTERING    #
##############################
def ratio_by_C_out(C_out:int)->float:
    if C_out <= 64:
        return 0.9   # very gentle
    elif C_out <= 128:
        return 0.4
    elif C_out <= 256:
        return 0.8
    else:
        return 0.6   # heaviest for big layers

def build_mask_from_classes(dataset, class_ids):
    """
    Returns a boolean mask selecting samples whose labels are in class_ids.
    """
    targets = dataset.targets if hasattr(dataset, "targets") else dataset.labels
    mask = torch.tensor([t in class_ids for t in targets])
    return mask

def cluster_filters_from_cka(sim_matrix, num_clusters,use_sparsity=False,save_path=None,seed=1000, gpu_id=0, verbosity=0 ):
    """
    sim_matrix: (C_out, C_out) CKA similarity between filters.
    We cluster each filter based on its similarity profile (row of sim_matrix).
    """
    X = sim_matrix.detach().cpu().numpy().astype(np.float32)  # (C_out, C_out)

    weight_compress = np.zeros((sim_matrix.shape[0], sim_matrix.shape[1]), dtype=np.float32)

    # Choose backend
    if use_sparsity:
        weight_compress, centers, labels = k_means_gpu_sparsity( X, num_clusters,save_path=save_path, seed=seed, gpu_id=gpu_id )
    else:
        weight_compress, centers, labels = k_means_gpu(X,num_clusters, verbosity=verbosity,save_path=save_path,seed=seed, gpu_id=gpu_id )
    # labels may be numpy; convert to torch.LongTensor
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    labels = labels.long()

    return labels  # shape (C_out,)


def compute_weight_centers(conv_layer, labels, num_clusters):
    """
    conv_layer.weight: (C_out, C_in, kh, kw)
    labels: (C_out,), cluster id in [0, num_clusters-1]
    """
    W = conv_layer.weight.data
    device_w = W.device

    C_out, C_in, kh, kw = W.shape

    centers = torch.zeros(num_clusters, C_in, kh, kw, device=device_w)
    counts = torch.zeros(num_clusters, device=device_w)

    for i in range(C_out):
        c = labels[i].item()
        centers[c] += W[i]
        counts[c] += 1

    counts = counts.view(-1, 1, 1, 1).clamp(min=1.0)
    centers /= counts
    return centers  # (K, C_in, kh, kw)


class ClusteredConv2d(nn.Module):
    """
    Conv layer that:
      1) Convolves with K center filters
      2) Expands back to C_out channels via labels
      3) Optionally uses per-filter scale and original bias
    """
    def __init__(self, original_conv, centers, labels, per_filter_scale=True):
        super().__init__()

        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups

        self.bias_flag = original_conv.bias is not None

        K, C_in, kh, kw = centers.shape
        C_out = labels.shape[0]

        self.centers = nn.Parameter(centers)           # (K, C_in, kh, kw)
        self.register_buffer('labels', labels.clone()) # (C_out,)

        if self.bias_flag:
            self.bias_param = nn.Parameter(original_conv.bias.data.clone())
        else:
            self.register_parameter('bias_param', None)

        # if per_filter_scale:
        #     self.scale = nn.Parameter(torch.ones(C_out))
        # else:
        #     self.register_parameter('scale', None)
        device_param = centers.device  # same as conv weights

        if per_filter_scale:
            # One scalar per original filter (optional), on same device as centers
            self.scale = nn.Parameter(torch.ones(C_out, device=device_param))
        else:
            self.register_parameter('scale', None)


    def forward(self, x):
        # 1) Conv with centers: (B, K, H, W)
        y_centers = F.conv2d(
            x,
            self.centers,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        # 2) Expand to original C_out channels via labels
        y = y_centers[:, self.labels, :, :]  # (B, C_out, H, W)

        if self.scale is not None:
            y = y * self.scale.view(1, -1, 1, 1)

        if self.bias_param is not None:
            y = y + self.bias_param.view(1, -1, 1, 1)

        return y


def replace_conv_with_clustered(model, target_conv, centers, labels):
    clustered_conv = ClusteredConv2d(target_conv, centers, labels)

    def _replace(module):
        for name, child in list(module.named_children()):
            if child is target_conv:
                setattr(module, name, clustered_conv)
            else:
                _replace(child)

    _replace(model)
    return model


##########################
# 3. METRICS / EVALUATION
##########################

@torch.no_grad()
def evaluate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
    return 100.0 * correct / total


@torch.no_grad()
def measure_latency(model, dataloader, num_batches=20):
    model.eval()
    # warmup
    for i, (x, _) in enumerate(dataloader):
        if i >= 3:
            break
        x = x.to(device)
        _ = model(x)

    start = time.perf_counter()
    count = 0
    for i, (x, _) in enumerate(dataloader):
        if i >= num_batches:
            break
        x = x.to(device)
        _ = model(x)
        count += x.size(0)
    end = time.perf_counter()

    avg_sec_per_sample = (end - start) / max(count, 1)
    return avg_sec_per_sample * 1000.0  # ms/sample


def conv_flops(conv, input_shape, out_channels=None):
    """
    Rough FLOPs estimate for one conv layer and one input.
    input_shape: (C_in, H_in, W_in)
    """
    C_in, H_in, W_in = input_shape
    kh, kw = conv.kernel_size
    stride_h, stride_w = conv.stride
    pad_h, pad_w = conv.padding

    H_out = (H_in + 2 * pad_h - kh) // stride_h + 1
    W_out = (W_in + 2 * pad_w - kw) // stride_w + 1

    C_out = out_channels if out_channels is not None else conv.out_channels
    flops = 2 * C_out * C_in * kh * kw * H_out * W_out
    return flops

def count_parameters_by_type(model):
    """
    Returns:
      {
        'conv': int,
        'fc': int,
        'total': int
      }
    """
    conv_params = 0
    fc_params = 0

    for m in model.modules():
        # Standard Conv
        if isinstance(m, nn.Conv2d):
            conv_params += sum(p.numel() for p in m.parameters() if p.requires_grad)

        # Clustered Conv (centers + scale + bias)
        elif isinstance(m, ClusteredConv2d):
            conv_params += sum(p.numel() for p in m.parameters() if p.requires_grad)

        # Fully connected
        elif isinstance(m, nn.Linear):
            fc_params += sum(p.numel() for p in m.parameters() if p.requires_grad)

    return {
        "conv": conv_params,
        "fc": fc_params,
        "total": conv_params + fc_params
    }

def total_conv_flops(model, input_shape=(3, 32, 32)):
    """
    Counts conv FLOPs, supports Conv2d and ClusteredConv2d.
    """
    total_flops = 0.0

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total_flops += conv_flops(
                m, input_shape, out_channels=m.out_channels
            )

        elif isinstance(m, ClusteredConv2d):
            # FLOPs only depend on number of centers (K)
            K = m.centers.shape[0]
            total_flops += clustered_conv_flops(
                m, input_shape
            )

    return total_flops

def clustered_conv_flops(clustered_conv, input_shape):
    """
    FLOPs for ClusteredConv2d.
    Only count convolution with K centers.
    Expansion via labels is indexing → no FLOPs.
    """
    C_in, H_in, W_in = input_shape
    K, _, kh, kw = clustered_conv.centers.shape

    stride_h, stride_w = clustered_conv.stride
    pad_h, pad_w = clustered_conv.padding

    H_out = (H_in + 2 * pad_h - kh) // stride_h + 1
    W_out = (W_in + 2 * pad_w - kw) // stride_w + 1

    flops = 2 * K * C_in * kh * kw * H_out * W_out
    return flops


##########################
# 4. END-TO-END PIPELINE #
##########################

def replace_classifier_head(model, num_classes):
    """
    Replace the final classification layer to match subset class count.
    Supports:
      - ResNet variants with .linear
      - Torchvision ResNet with .fc
      - VGG / AlexNet with .classifier
    """
    device = next(model.parameters()).device

    # --- YOUR ResNet (custom) ---
    if hasattr(model, "linear") and isinstance(model.linear, nn.Linear):
        in_features = model.linear.in_features
        model.linear = nn.Linear(in_features, num_classes).to(device)
        return model

    # --- Torchvision ResNet ---
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes).to(device)
        return model

    # --- VGG / AlexNet ---
    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            last = model.classifier[-1]
            if isinstance(last, nn.Linear):
                in_features = last.in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes).to(device)
                return model
        elif isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes).to(device)
            return model

    raise ValueError(
        f"Unknown model architecture for classifier replacement: {type(model)}"
    )




def check_subset_class_balance(dataloader):
    counts = Counter()
    for _, labels in dataloader:
        counts.update(labels.tolist())

    total = sum(counts.values())

    print("\n[CLASS DISTRIBUTION]")
    for c in sorted(counts):
        print(
            f"Class {c:2d}: {counts[c]:5d} samples "
            f"({100.0 * counts[c] / total:.2f}%)"
        )

    max_c = max(counts.values())
    min_c = min(counts.values())
    print(f"Total samples: {total}")
    print(f"Imbalance ratio (max/min): {max_c / min_c:.2f}x")


def run_cka_compression_all_convs_new(
    model_name,
    subset_trainloader,
    test_loader,
    num_clusters_ratio=0.5,      # unused now, kept for API compatibility
    layer_ratios=None,           # unused now
    batch_size=64,
    drop_threshold=5.0,          # max allowed drop vs ORIGINAL accuracy (in % points)
    save_path='./cka_models/ResNet50_c100_compr.pth'
):
    """
    CKA-based clustering for *every* Conv2d layer, with a global accuracy gate:

      - Compute ORIGINAL accuracy once (uncompressed model).
      - For each conv layer:
          * Make a trial copy of the CURRENT model.
          * Compress that layer in the trial copy.
      - Track FLOPs.
      - At the end, report final accuracy, latency, FLOPs and save compressed model.
    """

    # 1) Data & original model
    trainloader, testloader = data_initialize(batch_size=batch_size)
    model = model_initialize(model_type=model_name)  # current working model

    conv_layers_template = get_conv_stream(model)
    print(f"Total conv layers (template): {len(conv_layers_template)}")
   
    print(f"Total conv layers (template): {len(conv_layers_template)}")

    # ORIGINAL global metrics (reference for all decisions)
    orig_acc = evaluate_accuracy(model, testloader)
    orig_lat = measure_latency(model, testloader, num_batches=20)
    print(f"[ORIG] Accuracy: {orig_acc:.2f}%, Latency: {orig_lat:.3f} ms/sample")

    example_input_shape = ( 3, 32, 32)
    total_base_flops = 0.0
    for conv in conv_layers_template:
        total_base_flops += conv_flops(conv, example_input_shape,
                                       out_channels=conv.out_channels)
    total_comp_flops = total_base_flops 
    print(f"[ORIG] Total conv FLOPs (approx): {total_base_flops:.2e}")

    # 2) Iterate conv layers by template index
    for idx, conv_template in enumerate(conv_layers_template):
        C_out_template = conv_template.out_channels

        # Decide ratio for THIS layer based on its C_out
        ratio = ratio_by_C_out(C_out_template)
        ratio = float(max(0.0, min(1.0, ratio)))

        print("\n" + "=" * 70)
        print(f"[LAYER {idx}] Template Conv2d: C_out={C_out_template}, "
              f"kernel={conv_template.kernel_size}, stride={conv_template.stride}")
        print(f"[LAYER {idx}] ratio(from C_out) = {ratio:.2f}")

        # ---- Make a TRIAL copy of the CURRENT model ----
        model_trial = copy.deepcopy(model)
        model_trial.to(device)
        model_trial.eval()
        conv_layers_trial = get_conv_stream(model_trial)

        print(f"Total conv layers (template): {len(conv_layers_trial)}")
     
        print(f"Total conv layers (template): {len(conv_layers_trial)}")

        print(f"index idx : {idx}, len of conv_layr trial= {len(conv_layers_trial)}")
        print(f"index idx : {idx}, len of conv_layr trial= {len(conv_layers_trial)}")
        
        if idx >= len(conv_layers_trial):
            print(f"[LAYER {idx}] Index out of range in trial conv list, skipping.")
            continue
        target_conv_trial = conv_layers_trial[idx]

        if isinstance(target_conv_trial, ClusteredConv2d):
            print(f"LAYER {idx}--ALREADY CLUSTERED, 'SKIPPING'")
            continue

        # 2.1) Compute reps & CKA for this trial-layer
        reps = compute_filter_representations_2( model_trial, target_conv_trial, subset_trainloader, device )
        print(f"[LAYER {idx}] Reps shape: {reps.shape}")  # (N_samples, C_eff)

        sim_matrix = filter_cka_matrix(reps)  # (C_eff, C_eff)
        C_eff = sim_matrix.shape[0]
        print(f"[LAYER {idx}] CKA matrix shape: {sim_matrix.shape} (C_eff={C_eff})")
        print(f"[LAYER {idx}] CKA stats: mean={sim_matrix.mean().item():.4f}, "
              f"max={sim_matrix.max().item():.4f}, min={sim_matrix.min().item():.4f}")

        if idx==8:
            print(f"SIMILARITY MATRIX AFTER CKA:------------------{sim_matrix}")
        
        # scaling mask to control number of cluster, preventing aggressive compression
        eigvals = torch.linalg.eigvalsh(sim_matrix)
        eigvals = torch.relu(eigvals)

        # Normalize
        p = eigvals / eigvals.sum()

        # --- Entropy-based effective rank (lower bound) ---
        entropy = -(p * torch.log(p + 1e-12)).sum()
        k_entropy = torch.exp(entropy).item()

        # --- Energy-based rank (upper stabilizer) ---
        energy = torch.cumsum(eigvals, dim=0) / eigvals.sum()
        k_energy = torch.searchsorted(energy, 0.90).item() + 1  # 90% energy

        
        if math.isnan(k_entropy) or math.isnan(k_energy):
            num_clusters=0
            continue
        else:
            alpha = 0.3
            beta = 0.3
            num_clusters = int(alpha * k_entropy + beta * k_energy)

            num_clusters = max(1, min(num_clusters, C_eff - 1))

            print(
                f"[CKA-CLUS] k_entropy={int(k_entropy)}, "
                f"k_energy={k_energy}, "
                f"num_clusters={num_clusters}"
                )
        # -------------------------------------------------------------------------------------------------


        if num_clusters >= C_eff:
            print(f"[LAYER {idx}] num_clusters={num_clusters} >= C_eff={C_eff}, skipping.")
            continue

        print(f"[LAYER {idx}] → num_clusters = {num_clusters}/{C_eff}")

        # 2.3) Cluster filters using k_means_gpu / k_means_gpu_sparsity
        labels = cluster_filters_from_cka(
            sim_matrix,
            num_clusters=num_clusters,
            use_sparsity=False,
            save_path=None,
            seed=1000,
            gpu_id=0,
            verbosity=0
        )
        print(f"[LAYER {idx}] labels shape: {labels.shape}, "
              f"unique clusters: {labels.unique().numel()}")

        labels = labels.to(next(target_conv_trial.parameters()).device)

        

        # 2.4) Compute weight centers
        centers = compute_weight_centers(
            target_conv_trial,
            # labels.to(target_conv_trial.weight.device),
            labels,
            num_clusters
        )

        # FLOPs for this layer in trial model
        base_flops_layer = conv_flops(target_conv_trial, example_input_shape,
                                      out_channels=C_eff)
        comp_flops_layer = conv_flops(target_conv_trial, example_input_shape,
                                      out_channels=num_clusters)
        print(f"[LAYER {idx}] Conv FLOPs: base={base_flops_layer:.2e}, "
              f"comp={comp_flops_layer:.2e}, "
              f"ratio={comp_flops_layer / base_flops_layer:.3f}")

        # 2.5) Replace this conv with clustered version in TRIAL model
        model_trial = replace_conv_with_clustered(
            model_trial,
            target_conv_trial,
            centers,
            labels.to(device)
        )

        # 2.6) Accuracy after compressing this layer (GLOBAL check vs original)
        acc_after = evaluate_accuracy(model_trial, testloader)
              
        model = model_trial
        total_comp_flops += (comp_flops_layer - base_flops_layer)

    # 3) Final metrics after all accepted layers
    print("\n" + "#" * 70)
    print("# Finished compressing Conv2d layers with global accuracy gate")
    print("#" * 70)

    final_acc = evaluate_accuracy(model, testloader)
    final_lat = measure_latency(model, testloader, num_batches=20)

    print(f"[FINAL] Accuracy: {final_acc:.2f}% "
          f"(drop {orig_acc - final_acc:.2f} points from original)")
    print(f"[FINAL] Latency: {final_lat:.3f} ms/sample "
          f"(x{final_lat / orig_lat:.3f} vs original)")

    print(f"[FLOPs] Total conv FLOPs approx: base={total_base_flops:.2e}, "
          f"comp={total_comp_flops:.2e}, "
          f"ratio={total_comp_flops / total_base_flops:.3f}")

    # 4) Save compressed model
    torch.save(model, save_path)
    print(f"[SAVE] Compressed model saved to: {save_path}")
    print("********************************************************************************")
    # print(f"Printing saved model: {model}")
    print("********************************************************************************")

    return model

def log_model_stats_to_csv(
    csv_path,
    model_name,
    subset_name,
    is_compressed,
    class_labels,
    accuracy,
    param_dict,
    conv_flops,
    epochs=None,
    lr=None
):
    """
    Append model statistics to a CSV file.
    Creates file + header automatically if it does not exist.
    """

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        # Write header once
        if not file_exists:
            writer.writerow([
                 "model_name",
                "subset_name",
                "is_compressed",
                "num_classes",
                "class_labels",
                "accuracy",
                "conv_params",
                "fc_params",
                "total_params",
                "conv_flops",
                "epochs",
                "lr"
            ])

        writer.writerow([
             model_name,
            subset_name,
            int(is_compressed),
            len(class_labels) if class_labels is not None else "all",
            class_labels if class_labels is not None else "all",
            round(accuracy, 4),
            param_dict["conv"],
            param_dict["fc"],
            param_dict["total"],
            float(conv_flops),
            epochs if epochs is not None else "NA",
            lr if lr is not None else "NA"
        ])


def make_balanced_sampler_from_loader(dataloader):
    labels = []
    for _, y in dataloader:
        labels.extend(y.tolist())

    labels = np.array(labels)
    class_count = np.bincount(labels)
    class_weights = 1.0 / class_count
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

def add_cifar_strong_aug(dataset):
    base = dataset.dataset if hasattr(dataset, "dataset") else dataset
    base.transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761)
        ),
    ])

def get_subset_training_schedule(class_ids):
    """
    Returns (epochs, lr) based on number of classes in subset.
    """
    n_classes = len(class_ids)

    if n_classes >= 9:
        return 25, 1e-4
    elif 5 <= n_classes <= 7:
        return 15, 5e-5
    else:  # <= 5 classes
        return 10, 3e-5


def freeze_early_layers(model, freeze_until=2):
    """
    For ≤ 5 classes, also freeze early layers to avoid overfitting:
    Freeze first `freeze_until` conv blocks.
    """
    count = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if count < freeze_until:
                for p in m.parameters():
                    p.requires_grad = False
                count += 1


if __name__ == "__main__":
    model = model_initialize('resnet50_c100')
    print("Model type:", type(model))

    model_base = model

    base_params = count_parameters_by_type(model_base)
    base_flops = total_conv_flops(model_base)
    
    


    print("\n[BASE MODEL]")
    print(f"Conv params: {base_params['conv']:,}")
    print(f"FC params:   {base_params['fc']:,}")
    print(f"Total params:{base_params['total']:,}")
    print(f"Conv FLOPs:  {base_flops:.2e}")

    # ---- 1. Load full data once ----
    trainloader, testloader = data_initialize(batch_size=64)
    base_acc = evaluate_accuracy(model_base, testloader)

    log_model_stats_to_csv(
        csv_path="./results_csv/resnet50_c100_model_stats_full.csv",
        # log_model_stats_to_csv(
    # csv_path="./results_csv/resnet50_c100_model_stats_subsets.csv",
        model_name="resnet50_c100",
        subset_name="full",
        is_compressed=False,
        class_labels=None,
        accuracy=base_acc,
        param_dict=base_params,
        conv_flops=base_flops,
        epochs=None,
        lr=None


    )

    print("[CSV] Logged base model stats")

    
    sieve = ClassSubsetSieve(
        train_dataset=trainloader.dataset,
        test_dataset=testloader.dataset,
        device=device,
        batch_size=64
    )
    # CIFAR-10 example
    # class_subsets = sieve.define_manual_subsets({
    #     "vehicles": [1, 9],              # automobile, truck
    #     "animals": [2, 3, 4, 5, 6, 7],    # bird, cat, deer, dog, frog, horse
    #     "others": [0, 8]                 # airplane, ship
    # })

    # for CIFAR100
    class_subsets = sieve.define_manual_subsets({
        #  Vehicles (road + heavy)
        "vehicles": [
            8, 13, 48, 58, 90,      # bicycle, bus, motorcycle, pickup_truck, train
            41, 81, 85, 89          # lawn_mower, streetcar, tank, tractor
        ],

        #  Large mammals
        "large_mammals": [
            3, 42, 43, 88, 97,      # bear, leopard, lion, tiger, wolf
            19, 31, 64              # cattle, elephant, kangaroo
        ],

        #  Small mammals
        "small_mammals": [
            34, 50, 65, 74, 80,     # hamster, mouse, rabbit, shrew, squirrel
            55, 72                  # otter, seal (similar body scale)
        ],

        #  Aquatic animals
        "aquatic_animals": [
            1, 4, 30, 32, 55, 67, 73, 91, 95
        ],

        #  Insects
        "insects": [
            6, 7, 14, 18, 24
        ],

        #  Reptiles
        "reptiles": [
            16, 33, 44, 78, 93
        ],

        #  Flowers
        "flowers": [
            54, 62, 70, 82, 92
        ],

        #  Trees & plants
        "plants": [
            47, 52, 56, 59, 96
        ],

        #  Food & produce
        "food": [
            0, 51, 53, 57, 83,
            9, 10, 28, 61
        ],

        #  Indoor objects
        "indoor_objects": [
            5, 20, 22, 25, 39, 40, 84, 86, 87, 94
        ],

        #  Outdoor scenes & structures
        "outdoor_scenes": [
            12, 17, 23, 36, 37, 49, 60, 68, 71, 76
        ],

        #  People
        "people": [
            2, 11, 35, 46, 98
        ],
        "misc": [
        15, 19, 21, 26, 63,
        64, 66, 75, 77, 79]
        })

    # ===================================================================================================
    # CKA compression per subset
    compressed_models = {}

    for subset_name, class_ids in class_subsets.items():
        print("\n" + "="*80)
        print(f"\n===== CKA-CLUS for subset: {subset_name} =====")

        print(f"[SIEVE] Processing subset: {subset_name} → classes {class_ids}")

        # subset_trainloader, subset_testloader = sieve.build_subset_loaders_2(class_ids)
        # check_subset_class_balance(subset_trainloader, class_ids)
        subset_trainloader, subset_testloader = build_cifar100_subset_loaders(
            train_dataset=trainloader.dataset,
            test_dataset=testloader.dataset,
            class_ids=class_ids,
            batch_size=64
        )

        add_cifar_strong_aug(subset_trainloader.dataset)

        # --- COPY BASE MODEL ---
        model_subset = copy.deepcopy(model)
        model_subset.to(device)

        # --- RUN YOUR EXISTING CKA PIPELINE ---
        model_subset = run_cka_compression_all_convs_new(
            model_name='resnet50_c100',
            subset_trainloader=subset_trainloader,
            # test_loader=testloader,
            test_loader = subset_testloader,
            batch_size=64,
            drop_threshold=12.0,
            save_path=f'./cka_models/resnet50_c100_{subset_name}_cka.pth'
        )

        

        # --- REPLACE CLASSIFIER HEAD (CRITICAL) ------------------------------------------
        num_subset_classes = len(class_ids)
        model_subset = replace_classifier_head(model_subset, num_subset_classes)
        # ---------------------------------------------------------------------------------

        print(f"[HEAD] Replaced classifier with {num_subset_classes} outputs")


        if len(class_ids) <= 5:
            freeze_early_layers(model_subset, freeze_until=2)
            print("[FREEZE] Early conv layers frozen for small subset")

        # --- SUBSET-AWARE TRAINING SCHEDULE ---
        epochs, lr = get_subset_training_schedule(class_ids)

        print(f"[TRAIN-SCHEDULE] Subset '{subset_name}': "
            f"{len(class_ids)} classes → epochs={epochs}, lr={lr}")

        model_subset = sieve.finetune_on_subset(
            model_subset,
            train_loader=subset_trainloader,
            test_loader=subset_testloader,
            epochs=epochs,
            lr=lr
        )


        # --- EVALUATE ONLY ON THIS SUBSET ---
        subset_acc = sieve.evaluate_on_subset(
            model_subset,
            subset_testloader
        )

        
        print(f" Subset accuracy ({subset_name}): {subset_acc:.2f}%")


        save_subset = f'./cka_models/resnet50_c100/resnet50_c100_cka_{subset_name}.pth'
        torch.save(model_subset,save_subset)
        print(f' Finetuned class reduced subset again saved at {save_subset}')
        # print(f"subset mdoel printing: {model_subset}")

        compressed_models[subset_name] = model_subset
        comp_params = count_parameters_by_type(model_subset)
        comp_flops = total_conv_flops(model_subset)

        print("\n[COMPRESSED MODEL]")
        print(f"Conv params: {comp_params['conv']:,} "
            f"(ratio {comp_params['conv']/base_params['conv']:.3f})")
        print(f"FC params:   {comp_params['fc']:,} "
            f"(ratio {comp_params['fc']/base_params['fc']:.3f})")
        print(f"Total params:{comp_params['total']:,} "
            f"(ratio {comp_params['total']/base_params['total']:.3f})")
        print(f"Conv FLOPs:  {comp_flops:.2e} "
            f"(ratio {comp_flops/base_flops:.3f})")
        
        log_model_stats_to_csv(
        csv_path="./results_csv/resnet50_c100_model_stats_subsets.csv",
        model_name="resnet50_c100",
        subset_name=subset_name,
        is_compressed=True,
        class_labels=class_ids,
        accuracy=subset_acc,
        param_dict=comp_params,
        conv_flops=comp_flops,
        epochs=epochs,
        lr=lr
        )

        print(f"[CSV] Logged compressed model stats for subset: {subset_name}")




