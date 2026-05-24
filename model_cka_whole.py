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

from models import *          
from cka_util import  filter_cka_matrix, compute_filter_representations_2
from compr import ClassSubsetSieve, build_mask_from_classes

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"


############################
# 1. DATA & MODEL HELPERS  #
############################

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
        train=False, download=True, transform=transform_test
    )

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

def model_initialize(model_type):
    if model_type == 'vgg16':
        path_vgg = './models/BaseVGGc10.pth'
        model = torch.load(path_vgg)
        print(f"VGG16 printed:")
        print(model)
    elif model_type == 'vgg_c100':
        path_anet = './models/BaseVGGc100.pth'
        model = torch.load(path_anet)
        print(f"VGG16_C100 printed:")
        print(model)
    elif model_type == 'alexnet':
        path_anet = './models/Alexnet_C10SGD_BASE.pth'
        model = torch.load(path_anet)
        print(f"alexnet printed:")
        print(model)
    elif model_type== 'resnet56':
        path_resnet = './models/PretrainedResNet56.pth'
        model = torch.load(path_resnet)
        print(f"ResNet50 printed:")
        print(model)
    elif model_type== 'resnet56_c100':
        path_resnet = './models/PretrainedResNet56_c100.pth'
        model = torch.load(path_resnet)
        print(f"ResNet56 C100 printed:")
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

def run_cka_compression_all_convs_new(
    model_name,
    subset_trainloader,
    test_loader,
    num_clusters_ratio=0.5,      # unused now, kept for API compatibility
    layer_ratios=None,           # unused now
    batch_size=64,
    drop_threshold=5.0,          # max allowed drop vs ORIGINAL accuracy (in % points)
    save_path='./cka_models/vgg16_c10_CKA_all_convs.pth'
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

    # FLOPs accounting (rough, CIFAR-like 32x32 input)
    example_input_shape = ( 3, 32, 32)
    total_base_flops = 0.0
    for conv in conv_layers_template:
        total_base_flops += conv_flops(conv, example_input_shape,
                                       out_channels=conv.out_channels)
    total_comp_flops = total_base_flops  # adjust only for accepted layers
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

        # --- Blend (THIS IS THE KEY LINE) ---
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
    # print("********************************************************************************")

    return model



if __name__ == "__main__":
    model = model_initialize('resnet56')
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


    sieve = ClassSubsetSieve(
        train_dataset=trainloader.dataset,
        test_dataset=testloader.dataset,
        device=device,
        batch_size=64
    )
    model_subset = run_cka_compression_all_convs_new(
            model_name='resnet56',
            subset_trainloader=trainloader,
            # test_loader=testloader,
            test_loader = testloader,
            batch_size=64,
            drop_threshold=12.0,
            save_path=f'./cka_models/resnet56_cka_0.3_whole.pth'
        )
    model_subset = sieve.finetune_on_subset(
            model_subset,
            train_loader=trainloader,
            test_loader=testloader,
            # epochs=10,
            epochs=150,
            lr=5e-4
        )

    # --- EVALUATE ONLY ON THIS SUBSET ---
    acc = evaluate_accuracy(
        model_subset,
        testloader
    )


    print(f"No Seive, accuracy ({acc:.2f}%")


    save_path_comp = f'./cka_models/resnet56_cka_0.3_whole.pth'
    torch.save(model_subset,save_path_comp)
    print(f' Finetuned class reduced subset again saved at {save_path_comp}')
    # print(f"subset mdoel printing: {model_subset}")

    # compressed_model= model_subset
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


