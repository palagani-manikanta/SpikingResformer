import os
import sys
import csv
import types
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset, DataLoader
# pyrefly: ignore [missing-import]
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Ensure SpikingJelly uses PyTorch backends when CuPy is not available
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

# pyrefly: ignore [missing-import]
from spikingjelly.activation_based import base, functional, neuron
try:
    # pyrefly: ignore [missing-import]
    import cupy
    cupy_available = True
except ImportError:
    cupy_available = False

if not cuda_available or not cupy_available:
    base.check_backend_library = lambda backend: None

from models.spikingresformer import spikingresformer_ti

class CUBFeatureDataset(Dataset):
    def __init__(self, csv_path, images_dir):
        self.images_dir = images_dir
        self.data = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append({
                    "image_id": int(row["image_id"]),
                    "image_path": row["image_path"],
                    "class_id": int(row["class_id"]),
                    "split": row["split"]
                })
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        full_path = os.path.join(self.images_dir, item["image_path"])
        img = Image.open(full_path).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, item["image_id"]

def load_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dict = None
    if isinstance(checkpoint, torch.nn.Module):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        if "model" in checkpoint:
            val = checkpoint["model"]
            state_dict = val.state_dict() if isinstance(val, torch.nn.Module) else val
        elif "state_dict" in checkpoint:
            val = checkpoint["state_dict"]
            state_dict = val.state_dict() if isinstance(val, torch.nn.Module) else val
        else:
            state_dict = checkpoint
    else:
        raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")
        
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
    print("Checkpoint loaded successfully with strict=True.\n")

def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    cub_dir = os.path.join(repo_root, "datasets/CUB_200_2011")
    csv_path = os.path.join(cub_dir, "processed_attributes.csv")
    images_dir = os.path.join(cub_dir, "images")
    checkpoint_path = os.path.join(repo_root, "checkpoints", "SpikingResformer-checkpoints", "spikingresformer_ti.pth")
    features_dir = os.path.join(cub_dir, "features")
    
    os.makedirs(features_dir, exist_ok=True)
    
    # 1. Load SpikingResformer-Ti Model
    print("Initializing SpikingResformer-Ti model...")
    model = spikingresformer_ti()
    model.to(device)
    load_checkpoint(model, checkpoint_path, device)
    
    # Configure SpikingJelly nodes
    for name, m in model.named_modules():
        if isinstance(m, neuron.BaseNode):
            m.backend = 'cupy' if (cuda_available and cupy_available) else 'torch'
            m.v_threshold = 1.0
    model.eval()
    
    # Target penultimate spiking module
    target_module = dict(model.named_modules()).get("layers.2.6.down.0")
    if target_module is None:
        print("[ERROR] Target module layers.2.6.down.0 not found!")
        sys.exit(1)
        
    orig_backend = target_module.backend
    orig_multi_step_forward = target_module.multi_step_forward
    
    # 2. Setup Dataset and DataLoader
    print("Preparing dataset and DataLoader...")
    dataset = CUBFeatureDataset(csv_path, images_dir)
    # Batch size of 64 is safe and fast
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    
    spike_rates_list = []
    post_reset_list = []
    pre_reset_list = []
    image_ids_list = []
    
    # Define custom pre-reset forward override
    def custom_multi_step_forward(self, x_seq: torch.Tensor):
        self.v_float_to_tensor(x_seq[0])
        T = x_seq.shape[0]
        spike_seq = []
        v_pre_seq = []
        for t in range(T):
            self.neuronal_charge(x_seq[t])
            v_pre_seq.append(self.v.clone())
            spike = self.neuronal_fire()
            self.neuronal_reset(spike)
            spike_seq.append(spike)
        self.v_seq = torch.stack(v_pre_seq)
        return torch.stack(spike_seq)
    
    print(f"Device: {device}, cuda_available: {cuda_available}, cupy_available: {cupy_available}")
    print("Starting feature extraction loop...")
    total_batches = len(dataloader)
    
    import time
    loop_start_time = time.time()
    for batch_idx, (batch_x, batch_ids) in enumerate(dataloader):
        batch_start_time = time.time()
        batch_x = batch_x.to(device)
        B = batch_x.size(0)
        
        # Pass 1: Spike-rate Extraction
        t0 = time.time()
        functional.reset_net(model)
        target_module.store_v_seq = False
        spikes_out = None
        def spike_hook_fn(module, input_args, output_spikes):
            nonlocal spikes_out
            spikes_out = output_spikes.detach()
        hook_handle = target_module.register_forward_hook(spike_hook_fn)
        with torch.no_grad():
            _ = model(batch_x)
        hook_handle.remove()
        
        # spikes_out shape: [T, B, C, H, W]
        # Spike-rate: mean over T
        spike_rate = spikes_out.mean(dim=0) # [B, C, H, W]
        # Spatial GAP: mean over H, W
        spike_rate_gap = spike_rate.mean(dim=(-2, -1)).cpu().numpy() # [B, C]
        spike_rates_list.append(spike_rate_gap)
        t1 = time.time()
        
        # Pass 2: Post-reset Vmem Extraction
        functional.reset_net(model)
        target_module.store_v_seq = True
        with torch.no_grad():
            _ = model(batch_x)
        # target_module.v_seq shape: [T, B, C, H, W]
        post_reset = target_module.v_seq.detach()
        # Spatial GAP
        post_reset_gap = post_reset.mean(dim=(-2, -1)) # [T, B, C]
        # Permute to [B, T, C] and flatten to [B, T * C]
        post_reset_flat = post_reset_gap.permute(1, 0, 2).reshape(B, -1).cpu().numpy()
        post_reset_list.append(post_reset_flat)
        t2 = time.time()
        
        # Pass 3: Pre-reset Vmem Extraction
        functional.reset_net(model)
        target_module.store_v_seq = False
        target_module.backend = 'torch'
        target_module.multi_step_forward = types.MethodType(custom_multi_step_forward, target_module)
        with torch.no_grad():
            _ = model(batch_x)
        # target_module.v_seq shape: [T, B, C, H, W]
        pre_reset = target_module.v_seq.detach()
        # Restore target module backend configurations
        target_module.backend = orig_backend
        target_module.multi_step_forward = orig_multi_step_forward
        
        # Spatial GAP
        pre_reset_gap = pre_reset.mean(dim=(-2, -1)) # [T, B, C]
        # Permute to [B, T, C] and flatten to [B, T * C]
        pre_reset_flat = pre_reset_gap.permute(1, 0, 2).reshape(B, -1).cpu().numpy()
        pre_reset_list.append(pre_reset_flat)
        t3 = time.time()
        
        # Store IDs
        image_ids_list.append(batch_ids.numpy())
        
        # Print progress every 10 batches or at the end
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.time() - loop_start_time
            avg_time = elapsed / (batch_idx + 1)
            eta = avg_time * (total_batches - (batch_idx + 1))
            print(f"Processed batch {batch_idx + 1}/{total_batches} | Elapsed: {elapsed:.2f}s | Avg/Batch: {avg_time:.2f}s | ETA: {eta:.2f}s")
            
    # Concatenate all batches
    print("Concatenating and saving features...")
    spike_rates = np.concatenate(spike_rates_list, axis=0)
    post_reset = np.concatenate(post_reset_list, axis=0)
    pre_reset = np.concatenate(pre_reset_list, axis=0)
    image_ids = np.concatenate(image_ids_list, axis=0)
    
    # Save arrays
    np.save(os.path.join(features_dir, "spike_rates.npy"), spike_rates)
    np.save(os.path.join(features_dir, "post_reset_vmem.npy"), post_reset)
    np.save(os.path.join(features_dir, "pre_reset_vmem.npy"), pre_reset)
    np.save(os.path.join(features_dir, "image_ids.npy"), image_ids)
    
    # Assertions for verification (total CUB-200-2011 images = 11788)
    assert len(image_ids) == 11788, f"Expected 11788 images, got {len(image_ids)}"
    assert spike_rates.shape[0] == 11788, f"Expected spike_rates.shape[0] to be 11788, got {spike_rates.shape[0]}"
    assert post_reset.shape[0] == 11788, f"Expected post_reset.shape[0] to be 11788, got {post_reset.shape[0]}"
    assert pre_reset.shape[0] == 11788, f"Expected pre_reset.shape[0] to be 11788, got {pre_reset.shape[0]}"

    total_extraction_time = time.time() - loop_start_time

    print("\n--- Feature Extraction Stats ---")
    print(f"Total processed images   : {len(image_ids)}")
    print(f"Spike-rate array shape   : {spike_rates.shape}")
    print(f"Post-reset array shape   : {post_reset.shape}")
    print(f"Pre-reset array shape    : {pre_reset.shape}")
    print(f"Image ID array shape     : {image_ids.shape}")
    print(f"Total extraction time    : {total_extraction_time:.2f}s ({total_extraction_time / 3600:.2f} hours)")

    # Check for NaN and Inf values
    print("\n--- Checking for Invalid Values ---")
    for name, arr in [("Spike-rate", spike_rates), ("Post-reset Vmem", post_reset), ("Pre-reset Vmem", pre_reset), ("Image ID", image_ids)]:
        has_nan = np.isnan(arr).any()
        has_inf = np.isinf(arr).any()
        print(f"{name} contains NaN: {has_nan}")
        print(f"{name} contains Inf: {has_inf}")

    print(f"All features successfully extracted and saved to {features_dir}.")

if __name__ == "__main__":
    main()
