import os
import sys
import csv
import time
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Ensure SpikingJelly uses PyTorch backends when CuPy is not available
from spikingjelly.activation_based import base, functional, neuron
cuda_available = torch.cuda.is_available()
try:
    import cupy
    cupy_available = True
except ImportError:
    cupy_available = False

if not cuda_available or not cupy_available:
    base.check_backend_library = lambda backend: None

from models.spikingresformer import spikingresformer_ti
from extract_features import CUBFeatureDataset, load_checkpoint

def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    cub_dir = os.path.join(repo_root, "datasets", "CUB_200_2011")
    csv_path = os.path.join(cub_dir, "processed_attributes.csv")
    images_dir = os.path.join(cub_dir, "images")
    checkpoint_path = os.path.join(repo_root, "checkpoints", "SpikingResformer-checkpoints", "spikingresformer_ti.pth")
    features_dir = os.path.join(cub_dir, "features")
    
    device = torch.device("cuda" if cuda_available else "cpu")
    
    print(f"Initializing SpikingResformer-Ti model on device: {device}...")
    model = spikingresformer_ti()
    model.to(device)
    load_checkpoint(model, checkpoint_path, device)
    
    # Configure SpikingJelly nodes
    for name, m in model.named_modules():
        if isinstance(m, neuron.BaseNode):
            m.backend = 'cupy' if (cuda_available and cupy_available) else 'torch'
            m.v_threshold = 0.01
    model.eval()
    
    target_module = dict(model.named_modules()).get("layers.2.6.down.0")
    if target_module is None:
        print("[ERROR] Target module layers.2.6.down.0 not found!")
        sys.exit(1)
        
    print("Preparing dataset and DataLoader...")
    dataset = CUBFeatureDataset(csv_path, images_dir)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    
    spike_temporal_list = []
    
    print("Starting temporal spike feature extraction loop...")
    total_batches = len(dataloader)
    start_time = time.time()
    
    for batch_idx, (batch_x, batch_ids) in enumerate(dataloader):
        batch_x = batch_x.to(device)
        B = batch_x.size(0)
        
        functional.reset_net(model)
        spikes_out = None
        
        def spike_hook_fn(module, input_args, output_spikes):
            nonlocal spikes_out
            spikes_out = output_spikes.detach()
            
        hook_handle = target_module.register_forward_hook(spike_hook_fn)
        with torch.no_grad():
            _ = model(batch_x)
        hook_handle.remove()
        
        # spikes_out shape: [T, B, C, H, W]
        # Spatial GAP: mean over H, W -> [T, B, C]
        spikes_gap = spikes_out.mean(dim=(-2, -1))
        # Permute to [B, T, C] and flatten to [B, T * C]
        spikes_flat = spikes_gap.permute(1, 0, 2).reshape(B, -1).cpu().numpy()
        spike_temporal_list.append(spikes_flat)
        
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.time() - start_time
            avg_time = elapsed / (batch_idx + 1)
            eta = avg_time * (total_batches - (batch_idx + 1))
            print(f"Processed batch {batch_idx + 1}/{total_batches} | Elapsed: {elapsed:.2f}s | Avg/Batch: {avg_time:.2f}s | ETA: {eta:.2f}s")
            
    # Concatenate and save
    print("\nConcatenating and saving temporal spike features...")
    spike_temporal = np.concatenate(spike_temporal_list, axis=0)
    
    save_path = os.path.join(features_dir, "spike_temporal_6144.npy")
    np.save(save_path, spike_temporal)
    
    # Verification
    print(f"\nSaved feature shape: {spike_temporal.shape}")
    has_nan = np.isnan(spike_temporal).any()
    has_inf = np.isinf(spike_temporal).any()
    mean_val = np.mean(spike_temporal)
    std_val = np.std(spike_temporal)
    
    print(f"Verify shape matches (11788, 6144): {spike_temporal.shape == (11788, 6144)}")
    print(f"Contains NaN: {has_nan}")
    print(f"Contains Inf: {has_inf}")
    print(f"Mean value:   {mean_val:.6f}")
    print(f"Std dev:      {std_val:.6f}")
    
    assert spike_temporal.shape == (11788, 6144), f"Expected shape (11788, 6144), got {spike_temporal.shape}"
    assert not has_nan, "Spike temporal features contain NaN values!"
    assert not has_inf, "Spike temporal features contain Inf values!"
    print("Verification passed successfully!")

if __name__ == "__main__":
    main()
