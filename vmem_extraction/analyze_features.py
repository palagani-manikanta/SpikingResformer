import sys
import os
import types
# pyrefly: ignore [missing-import]
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

def pearson_correlation(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x - mean_x
    ym = y - mean_y
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))
    return (r_num / r_den).item() if r_den != 0 else 0.0

def main():
    print(f"Loading SpikingResformer-Ti on Device: {device}...")
    model = spikingresformer_ti()
    model.to(device)
    
    for name, m in model.named_modules():
        if isinstance(m, neuron.BaseNode):
            m.backend = 'cupy' if (cuda_available and cupy_available) else 'torch'
            m.v_threshold = 0.01
    model.eval()

    target_module = dict(model.named_modules()).get("layers.2.6.down.0")
    
    torch.manual_seed(42)
    x = torch.randn(2, 3, 224, 224, device=device) * 10.0

    # 1. Extract Spike-rate
    functional.reset_net(model)
    target_module.store_v_seq = False
    spikes_out = None
    def spike_hook_fn(module, input_args, output_spikes):
        nonlocal spikes_out
        spikes_out = output_spikes.detach()
    hook_handle = target_module.register_forward_hook(spike_hook_fn)
    with torch.no_grad():
        _ = model(x)
    hook_handle.remove()
    
    # 2. Extract Post-reset Vmem
    functional.reset_net(model)
    target_module.store_v_seq = True
    with torch.no_grad():
        _ = model(x)
    post_reset_vmem = target_module.v_seq.detach()

    # 3. Extract Pre-reset Vmem
    functional.reset_net(model)
    target_module.backend = 'torch'
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
    target_module.multi_step_forward = types.MethodType(custom_multi_step_forward, target_module)
    with torch.no_grad():
        _ = model(x)
    pre_reset_vmem = target_module.v_seq.detach()
    
    functional.reset_net(model)

    # Flatten for statistical calculations
    spikes_flat = spikes_out.flatten()
    post_flat = post_reset_vmem.flatten()
    pre_flat = pre_reset_vmem.flatten()

    print("\n==================================================")
    print("                FEATURE DENSITY & SUMMARY STATS   ")
    print("==================================================")
    for name, tensor in [("Spike Output", spikes_flat), ("Post-Reset Vmem", post_flat), ("Pre-Reset Vmem", pre_flat)]:
        nz_pct = torch.count_nonzero(tensor).item() / tensor.numel() * 100
        mean_val = tensor.mean().item()
        std_val = tensor.std().item()
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        print(f"{name:16s} | Density (Non-Zero): {nz_pct:7.3f}% | Mean: {mean_val:8.5f} | Std: {std_val:8.5f} | Min: {min_val:8.5f} | Max: {max_val:8.5f}")

    print("\n==================================================")
    print("                PERCENTILE DISTRIBUTION           ")
    print("==================================================")
    percentiles = torch.tensor([0.25, 0.50, 0.75, 0.90, 0.99], device=device)
    for name, tensor in [("Post-Reset Vmem", post_flat), ("Pre-Reset Vmem", pre_flat)]:
        q = torch.quantile(tensor, percentiles)
        q_list = [f"{p*100:.0f}%: {val:.5f}" for p, val in zip(percentiles.tolist(), q.tolist())]
        print(f"{name:16s} | " + " | ".join(q_list))

    print("\n==================================================")
    print("                CONDITIONAL DISTRIBUTIONS         ")
    print("==================================================")
    fired_mask = (spikes_flat == 1.0)
    not_fired_mask = (spikes_flat == 0.0)
    
    print(f"Fired states (count = {fired_mask.sum().item()}):")
    print(f"  Post-Reset Vmem - Mean: {post_flat[fired_mask].mean().item():8.5f} | Std: {post_flat[fired_mask].std().item():8.5f}")
    print(f"  Pre-Reset Vmem  - Mean: {pre_flat[fired_mask].mean().item():8.5f} | Std: {pre_flat[fired_mask].std().item():8.5f}")
    
    print(f"Non-fired states (count = {not_fired_mask.sum().item()}):")
    print(f"  Post-Reset Vmem - Mean: {post_flat[not_fired_mask].mean().item():8.5f} | Std: {post_flat[not_fired_mask].std().item():8.5f}")
    print(f"  Pre-Reset Vmem  - Mean: {pre_flat[not_fired_mask].mean().item():8.5f} | Std: {pre_flat[not_fired_mask].std().item():8.5f}")

    print("\n==================================================")
    print("                PEARSON CORRELATION MATRIX        ")
    print("==================================================")
    # Average spikes over T (Spike-rate) for correlation comparison, but to keep shapes aligned
    # we can also correlate the time-step flattened representations.
    r_spike_post = pearson_correlation(spikes_flat, post_flat)
    r_spike_pre = pearson_correlation(spikes_flat, pre_flat)
    r_post_pre = pearson_correlation(post_flat, pre_flat)
    print(f"Correlation (Spikes, Post-Reset Vmem): {r_spike_post:8.5f}")
    print(f"Correlation (Spikes, Pre-Reset Vmem) : {r_spike_pre:8.5f}")
    print(f"Correlation (Post-Reset, Pre-Reset)  : {r_post_pre:8.5f}")
    
if __name__ == "__main__":
    main()
