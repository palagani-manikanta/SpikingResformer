import sys
import os
import types
# pyrefly: ignore [missing-import]
import torch    

# Add the parent directory to the python path to allow importing models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Check for CUDA availability
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

# Patch CuPy library check if CUDA is not available or if cupy cannot be imported
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

def main():
    print(f"--- STEP 1: Constructing SpikingResformer-Ti on Device: {device} ---")
    model = spikingresformer_ti()
    model.to(device)
    
    # Configure backends and thresholds
    # On CPU, we must use 'torch' backend.
    # On GPU, we use 'cupy' for performance, but we will override the target module's backend.
    for name, m in model.named_modules():
        if isinstance(m, neuron.BaseNode):
            if not cuda_available or not cupy_available:
                m.backend = 'torch'
            else:
                m.backend = 'cupy'
            # Lower threshold for all neurons to ensure spike propagation and active validation
            m.v_threshold = 0.01
            
    # Set model to evaluation mode
    model.eval()
    print(f"Model constructed and moved to {device} successfully.")

    # Locate target module layers.2.6.down.0
    print("\n--- STEP 2: Locating Target Penultimate Spiking Module ---")
    named_modules = dict(model.named_modules())
    target_module_name = "layers.2.6.down.0"
    target_module = named_modules.get(target_module_name)
    
    if target_module is None:
        raise ValueError(f"Could not find target module '{target_module_name}' in the model!")
    print(f"Target module found: {target_module}")
    print(f"Target module class: {target_module.__class__.__name__}")

    # Generate synthetic input
    torch.manual_seed(42)
    x = torch.randn(2, 3, 224, 224, device=device) * 10.0
    print(f"\nGenerated random input tensor of shape {list(x.shape)} scaled by 10.0 on {device}")

    # --- FEATURE 1: Spike-rate ---
    print("\n--- STEP 3: Extracting Feature 1 (Spike-rate) ---")
    functional.reset_net(model)
    target_module.store_v_seq = False
    
    spikes_out = None
    def spike_hook_fn(module, input_args, output_spikes):
        nonlocal spikes_out
        spikes_out = output_spikes
        
    hook_handle = target_module.register_forward_hook(spike_hook_fn)
    with torch.no_grad():
        _ = model(x)
    hook_handle.remove()
    
    spike_rate = spikes_out.mean(dim=0)
    print(f"Spike output shape: {list(spikes_out.shape)}")
    print(f"Spike rate shape (mean over T): {list(spike_rate.shape)}")
    print(f"Spike output non-zero %: {torch.count_nonzero(spikes_out).item() / spikes_out.numel() * 100:.4f}%")

    # --- FEATURE 2: Post-reset V_mem (Native store_v_seq) ---
    print("\n--- STEP 4: Extracting Feature 2 (Post-reset V_mem) ---")
    functional.reset_net(model)
    target_module.store_v_seq = True
    
    with torch.no_grad():
        _ = model(x)
        
    post_reset_vmem = target_module.v_seq
    print(f"Post-reset V_mem shape: {list(post_reset_vmem.shape)}")
    print(f"Post-reset V_mem non-zero %: {torch.count_nonzero(post_reset_vmem).item() / post_reset_vmem.numel() * 100:.4f}%")

    # --- FEATURE 3: Pre-reset V_mem (Custom Override) ---
    print("\n--- STEP 5: Extracting Feature 3 (Pre-reset V_mem) ---")
    functional.reset_net(model)
    
    # Crucial: Override the target module backend to 'torch' even on GPU
    # to allow Python loop interception of neuronal_charge/fire/reset
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
        
    pre_reset_vmem = target_module.v_seq
    print(f"Pre-reset V_mem shape: {list(pre_reset_vmem.shape)}")
    print(f"Pre-reset V_mem non-zero %: {torch.count_nonzero(pre_reset_vmem).item() / pre_reset_vmem.numel() * 100:.4f}%")

    functional.reset_net(model)

    # --- STEP 6: Assertions and Comparative Analysis ---
    print("\n--- STEP 6: Verifying Correctness & Assertions ---")
    
    assert spikes_out.shape == post_reset_vmem.shape == pre_reset_vmem.shape, "Shape mismatch between extracted tensors!"
    print("Assertion passed: All extracted sequence tensors have the same shape.")
    
    diff_post_pre = torch.max(torch.abs(post_reset_vmem - pre_reset_vmem)).item()
    print(f"Maximum absolute difference between post-reset and pre-reset Vmem: {diff_post_pre:.6f}")
    assert diff_post_pre > 0.0, "Post-reset and Pre-reset Vmem are identical!"
    print("Assertion passed: Post-reset and Pre-reset membrane potentials are numerically distinct.")

    fired_mask = (spikes_out == 1.0)
    num_fired = fired_mask.sum().item()
    total_elements = spikes_out.numel()
    print(f"Total neural elements: {total_elements}, Fired events: {num_fired} ({num_fired / total_elements * 100:.4f}%)")
    
    if num_fired > 0:
        fired_post_vals = post_reset_vmem[fired_mask]
        fired_pre_vals = pre_reset_vmem[fired_mask]
        
        # Native post-reset Vmem should be exactly 0.0 (v_reset)
        # Custom pre-reset Vmem should be >= v_threshold (0.01)
        assert torch.allclose(fired_post_vals, torch.tensor(0.0, device=device), atol=1e-6), "Post-reset Vmem at firing positions is not 0.0!"
        assert (fired_pre_vals >= target_module.v_threshold).all(), f"Pre-reset Vmem at firing positions contains values below threshold: {fired_pre_vals.min().item()}"
        print(f"Assertion passed: At all firing locations, Post-reset Vmem is 0.0 and Pre-reset Vmem is >= {target_module.v_threshold}.")
    else:
        raise ValueError("Error: No neurons fired in the target module. Try increasing input scale further.")

    # 4. Compare feature summary stats
    print("\nSummary Statistics of Penultimate Spiking Module:")
    print(f"Spike Output:     mean = {spikes_out.mean().item():.6f}, std = {spikes_out.std().item():.6f}")
    print(f"Post-Reset Vmem:  mean = {post_reset_vmem.mean().item():.6f}, std = {post_reset_vmem.std().item():.6f}")
    print(f"Pre-Reset Vmem:   mean = {pre_reset_vmem.mean().item():.6f}, std = {pre_reset_vmem.std().item():.6f}")
    
    print("\nVerification successful! The extraction pipeline is fully correct and ready.")

if __name__ == "__main__":
    main()
