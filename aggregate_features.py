import os
import sys
# pyrefly: ignore [missing-import]
import numpy as np

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    features_dir = os.path.join(script_dir, "datasets", "CUB_200_2011", "features")
    
    post_path = os.path.join(features_dir, "post_reset_vmem.npy")
    pre_path = os.path.join(features_dir, "pre_reset_vmem.npy")
    
    # 1. Load files
    print("Loading original feature files...")
    if not os.path.exists(post_path):
        sys.exit(f"Error: Post-reset file not found: {post_path}")
    if not os.path.exists(pre_path):
        sys.exit(f"Error: Pre-reset file not found: {pre_path}")
        
    post_vmem = np.load(post_path)
    pre_vmem = np.load(pre_path)
    
    # 2. Verify shapes
    print(f"\nOriginal shapes:")
    print(f"  - post_reset_vmem.npy: {post_vmem.shape}")
    print(f"  - pre_reset_vmem.npy:  {pre_vmem.shape}")
    
    if post_vmem.shape != (11788, 6144):
        sys.exit(f"Error: post_reset_vmem shape is {post_vmem.shape}, expected (11788, 6144)")
    if pre_vmem.shape != (11788, 6144):
        sys.exit(f"Error: pre_reset_vmem shape is {pre_vmem.shape}, expected (11788, 6144)")
        
    # Stats before aggregation
    post_mean_before = np.mean(post_vmem)
    post_std_before = np.std(post_vmem)
    pre_mean_before = np.mean(pre_vmem)
    pre_std_before = np.std(pre_vmem)
    
    # 3. Reshape each array to (11788, 4, 1536)
    post_reshaped = post_vmem.reshape(-1, 4, 1536)
    pre_reshaped = pre_vmem.reshape(-1, 4, 1536)
    
    print(f"\nReshaped dimensions:")
    print(f"  - post_reset_vmem: {post_reshaped.shape}")
    print(f"  - pre_reset_vmem:  {pre_reshaped.shape}")
    
    # 4. Aggregate along the time dimension (axis 1) using sum
    post_sum = post_reshaped.sum(axis=1)
    pre_sum = pre_reshaped.sum(axis=1)
    
    # 5. Save paths
    post_save_path = os.path.join(features_dir, "post_reset_vmem_sum.npy")
    pre_save_path = os.path.join(features_dir, "pre_reset_vmem_sum.npy")
    
    # Ensure we don't modify the original files
    assert post_save_path != post_path, "Error: Target save path is the same as the original post-reset file!"
    assert pre_save_path != pre_path, "Error: Target save path is the same as the original pre-reset file!"
    
    np.save(post_save_path, post_sum)
    np.save(pre_save_path, pre_sum)
    print(f"\nSaved aggregated sum features to:")
    print(f"  - {post_save_path}")
    print(f"  - {pre_save_path}")
    
    # Stats after aggregation
    post_mean_after = np.mean(post_sum)
    post_std_after = np.std(post_sum)
    pre_mean_after = np.mean(pre_sum)
    pre_std_after = np.std(pre_sum)
    
    # 7. Print stats
    print(f"\nFinal shapes:")
    print(f"  - post_reset_vmem_sum.npy: {post_sum.shape}")
    print(f"  - pre_reset_vmem_sum.npy:  {pre_sum.shape}")
    
    # Verify no NaN or Inf values
    post_nan = np.isnan(post_sum).any()
    post_inf = np.isinf(post_sum).any()
    pre_nan = np.isnan(pre_sum).any()
    pre_inf = np.isinf(pre_sum).any()
    
    print(f"\nValue validation:")
    print(f"  - post_reset_vmem_sum contains NaN: {post_nan} | Inf: {post_inf}")
    print(f"  - pre_reset_vmem_sum contains NaN:  {pre_nan} | Inf: {pre_inf}")
    
    print(f"\nStats before and after aggregation:")
    print(f"  - Post-reset Vmem mean before: {post_mean_before:.6f} | after: {post_mean_after:.6f}")
    print(f"  - Post-reset Vmem std before:  {post_std_before:.6f} | after: {post_std_after:.6f}")
    print(f"  - Pre-reset Vmem mean before:  {pre_mean_before:.6f} | after: {pre_mean_after:.6f}")
    print(f"  - Pre-reset Vmem std before:   {pre_std_before:.6f} | after: {pre_std_after:.6f}")
    
    print("\nAggregation completed successfully!")

if __name__ == "__main__":
    main()
