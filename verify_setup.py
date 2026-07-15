import os
import sys
import numpy as np
import pandas as pd

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    features_dir = os.path.join(script_dir, "datasets", "CUB_200_2011", "features")
    csv_path = os.path.join(script_dir, "datasets", "CUB_200_2011", "processed_attributes.csv")
    
    # 1. Check file existence
    files_to_check = {
        "spike_rates.npy": os.path.join(features_dir, "spike_rates.npy"),
        "post_reset_vmem.npy": os.path.join(features_dir, "post_reset_vmem.npy"),
        "pre_reset_vmem.npy": os.path.join(features_dir, "pre_reset_vmem.npy"),
        "image_ids.npy": os.path.join(features_dir, "image_ids.npy"),
        "processed_attributes.csv": csv_path
    }
    
    print("==================================================")
    print("PHASE 1 - VERIFYING EXPERIMENT SETUP")
    print("==================================================")
    
    all_exist = True
    for name, path in files_to_check.items():
        exists = os.path.exists(path)
        print(f"File '{name}' exists: {exists}")
        if not exists:
            all_exist = False
            
    if not all_exist:
        print("ERROR: One or more required files are missing. Verification failed.")
        sys.exit(1)
        
    # 2. Load files and verify properties
    print("\nLoading files...")
    spike_rates = np.load(files_to_check["spike_rates.npy"])
    post_reset = np.load(files_to_check["post_reset_vmem.npy"])
    pre_reset = np.load(files_to_check["pre_reset_vmem.npy"])
    image_ids = np.load(files_to_check["image_ids.npy"])
    df = pd.read_csv(csv_path)
    
    print("\nVerifying properties:")
    
    # - Feature dimensions
    print(f"  - spike_rates shape      : {spike_rates.shape} (Expected: (11788, 1536))")
    print(f"  - post_reset_vmem shape  : {post_reset.shape} (Expected: (11788, 6144))")
    print(f"  - pre_reset_vmem shape   : {pre_reset.shape} (Expected: (11788, 6144))")
    print(f"  - image_ids shape        : {image_ids.shape} (Expected: (11788,))")
    print(f"  - processed_attributes   : {df.shape} (Expected: (11788, 116))")
    
    assert spike_rates.shape == (11788, 1536), "Incorrect spike_rates shape"
    assert post_reset.shape == (11788, 6144), "Incorrect post_reset shape"
    assert pre_reset.shape == (11788, 6144), "Incorrect pre_reset shape"
    assert image_ids.shape == (11788,), "Incorrect image_ids shape"
    
    # - image IDs match
    ids_match = np.array_equal(image_ids, df["image_id"].values)
    print(f"  - Image IDs match between npy and csv: {ids_match}")
    assert ids_match, "Image IDs between image_ids.npy and processed_attributes.csv do not match!"
    
    # - train/test split is correct
    splits = df["split"].value_counts()
    print(f"  - Data splits:")
    print(f"      train count: {splits.get('train', 0)} (Expected: 5994)")
    print(f"      test count : {splits.get('test', 0)} (Expected: 5794)")
    assert splits.get('train', 0) == 5994, "Incorrect train split size"
    assert splits.get('test', 0) == 5794, "Incorrect test split size"
    
    # - exactly 112 concept labels exist
    metadata_cols = ["image_id", "image_path", "class_id", "split"]
    concept_cols = [col for col in df.columns if col not in metadata_cols]
    print(f"  - Number of concept columns: {len(concept_cols)} (Expected: 112)")
    assert len(concept_cols) == 112, f"Expected 112 concepts, found {len(concept_cols)}"
    
    # - no NaN or corrupted feature values
    nan_spike = np.isnan(spike_rates).any() or np.isinf(spike_rates).any()
    nan_post = np.isnan(post_reset).any() or np.isinf(post_reset).any()
    nan_pre = np.isnan(pre_reset).any() or np.isinf(pre_reset).any()
    print(f"  - NaNs/Infs in spike_rates: {nan_spike} (Expected: False)")
    print(f"  - NaNs/Infs in post_reset : {nan_post} (Expected: False)")
    print(f"  - NaNs/Infs in pre_reset  : {nan_pre} (Expected: False)")
    
    assert not nan_spike, "spike_rates contains NaNs or Infs"
    assert not nan_post, "post_reset contains NaNs or Infs"
    assert not nan_pre, "pre_reset contains NaNs or Infs"
    
    print("\nVerification successful! All checks passed.")
    print("==================================================")

if __name__ == "__main__":
    main()
