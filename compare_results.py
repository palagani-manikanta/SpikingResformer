import os
import sys
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    
    spike_path = os.path.join(results_dir, "probe_results_spike_rate.csv")
    post_path = os.path.join(results_dir, "probe_results_post_reset.csv")
    pre_path = os.path.join(results_dir, "probe_results_pre_reset.csv")
    
    # Check if files exist
    for path in [spike_path, post_path, pre_path]:
        if not os.path.exists(path):
            sys.exit(f"Error: Required file not found: {path}")
            
    df_spike = pd.read_csv(spike_path)
    df_post = pd.read_csv(post_path)
    df_pre = pd.read_csv(pre_path)
    
    # 2. Verification
    # Check row counts
    for df, name in [(df_spike, "probe_results_spike_rate.csv"), 
                     (df_post, "probe_results_post_reset.csv"), 
                     (df_pre, "probe_results_pre_reset.csv")]:
        if len(df) != 112:
            sys.exit(f"Error: File {name} contains {len(df)} rows, expected exactly 112.")
            
    # Check uniqueness of concept_name
    for df, name in [(df_spike, "probe_results_spike_rate.csv"), 
                     (df_post, "probe_results_post_reset.csv"), 
                     (df_pre, "probe_results_pre_reset.csv")]:
        if df["concept_name"].nunique() != 112:
            sys.exit(f"Error: concept_name is not unique in {name}.")
            
    # Check that all three contain exactly the same concepts
    concepts_spike = set(df_spike["concept_name"])
    concepts_post = set(df_post["concept_name"])
    concepts_pre = set(df_pre["concept_name"])
    
    if concepts_spike != concepts_post:
        sys.exit("Error: concept_name sets in spike_rate and post_reset do not match.")
    if concepts_spike != concepts_pre:
        sys.exit("Error: concept_name sets in spike_rate and pre_reset do not match.")
        
    print("Verification passed: Exactly 112 unique matching concepts in all three files.")

    # 3 & 4. Merge and rename columns
    df_spike_rename = df_spike.rename(columns={"roc_auc": "roc_auc_spike"})
    df_post_rename = df_post.rename(columns={"roc_auc": "roc_auc_post"})
    df_pre_rename = df_pre.rename(columns={"roc_auc": "roc_auc_pre"})
    
    merged_df = df_spike_rename[["concept_index", "concept_name", "roc_auc_spike"]].merge(
        df_post_rename[["concept_name", "roc_auc_post"]], on="concept_name"
    ).merge(
        df_pre_rename[["concept_name", "roc_auc_pre"]], on="concept_name"
    )
    
    # 5. Compute deltas
    merged_df["delta_post_minus_spike"] = merged_df["roc_auc_post"] - merged_df["roc_auc_spike"]
    merged_df["delta_pre_minus_spike"] = merged_df["roc_auc_pre"] - merged_df["roc_auc_spike"]
    merged_df["delta_pre_minus_post"] = merged_df["roc_auc_pre"] - merged_df["roc_auc_post"]
    
    # 6. Save the merged table
    output_path = os.path.join(results_dir, "probe_results_comparison.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"Merged comparison saved to: {output_path}")
    
    # 7. Print summary metrics
    print(f"\nNumber of merged concepts: {len(merged_df)}")
    
    print("\nFirst five rows:")
    print(merged_df.head(5).to_string(index=False))
    
    print("\nMean of each delta column:")
    print(f"  - Mean delta_post_minus_spike: {merged_df['delta_post_minus_spike'].mean():+.6f}")
    print(f"  - Mean delta_pre_minus_spike:  {merged_df['delta_pre_minus_spike'].mean():+.6f}")
    print(f"  - Mean delta_pre_minus_post:  {merged_df['delta_pre_minus_post'].mean():+.6f}")
    
    for col in ["delta_post_minus_spike", "delta_pre_minus_spike", "delta_pre_minus_post"]:
        max_imp_idx = merged_df[col].idxmax()
        max_dec_idx = merged_df[col].idxmin()
        
        max_imp_row = merged_df.loc[max_imp_idx]
        max_dec_row = merged_df.loc[max_dec_idx]
        
        print(f"\nStats for {col}:")
        print(f"  - Maximum improvement: {max_imp_row[col]:+.6f} ({max_imp_row['concept_name']})")
        print(f"  - Maximum decrease:    {max_dec_row[col]:+.6f} ({max_dec_row['concept_name']})")

if __name__ == "__main__":
    main()
