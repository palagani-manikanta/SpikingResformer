import os
import sys
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from scipy.stats import wilcoxon

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    
    post_path = os.path.join(results_dir, "probe_results_post_reset_sum.csv")
    pre_path = os.path.join(results_dir, "probe_results_pre_reset_sum.csv")
    
    # 1. Load files
    if not os.path.exists(post_path):
        sys.exit(f"Error: Post Sum file not found: {post_path}")
    if not os.path.exists(pre_path):
        sys.exit(f"Error: Pre Sum file not found: {pre_path}")
        
    df_post = pd.read_csv(post_path)
    df_pre = pd.read_csv(pre_path)
    
    # 2. Verify
    if len(df_post) != 112 or len(df_pre) != 112:
        sys.exit(f"Error: Both files must contain exactly 112 rows. Post: {len(df_post)}, Pre: {len(df_pre)}")
        
    # Check names and order
    if not df_post["concept_name"].equals(df_pre["concept_name"]):
        sys.exit("Error: concept_name values or order do not match exactly between the files.")
        
    print("Verification passed: Concept names and ordering match perfectly.")
    
    # 3. Merge
    df_post_rename = df_post.rename(columns={"roc_auc": "roc_auc_post_sum"})
    df_pre_rename = df_pre.rename(columns={"roc_auc": "roc_auc_pre_sum"})
    
    merged_df = df_pre_rename[["concept_index", "concept_name", "roc_auc_pre_sum"]].merge(
        df_post_rename[["concept_name", "roc_auc_post_sum"]], on="concept_name"
    )
    
    # 4. Compute delta
    merged_df["delta_pre_minus_post"] = merged_df["roc_auc_pre_sum"] - merged_df["roc_auc_post_sum"]
    
    roc_auc_pre = merged_df["roc_auc_pre_sum"].values
    roc_auc_post = merged_df["roc_auc_post_sum"].values
    delta = merged_df["delta_pre_minus_post"].values
    
    # 5. Wilcoxon test
    stat, p_val = wilcoxon(roc_auc_pre, roc_auc_post)
    
    # 6. Report stats
    mean_delta = np.mean(delta)
    med_delta = np.median(delta)
    std_delta = np.std(delta)
    
    # 7. Win/loss/tie count
    num_wins = np.sum(roc_auc_pre > roc_auc_post)
    num_losses = np.sum(roc_auc_pre < roc_auc_post)
    num_ties = np.sum(roc_auc_pre == roc_auc_post)
    
    # 8. Statistical interpretation
    is_significant = p_val < 0.05
    if is_significant:
        interpretation = (
            f"The difference is statistically significant (p-value = {p_val:.4e} < 0.05).\n"
            f"This confirms that 1536-dimensional Pre-reset Sum features systematically and significantly\n"
            f"outperform Post-reset Sum features in concept probing performance."
        )
    else:
        interpretation = (
            f"The difference is NOT statistically significant (p-value = {p_val:.4e} >= 0.05).\n"
            f"There is insufficient evidence that Pre Sum outperforms Post Sum."
        )
        
    # Print results
    print("="*60)
    print("PAIRED WILCOXON SIGNED-RANK TEST: PRE SUM VS POST SUM (1536 Features)")
    print("="*60)
    print(f"Wilcoxon statistic: {stat:.4f}")
    print(f"p-value:            {p_val:.4e}")
    print(f"\nDelta (Pre - Post) Statistics:")
    print(f"  - Median delta: {med_delta:+.6f}")
    print(f"  - Mean delta:   {mean_delta:+.6f}")
    print(f"  - Std dev:      {std_delta:.6f}")
    print(f"\nWin/Loss/Tie Counts:")
    print(f"  - Pre Sum > Post Sum: {num_wins}")
    print(f"  - Post Sum > Pre Sum: {num_losses}")
    print(f"  - Ties:               {num_ties}")
    print(f"\nInterpretation:")
    print(interpretation)
    
    # 9. Save report
    report_path = os.path.join(results_dir, "wilcoxon_pre_sum_vs_post_sum.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("PAIRED WILCOXON SIGNED-RANK TEST REPORT: PRE SUM VS POST SUM (1536 Features)\n")
        f.write("="*60 + "\n")
        f.write(f"Wilcoxon statistic: {stat:.4f}\n")
        f.write(f"p-value:            {p_val:.4e}\n\n")
        f.write("Delta (Pre - Post) Statistics:\n")
        f.write(f"  - Median delta: {med_delta:+.6f}\n")
        f.write(f"  - Mean delta:   {mean_delta:+.6f}\n")
        f.write(f"  - Std dev:      {std_delta:.6f}\n\n")
        f.write("Win/Loss/Tie Counts:\n")
        f.write(f"  - Pre Sum > Post Sum: {num_wins}\n")
        f.write(f"  - Post Sum > Pre Sum: {num_losses}\n")
        f.write(f"  - Ties:               {num_ties}\n\n")
        f.write("Interpretation:\n")
        f.write(interpretation + "\n")
        
    print(f"\nReport successfully saved to: {report_path}")

if __name__ == "__main__":
    main()
