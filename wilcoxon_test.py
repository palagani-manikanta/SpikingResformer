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
    csv_path = os.path.join(script_dir, "results", "probe_results_comparison.csv")
    
    if not os.path.exists(csv_path):
        sys.exit(f"Error: Comparison CSV not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    roc_auc_pre = df["roc_auc_pre"].values
    roc_auc_post = df["roc_auc_post"].values
    delta = df["delta_pre_minus_post"].values
    
    # 3. Perform Wilcoxon signed-rank test
    stat, p_val = wilcoxon(roc_auc_pre, roc_auc_post)
    
    # 5. Compute statistics
    med_delta = np.median(delta)
    mean_delta = np.mean(delta)
    std_delta = np.std(delta)
    
    # 6. Interpretation text
    is_significant = p_val < 0.05
    if is_significant:
        interpretation = (
            f"The difference is statistically significant (p-value = {p_val:.4e} < 0.05).\n"
            f"This confirms that Pre-reset membrane potential features systematically and significantly outperform\n"
            f"Post-reset features in concept probing performance."
        )
    else:
        interpretation = (
            f"The difference is NOT statistically significant (p-value = {p_val:.4e} >= 0.05).\n"
            f"There is insufficient evidence that Pre-reset outperforms Post-reset."
        )
        
    # Print results to stdout
    print("="*60)
    print("PAIRED WILCOXON SIGNED-RANK TEST: PRE-RESET VS POST-RESET")
    print("="*60)
    print(f"Wilcoxon statistic: {stat:.4f}")
    print(f"p-value:            {p_val:.4e}")
    print(f"\nDelta (Pre - Post) Statistics:")
    print(f"  - Median delta: {med_delta:+.6f}")
    print(f"  - Mean delta:   {mean_delta:+.6f}")
    print(f"  - Std dev:      {std_delta:.6f}")
    print(f"\nInterpretation:")
    print(interpretation)
    
    # 8. Save text report to results/wilcoxon_pre_vs_post.txt
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "wilcoxon_pre_vs_post.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("PAIRED WILCOXON SIGNED-RANK TEST REPORT: PRE-RESET VS POST-RESET\n")
        f.write("="*60 + "\n")
        f.write(f"Wilcoxon statistic: {stat:.4f}\n")
        f.write(f"p-value:            {p_val:.4e}\n\n")
        f.write("Delta (Pre - Post) Statistics:\n")
        f.write(f"  - Median delta: {med_delta:+.6f}\n")
        f.write(f"  - Mean delta:   {mean_delta:+.6f}\n")
        f.write(f"  - Std dev:      {std_delta:.6f}\n\n")
        f.write("Interpretation:\n")
        f.write(interpretation + "\n")
        
    print(f"\nReport successfully saved to: {report_path}")

if __name__ == "__main__":
    main()
