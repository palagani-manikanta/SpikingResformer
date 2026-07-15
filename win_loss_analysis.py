import os
import sys
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "results", "probe_results_comparison.csv")
    
    if not os.path.exists(csv_path):
        sys.exit(f"Error: Comparison CSV not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    num_concepts = len(df)
    
    # Count wins, losses, ties
    wins = df[df["roc_auc_pre"] > df["roc_auc_post"]]
    losses = df[df["roc_auc_pre"] < df["roc_auc_post"]]
    ties = df[df["roc_auc_pre"] == df["roc_auc_post"]]
    
    num_wins = len(wins)
    num_losses = len(losses)
    num_ties = len(ties)
    
    # Compute percentages
    p_wins = (num_wins / num_concepts) * 100
    p_losses = (num_losses / num_concepts) * 100
    p_ties = (num_ties / num_concepts) * 100
    
    print("="*60)
    print("WIN/LOSS ANALYSIS: PRE-RESET VS POST-RESET MEMBRANE POTENTIALS")
    print("="*60)
    print(f"Total concepts analyzed: {num_concepts}")
    print(f"Pre-reset wins (Pre > Post) : {num_wins} ({p_wins:.2f}%)")
    print(f"Post-reset wins (Pre < Post): {num_losses} ({p_losses:.2f}%)")
    print(f"Ties (Pre == Post)          : {num_ties} ({p_ties:.2f}%)")
    
    # Produce ranked tables
    print("\n" + "="*60)
    print("TOP 10 LARGEST POSITIVE IMPROVEMENTS (PRE OVER POST)")
    print("="*60)
    top_improvements = df.sort_values(by="delta_pre_minus_post", ascending=False).head(10)
    cols = ["concept_name", "roc_auc_post", "roc_auc_pre", "delta_pre_minus_post"]
    print(top_improvements[cols].to_string(index=False))
    
    print("\n" + "="*60)
    print("TOP 10 LARGEST DECREASES (PRE OVER POST)")
    print("="*60)
    top_decreases = df.sort_values(by="delta_pre_minus_post", ascending=True).head(10)
    print(top_decreases[cols].to_string(index=False))
    
    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    if num_wins > num_losses:
        print(f"The win/loss distribution suggests a consistent improvement of Pre-reset over Post-reset.")
        print(f"Pre-reset features outperform Post-reset features on {num_wins} out of {num_concepts} concepts ({p_wins:.1f}%),")
        print(f"while Post-reset wins on only {num_losses} concepts ({p_losses:.1f}%).")
    elif num_losses > num_wins:
        print(f"The win/loss distribution suggests a consistent improvement of Post-reset over Pre-reset.")
        print(f"Post-reset features outperform Pre-reset features on {num_losses} out of {num_concepts} concepts ({p_losses:.1f}%),")
        print(f"while Pre-reset wins on only {num_wins} concepts ({p_wins:.1f}%).")
    else:
        print(f"The win/loss distribution suggests an even split between Pre-reset and Post-reset features.")

if __name__ == "__main__":
    main()
