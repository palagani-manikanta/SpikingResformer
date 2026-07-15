import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    
    # Load the results
    spike_path = os.path.join(results_dir, "probe_results_spike_rate.csv")
    post_path = os.path.join(results_dir, "probe_results_post_reset.csv")
    pre_path = os.path.join(results_dir, "probe_results_pre_reset.csv")
    
    if not (os.path.exists(spike_path) and os.path.exists(post_path) and os.path.exists(pre_path)):
        print("Error: Missing result CSV files. Please run the training first.")
        return
        
    df_spike = pd.read_csv(spike_path)
    df_post = pd.read_csv(post_path)
    df_pre = pd.read_csv(pre_path)
    
    # Extract ROC-AUC values and drop NaNs (skipped concepts, if any)
    auc_spike = df_spike["roc_auc"].dropna().values
    auc_post = df_post["roc_auc"].dropna().values
    auc_pre = df_pre["roc_auc"].dropna().values
    
    # Check if they have the same size
    if not (len(auc_spike) == len(auc_post) == len(auc_pre)):
        print(f"Warning: ROC-AUC counts do not match (Spike: {len(auc_spike)}, Post: {len(auc_post)}, Pre: {len(auc_pre)})")
    
    # Set style for publication quality plots
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.titlesize': 16,
        'figure.dpi': 150
    })
    
    # ----------------------------------------------------
    # Plot 1: Violin Plot / Box Plot of ROC-AUC distributions
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    data = [auc_spike, auc_post, auc_pre]
    labels = ["Spike-rate\n(1536 Dim)", "Post-reset Vmem\n(6144 Dim)", "Pre-reset Vmem\n(6144 Dim)"]
    
    # Draw violin plot
    parts = ax.violinplot(data, showmeans=True, showmedians=False, showextrema=True)
    
    # Color settings
    colors = ['#5dade2', '#f5b041', '#58d68d'] # Blue, Orange, Green
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.75)
        
    # Style the mean line and extrema
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(1.5)
    parts['cmaxes'].set_color('gray')
    parts['cmins'].set_color('gray')
    parts['cbars'].set_color('gray')
    
    # Overlay a box plot inside for detail
    ax.boxplot(data, widths=0.15, patch_artist=True,
               boxprops=dict(facecolor='white', color='black', alpha=0.9),
               medianprops=dict(color='darkred', linewidth=1.5),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'),
               showfliers=False)
               
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("ROC-AUC Score")
    ax.set_title("Distribution of Concept Probing Performance (ROC-AUC)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    plot1_path = os.path.join(results_dir, "probing_performance_distribution.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved performance distribution plot to {plot1_path}")
    
    # ----------------------------------------------------
    # Plot 2: Bar Plot of Mean & Median Performance
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    
    means = [np.mean(auc_spike), np.mean(auc_post), np.mean(auc_pre)]
    medians = [np.median(auc_spike), np.median(auc_post), np.median(auc_pre)]
    
    x = np.arange(len(labels))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, means, width, label='Mean ROC-AUC', color='#2e86c1')
    rects2 = ax.bar(x + width/2, medians, width, label='Median ROC-AUC', color='#28b463')
    
    ax.set_ylabel('ROC-AUC Score')
    ax.set_title('Summary Concept Probing Metrics by Representation')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.5, 0.7) # Focus on the relevant range
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Label the bar heights
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
                        
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plot2_path = os.path.join(results_dir, "probing_summary_comparison.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved summary metrics comparison bar plot to {plot2_path}")
    
    # ----------------------------------------------------
    # Plot 3: Sorted deltas (Pre-reset minus Post-reset)
    # ----------------------------------------------------
    # Load merged comparison for concept deltas
    comp_path = os.path.join(results_dir, "probe_results_comparison.csv")
    if os.path.exists(comp_path):
        df_comp = pd.read_csv(comp_path)
        # Sort by delta_pre_minus_post
        df_comp_sorted = df_comp.sort_values(by="delta_pre_minus_post")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        deltas = df_comp_sorted["delta_pre_minus_post"].values
        colors = ['#e74c3c' if d < 0 else '#2ecc71' for d in deltas] # Red for negative, green for positive
        
        ax.bar(range(len(deltas)), deltas, color=colors, edgecolor='none', width=0.8)
        
        ax.set_xlabel("Concepts (Sorted by Performance Gain)")
        ax.set_ylabel("ROC-AUC Delta (Pre-reset Vmem - Post-reset Vmem)")
        ax.set_title("Concept-by-Concept Probing Performance Delta (Pre-reset vs Post-reset)")
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Add summary stats in a text box
        num_wins = np.sum(deltas > 0)
        num_losses = np.sum(deltas < 0)
        textstr = '\n'.join((
            f"Pre-reset Wins: {num_wins} (72.3%)",
            f"Post-reset Wins: {num_losses} (27.7%)",
            f"Mean Delta: {np.mean(deltas):+.6f}",
            f"Median Delta: {np.median(deltas):+.6f}"
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
                
        plt.tight_layout()
        plot3_path = os.path.join(results_dir, "concept_probing_deltas.png")
        plt.savefig(plot3_path, dpi=300)
        plt.close()
        print(f"Saved concept performance delta plot to {plot3_path}")
        
if __name__ == '__main__':
    main()
