"""
visualize_results.py - Generate publication-quality comparison visualizations

Creates comprehensive comparison plots for the academic report:
1. Model performance comparison (Accuracy, F1 scores)
2. Per-class F1 comparison
3. Training time and model size comparison
4. Example predictions comparison
5. Compression results comparison

All plots saved to outputs/plots/

Usage:
    python visualize_results.py
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-quality figures
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
COLORS = {'BERT': '#3498db', 'RoBERTa': '#2ecc71', 'ELECTRA': '#e74c3c'}

# Create output directory for plots
PLOTS_DIR = "outputs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_results():
    """Load all result JSON files."""
    results = {}
    
    result_files = {
        "BERT": "outputs/bert_results.json",
        "RoBERTa": "outputs/roberta_results.json",
        "ELECTRA": "outputs/electra_results.json"
    }
    
    for model_name, filepath in result_files.items():
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                results[model_name] = json.load(f)
        else:
            print(f"Warning: {filepath} not found")
    
    return results


def load_compression_results():
    """Load compression results."""
    filepath = "outputs/compression_results.json"
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None


def get_size_mb(result_dict):
    """Get file size from result dict, handling different key names."""
    # Try different possible key names
    for key in ['file_size_mb', 'size_mb', 'size']:
        if key in result_dict:
            return result_dict[key]
    return 0


# =============================================================================
# PLOT 1: Model Performance Comparison (Improved)
# =============================================================================
def plot_model_performance(results):
    """Create comprehensive performance comparison."""
    
    models = list(results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    
    # Plot 1: Accuracy
    accuracies = [results[m]["accuracy"] for m in models]
    bars1 = axes[0].bar(models, accuracies, color=[COLORS[m] for m in models], edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy')
    axes[0].set_ylim(0.90, 0.95)
    axes[0].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Min. Required (80%)')
    for bar, acc in zip(bars1, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: F1 Scores (Macro and Weighted)
    x = np.arange(len(models))
    width = 0.35
    f1_macro = [results[m]["f1_macro"] for m in models]
    f1_weighted = [results[m]["f1_weighted"] for m in models]
    
    bars2a = axes[1].bar(x - width/2, f1_macro, width, label='F1 Macro', color='#3498db', edgecolor='black')
    bars2b = axes[1].bar(x + width/2, f1_weighted, width, label='F1 Weighted', color='#9b59b6', edgecolor='black')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('F1 Scores')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].set_ylim(0.88, 0.95)
    axes[1].legend(loc='upper right')
    axes[1].grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars2a, f1_macro):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2b, f1_weighted):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Overall Ranking
    rankings = {m: [] for m in models}
    metrics_for_rank = ['accuracy', 'f1_macro', 'f1_weighted']
    for metric in metrics_for_rank:
        sorted_models = sorted(models, key=lambda x: results[x][metric], reverse=True)
        for rank, model in enumerate(sorted_models):
            rankings[model].append(rank + 1)
    
    avg_ranks = [np.mean(rankings[m]) for m in models]
    bars3 = axes[2].bar(models, avg_ranks, color=[COLORS[m] for m in models], edgecolor='black', linewidth=1.2)
    axes[2].set_ylabel('Average Rank (lower is better)')
    axes[2].set_title('Overall Ranking')
    axes[2].set_ylim(0, 4)
    axes[2].invert_yaxis()
    for bar, rank in zip(bars3, avg_ranks):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{rank:.2f}', ha='center', va='top', fontsize=11, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = f"{PLOTS_DIR}/01_model_performance_comparison.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# PLOT 2: Per-Class F1 Comparison
# =============================================================================
def plot_per_class_f1(results):
    """Create detailed per-class F1 comparison."""
    
    models = list(results.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Per-Class Performance Analysis', fontsize=16, fontweight='bold', y=1.02)
    
    # Plot 1: Grouped bar chart
    x = np.arange(len(LABEL_NAMES))
    width = 0.25
    
    for i, model in enumerate(models):
        f1_scores = [results[model][f"f1_{label}"] for label in LABEL_NAMES]
        bars = axes[0].bar(x + i*width, f1_scores, width, label=model, 
                          color=COLORS[model], edgecolor='black', linewidth=0.8)
    
    axes[0].set_xlabel('Emotion Class')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_title('F1 Score by Emotion Class')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(LABEL_NAMES, rotation=45, ha='right')
    axes[0].legend(loc='upper right')
    axes[0].set_ylim(0.80, 1.0)
    axes[0].axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Heatmap
    f1_matrix = []
    for model in models:
        f1_scores = [results[model][f"f1_{label}"] for label in LABEL_NAMES]
        f1_matrix.append(f1_scores)
    
    f1_df = pd.DataFrame(f1_matrix, index=models, columns=LABEL_NAMES)
    
    sns.heatmap(f1_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0.85, vmax=0.98, ax=axes[1],
                cbar_kws={'label': 'F1 Score'}, linewidths=0.5)
    axes[1].set_title('F1 Score Heatmap')
    axes[1].set_xlabel('Emotion Class')
    axes[1].set_ylabel('Model')
    
    plt.tight_layout()
    save_path = f"{PLOTS_DIR}/02_per_class_f1_comparison.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# PLOT 3: Training Resources Comparison
# =============================================================================
def plot_training_resources(results):
    """Compare training time, model size, and parameters."""
    
    models = list(results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Training Resources Comparison', fontsize=16, fontweight='bold', y=1.02)
    
    # Plot 1: Training Time
    times = [results[m]["training_time"] / 60 for m in models]
    bars1 = axes[0].bar(models, times, color=[COLORS[m] for m in models], edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Training Time (minutes)')
    axes[0].set_title('Training Time')
    for bar, t in zip(bars1, times):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{t:.1f} min', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Model Size
    sizes = [results[m]["size_mb"] for m in models]
    bars2 = axes[1].bar(models, sizes, color=[COLORS[m] for m in models], edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('Model Size (MB)')
    axes[1].set_title('Model Size')
    for bar, s in zip(bars2, sizes):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{s:.1f} MB', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Number of Parameters
    params = [results[m]["parameters"] / 1e6 for m in models]
    bars3 = axes[2].bar(models, params, color=[COLORS[m] for m in models], edgecolor='black', linewidth=1.2)
    axes[2].set_ylabel('Parameters (Millions)')
    axes[2].set_title('Number of Parameters')
    for bar, p in zip(bars3, params):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{p:.1f}M', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = f"{PLOTS_DIR}/03_training_resources_comparison.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# PLOT 4: Efficiency Analysis (Performance vs Resources)
# =============================================================================
def plot_efficiency_analysis(results):
    """Analyze efficiency: performance relative to resources."""
    
    models = list(results.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Efficiency Analysis', fontsize=16, fontweight='bold', y=1.02)
    
    # Plot 1: Accuracy vs Training Time (Scatter)
    times = [results[m]["training_time"] / 60 for m in models]
    accuracies = [results[m]["accuracy"] for m in models]
    
    for i, model in enumerate(models):
        axes[0].scatter(times[i], accuracies[i], s=200, c=COLORS[model], 
                       label=model, edgecolors='black', linewidth=2, zorder=3)
    
    axes[0].set_xlabel('Training Time (minutes)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy vs Training Time')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    for i, model in enumerate(models):
        axes[0].annotate(model, (times[i], accuracies[i]), 
                        xytext=(10, 10), textcoords='offset points', fontsize=9)
    
    # Plot 2: F1 Macro vs Model Size (Scatter)
    sizes = [results[m]["size_mb"] for m in models]
    f1_macros = [results[m]["f1_macro"] for m in models]
    
    for i, model in enumerate(models):
        axes[1].scatter(sizes[i], f1_macros[i], s=200, c=COLORS[model], 
                       label=model, edgecolors='black', linewidth=2, zorder=3)
    
    axes[1].set_xlabel('Model Size (MB)')
    axes[1].set_ylabel('F1 Macro')
    axes[1].set_title('F1 Macro vs Model Size')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    for i, model in enumerate(models):
        axes[1].annotate(model, (sizes[i], f1_macros[i]), 
                        xytext=(10, 10), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    save_path = f"{PLOTS_DIR}/04_efficiency_analysis.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# PLOT 5: Comprehensive Summary Table (as figure)
# =============================================================================
def plot_summary_table(results):
    """Create a visual summary table."""
    
    models = list(results.keys())
    
    data = []
    for model in models:
        r = results[model]
        data.append([
            model,
            f"{r['parameters']/1e6:.1f}M",
            f"{r['size_mb']:.1f}",
            f"{r['training_time']/60:.1f}",
            f"{r['accuracy']*100:.2f}%",
            f"{r['f1_macro']:.4f}",
            f"{r['f1_weighted']:.4f}"
        ])
    
    columns = ['Model', 'Parameters', 'Size (MB)', 'Time (min)', 'Accuracy', 'F1 Macro', 'F1 Weighted']
    
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')
    
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Highlight best model row (BERT - row 1)
    for i in range(len(columns)):
        table[(1, i)].set_facecolor('#d5f4e6')
    
    plt.title('Model Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    
    save_path = f"{PLOTS_DIR}/05_summary_table.png"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# PLOT 6: Compression Results
# =============================================================================
def plot_compression_results():
    """Create comprehensive compression comparison."""
    
    results = load_compression_results()
    if results is None:
        print("Compression results not found. Skipping compression plots.")
        return
    
    models = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Compression Analysis', fontsize=16, fontweight='bold', y=1.02)
    
    colors = ['#3498db', '#9b59b6', '#e67e22', '#e74c3c']
    
    # Plot 1: File Size - use helper function to handle different key names
    sizes = [get_size_mb(results[m]) for m in models]
    bars1 = axes[0, 0].bar(models, sizes, color=colors, edgecolor='black', linewidth=1.2)
    axes[0, 0].set_ylabel('File Size (MB)')
    axes[0, 0].set_title('Model File Size After Compression')
    axes[0, 0].tick_params(axis='x', rotation=45)
    for bar, s in zip(bars1, sizes):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                       f'{s:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Accuracy
    accuracies = [results[m]["accuracy"] for m in models]
    bars2 = axes[0, 1].bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.2)
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy After Compression')
    axes[0, 1].set_ylim(0.90, 0.95)
    axes[0, 1].tick_params(axis='x', rotation=45)
    for bar, acc in zip(bars2, accuracies):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                       f'{acc:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: F1 Macro
    f1_macros = [results[m]["f1_macro"] for m in models]
    bars3 = axes[1, 0].bar(models, f1_macros, color=colors, edgecolor='black', linewidth=1.2)
    axes[1, 0].set_ylabel('F1 Macro')
    axes[1, 0].set_title('F1 Macro After Compression')
    axes[1, 0].set_ylim(0.88, 0.93)
    axes[1, 0].tick_params(axis='x', rotation=45)
    for bar, f1 in zip(bars3, f1_macros):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                       f'{f1:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Inference Time
    inf_times = [results[m]["inference_time"] for m in models]
    bars4 = axes[1, 1].bar(models, inf_times, color=colors, edgecolor='black', linewidth=1.2)
    axes[1, 1].set_ylabel('Inference Time (seconds)')
    axes[1, 1].set_title('Inference Time')
    axes[1, 1].tick_params(axis='x', rotation=45)
    for bar, t in zip(bars4, inf_times):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{t:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = f"{PLOTS_DIR}/06_compression_comparison.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# PLOT 7: Compression Trade-off Analysis
# =============================================================================
def plot_compression_tradeoff():
    """Analyze compression trade-offs."""
    
    results = load_compression_results()
    if results is None:
        return
    
    models = list(results.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3498db', '#9b59b6', '#e67e22', '#e74c3c']
    markers = ['o', 's', '^', 'D']
    
    # Scatter plot: Size vs F1 Macro
    for i, model in enumerate(models):
        size = get_size_mb(results[model])
        ax.scatter(size, results[model]["f1_macro"],
                  s=300, c=colors[i], marker=markers[i], label=model,
                  edgecolors='black', linewidth=2, zorder=3)
    
    ax.set_xlabel('Model File Size (MB)', fontsize=12)
    ax.set_ylabel('F1 Macro Score', fontsize=12)
    ax.set_title('Compression Trade-off: Size vs Performance', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotation for best trade-off (quantized model)
    quantized_size = get_size_mb(results.get("Quantized (8-bit)", {}))
    if quantized_size > 0:
        ax.annotate('Best compression\n(~2.4x smaller)', 
                   xy=(quantized_size, results["Quantized (8-bit)"]["f1_macro"]),
                   xytext=(quantized_size + 80, results["Quantized (8-bit)"]["f1_macro"] - 0.01),
                   arrowprops=dict(arrowstyle='->', color='black'),
                   fontsize=10)
    
    plt.tight_layout()
    save_path = f"{PLOTS_DIR}/07_compression_tradeoff.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# PLOT 8: Class Distribution with Model Performance
# =============================================================================
def plot_class_analysis(results):
    """Analyze performance relative to class distribution."""
    
    # Class distribution (from training data)
    class_counts = {
        "sadness": 4666,
        "joy": 5362,
        "love": 1304,
        "anger": 2159,
        "fear": 1937,
        "surprise": 572
    }
    total = sum(class_counts.values())
    class_pcts = {k: v/total*100 for k, v in class_counts.items()}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Class Distribution vs Model Performance', fontsize=16, fontweight='bold', y=1.02)
    
    # Plot 1: Class distribution
    colors_class = sns.color_palette("husl", 6)
    bars = axes[0].bar(LABEL_NAMES, [class_pcts[l] for l in LABEL_NAMES], 
                       color=colors_class, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Percentage of Training Data')
    axes[0].set_title('Training Data Class Distribution')
    axes[0].tick_params(axis='x', rotation=45)
    for bar, pct in zip(bars, [class_pcts[l] for l in LABEL_NAMES]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Scatter - Class size vs Best model F1
    best_model = "BERT"
    f1_scores = [results[best_model][f"f1_{label}"] for label in LABEL_NAMES]
    sizes_scatter = [class_pcts[label] for label in LABEL_NAMES]
    
    for i, label in enumerate(LABEL_NAMES):
        axes[1].scatter(sizes_scatter[i], f1_scores[i], s=200, c=[colors_class[i]], 
                       label=label, edgecolors='black', linewidth=2, zorder=3)
    
    axes[1].set_xlabel('Class Size (% of training data)')
    axes[1].set_ylabel('F1 Score (BERT)')
    axes[1].set_title('Class Size vs F1 Score')
    axes[1].legend(loc='lower right', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # Add trend observation
    axes[1].annotate('Minority classes\n(harder to predict)', 
                    xy=(5, 0.87), fontsize=9, style='italic')
    axes[1].annotate('Majority classes\n(easier to predict)', 
                    xy=(28, 0.95), fontsize=9, style='italic')
    
    plt.tight_layout()
    save_path = f"{PLOTS_DIR}/08_class_distribution_analysis.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*60)
    print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
    print("="*60)
    print(f"\nOutput directory: {PLOTS_DIR}/")
    
    # Load results
    results = load_results()
    
    if not results:
        print("\nNo results found. Please run training scripts first.")
        return
    
    print(f"\nLoaded results for: {list(results.keys())}")
    
    # Generate all plots
    print("\nGenerating plots...")
    
    plot_model_performance(results)
    plot_per_class_f1(results)
    plot_training_resources(results)
    plot_efficiency_analysis(results)
    plot_summary_table(results)
    plot_compression_results()
    plot_compression_tradeoff()
    plot_class_analysis(results)
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS GENERATED!")
    print("="*60)
    print(f"\nPlots saved in: {PLOTS_DIR}/")
    print("\nFiles created:")
    for f in sorted(os.listdir(PLOTS_DIR)):
        if f.endswith('.png'):
            print(f"  - {f}")


if __name__ == "__main__":
    main()