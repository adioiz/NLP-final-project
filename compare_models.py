"""Generate publication-quality comparison visualizations for academic report."""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    for key in ['file_size_mb', 'size_mb', 'size']:
        if key in result_dict:
            return result_dict[key]
    return 0


def plot_model_performance(results):
    """Create comprehensive performance comparison."""

    models = list(results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)

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


def plot_per_class_f1(results):
    """Create detailed per-class F1 comparison."""

    models = list(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Per-Class Performance Analysis', fontsize=16, fontweight='bold', y=1.02)

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


def plot_training_resources(results):
    """Compare training time, model size, and parameters."""

    models = list(results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Training Resources Comparison', fontsize=16, fontweight='bold', y=1.02)

    times = [results[m]["training_time"] / 60 for m in models]
    bars1 = axes[0].bar(models, times, color=[COLORS[m] for m in models], edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Training Time (minutes)')
    axes[0].set_title('Training Time')
    for bar, t in zip(bars1, times):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{t:.1f} min', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    sizes = [results[m]["size_mb"] for m in models]
    bars2 = axes[1].bar(models, sizes, color=[COLORS[m] for m in models], edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('Model Size (MB)')
    axes[1].set_title('Model Size')
    for bar, s in zip(bars2, sizes):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{s:.1f} MB', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

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

    for i in range(len(columns)):
        table[(2, i)].set_facecolor('#d5f4e6')

    plt.title('Model Comparison Summary', fontsize=14, fontweight='bold', pad=20)

    save_path = f"{PLOTS_DIR}/04_summary_table.png"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(f"Saved: {save_path}")


def plot_compression_tradeoff():
    """Analyze compression trade-offs."""

    results = load_compression_results()
    if results is None:
        return

    models = list(results.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#3498db', '#9b59b6', '#e67e22', '#e74c3c']
    markers = ['o', 's', '^', 'D']

    for i, model in enumerate(models):
        size = get_size_mb(results[model])
        ax.scatter(size, results[model]["f1_macro"],
                  s=300, c=colors[i], marker=markers[i], label=model,
                  edgecolors='black', linewidth=2, zorder=3)

    ax.set_xlabel('Model File Size (MB)', fontsize=12)
    ax.set_ylabel('F1 Macro Score', fontsize=12)
    ax.set_title('Compression Trade-off: Size vs Performance', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    quantized_size = get_size_mb(results.get("Quantized (8-bit)", {}))
    if quantized_size > 0:
        ax.annotate('Best compression\n(~2.4x smaller)',
                   xy=(quantized_size, results["Quantized (8-bit)"]["f1_macro"]),
                   xytext=(quantized_size + 80, results["Quantized (8-bit)"]["f1_macro"] - 0.01),
                   arrowprops=dict(arrowstyle='->', color='black'),
                   fontsize=10)

    plt.tight_layout()
    save_path = f"{PLOTS_DIR}/05_compression_tradeoff.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
    print(f"\nOutput directory: {PLOTS_DIR}/")

    results = load_results()

    if not results:
        print("\nNo results found. Please run training scripts first.")
        return

    print(f"\nLoaded results for: {list(results.keys())}")

    print("\nGenerating plots...")

    plot_model_performance(results)
    plot_per_class_f1(results)
    plot_training_resources(results)
    plot_summary_table(results)
    plot_compression_tradeoff()

    print("\nALL VISUALIZATIONS GENERATED!")
    print(f"\nPlots saved in: {PLOTS_DIR}/")
    print("\nFiles created:")
    for f in sorted(os.listdir(PLOTS_DIR)):
        if f.endswith('.png'):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
