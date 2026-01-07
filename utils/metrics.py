"""
Evaluation metrics for Emotion Classification.
Includes accuracy, F1-score, and confusion matrix visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)


# Label names for visualization
LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def compute_metrics(y_true, y_pred):
    """
    Compute comprehensive metrics for classification.
    
    Args:
        y_true: Ground truth labels (list or array)
        y_pred: Predicted labels (list or array)
    
    Returns:
        Dictionary with all metrics:
            - Accuracy: % of correct predictions
            - Macro F1: Average F1 across all classes (treats all classes equally)
            - Weighted F1: F1 weighted by class frequency (accounts for imbalance)
            - Per-class metrics: Shows which emotions are hard to predict
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
    }
    
    # Per-class F1 scores
    f1_per_class = f1_score(y_true, y_pred, average=None)
    for i, label in enumerate(LABEL_NAMES):
        metrics[f"f1_{label}"] = f1_per_class[i]
    
    return metrics


def print_metrics(metrics, model_name="Model"):
    """Pretty print metrics."""
    print(f"\n{'='*50}")
    print(f"Results for {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"F1 Macro:        {metrics['f1_macro']:.4f}")
    print(f"F1 Weighted:     {metrics['f1_weighted']:.4f}")
    print(f"Precision Macro: {metrics['precision_macro']:.4f}")
    print(f"Recall Macro:    {metrics['recall_macro']:.4f}")
    print(f"\nPer-class F1 scores:")
    for label in LABEL_NAMES:
        print(f"  {label:10s}: {metrics[f'f1_{label}']:.4f}")


def print_classification_report(y_true, y_pred):
    """
    Print detailed classification report.
    Shows precision, recall, F1 for each class.
    """
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True,           # Show numbers in cells
        fmt="d",              # Integer format
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()  # Close to free memory
    return cm


def plot_metrics_comparison(results_dict, save_path=None):
    """
    Plot bar chart comparing metrics across models.
    
    Args:
        results_dict: Dict of {model_name: metrics_dict}
        save_path: Path to save the plot
    """
    models = list(results_dict.keys())
    metrics_to_plot = ["accuracy", "f1_macro", "f1_weighted"]
    
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, model in enumerate(models):
        values = [results_dict[model][m] for m in metrics_to_plot]
        ax.bar(x + i*width, values, width, label=model)
    
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Accuracy", "F1 Macro", "F1 Weighted"])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison plot saved to {save_path}")
    
    plt.close()