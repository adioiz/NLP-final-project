"""Evaluation metrics for emotion classification."""

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

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def compute_metrics(y_true, y_pred):
    """Compute comprehensive classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
    }

    f1_per_class = f1_score(y_true, y_pred, average=None)
    for i, label in enumerate(LABEL_NAMES):
        metrics[f"f1_{label}"] = f1_per_class[i]

    return metrics


def print_metrics(metrics, model_name="Model"):
    """Print metrics in formatted output."""
    print(f"\nResults for {model_name}")
    print(f"Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"F1 Macro:        {metrics['f1_macro']:.4f}")
    print(f"F1 Weighted:     {metrics['f1_weighted']:.4f}")
    print(f"Precision Macro: {metrics['precision_macro']:.4f}")
    print(f"Recall Macro:    {metrics['recall_macro']:.4f}")
    print(f"\nPer-class F1 scores:")
    for label in LABEL_NAMES:
        print(f"  {label:10s}: {metrics[f'f1_{label}']:.4f}")


def print_classification_report(y_true, y_pred):
    """Print detailed classification report."""
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
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

    plt.close()
    return cm
