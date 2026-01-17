"""Apply compression techniques to the best model using quantization and pruning."""

import os
import json
import time
import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import AutoModelForSequenceClassification

from config import (
    TRAIN_PATH, VAL_PATH, WEIGHTS_DIR,
    NUM_CLASSES, MODEL_CONFIGS, LABEL_NAMES,
    MAX_LENGTH, BATCH_SIZE, LEARNING_RATE,
    CLASS_WEIGHTS, DEVICE, SEED
)

from utils.data_loader import load_data, create_dataloaders, get_tokenizer
from utils.metrics import compute_metrics, plot_confusion_matrix
from utils.trainer import count_parameters


def evaluate_model(model, val_loader, device):
    """Evaluate model and return metrics, inference time, predictions, and labels."""
    model.eval()
    all_preds = []
    all_labels = []

    start_time = time.time()

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    inference_time = time.time() - start_time
    metrics = compute_metrics(all_labels, all_preds)

    return metrics, inference_time, all_preds, all_labels


def get_model_file_size(model, save_path="temp_model.pt"):
    """Get actual file size of saved model in MB."""
    torch.save(model.state_dict(), save_path)
    size_mb = os.path.getsize(save_path) / 1024 / 1024
    os.remove(save_path)
    return size_mb


def apply_quantization(model):
    """Apply dynamic quantization to reduce model size."""
    print("\nAPPLYING DYNAMIC QUANTIZATION")
    torch.backends.quantized.engine = 'qnnpack'
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )

    return quantized_model


def apply_pruning(model, pruning_amount=0.3):
    """Apply unstructured magnitude-based pruning to remove smallest weights."""
    print(f"\nAPPLYING MAGNITUDE PRUNING ({pruning_amount*100:.0f}%)")

    pruned_model = copy.deepcopy(model)

    total_params = 0
    zero_params = 0

    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_amount)
            prune.remove(module, 'weight')

            total_params += module.weight.numel()
            zero_params += (module.weight == 0).sum().item()

    sparsity = 100 * zero_params / total_params if total_params > 0 else 0
    print(f"Pruning complete. Sparsity: {sparsity:.1f}%")

    return pruned_model


def main():
    print("MODEL COMPRESSION - RoBERTa")

    print("\nLoading Data and Model...")

    train_df, val_df = load_data(TRAIN_PATH, VAL_PATH)

    model_name = MODEL_CONFIGS["roberta"]
    tokenizer = get_tokenizer(model_name)

    _, val_loader = create_dataloaders(
        train_df, val_df, tokenizer, MAX_LENGTH, BATCH_SIZE
    )

    print(f"\nLoading trained RoBERTa model from {WEIGHTS_DIR}/roberta_best.pt")

    original_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES
    )
    original_model.load_state_dict(
        torch.load(f"{WEIGHTS_DIR}/roberta_best.pt", map_location="cpu")
    )
    original_model.eval()

    print("\nEvaluating Original Model...")

    original_size = get_model_file_size(original_model)
    original_params = count_parameters(original_model)

    original_model_cpu = original_model.to("cpu")
    original_metrics, original_time, _, _ = evaluate_model(
        original_model_cpu, val_loader, "cpu"
    )

    print(f"Original Model:")
    print(f"  Size: {original_size:.1f} MB")
    print(f"  Parameters: {original_params:,}")
    print(f"  Accuracy: {original_metrics['accuracy']:.4f}")
    print(f"  F1 Macro: {original_metrics['f1_macro']:.4f}")
    print(f"  Inference Time: {original_time:.2f}s")

    quantized_model = apply_quantization(copy.deepcopy(original_model_cpu))

    quantized_size = get_model_file_size(quantized_model)
    quantized_metrics, quantized_time, q_preds, q_labels = evaluate_model(
        quantized_model, val_loader, "cpu"
    )

    print(f"\nQuantized Model Results:")
    print(f"  Size: {quantized_size:.1f} MB (↓{(1-quantized_size/original_size)*100:.1f}%)")
    print(f"  Accuracy: {quantized_metrics['accuracy']:.4f}")
    print(f"  F1 Macro: {quantized_metrics['f1_macro']:.4f}")
    print(f"  Inference Time: {quantized_time:.2f}s")

    torch.save(quantized_model.state_dict(), f"{WEIGHTS_DIR}/roberta_quantized.pt")
    print(f"  Saved to: {WEIGHTS_DIR}/roberta_quantized.pt")

    pruned_model_30 = apply_pruning(copy.deepcopy(original_model_cpu), pruning_amount=0.3)

    pruned_30_size = get_model_file_size(pruned_model_30)
    pruned_30_metrics, pruned_30_time, p30_preds, p30_labels = evaluate_model(
        pruned_model_30, val_loader, "cpu"
    )

    print(f"\nPruned Model (30%) Results:")
    print(f"  Size: {pruned_30_size:.1f} MB")
    print(f"  Accuracy: {pruned_30_metrics['accuracy']:.4f}")
    print(f"  F1 Macro: {pruned_30_metrics['f1_macro']:.4f}")
    print(f"  Inference Time: {pruned_30_time:.2f}s")

    torch.save(pruned_model_30.state_dict(), f"{WEIGHTS_DIR}/roberta_pruned_30.pt")
    print(f"  Saved to: {WEIGHTS_DIR}/roberta_pruned_30.pt")

    pruned_model_50 = apply_pruning(copy.deepcopy(original_model_cpu), pruning_amount=0.5)

    pruned_50_size = get_model_file_size(pruned_model_50)
    pruned_50_metrics, pruned_50_time, p50_preds, p50_labels = evaluate_model(
        pruned_model_50, val_loader, "cpu"
    )

    print(f"\nPruned Model (50%) Results:")
    print(f"  Size: {pruned_50_size:.1f} MB")
    print(f"  Accuracy: {pruned_50_metrics['accuracy']:.4f}")
    print(f"  F1 Macro: {pruned_50_metrics['f1_macro']:.4f}")
    print(f"  Inference Time: {pruned_50_time:.2f}s")

    torch.save(pruned_model_50.state_dict(), f"{WEIGHTS_DIR}/roberta_pruned_50.pt")
    print(f"  Saved to: {WEIGHTS_DIR}/roberta_pruned_50.pt")

    print("\nCOMPRESSION COMPARISON SUMMARY")

    results = {
        "Original RoBERTa": {
            "size_mb": original_size,
            "accuracy": original_metrics["accuracy"],
            "f1_macro": original_metrics["f1_macro"],
            "f1_weighted": original_metrics["f1_weighted"],
            "inference_time": original_time,
            "compression_ratio": 1.0
        },
        "Quantized (8-bit)": {
            "size_mb": quantized_size,
            "accuracy": quantized_metrics["accuracy"],
            "f1_macro": quantized_metrics["f1_macro"],
            "f1_weighted": quantized_metrics["f1_weighted"],
            "inference_time": quantized_time,
            "compression_ratio": original_size / quantized_size
        },
        "Pruned (30%)": {
            "size_mb": pruned_30_size,
            "accuracy": pruned_30_metrics["accuracy"],
            "f1_macro": pruned_30_metrics["f1_macro"],
            "f1_weighted": pruned_30_metrics["f1_weighted"],
            "inference_time": pruned_30_time,
            "compression_ratio": 1.0
        },
        "Pruned (50%)": {
            "size_mb": pruned_50_size,
            "accuracy": pruned_50_metrics["accuracy"],
            "f1_macro": pruned_50_metrics["f1_macro"],
            "f1_weighted": pruned_50_metrics["f1_weighted"],
            "inference_time": pruned_50_time,
            "compression_ratio": 1.0
        }
    }

    print(f"\n{'Model':<20} {'Size (MB)':<12} {'Accuracy':<10} {'F1 Macro':<10} {'Inf. Time':<10}")
    print("-" * 62)
    for model_name, data in results.items():
        print(f"{model_name:<20} {data['size_mb']:<12.1f} {data['accuracy']:<10.4f} {data['f1_macro']:<10.4f} {data['inference_time']:<10.2f}s")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/compression_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to outputs/compression_results.json")

    print("\nGenerating Confusion Matrices...")

    plot_confusion_matrix(
        q_labels, q_preds,
        "RoBERTa Quantized",
        save_path="outputs/roberta_quantized_confusion_matrix.png"
    )

    plot_confusion_matrix(
        p30_labels, p30_preds,
        "RoBERTa Pruned 30%",
        save_path="outputs/roberta_pruned_30_confusion_matrix.png"
    )

    plot_confusion_matrix(
        p50_labels, p50_preds,
        "RoBERTa Pruned 50%",
        save_path="outputs/roberta_pruned_50_confusion_matrix.png"
    )

    print("\nCOMPRESSION COMPLETE!")

    return results


if __name__ == "__main__":
    results = main()
