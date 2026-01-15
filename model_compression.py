"""
Apply compression techniques to the best model

Two compression methods implemented:
1. Dynamic Quantization - Reduce weight precision from 32-bit to 8-bit
2. Magnitude Pruning - Remove weights with smallest absolute values

- Quantization: Computers use 32/64 bits, but brain synapses use ~4.6 bits
- Pruning: Remove connections below a threshold, then retrain

"""

import os
import json
import time
import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    TRAIN_PATH, VAL_PATH, WEIGHTS_DIR,
    NUM_CLASSES, MODEL_CONFIGS, LABEL_NAMES,
    MAX_LENGTH, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, WARMUP_RATIO,
    CLASS_WEIGHTS, DEVICE, SEED
)

from utils.data_loader import load_data, create_dataloaders, get_tokenizer
from utils.metrics import compute_metrics, print_metrics, plot_confusion_matrix, print_classification_report
from utils.trainer import train_model, count_parameters, get_model_size_mb


def evaluate_model(model, val_loader, device):
    """Simplified Evaluation for the model only returns predictions and labels."""
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
    """Get actual file size of saved model."""
    torch.save(model.state_dict(), save_path)
    size_mb = os.path.getsize(save_path) / 1024 / 1024
    os.remove(save_path)
    return size_mb


# =============================================================================
# METHOD 1: DYNAMIC QUANTIZATION
# =============================================================================

def apply_quantization(model):
    """
    Apply dynamic quantization to the model.
    
    Dynamic Quantization:
    - Weights are converted from float32 to int8
    - Activations remain in float32 during computation
    - ~4x reduction in model size
    - Minimal accuracy loss

    We only quantize `nn.Linear` layers because:
    - Linear layers have the most parameters (weight matrices)
    - They benefit most from quantization
    - Embedding layers are kept in float32 (they're lookup tables, quantization hurts them)
    ### What Happens Inside `quantize_dynamic`:
    ```
    For each Linear layer in the model:

    1. Find min/max of weights

    2. Calculate scale and zero_point:
    scale = (max - min) / 255  # Map float range to int8 range
    zero_point = round(-min / scale)

    3. Quantize each weight:
    int8_weight = round(float_weight / scale) + zero_point

    4. Store: int8_weights + scale + zero_point

    args:
        model: The model to quantize
    returns:
        quantized_model: The quantized model
    """
    print("\n" + "="*50)
    print("APPLYING DYNAMIC QUANTIZATION")
    print("="*50)
    torch.backends.quantized.engine = 'qnnpack'
    # Quantize the model - only Linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},  # Which layers to quantize
        dtype=torch.qint8  # Target dtype
    )
    
    return quantized_model


# =============================================================================
# METHOD 2: MAGNITUDE-BASED PRUNING
# =============================================================================

def apply_pruning(model, pruning_amount=0.3):
    """
    Apply unstructured magnitude-based pruning to the model.
    
    Magnitude Pruning:
    - Remove weights with smallest absolute values
    - Based on the idea that small weights contribute less

    We only prune Linear layers because:
    - Linear layers have the most parameters (weight matrices)
    - Embedding layers shouldn't be pruned

    Apply L1 unstructured pruning to each Linear layer explained:
    - L1 = Magnitude (absolute value)
    - Unstructured = Individual weights (not entire neurons)  
    - Process:
        1. Take all weights in the layer
        2. Compute absolute value: |weight|
        3. Sort by magnitude
        4. Remove the smallest % of weights (set to zero)

    Args:
        model: The model to prune
        pruning_amount: Fraction of weights to remove (0.3 = 30%)
    Returns:
        pruned_model: The pruned model
    """
    print("\n" + "="*50)
    print(f"APPLYING MAGNITUDE PRUNING ({pruning_amount*100:.0f}%)")
    print("="*50)
    
    pruned_model = copy.deepcopy(model)

    total_params = 0
    zero_params = 0
    
    # Apply pruning to each Linear layers
    for name, module in pruned_model.named_modules(): # Iterate Over All Modules (the layers)
        if isinstance(module, nn.Linear): # Target Linear Layers
            # Apply L1 unstructured pruning
            prune.l1_unstructured(module, name='weight', amount=pruning_amount) 
            
            # Make pruning permanent (remove the mask)
            prune.remove(module, 'weight')
            
            # Count zeros
            total_params += module.weight.numel()
            zero_params += (module.weight == 0).sum().item()
    
    sparsity = 100 * zero_params / total_params if total_params > 0 else 0
    print(f"Pruning complete. Sparsity: {sparsity:.1f}%")
    
    return pruned_model


def main():
    print("="*60)
    print("MODEL COMPRESSION - BERT")
    print("="*60)
    
    # =========================================================================
    # LOAD DATA AND ORIGINAL MODEL
    # =========================================================================
    print("\n" + "-"*40)
    print("Loading Data and Model...")
    print("-"*40)
    
    # Load data
    train_df, val_df = load_data(TRAIN_PATH, VAL_PATH)
    
    # Load tokenizer
    model_name = MODEL_CONFIGS["bert"]
    tokenizer = get_tokenizer(model_name)
    
    # Create validation dataloader
    _, val_loader = create_dataloaders(
        train_df, val_df, tokenizer, MAX_LENGTH, BATCH_SIZE
    )
    
    # Load the trained BERT model
    print(f"\nLoading trained BERT model from {WEIGHTS_DIR}/bert_best.pt")
    
    original_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES
    )
    original_model.load_state_dict(
        torch.load(f"{WEIGHTS_DIR}/bert_best.pt", map_location="cpu")
    )
    original_model.eval()
    
    # =========================================================================
    # EVALUATE ORIGINAL MODEL
    # =========================================================================
    print("\n" + "-"*40)
    print("Evaluating Original Model...")
    print("-"*40)
    
    original_size = get_model_file_size(original_model)
    original_params = count_parameters(original_model)
    
    # Move to CPU (quantization only works on CPU)
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
    
    # =========================================================================
    # METHOD 1: QUANTIZATION
    # =========================================================================
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
    
    # Save quantized model
    torch.save(quantized_model.state_dict(), f"{WEIGHTS_DIR}/bert_quantized.pt")
    print(f"  Saved to: {WEIGHTS_DIR}/bert_quantized.pt")
    
    # =========================================================================
    # METHOD 2: PRUNING (30%)
    # =========================================================================
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
    
    # Save pruned model
    torch.save(pruned_model_30.state_dict(), f"{WEIGHTS_DIR}/bert_pruned_30.pt")
    print(f"  Saved to: {WEIGHTS_DIR}/bert_pruned_30.pt")
    
    # =========================================================================
    # METHOD 2: PRUNING (50%)
    # =========================================================================
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
    
    # Save pruned model
    torch.save(pruned_model_50.state_dict(), f"{WEIGHTS_DIR}/bert_pruned_50.pt")
    print(f"  Saved to: {WEIGHTS_DIR}/bert_pruned_50.pt")
    
    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    print("\n" + "="*60)
    print("COMPRESSION COMPARISON SUMMARY")
    print("="*60)
    
    results = {
        "Original BERT": {
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
            "compression_ratio": 1.0  # Sparse format doesn't reduce file size
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
    
    # Print comparison table
    print(f"\n{'Model':<20} {'Size (MB)':<12} {'Accuracy':<10} {'F1 Macro':<10} {'Inf. Time':<10}")
    print("-" * 62)
    for model_name, data in results.items():
        print(f"{model_name:<20} {data['size_mb']:<12.1f} {data['accuracy']:<10.4f} {data['f1_macro']:<10.4f} {data['inference_time']:<10.2f}s")
    
    # Save results
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/compression_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to outputs/compression_results.json")
    
    # =========================================================================
    # PLOT CONFUSION MATRICES FOR COMPRESSED MODELS
    # =========================================================================
    print("\n" + "-"*40)
    print("Generating Confusion Matrices...")
    print("-"*40)
    
    plot_confusion_matrix(
        q_labels, q_preds,
        "BERT Quantized",
        save_path="outputs/bert_quantized_confusion_matrix.png"
    )
    
    plot_confusion_matrix(
        p30_labels, p30_preds,
        "BERT Pruned 30%",
        save_path="outputs/bert_pruned_30_confusion_matrix.png"
    )
    
    plot_confusion_matrix(
        p50_labels, p50_preds,
        "BERT Pruned 50%",
        save_path="outputs/bert_pruned_50_confusion_matrix.png"
    )
    
    print("\n" + "="*60)
    print("COMPRESSION COMPLETE!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = main()