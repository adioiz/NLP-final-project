"""Train ELECTRA for emotion classification."""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from config import (
    TRAIN_PATH, VAL_PATH, WEIGHTS_DIR,
    NUM_CLASSES, MODEL_CONFIGS, LABEL_NAMES,
    MAX_LENGTH, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, WARMUP_RATIO, WEIGHT_DECAY,
    CLASS_WEIGHTS, DEVICE, SEED
)

from utils.data_loader import load_data, create_dataloaders, get_tokenizer
from utils.metrics import plot_confusion_matrix, print_classification_report
from utils.trainer import train_model, count_parameters, get_model_size_mb
from train_bert import set_seed

def main():
    print("EMOTION CLASSIFICATION - ELECTRA TRAINING")

    set_seed(SEED)

    print(f"\nDevice: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_df, val_df = load_data(TRAIN_PATH, VAL_PATH)

    print("\nClass distribution (training):")
    for label_id, label_name in LABEL_NAMES.items():
        count = (train_df['label'] == label_id).sum()
        pct = 100 * count / len(train_df)
        print(f"  {label_name:10s}: {count:5d} ({pct:.1f}%)")

    model_name = MODEL_CONFIGS["electra"]
    print(f"\nModel: {model_name}")

    tokenizer = get_tokenizer(model_name)
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_CLASSES, use_safetensors=True)
    model.to(DEVICE)

    n_params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    print(f"Total parameters: {n_params:,}")
    print(f"Model size: {size_mb:.1f} MB")

    train_loader, val_loader = create_dataloaders(train_df, val_df, tokenizer, MAX_LENGTH, BATCH_SIZE)
    print(f"\nTraining batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    history, best_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        model_name="electra",
        weights_dir=WEIGHTS_DIR
    )

    print_classification_report(history["best_labels"], history["best_preds"])

    os.makedirs("outputs", exist_ok=True)
    plot_confusion_matrix(history["best_labels"], history["best_preds"], "ELECTRA", save_path="outputs/electra_confusion_matrix.png")

    results = {
        "model": "ELECTRA",
        "model_name": model_name,
        "parameters": n_params,
        "size_mb": size_mb,
        "training_time": history["total_time"],
        "accuracy": best_metrics["accuracy"],
        "f1_macro": best_metrics["f1_macro"],
        "f1_weighted": best_metrics["f1_weighted"],
    }

    for label_name in LABEL_NAMES.values():
        results[f"f1_{label_name}"] = best_metrics[f"f1_{label_name}"]

    with open("outputs/electra_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to outputs/electra_results.json")

    print(f"\nBest model saved to: {WEIGHTS_DIR}/electra_best.pt")
    print(f"Accuracy: {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.1f}%)")
    print(f"F1 Macro: {best_metrics['f1_macro']:.4f}")

    return results, history


if __name__ == "__main__":
    results, history = main()