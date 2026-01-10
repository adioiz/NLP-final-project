"""
Train RoBERTa for Emotion Classification

RoBERTa improvements over BERT:
- More training data (16GB → 161GB)
- Dynamic masking (new mask each epoch)
- No NSP (Next Sentence Prediction) task
- Larger batch sizes
- BPE tokenization with 50k vocabulary (vs BERT's 30k WordPiece)
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from config import (
    TRAIN_PATH, VAL_PATH, WEIGHTS_DIR,
    NUM_CLASSES, MODEL_CONFIGS, LABEL_NAMES,
    MAX_LENGTH, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, WARMUP_RATIO,
    CLASS_WEIGHTS, DEVICE, SEED
)

from utils.data_loader import load_data, create_dataloaders, get_tokenizer
from utils.metrics import compute_metrics, print_metrics, plot_confusion_matrix, print_classification_report
from utils.trainer import train_model, count_parameters, get_model_size_mb

from train_bert import set_seed  # Reuse the same set_seed function

def main():
    # =========================================================================
    # SETUP
    # =========================================================================
    print("="*60)
    print("EMOTION CLASSIFICATION - RoBERTa TRAINING")
    print("="*60)
    
    set_seed(SEED)
    
    print(f"\nDevice: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("\n" + "-"*40)
    print("Loading Data...")
    print("-"*40)
    
    train_df, val_df = load_data(TRAIN_PATH, VAL_PATH)

    # Show class distribution
    print("\nClass distribution (training):")
    for label_id, label_name in LABEL_NAMES.items():
        count = (train_df['label'] == label_id).sum()
        pct = 100 * count / len(train_df)
        print(f"  {label_name:10s}: {count:5d} ({pct:.1f}%)")
    
    # =========================================================================
    # LOAD MODEL AND TOKENIZER
    # =========================================================================
    print("\n" + "-"*40)
    print("Loading RoBERTa Model...")
    print("-"*40)
    
    model_name = MODEL_CONFIGS["roberta"]  # "roberta-base"
    print(f"Model: {model_name}")
    
    tokenizer = get_tokenizer(model_name)
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

    # Load model with classification head
    # AutoModelForSequenceClassification adds a linear layer on top of RoBERTa
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES
    )
    model.to(DEVICE)

    # Print model info
    n_params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    print(f"Total parameters: {n_params:,}")
    print(f"Model size: {size_mb:.1f} MB")
    
    # =========================================================================
    # CREATE DATALOADERS
    # =========================================================================
    print("\n" + "-"*40)
    print("Creating DataLoaders...")
    print("-"*40)
    
    train_loader, val_loader = create_dataloaders(
        train_df, val_df, tokenizer, MAX_LENGTH, BATCH_SIZE
    )
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max sequence length: {MAX_LENGTH}")
    
    # =========================================================================
    # SETUP TRAINING
    # =========================================================================
    print("\n" + "-"*40)
    print("Setting up Training...")
    print("-"*40)
    
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE))
    print("Using weighted CrossEntropyLoss for class imbalance")
    print("Class weights:", {LABEL_NAMES[i]: f"{w:.2f}" for i, w in enumerate(CLASS_WEIGHTS.tolist())})
    # Optimizer: AdamW (Adam with weight decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"Optimizer: AdamW with LR={LEARNING_RATE}")
    
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    print(f"Scheduler: Linear warmup ({warmup_steps} steps) + decay")
    print(f"Total training steps: {total_steps}")
    
    # =========================================================================
    # TRAIN MODEL
    # =========================================================================
    print("\n" + "-"*40)
    print("Starting Training...")
    print("-"*40)
    
    history, best_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        model_name="roberta",
        weights_dir=WEIGHTS_DIR
    )
    
    # =========================================================================
    # FINAL EVALUATION
    # =========================================================================
    print("\n" + "-"*40)
    print("Final Evaluation")
    print("-"*40)
    
    print_classification_report(history["best_labels"], history["best_preds"])
    
    os.makedirs("outputs", exist_ok=True)
    plot_confusion_matrix(
        history["best_labels"],
        history["best_preds"],
        "RoBERTa",
        save_path="outputs/roberta_confusion_matrix.png"
    )
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results = {
        "model": "RoBERTa",
        "model_name": model_name,
        "parameters": n_params,
        "size_mb": size_mb,
        "training_time": history["total_time"],
        "accuracy": best_metrics["accuracy"],
        "f1_macro": best_metrics["f1_macro"],
        "f1_weighted": best_metrics["f1_weighted"],
    }
    
    # Add per-class F1
    for label_name in LABEL_NAMES.values():
        results[f"f1_{label_name}"] = best_metrics[f"f1_{label_name}"]
    
    # Save results to file
    with open("outputs/roberta_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to outputs/roberta_results.json")
    
    print("\n" + "="*60)
    print("RoBERTa TRAINING COMPLETE!")
    print("="*60)
    print(f"Best model saved to: {WEIGHTS_DIR}/roberta_best.pt")
    print(f"Accuracy: {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.1f}%)")
    print(f"F1 Macro: {best_metrics['f1_macro']:.4f}")
    
    return results, history


if __name__ == "__main__":
    results, history = main()