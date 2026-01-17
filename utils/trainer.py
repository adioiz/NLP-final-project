"""Training utilities for emotion classification."""

import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """Train one epoch and return average loss."""
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        logits = outputs.logits
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation set and return loss, predictions, and labels."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar_eval = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in progress_bar_eval:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits
            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), all_preds, all_labels


def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs, model_name, weights_dir="weights/"):
    """Full training loop with validation and checkpoint saving."""
    from utils.metrics import compute_metrics, print_metrics

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1_macro": []
    }

    best_f1 = 0
    best_metrics = None
    best_preds = None
    best_labels = None

    os.makedirs(weights_dir, exist_ok=True)

    print(f"\nTraining {model_name}")

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

        metrics = compute_metrics(val_labels, val_preds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(metrics["accuracy"])
        history["val_f1_macro"].append(metrics["f1_macro"])

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {metrics['accuracy']:.4f}")
        print(f"Val F1 Macro: {metrics['f1_macro']:.4f}")

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_metrics = metrics
            best_preds = val_preds
            best_labels = val_labels

            save_path = os.path.join(weights_dir, f"{model_name}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved! F1: {best_f1:.4f}")

    total_time = time.time() - start_time

    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.1f}min)")
    print_metrics(best_metrics, model_name)

    history["total_time"] = total_time
    history["best_preds"] = best_preds
    history["best_labels"] = best_labels

    return history, best_metrics


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """Get model size in megabytes."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024
