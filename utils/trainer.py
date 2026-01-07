"""
Train utilities for Emotion Classification.
Handles training loop, validation, and checkpoint saving.
"""

import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """
    Train one epoch.
    
    Args:
        model: The transformer model
        dataloader: Training DataLoader
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function (CrossEntropyLoss with class weights)
        device: cuda or cpu
    
    Returns:
        Average loss for the epoch
    """
    model.train()  # Set to training mode (enables dropout)
    total_loss = 0
    
    # tqdm creates a progress bar
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in progress_bar:
        # Move data to device (GPU if available)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        # Forward pass
        optimizer.zero_grad()  # Clear gradients from previous step
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # outputs.loss uses CrossEntropy internally, but we use our weighted criterion
        logits = outputs.logits
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()  # Compute gradients
        
        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()   # Update weights
        scheduler.step()   # Update learning rate
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on validation set.
    
    Returns:
        loss: Average validation loss
        predictions: All predicted labels
        true_labels: All true labels
    """
    model.eval()  # Set to evaluation mode (disables dropout)
    total_loss = 0
    all_preds = []
    all_labels = []
    
    # No gradient computation during evaluation
    with torch.no_grad():
        progress_bar_eval = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in progress_bar_eval:
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(dataloader), all_preds, all_labels


def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs, model_name,weights_dir="weights/"):
    """
    Full training loop with validation and checkpoint saving.
    
    Args:
        model: The transformer model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        optimizer: Optimizer
        scheduler: LR scheduler
        criterion: Loss function
        device: Device to train on
        num_epochs: Number of epochs
        model_name: Name for saving (e.g., "bert")
        weights_dir: Directory to save weights
    
    Returns:
        history: Dictionary with training history
        best_metrics: Metrics from the best epoch
    """
    # Import here to avoid circular imports
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
    
    # Create weights directory if needed
    os.makedirs(weights_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        
        # Validate
        val_loss, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        
        # Compute metrics
        metrics = compute_metrics(val_labels, val_preds)
        
        # Store history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(metrics["accuracy"])
        history["val_f1_macro"].append(metrics["f1_macro"])
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {metrics['accuracy']:.4f}")
        print(f"Val F1 Macro: {metrics['f1_macro']:.4f}")
        
        # Save best model (based on macro F1 - better for imbalanced data)
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_metrics = metrics
            best_preds = val_preds
            best_labels = val_labels
            
            save_path = os.path.join(weights_dir, f"{model_name}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved! F1: {best_f1:.4f}")
    
    total_time = time.time() - start_time
    
    print(f"\nTraining completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print_metrics(best_metrics, model_name)
    
    history["total_time"] = total_time
    history["best_preds"] = best_preds
    history["best_labels"] = best_labels

    return history, best_metrics


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    """Get model size in megabytes."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024
