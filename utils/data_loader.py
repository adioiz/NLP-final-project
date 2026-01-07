"""
Data loading utilities for Emotion Classification.
Handles CSV loading, tokenization, and DataLoader creation.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class EmotionDataset(Dataset):
    """
    Custom PyTorch Dataset for emotion classification.
    
    This class:
    - Gives us full control over how data is processed
    - Tokenizes text on-the-fly when accessed
    - Returns tensors ready for the model
    """
    
    def __init__(self, texts, labels, tokenizer, max_length):
        """
        Args:
            texts: List of text strings
            labels: List of integer labels (0-5)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Returns a single tokenized sample.
        
        The tokenizer returns:
        - input_ids: Token IDs (integers representing words/subwords)
        - attention_mask: 1 for real tokens, 0 for padding
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text: converts text → tokens → integer IDs
        # Example: "I feel happy" → ["i", "feel", "happy"] → [1045, 2514, 3407]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",      # Pad to max_length
            truncation=True,           # Cut if too long
            return_tensors="pt"        # Return PyTorch tensors
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),      # Shape: (max_length,)
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


def load_data(train_path, val_path):
    """
    Load training and validation data from CSV files.
    
    Returns:
        train_df, val_df
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    return train_df, val_df


def create_dataloaders(train_df, val_df, tokenizer, max_length, batch_size):
    """
    Create PyTorch DataLoaders for training and validation.
    
    - Automatic batching of samples
    - Shuffling for training (important for SGD)
    
    Args:
        train_df: Training DataFrame with 'text' and 'label' columns
        val_df: Validation DataFrame
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size for training
    
    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """
    # Create datasets
    train_dataset = EmotionDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = EmotionDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,    # Shuffle training data each epoch
        num_workers=2    # Parallel data loading
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,   # No need to shuffle validation
        num_workers=2
    )
    
    return train_loader, val_loader


def get_tokenizer(model_name):
    """
    Load the appropriate tokenizer for a model.
    
    - Automatically picks the right tokenizer for each model
    - BERT uses WordPiece, RoBERTa uses BPE, etc.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer