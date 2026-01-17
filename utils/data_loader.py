"""Data loading utilities for emotion classification."""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class EmotionDataset(Dataset):
    """Custom PyTorch Dataset for emotion classification."""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """Returns a single tokenized sample."""
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


def load_data(train_path, val_path):
    """Load training and validation data from CSV files."""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    return train_df, val_df


def create_dataloaders(train_df, val_df, tokenizer, max_length, batch_size):
    """Create PyTorch DataLoaders for training and validation."""
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader


def get_tokenizer(model_name):
    """Load the appropriate tokenizer for a model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer
