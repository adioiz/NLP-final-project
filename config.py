"""
Configuration file for The Emotion Project
All hyperparameters and settings are centralized here for easy tuning.
"""
import torch

# =============================================================================
# PATHS
# =============================================================================
TRAIN_PATH = "data/train.csv"
VAL_PATH = "data/validation.csv"
WEIGHTS_DIR = "weights/"
OUTPUTS_DIR = "outputs/"

# =============================================================================
# MODEL SETTINGS
# =============================================================================
# Number of emotion classes: sadness, joy, love, anger, fear, surprise
NUM_CLASSES = 6

# Label mapping for reference
LABEL_NAMES = {
    0: "sadness",
    1: "joy", 
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# Pre-trained model names from HuggingFace
MODEL_CONFIGS = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "electra": "google/electra-base-discriminator",
}

# =============================================================================
MAX_LENGTH = 128          # Tokens length
BATCH_SIZE = 32           # Standard for finetuning on a CPU
LEARNING_RATE = 2e-5      # Standard LR for fine-tuning transformers
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1        # Warmup steps as ratio of total steps

# =============================================================================
# CLASS WEIGHTS (to handle imbalance)
# =============================================================================
# Calculated as: total_samples / (num_classes * class_count)
# This gives higher weight to underrepresented classes
# Based on training data distribution:
#   sadness: 4666, joy: 5362, love: 1304, anger: 2159, fear: 1937, surprise: 572
CLASS_WEIGHTS = torch.tensor([
    16000 / (6 * 4666),   # sadness: 0.57
    16000 / (6 * 5362),   # joy: 0.50
    16000 / (6 * 1304),   # love: 2.04
    16000 / (6 * 2159),   # anger: 1.23
    16000 / (6 * 1937),   # fear: 1.38
    16000 / (6 * 572),    # surprise: 4.66
])

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# RANDOM SEED
# =============================================================================
SEED = 42
