# Project Tour: Emotion Classification Using Transformers

## A Comprehensive Guide for Presentation

This document provides a detailed walkthrough of every component in the project, explaining how things work behind the scenes, what design choices were made, and why.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Pipeline](#2-data-pipeline)
3. [Model Architecture & Selection](#3-model-architecture--selection)
4. [Training Process Deep Dive](#4-training-process-deep-dive)
5. [Handling Class Imbalance](#5-handling-class-imbalance)
6. [Model Compression Techniques](#6-model-compression-techniques)
7. [Evaluation & Metrics](#7-evaluation--metrics)
8. [Inference Pipeline](#8-inference-pipeline)
9. [Key Design Decisions](#9-key-design-decisions)
10. [Results Summary](#10-results-summary)

---

## 1. Project Overview

### The Problem
Classify tweets into **6 emotion categories**: sadness, joy, love, anger, fear, and surprise.

### The Challenges
| Challenge | Description | Our Solution |
|-----------|-------------|--------------|
| **Class Imbalance** | Joy has 5,362 samples (33.5%) while Surprise has only 572 (3.6%) - a 9.4x difference | Weighted Cross-Entropy Loss |
| **Informal Text** | Twitter text contains slang, abbreviations, missing punctuation | Comprehensive preprocessing pipeline |
| **Short Context** | Tweets are limited to 280 characters | Transformer attention captures context efficiently |
| **Nuanced Emotions** | Love vs Joy, Fear vs Sadness can be subtle | Pre-trained language models capture semantic nuances |

### Dataset Statistics
```
Total Samples: 18,000 tweets
├── Training: 16,000 samples
└── Validation: 2,000 samples

Class Distribution:
  Joy:       5,362 (33.5%)  ← Majority class
  Sadness:   4,666 (29.2%)
  Anger:     2,159 (13.5%)
  Fear:      1,937 (12.1%)
  Love:      1,304 (8.2%)
  Surprise:    572 (3.6%)   ← Minority class (9.4x smaller than Joy)
```

---

## 2. Data Pipeline

### 2.1 Data Preprocessing (`data_preprocessing.py`)

The preprocessing pipeline normalizes Twitter-specific text patterns for better tokenization.

#### Step-by-Step Cleaning Process

```
Original Tweet: "Im soooo happy!!! 😊 @friend check this https://t.co/abc #blessed"
                                    ↓
Step 1: Remove URLs         → "Im soooo happy!!! 😊 @friend check this  #blessed"
Step 2: Remove @mentions    → "Im soooo happy!!! 😊  check this  #blessed"
Step 3: Handle hashtags     → "Im soooo happy!!! 😊  check this  blessed"
Step 4: Expand contractions → "I'm soooo happy!!! 😊  check this  blessed"
Step 5: Remove emojis       → "I'm soooo happy!!!   check this  blessed"
Step 6: Remove special chars→ "I'm soooo happy   check this  blessed"
Step 7: Reduce repetition   → "I'm soo happy   check this  blessed"
Step 8: Clean whitespace    → "I'm soo happy check this blessed"
                                    ↓
                             Final cleaned text
```

#### Why Each Step Matters

| Step | Function | Why It's Necessary |
|------|----------|-------------------|
| **URL Removal** | `remove_urls()` | URLs don't carry emotional content; they add noise |
| **Mention Removal** | `remove_mentions()` | @usernames are identifiers, not emotional signals |
| **Hashtag Handling** | `handle_hashtags()` | Keep the word (e.g., "blessed") but remove # symbol |
| **Contraction Expansion** | `expand_contractions()` | "Im" → "I'm" helps tokenizer recognize words correctly |
| **Emoji Removal** | `handle_emojis()` | Emojis aren't in BERT's vocabulary; could explore emoji2vec in future |
| **Character Reduction** | `remove_repeated_characters()` | "soooo" → "soo" normalizes emphasis patterns |

#### The Contractions Dictionary
The code includes **70+ contractions** covering:
- Standard contractions: "don't", "I'm", "you're"
- Informal spellings: "dont", "im", "youre"
- Slang: "gonna", "wanna", "gotta", "tryna", "finna"

**Design Choice**: We keep basic punctuation (`.`, `,`, `!`, `?`, `'`, `"`, `-`) because:
- Exclamation marks signal intensity ("I'm happy!" vs "I'm happy")
- Question marks change meaning entirely
- These are in BERT's vocabulary

### 2.2 Data Loading (`utils/data_loader.py`)

#### The EmotionDataset Class
```python
class EmotionDataset(Dataset):
    def __getitem__(self, idx):
        # Tokenization happens here, on-the-fly
        encoding = self.tokenizer(
            text,
            max_length=128,        # Truncate long texts
            padding="max_length",  # Pad short texts to 128
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": ...,       # Token IDs [101, 2003, 2061, ...]
            "attention_mask": ...,  # 1s for real tokens, 0s for padding
            "label": ...            # 0-5 for the emotion
        }
```

**What happens during tokenization:**
```
Text: "I'm so happy today"
         ↓
Tokens: ["[CLS]", "i", "'", "m", "so", "happy", "today", "[SEP]", "[PAD]", ...]
         ↓
IDs:    [101, 1045, 1005, 1049, 2061, 3407, 2651, 102, 0, 0, ...]
         ↓
Mask:   [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ...]
```

#### DataLoader Configuration
```python
train_loader = DataLoader(
    dataset,
    batch_size=32,      # 32 samples per batch
    shuffle=True,       # Randomize order each epoch (training only)
    num_workers=2       # Parallel data loading
)
```

**Why batch_size=32?**
- Standard for transformer fine-tuning
- Balances memory usage vs. gradient stability
- Fits comfortably on GPU/CPU memory

---

## 3. Model Architecture & Selection

### 3.1 The Three Models Compared

| Model | Architecture | Pre-training | Parameters | Key Strength |
|-------|-------------|--------------|------------|--------------|
| **BERT** | Bidirectional Encoder | Masked Language Modeling (MLM) | 110M | Robust, well-studied baseline |
| **RoBERTa** | Same as BERT | MLM (optimized training) | 125M | Better pre-training, more data |
| **ELECTRA** | Discriminator | Replaced Token Detection | 110M | Sample-efficient pre-training |

### 3.2 How Each Model Was Pre-trained

#### BERT: Masked Language Modeling
```
Input:  "The cat [MASK] on the mat"
Task:   Predict [MASK] = "sat"

BERT sees bidirectional context (both left and right).
Only ~15% of tokens are masked per sentence.
```

#### RoBERTa: Optimized BERT
Same architecture as BERT, but:
- Trained on 10x more data (160GB vs 16GB)
- Trained for longer (500K steps vs 100K)
- Dynamic masking (new masks each epoch)
- No Next Sentence Prediction task
- Larger batch sizes

#### ELECTRA: Replaced Token Detection
```
Generator creates: "The cat sat on the mat"
                         ↓ replace "sat" with "ran"
Corrupted:         "The cat ran on the mat"
                         ↓
Discriminator predicts: [original, original, REPLACED, original, original, original]
```
**Advantage**: Every token provides a training signal, not just masked ones.

### 3.3 Fine-tuning Architecture

All three models use the same classification head:

```
                    ┌─────────────────────────────────┐
Input Tokens  ───→  │   Transformer Encoder           │
                    │   (12 layers, 768 hidden dim)   │
                    └───────────────┬─────────────────┘
                                    ↓
                    ┌─────────────────────────────────┐
                    │   [CLS] Token Embedding         │
                    │   (768-dimensional vector)      │
                    └───────────────┬─────────────────┘
                                    ↓
                    ┌─────────────────────────────────┐
                    │   Linear Layer (768 → 6)        │
                    │   + Softmax                     │
                    └───────────────┬─────────────────┘
                                    ↓
                    [sadness, joy, love, anger, fear, surprise]
                    [0.05, 0.80, 0.05, 0.03, 0.04, 0.03]
```

**Why use [CLS] token?**
- Specifically designed to aggregate sequence-level information
- Positioned at start, attends to all tokens
- Pre-trained to represent the whole sentence

---

## 4. Training Process Deep Dive

### 4.1 The Training Loop (`utils/trainer.py`)

```python
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()  # Enable dropout, batch norm training mode

    for batch in dataloader:
        # 1. Forward Pass
        outputs = model(input_ids, attention_mask, labels)
        logits = outputs.logits  # Raw scores [batch, 6]

        # 2. Compute Loss (with class weights)
        loss = criterion(logits, labels)

        # 3. Backward Pass
        loss.backward()  # Compute gradients

        # 4. Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 5. Update Weights
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

### 4.2 Key Training Components

#### Optimizer: AdamW
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,           # Learning rate
    weight_decay=0.05  # L2 regularization
)
```

**Why AdamW?**
- Adam with proper weight decay (not L2 regularization)
- Standard for transformer fine-tuning
- Adaptive learning rates per parameter

**Why lr=2e-5?**
- Pre-trained models need small learning rates
- Too high: destroy pre-trained knowledge
- Too low: won't adapt to task
- 2e-5 is empirically optimal for BERT-family

#### Learning Rate Scheduler: Linear Warmup + Decay
```python
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,    # 10% of total steps
    num_training_steps=total_steps
)
```

**The schedule looks like:**
```
Learning Rate
    ^
2e-5|     /\
    |    /  \
    |   /    \
    |  /      \
  0 | /        \______________
    └─────────────────────────→ Steps
      warmup    decay
      (10%)     (90%)
```

**Why warmup?**
- Prevents large updates at start when gradients are unstable
- Allows model to "settle" before full learning rate

#### Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**What it does:**
- If gradient norm > 1.0, scale down all gradients proportionally
- Prevents exploding gradients
- Essential for stable transformer training

### 4.3 Hyperparameters Summary (`config.py`)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `MAX_LENGTH` | 128 tokens | Tweets are short; 128 covers ~99% of texts |
| `BATCH_SIZE` | 32 | Standard for fine-tuning, balances speed/stability |
| `LEARNING_RATE` | 2e-5 | Optimal for BERT-family fine-tuning |
| `NUM_EPOCHS` | 3 | Standard for fine-tuning; more risks overfitting |
| `WARMUP_RATIO` | 0.1 | 10% warmup is standard practice |
| `WEIGHT_DECAY` | 0.05 | Light regularization to prevent overfitting |
| `SEED` | 42 | Reproducibility (the universal answer) |

---

## 5. Handling Class Imbalance

### 5.1 The Problem

```
Class Distribution:
Joy:      ████████████████████████████████████ 33.5%
Sadness:  ██████████████████████████████ 29.2%
Anger:    ██████████████ 13.5%
Fear:     ████████████ 12.1%
Love:     ████████ 8.2%
Surprise: ███ 3.6%  ← 9.4x less than Joy!
```

Without handling this, the model would:
- Always predict "Joy" for borderline cases
- Rarely predict "Surprise" (only when extremely confident)
- Achieve high accuracy but poor minority-class recall

### 5.2 The Solution: Weighted Cross-Entropy Loss

#### How Weights Are Calculated (`config.py`)

```python
# Formula: weight_c = total_samples / (num_classes × class_count)
CLASS_WEIGHTS = torch.tensor([
    16000 / (6 * 4666),   # sadness: 0.57
    16000 / (6 * 5362),   # joy: 0.50      ← Lowest weight (majority class)
    16000 / (6 * 1304),   # love: 2.04
    16000 / (6 * 2159),   # anger: 1.23
    16000 / (6 * 1937),   # fear: 1.38
    16000 / (6 * 572),    # surprise: 4.66 ← Highest weight (minority class)
])
```

#### How It Works Mathematically

Standard Cross-Entropy:
```
L = -log(p_correct)
```

Weighted Cross-Entropy:
```
L = -w_class × log(p_correct)
```

**Example:**
- Model predicts Joy with 90% confidence for a Joy sample: Loss = 0.50 × 0.105 = 0.053
- Model predicts Surprise with 90% confidence for Surprise sample: Loss = 4.66 × 0.105 = 0.49

The model is penalized **9.3x more** for mistakes on Surprise than on Joy!

#### The Effect on Gradients

```
                     Without Weighting           With Weighting
                     ────────────────────        ────────────────────
Misclassify Joy:     Large gradient update  →    Small gradient update
Misclassify Surprise: Small gradient update  →    Large gradient update
```

**Result**: Model learns to pay attention to minority classes.

---

## 6. Model Compression Techniques

### 6.1 Why Compress?

| Model | Original Size | Challenge |
|-------|--------------|-----------|
| RoBERTa | 475.5 MB | Too large for mobile deployment |
| | | Slow inference on CPU |
| | | High memory requirements |

### 6.2 Dynamic Quantization (`model_compression.py`)

#### What Is Quantization?
Convert 32-bit floating point weights to 8-bit integers.

```
FP32: 3.14159265 → Uses 32 bits
INT8: 3          → Uses 8 bits (4x smaller!)
```

#### How It's Applied
```python
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},  # Only quantize Linear layers
    dtype=torch.qint8
)
```

**What happens to the weights:**
```
Original Linear Layer:
  Weight Matrix: [768 × 768] × 32 bits = 2.25 MB

Quantized Linear Layer:
  Weight Matrix: [768 × 768] × 8 bits = 0.56 MB
  Scale & Zero-Point: Few bytes for dequantization
```

#### Results
| Metric | Original | Quantized |
|--------|----------|-----------|
| Size | 475.6 MB | 230.9 MB |
| Compression | 1.0x | **2.06x** |
| Accuracy | 93.40% | 92.90% |
| F1 Macro | 91.32% | 90.74% |

**Key Finding**: 2x smaller with only 0.5% accuracy drop - excellent trade-off!

### 6.3 Magnitude-Based Pruning

#### What Is Pruning?
Remove the smallest weights (set them to zero).

```
Original weights: [0.8, -0.5, 0.01, -0.02, 0.7]
After 40% pruning: [0.8, -0.5, 0.00, 0.00, 0.7]
                            ↑       ↑
                         Pruned (too small)
```

#### The Theory
- Small weights contribute little to output
- Removing them shouldn't hurt performance much
- Creates sparse matrices (potentially faster inference)

#### How It's Applied
```python
for module in model.modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)  # Remove 30%
```

#### Results

| Pruning Level | Accuracy | F1 Macro | Analysis |
|---------------|----------|----------|----------|
| 0% (Original) | 93.40% | 91.32% | Baseline |
| 30% | 90.20% | 86.49% | Significant degradation |
| 50% | 40.20% | 19.36% | **Model collapse!** |

**Key Finding**: Aggressive pruning destroys the model. Transformers are sensitive to weight removal - all weights seem important for the complex attention patterns.

### 6.4 Compression Recommendations

```
For Production Deployment:
┌─────────────────────────────────────────────────────────────────┐
│  ✅ USE: Dynamic Quantization (INT8)                            │
│     - 2x smaller                                                │
│     - <1% accuracy loss                                         │
│     - No retraining needed                                      │
│                                                                 │
│  ❌ AVOID: Pruning > 20%                                        │
│     - Transformers don't prune well                             │
│     - Need expensive retraining                                 │
│     - Risk of catastrophic performance loss                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Evaluation & Metrics

### 7.1 Metrics Computed (`utils/metrics.py`)

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **Accuracy** | Correct / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | Of predicted positives, how many are correct |
| **Recall** | TP / (TP + FN) | Of actual positives, how many were found |
| **F1 Score** | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |

### 7.2 Macro vs Weighted F1

**F1 Macro**: Simple average across classes
```
F1_macro = (F1_sadness + F1_joy + F1_love + F1_anger + F1_fear + F1_surprise) / 6
```
- Each class counts equally
- Better for evaluating minority class performance
- **We use this for model selection**

**F1 Weighted**: Weighted by class frequency
```
F1_weighted = Σ (class_count × F1_class) / total_samples
```
- Accounts for class imbalance
- Dominated by majority classes
- Closer to accuracy interpretation

### 7.3 Why F1 Macro for Model Selection?

```python
# In trainer.py
if metrics["f1_macro"] > best_f1:  # Using F1 Macro, not accuracy!
    best_f1 = metrics["f1_macro"]
    torch.save(model.state_dict(), save_path)
```

**Reason**: A model with 93% accuracy could achieve that by always predicting Joy/Sadness. F1 Macro ensures the model performs well on ALL classes, including Surprise.

### 7.4 Confusion Matrix Interpretation

```
                    Predicted
                 sad  joy  lov  ang  fea  sur
              ┌─────────────────────────────┐
        sad   │ 550 │  12 │   3 │  10 │   5 │   1 │  ← Most sadness correct
Actual  joy   │  15 │ 665 │  10 │   3 │   2 │   0 │  ← Most joy correct
        lov   │   5 │  40 │ 100 │   8 │   4 │   2 │  ← Some confusion with joy
        ang   │  15 │   5 │   5 │ 240 │   8 │   2 │  ← Good anger detection
        fea   │  10 │   5 │   3 │  12 │ 190 │   4 │  ← Some confusion with anger
        sur   │   3 │   8 │   5 │   4 │   6 │  40 │  ← Hardest class
              └─────────────────────────────┘
```

**Common Confusions:**
- Love ↔ Joy (both positive emotions)
- Fear ↔ Anger (both negative, intense)
- Surprise is hardest (fewest samples, ambiguous)

---

## 8. Inference Pipeline

### 8.1 The Inference Flow (`run_inference.py`)

```
Input CSV with 'text' column
            ↓
    ┌───────────────────┐
    │ Load Model Weights│
    │ (roberta_best.pt) │
    └─────────┬─────────┘
              ↓
    ┌───────────────────┐
    │ For each text:    │
    │ 1. Tokenize       │
    │ 2. Forward pass   │
    │ 3. argmax(logits) │
    └─────────┬─────────┘
              ↓
    ┌───────────────────┐
    │ Save predictions  │
    │ to predictions.csv│
    └───────────────────┘
```

### 8.2 The Required Interface

```python
def run_inference(weights, csv):
    """Required interface: run_inference(weights, csv) -> predictions"""
    return inference(
        weights_path=weights,
        csv_path=csv,
        model_type="roberta",
        output_path="predictions.csv"
    )
```

This standardized interface allows external testing with:
```bash
python run_inference.py --weights weights/roberta_best.pt --csv data/test.csv
```

### 8.3 Output Format

```csv
text,prediction,predicted_emotion,true_label,true_emotion,correct
"i am so happy today",1,joy,1,joy,True
"feeling down again",0,sadness,0,sadness,True
"this makes me angry",3,anger,4,fear,False
```

---

## 9. Key Design Decisions

### 9.1 What Was Included

| Decision | Rationale |
|----------|-----------|
| **Three transformer models** | Compare different pre-training strategies |
| **Weighted loss** | Essential for class imbalance |
| **F1 Macro selection** | Ensures minority class performance |
| **Data preprocessing** | Twitter text is noisy |
| **Model compression** | Practical deployment consideration |
| **Confusion matrices** | Understand error patterns |

### 9.2 What Was NOT Included (and Why)

| Omission | Reason |
|----------|--------|
| **Data augmentation** | Pre-trained models already generalize well; tweets are too short for meaningful augmentation |
| **Ensemble methods** | Would complicate compression; single model preferred for deployment |
| **Custom architectures** | Fine-tuning pre-trained is more effective than training from scratch |
| **Emoji embeddings** | Out of scope; requires separate emoji2vec integration |
| **Multi-task learning** | Would require additional labels (sentiment, etc.) |
| **Knowledge distillation** | Quantization achieved sufficient compression |

### 9.3 Hyperparameter Choices

| Choice | Alternative Considered | Why We Chose This |
|--------|----------------------|-------------------|
| 3 epochs | 5-10 epochs | Pre-trained models converge fast; more epochs risk overfitting |
| lr=2e-5 | 1e-4, 5e-5 | Standard BERT fine-tuning; validated empirically |
| MAX_LENGTH=128 | 64, 256 | 128 covers 99%+ of tweets; efficient padding |
| batch_size=32 | 16, 64 | Balance of gradient stability and memory |

---

## 10. Results Summary

### 10.1 Model Performance

| Model | Accuracy | F1 Macro | F1 Weighted | Size | Training Time |
|-------|----------|----------|-------------|------|---------------|
| BERT | 93.40% | 91.08% | 93.50% | 417.7 MB | ~45 min |
| **RoBERTa** | **93.40%** | **91.32%** | **93.52%** | 475.5 MB | ~50 min |
| ELECTRA | 93.10% | 90.82% | 93.21% | 417.7 MB | ~45 min |

**Winner**: RoBERTa (highest F1 Macro)

### 10.2 Per-Class Performance (RoBERTa)

| Emotion | F1 Score | Samples | Analysis |
|---------|----------|---------|----------|
| Joy | 0.95 | 5,362 | Excellent (majority class) |
| Sadness | 0.94 | 4,666 | Excellent |
| Anger | 0.91 | 2,159 | Very good |
| Fear | 0.89 | 1,937 | Good |
| Love | 0.87 | 1,304 | Good (often confused with joy) |
| Surprise | 0.82 | 572 | Acceptable (minority class challenge) |

### 10.3 Compression Results

| Technique | Size Reduction | Accuracy Impact | Recommendation |
|-----------|---------------|-----------------|----------------|
| Quantization (INT8) | 2.06x smaller | -0.5% | ✅ **Recommended** |
| Pruning 30% | No reduction | -3.2% | ⚠️ Not recommended |
| Pruning 50% | No reduction | -53.2% | ❌ Catastrophic |

### 10.4 Key Takeaways

1. **RoBERTa performs best** due to optimized pre-training on more data
2. **Weighted loss is essential** for handling the 9.4x class imbalance
3. **Quantization is practical** for deployment (2x smaller, minimal loss)
4. **Pruning doesn't work well** for transformers without expensive retraining
5. **F1 Macro is the right metric** for imbalanced classification

---

## File Quick Reference

| File | Purpose | When to Discuss |
|------|---------|-----------------|
| `config.py` | All hyperparameters | Architecture decisions |
| `data_preprocessing.py` | Text cleaning | Data pipeline |
| `train_*.py` | Model training | Training process |
| `utils/data_loader.py` | Dataset & batching | Data pipeline |
| `utils/trainer.py` | Training loop | Training process |
| `utils/metrics.py` | Evaluation | Results discussion |
| `model_compression.py` | Quantization & pruning | Compression |
| `run_inference.py` | Prediction | Demo |
| `compare_models.py` | Visualizations | Results |

---

## Presentation Flow Suggestion

1. **Introduction** (2 min): Problem, dataset, challenges
2. **Data Pipeline** (3 min): Preprocessing choices, tokenization
3. **Models** (3 min): BERT vs RoBERTa vs ELECTRA
4. **Training** (3 min): Loss function, optimizer, class weighting
5. **Results** (3 min): Performance comparison, confusion matrices
6. **Compression** (2 min): Quantization success, pruning failure
7. **Demo** (2 min): Run inference on sample texts
8. **Conclusions** (2 min): Key findings, future work

---

*Document generated for presentation preparation. Good luck with your presentation!*
