# Testing Guide - Complete Project Walkthrough

This guide walks you through testing the entire project from scratch, including the `run_inference` function as the professor will test it.

---

## Prerequisites

Make sure you have all dependencies installed:
```bash
pip install -r requirements.txt
```

---

## Testing Overview

We'll test the project in this order:
1. Data preprocessing (if needed)
2. Model training (one model for quick test)
3. Model compression
4. **run_inference function** (professor's test)
5. Model comparison

---

## Step 1: Verify Data is Ready

Check that cleaned data exists:
```bash
ls -lh data/
```

You should see:
- `train_cleaned.csv` (16,000 samples)
- `validation_cleaned.csv` (2,000 samples)
- `test_sample.csv` (1,600 samples) - for testing inference

If cleaned data is missing, run:
```bash
python data_preprocessing.py
```

---

## Step 2: Quick Training Test (Optional - Takes ~20 min)

**Option A: Use existing trained weights** (RECOMMENDED - Fast)
```bash
ls weights/
```

If you see `roberta_best.pt`, skip to Step 3.

**Option B: Train a model from scratch** (Takes time)
```bash
# Train RoBERTa (best model, ~20-30 minutes on GPU)
python train_roberta.py
```

---

## Step 3: Test run_inference Function (CRITICAL - Professor's Test)

This is the **required interface** specified in the project requirements.

### Test 3A: Using the simple wrapper script
```bash
python test_inference_simple.py
```

**Expected output:**
```
Testing run_inference function as required by project specs
Function signature: run_inference(weights, csv)

Weights: weights/roberta_best.pt
Input CSV: data/test_sample.csv

Loading roberta-base on cuda
Processing 1600 samples
Predicting: 100%|████████████████| 1600/1600
Predictions saved to predictions.csv

Returned predictions (first 10): [1, 0, 2, 4, 1, ...]
Total predictions: 1600
Predictions type: <class 'list'>

Output saved to: predictions.csv
```

### Test 3B: Using Python REPL (Professor's likely approach)
```python
# Start Python
python

# Import the required function
from run_inference import run_inference

# Test with the exact signature from specs
predictions = run_inference("weights/roberta_best.pt", "data/test_sample.csv")

# Verify output
print(type(predictions))  # Should be: <class 'list'>
print(len(predictions))   # Should be: 1600
print(predictions[:5])    # First 5 predictions

# Check saved CSV
import pandas as pd
results = pd.read_csv("predictions.csv")
print(results.head())
```

### Test 3C: Using CLI (Advanced features)
```bash
python run_inference.py \
  --weights weights/roberta_best.pt \
  --csv data/test_sample.csv \
  --model roberta \
  --output test_predictions.csv
```

---

## Step 4: Verify Predictions Output

Check the predictions CSV:
```bash
head -10 predictions.csv
```

**Expected columns:**
- `text`: Original tweet text
- `prediction`: Predicted label (0-5)
- `predicted_emotion`: Emotion name (sadness, joy, love, anger, fear, surprise)
- `true_label`: Ground truth label (if available in input CSV)
- `true_emotion`: Ground truth emotion name
- `correct`: Boolean - whether prediction matches truth

---

## Step 5: Test Model Compression (Optional)

Test quantization and pruning:
```bash
python model_compression.py
```

**Expected output:**
- Quantized model results
- Pruned model results (30% and 50%)
- Comparison saved to `outputs/compression_results.json`

---

## Step 6: Test Model Comparison (Optional)

If you have all three models trained:
```bash
python compare_models.py
```

**Expected output:**
- Plots saved to `outputs/plots/`
- Model performance comparison
- Per-class F1 comparison
- Training resources comparison

---

## Quick Test Checklist

Use this checklist to verify everything works:

### Data
- [ ] `data/train_cleaned.csv` exists (16,000 samples)
- [ ] `data/validation_cleaned.csv` exists (2,000 samples)
- [ ] `data/test_sample.csv` exists (1,600 samples)

### Weights
- [ ] At least one model weight file exists in `weights/`
- [ ] Recommended: `weights/roberta_best.pt` (best model)

### Inference Function (CRITICAL)
- [ ] Can import: `from run_inference import run_inference`
- [ ] Function signature works: `run_inference(weights, csv)`
- [ ] Returns a list of predictions
- [ ] Saves output to `predictions.csv`

### Outputs
- [ ] `predictions.csv` created and contains correct columns
- [ ] Predictions are integers 0-5
- [ ] Number of predictions matches input CSV rows

---

## Common Issues & Solutions

### Issue 1: "No module named 'torch'"
**Solution:** Install requirements
```bash
pip install -r requirements.txt
```

### Issue 2: "FileNotFoundError: weights/roberta_best.pt"
**Solution:** Train at least one model first
```bash
python train_roberta.py
```

### Issue 3: CUDA out of memory
**Solution:** Use CPU or reduce batch size in config.py
```python
DEVICE = "cpu"  # or
BATCH_SIZE = 16  # reduce from 32
```

### Issue 4: Predictions CSV missing columns
**Solution:** Make sure input CSV has both 'text' and 'label' columns

---

## Expected Performance (RoBERTa)

When testing on validation data, expect:
- **Accuracy:** ~93.4%
- **F1 Macro:** ~91.3%
- **Inference speed:** ~100-200 samples/sec on GPU

---

## Files Generated During Testing

After running all tests, you should have:

```
outputs/
  ├── roberta_results.json
  ├── roberta_confusion_matrix.png
  ├── compression_results.json
  └── plots/
      ├── 01_model_performance_comparison.png
      ├── 02_per_class_f1_comparison.png
      ├── 03_training_resources_comparison.png
      ├── 04_summary_table.png
      └── 05_compression_tradeoff.png

predictions.csv
test_predictions.csv (if using CLI)
```

---

## Professor Testing Scenario

Your professor will likely do this:

1. Clone your repository
2. Install requirements: `pip install -r requirements.txt`
3. Download your trained weights (from submission)
4. Run in Python:
   ```python
   from run_inference import run_inference
   predictions = run_inference("weights/roberta_best.pt", "test_data.csv")
   ```
5. Verify:
   - Returns a list
   - Predictions are correct format
   - Creates predictions.csv

**Make sure your weights file is included in submission or provide download link!**

---

## Full Project Test (Complete Run)

If you want to test everything from scratch (takes ~1-2 hours):

```bash
# 1. Preprocess data (if needed)
python data_preprocessing.py

# 2. Train all three models (~1 hour on GPU)
python train_bert.py      # ~20 min
python train_roberta.py   # ~20 min
python train_electra.py   # ~20 min

# 3. Test inference on each model
python test_inference_simple.py

# 4. Compress best model
python model_compression.py  # ~10 min

# 5. Compare all models
python compare_models.py

# 6. Verify all outputs exist
ls outputs/
ls weights/
ls predictions.csv
```

---

## Success Criteria

Your project is working correctly if:

1. ✅ `run_inference(weights, csv)` function exists and works
2. ✅ Returns a Python list of predictions (0-5)
3. ✅ Saves predictions.csv with correct format
4. ✅ At least one trained model achieves >80% accuracy
5. ✅ Model compression techniques applied (2+ methods)
6. ✅ All scripts run without errors

---

## Need Help?

If you encounter issues:
1. Check error messages carefully
2. Verify all files in `data/` and `weights/` exist
3. Check Python version: `python --version` (should be 3.8+)
4. Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`

---

Good luck with your testing!
