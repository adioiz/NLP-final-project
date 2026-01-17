# Quick Start - Testing Your Project

## Fastest Way to Test Everything

```bash
# Run the automated test script
./run_full_test.sh
```

This will:
1. ✓ Create test dataset (if needed)
2. ✓ Verify model weights exist
3. ✓ Test the `run_inference()` function
4. ✓ Verify predictions output

---

## What the Professor Will Do

The professor will likely test your code like this:

```python
# In Python REPL or script
from run_inference import run_inference

# Call with exact signature from project specs
predictions = run_inference("weights/roberta_best.pt", "test_data.csv")

# Verify it works
print(type(predictions))  # Should be: list
print(len(predictions))   # Should match CSV rows
print(predictions[:10])   # First 10 predictions
```

**Your function is ready!** ✓

---

## Test Files Created

- **`data/test_sample.csv`** - 1,600 test samples (10% of training data)
- **`test_inference_simple.py`** - Simple test demonstrating run_inference
- **`run_full_test.sh`** - Automated test script
- **`TESTING_GUIDE.md`** - Complete testing documentation

---

## Manual Test (Step-by-Step)

If you prefer to test manually:

### 1. Check data
```bash
ls data/test_sample.csv
```

### 2. Check weights
```bash
ls weights/roberta_best.pt
```

### 3. Test inference
```bash
python test_inference_simple.py
```

### 4. Check output
```bash
head predictions.csv
```

---

## Expected Results

When you run the test, you should see:

```
Testing run_inference function as required by project specs
Function signature: run_inference(weights, csv)

Weights: weights/roberta_best.pt
Input CSV: data/test_sample.csv

Loading roberta-base on cuda
Processing 1600 samples
Predicting: 100%|████████████████| 1600/1600

Returned predictions (first 10): [1, 0, 2, 4, 1, 3, 0, 1, 5, 2]
Total predictions: 1600
Predictions type: <class 'list'>

Output saved to: predictions.csv
```

**If you see this, everything works!** ✓

---

## Troubleshooting

### "No module named 'torch'"
```bash
pip install -r requirements.txt
```

### "FileNotFoundError: weights/roberta_best.pt"
```bash
# You need to train the model first (or use existing weights)
python train_roberta.py
```

### "CUDA out of memory"
Edit `config.py` and change:
```python
DEVICE = "cpu"
```

---

## What to Submit

Make sure you include:
1. ✓ All code files (already in repo)
2. ✓ `run_inference.py` with the required function ✓
3. ✓ Trained model weights (`weights/roberta_best.pt`)
4. ✓ Report PDF (`פרויקט חלק ב.pdf`) ✓
5. ✓ Requirements.txt ✓
6. ✓ README.md ✓

**Your project is complete and ready for submission!**

---

For detailed testing instructions, see **TESTING_GUIDE.md**
