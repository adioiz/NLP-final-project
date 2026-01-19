# PowerPoint Quick Reference
# Use this for fast slide creation

---

## VISUALIZATION USAGE MAP

### Where to Use Each Image:

| Slide # | Title | Image File | Placement |
|---------|-------|------------|-----------|
| 5 | Model Performance | `01_model_performance_comparison.png` | Right side (50%) |
| 6 | Per-Class Performance | `02_per_class_f1_comparison.png` | Right side (50%) |
| 7 | Confusion Matrix | `roberta_confusion_matrix.png` | **FULL SLIDE (80%)** ⭐ |
| 8 | Training Resources | `03_training_resources_comparison.png` | Right side (50%) |
| 10 | Compression Tradeoff | `05_compression_tradeoff.png` | **LARGE (70%)** ⭐ |
| 11 | Summary Table | `04_summary_table.png` | **FULL SLIDE** ⭐ |

**Image folder:** `outputs/plots/` or `outputs/`

---

## KEY NUMBERS TO HIGHLIGHT

### Model Performance:
- **Best Model:** RoBERTa
- **Accuracy:** 93.4%
- **F1 Macro:** 91.32%
- **Training Time:** 90 minutes

### Model Comparison:
```
BERT:     93.4% accuracy, 91.08% F1, 418 MB, 85 min
RoBERTa:  93.4% accuracy, 91.32% F1, 476 MB, 90 min  ⭐ WINNER
ELECTRA:  93.1% accuracy, 90.82% F1, 418 MB, 84 min
```

### Per-Class F1 (RoBERTa):
```
Sadness:  96.2%  (Best)
Joy:      95.1%  (Best)
Anger:    93.5%  (Excellent)
Fear:     88.9%  (Good)
Love:     88.1%  (Good - only 8% of data)
Surprise: 86.2%  (Good - only 3.6% of data!)
```

### Compression Results:
```
Original:      476 MB, 93.4% accuracy, 91.3% F1
Quantized:     231 MB, 92.9% accuracy, 90.7% F1  ⭐ 2× SMALLER!
Pruned 30%:    476 MB, 90.2% accuracy, 86.5% F1
Pruned 50%:    476 MB, 40.2% accuracy, 19.4% F1  ❌ FAILED
```

### Dataset:
```
Total: 18,000 tweets
Train: 16,000 samples
Val:    2,000 samples

Joy:      33.5% (largest)
Sadness:  29.2%
Anger:    13.5%
Fear:     12.1%
Love:      8.2%
Surprise:  3.6% (smallest - 9× less than Joy!)
```

---

## SLIDE STRUCTURE (17 slides)

### Section 1: Introduction (Slides 1-4)
1. Title
2. Project Overview
3. Class Imbalance Challenge
4. Models Compared

### Section 2: Results (Slides 5-8) ⭐ CORE SECTION
5. **Model Performance** (use chart 01)
6. **Per-Class Performance** (use chart 02)
7. **Confusion Matrix** (use roberta confusion matrix) - FULL SLIDE
8. **Training Resources** (use chart 03)

### Section 3: Compression (Slides 9-11)
9. Compression Methods (text only)
10. **Compression Results** (use chart 05) - LARGE IMAGE
11. **Summary Table** (use chart 04) - FULL SLIDE

### Section 4: Wrap-Up (Slides 12-17)
12. Technical Implementation
13. Key Contributions
14. Key Findings
15. Future Work
16. Conclusion
17. Thank You / Questions

---

## 5-MINUTE EXPRESS BUILD

If you need to build FAST:

### Keep These 10 Slides Only:
1. Title
2. Project Overview
5. Model Performance (+ chart 01)
6. Per-Class Performance (+ chart 02)
7. Confusion Matrix (+ roberta matrix)
10. Compression Results (+ chart 05)
11. Summary Table (+ chart 04)
13. Key Contributions
16. Conclusion
17. Questions

**Skip:** Slides 3, 4, 8, 9, 12, 14, 15

---

## TEXT TEMPLATES

### Slide 2: Project Overview
```
GOAL: Classify Twitter posts into 6 emotions

EMOTIONS:
  Sadness, Joy, Love, Anger, Fear, Surprise

DATASET:
  18,000 tweets (16K train, 2K validation)

CHALLENGE:
  Class imbalance (Joy: 33.5% vs Surprise: 3.6%)
```

### Slide 5: Model Performance
```
RESULTS: All Models Exceed 93% Accuracy

         Accuracy    F1 Macro    F1 Weighted
BERT      93.4%      91.08%       93.50%
RoBERTa   93.4%      91.32%       93.52%  ⭐ WINNER
ELECTRA   93.1%      90.82%       93.21%

Winner: RoBERTa (best F1 Macro for imbalanced data)
```

### Slide 10: Compression
```
COMPRESSION RESULTS

Method          Size      Accuracy    Drop
Original        476 MB    93.4%       -
Quantized       231 MB    92.9%       0.5%  ⭐
Pruned (30%)    476 MB    90.2%       3.2%
Pruned (50%)    476 MB    40.2%       53.2%  ❌

Recommendation: Quantization (2× smaller, minimal loss)
```

### Slide 13: Key Contributions
```
WHAT WE DELIVERED

✅ Three transformer models (all >93% accuracy)
✅ Handled class imbalance (even 3.6% class: 86% F1)
✅ Two compression techniques tested
✅ Production-ready inference API
✅ Comprehensive documentation
```

### Slide 16: Conclusion
```
SUMMARY

Problem: Twitter emotion classification (imbalanced data)
Solution: Fine-tuned transformers + weighted loss

Results:
  • 93.4% accuracy (RoBERTa)
  • 91.3% F1 Macro
  • Works for minority classes

Compression:
  • 2× smaller (quantization)
  • <1% accuracy loss

Status: Production-ready ✅
```

---

## POWERPOINT TIPS

### Fonts:
- **Title:** Arial Bold, 44pt
- **Headers:** Arial Bold, 36pt
- **Body:** Arial, 24pt
- **Tables:** Arial, 18pt

### Colors:
- **Headers:** Dark blue (#003366)
- **Highlights:** Orange (#FF6600)
- **Good news:** Green (#00AA00)
- **Bad news:** Red (#CC0000)
- **Background:** White

### Layout:
- **Title slide:** Centered
- **Content slides:** Title at top, content below
- **Image slides:** Title + large centered image
- **Max bullets:** 6 per slide
- **Tables:** Use PowerPoint's table tool (not images)

### Animations:
- **None** or simple fade-in
- Don't distract from content

---

## FILES YOU NEED

Copy these from `outputs/` folder:

```
outputs/plots/01_model_performance_comparison.png
outputs/plots/02_per_class_f1_comparison.png
outputs/plots/03_training_resources_comparison.png
outputs/plots/04_summary_table.png
outputs/plots/05_compression_tradeoff.png
outputs/roberta_confusion_matrix.png
```

**Total: 6 images** for a complete presentation

---

## PRESENTATION TIMING

**17 slides = 15 minutes** (recommended)
- Intro: 3 minutes (slides 1-4)
- Results: 6 minutes (slides 5-8)
- Compression: 3 minutes (slides 9-11)
- Wrap-up: 3 minutes (slides 12-17)

**10 slides = 8 minutes** (express version)

**Practice 2-3 times** before presenting!

---

## COMMON QUESTIONS TO PREPARE FOR

**Q: Why RoBERTa over BERT?**
A: Higher F1 Macro (91.32% vs 91.08%) - better for imbalanced data

**Q: How did you handle class imbalance?**
A: Weighted loss (4.66× for Surprise) + F1 Macro evaluation

**Q: Why did 50% pruning fail?**
A: Too aggressive - removed critical weights, model couldn't recover

**Q: Deployment recommendation?**
A: Quantized RoBERTa (231 MB, 92.9% accuracy, 2× smaller)

**Q: Training time?**
A: ~90 minutes on GPU (NVIDIA)

**Q: Can you handle new emotions?**
A: Would need retraining with new labeled data

---

## BUILD CHECKLIST

- [ ] Create PowerPoint file
- [ ] Set up master slides (consistent headers/footers)
- [ ] Add title slide with authors
- [ ] Copy 6 visualization images into slides folder
- [ ] Build slides 5-11 with images first (core content)
- [ ] Add intro slides (1-4)
- [ ] Add conclusion slides (12-17)
- [ ] Add slide numbers
- [ ] Check all images are visible/clear
- [ ] Review for typos
- [ ] Practice timing
- [ ] Export to PDF (backup)

---

Good luck! 🚀
