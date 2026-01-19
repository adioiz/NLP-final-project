# PowerPoint Presentation Structure
# Twitter Emotion Classification Using Transformers

---

## SLIDE 1: Title Slide
**Title:** Emotion Classification in Twitter Data Using Transformer-Based Models

**Subtitle:** A Comparative Study with Model Compression

**Authors:** Adi Oizerovich & Roei Michael

**Institution:** Hebrew University of Jerusalem

**Course:** NLP - Part B

**Visual:** Clean title slide (no images)

---

## SLIDE 2: Project Overview

**Title:** Project Goal

**Content:**
- **Task:** Classify Twitter posts into 6 emotions
  - 0: Sadness
  - 1: Joy
  - 2: Love
  - 3: Anger
  - 4: Fear
  - 5: Surprise

- **Dataset:** 18,000 English tweets
  - Training: 16,000 samples
  - Validation: 2,000 samples

- **Challenge:** Class imbalance (Joy: 33.5% vs Surprise: 3.6%)

**Visual:** Simple table showing class distribution

**Note:** Keep text minimal, use bullet points

---

## SLIDE 3: Key Challenge - Class Imbalance

**Title:** Dataset Challenge: Class Imbalance

**Content:**

**Class Distribution:**
| Emotion | Count | Percentage |
|---------|-------|------------|
| Joy | 5,362 | 33.5% |
| Sadness | 4,666 | 29.2% |
| Anger | 2,159 | 13.5% |
| Fear | 1,937 | 12.1% |
| Love | 1,304 | 8.2% |
| Surprise | 572 | 3.6% |

**Our Solution:**
- Weighted Cross-Entropy Loss
- Higher weights for minority classes (Surprise: 4.66×, Love: 2.04×)
- F1 Macro score as primary metric

**Visual:** Bar chart showing class distribution (create simple chart in PowerPoint)

---

## SLIDE 4: Models Compared

**Title:** Three Transformer Models Tested

**Content:**

| Model | Parameters | Size | Architecture Highlight |
|-------|-----------|------|------------------------|
| **BERT** | 109M | 418 MB | Bidirectional encoder |
| **RoBERTa** | 125M | 476 MB | Optimized BERT (dynamic masking) |
| **ELECTRA** | 109M | 418 MB | Discriminator-based pretraining |

**All models:**
- Pre-trained on large text corpora
- Fine-tuned on our Twitter emotion dataset
- 3 epochs training
- AdamW optimizer (LR: 2e-5)

**Visual:** None needed (table is visual enough)

---

## SLIDE 5: Model Performance Comparison

**Title:** Results: All Models Exceed 93% Accuracy

**Content:**

**Overall Performance:**
| Model | Accuracy | F1 Macro | F1 Weighted | Training Time |
|-------|----------|----------|-------------|---------------|
| BERT | 93.4% | 91.08% | 93.50% | 85 min |
| **RoBERTa** ⭐ | **93.4%** | **91.32%** | **93.52%** | 90 min |
| ELECTRA | 93.1% | 90.82% | 93.21% | 84 min |

**Winner:** RoBERTa (highest F1 Macro - better for imbalanced data)

**Visual:** Use `outputs/plots/01_model_performance_comparison.png`
- Place as full-slide background or large image on right side
- Shows accuracy, F1 scores in bar chart format

---

## SLIDE 6: Per-Class Performance Analysis

**Title:** Performance Breakdown by Emotion

**Content:**

**RoBERTa F1 Scores per Class:**
| Emotion | F1 Score | Comment |
|---------|----------|---------|
| Sadness | 96.2% | Excellent |
| Joy | 95.1% | Excellent |
| Anger | 93.5% | Very good |
| Fear | 88.9% | Good |
| Love | 88.1% | Good (despite 8% of data) |
| Surprise | 86.2% | Good (despite only 3.6% of data!) |

**Key Finding:** Even minority classes (Love, Surprise) achieve >86% F1

**Visual:** Use `outputs/plots/02_per_class_f1_comparison.png`
- Shows comparison across all 3 models for each emotion
- Demonstrates consistency across models

---

## SLIDE 7: Confusion Matrix - Best Model

**Title:** RoBERTa Confusion Matrix

**Content:**
- Shows where the model makes mistakes
- Most errors between similar emotions (e.g., Joy ↔ Love, Anger ↔ Fear)
- Strong diagonal = accurate predictions

**Visual:** Use `outputs/roberta_confusion_matrix.png`
- **Place as LARGE centered image** (this is the main visual)
- Should take up 70-80% of slide

**Optional text (bottom):**
"Darker blue diagonal = correct predictions. Off-diagonal = confusion between emotions."

---

## SLIDE 8: Training Resources Comparison

**Title:** Computational Efficiency

**Content:**

**Training Resources:**
- All models trained in ~85-90 minutes
- Similar GPU memory requirements
- Batch size: 32
- Max sequence length: 128 tokens

**Model Sizes:**
- BERT/ELECTRA: ~418 MB
- RoBERTa: ~476 MB (13% larger, minimal accuracy gain)

**Takeaway:** BERT offers best size/performance tradeoff if space is critical

**Visual:** Use `outputs/plots/03_training_resources_comparison.png`
- Shows training time, model size, parameters comparison
- Bar charts for easy comparison

---

## SLIDE 9: Model Compression - Part 1

**Title:** Model Compression: Making Models Smaller

**Goal:** Deploy on mobile/edge devices with limited resources

**Two Methods Tested:**

**1. Quantization (8-bit)**
- Convert weights from 32-bit to 8-bit integers
- **Size:** 476 MB → 231 MB (2.06× smaller)
- **Accuracy:** 93.4% → 92.9% (0.5% drop)
- **Verdict:** ✅ Excellent tradeoff!

**2. Pruning**
- Remove small-magnitude weights
- **30% Pruning:** 90.2% accuracy (3.2% drop) - Acceptable
- **50% Pruning:** 40.2% accuracy - ❌ Model collapse!

**Visual:** None (save space for next slide)

---

## SLIDE 10: Model Compression - Results

**Title:** Compression Comparison

**Content:**

| Method | Size | Accuracy | F1 Macro | Compression |
|--------|------|----------|----------|-------------|
| Original | 476 MB | 93.4% | 91.3% | 1.0× |
| **Quantized (8-bit)** ⭐ | **231 MB** | **92.9%** | **90.7%** | **2.06×** |
| Pruned (30%) | 476 MB | 90.2% | 86.5% | 1.0× |
| Pruned (50%) | 476 MB | 40.2% | 19.4% | 1.0× |

**Recommendation:** Quantization for deployment (2× smaller, minimal loss)

**Visual:** Use `outputs/plots/05_compression_tradeoff.png`
- **LARGE image** showing accuracy vs compression tradeoff
- Clearly shows quantization is the sweet spot

---

## SLIDE 11: Visual Summary Table

**Title:** Complete Results Summary

**Visual:** Use `outputs/plots/04_summary_table.png`
- **Full-slide image**
- Shows all metrics in one comprehensive table
- Easy reference for questions

**No additional text needed** - the table is self-explanatory

---

## SLIDE 12: Technical Implementation Highlights

**Title:** Implementation Details

**Key Technologies:**
- **Framework:** PyTorch + Hugging Face Transformers
- **Hardware:** NVIDIA GPU (CUDA enabled)
- **Training Strategy:**
  - Weighted loss for class imbalance
  - Linear warmup + learning rate decay
  - Gradient clipping (max_norm=1.0)
  - Best model selection via F1 Macro

**Code Quality:**
- Modular architecture (utils for data, metrics, training)
- Reproducible (fixed random seed)
- Well-documented and tested

**GitHub:** [Your Repository Link]

**Visual:** None (text slide)

---

## SLIDE 13: Key Contributions

**Title:** What We Delivered

**Checkmark list:**

✅ **Three transformer models** trained and compared
   - All exceed 80% accuracy requirement (achieved 93%!)

✅ **Handling class imbalance**
   - Weighted loss + F1 Macro evaluation
   - Minority classes achieve >86% F1

✅ **Two compression techniques**
   - Quantization: 2× smaller, <1% accuracy loss
   - Pruning: Showed limits (50% causes collapse)

✅ **Production-ready inference function**
   - Clean API: `run_inference(weights, csv)`
   - Returns predictions as list
   - Saves results to CSV

✅ **Comprehensive documentation**
   - Testing guide, API docs, README

**Visual:** None (use checkmarks/icons)

---

## SLIDE 14: Key Findings & Insights

**Title:** Main Takeaways

**1. Model Selection:**
- RoBERTa wins for imbalanced data (highest F1 Macro)
- BERT close second (smaller size, similar accuracy)
- All transformers performed well (93%+ accuracy)

**2. Class Imbalance:**
- Weighted loss critical for minority classes
- Even 3.6% class (Surprise) achieved 86% F1

**3. Compression:**
- Quantization is production-ready (2× smaller, <1% loss)
- Pruning >50% causes model failure
- Future work: Knowledge distillation

**4. Real-World Impact:**
- Social media sentiment analysis
- Mental health monitoring
- Customer feedback classification

**Visual:** None (keep focus on text)

---

## SLIDE 15: Future Work

**Title:** Potential Improvements

**Model Enhancements:**
- Test larger models (BERT-Large, RoBERTa-Large)
- Knowledge distillation (teacher-student)
- Ensemble methods

**Data Improvements:**
- More training data for minority classes
- Data augmentation (back-translation, paraphrasing)
- Emoji and emoticon handling

**Deployment:**
- Deploy quantized model to mobile app
- Real-time Twitter stream classification
- Multi-language support

**Visual:** None

---

## SLIDE 16: Conclusion

**Title:** Summary

**Problem:** Classify Twitter emotions with imbalanced data

**Solution:** Fine-tuned transformers with weighted loss

**Results:**
- 93.4% accuracy (RoBERTa)
- 91.3% F1 Macro
- Works well even for minority classes (3.6% of data)

**Compression:**
- 2× size reduction with minimal accuracy loss (quantization)

**Deliverables:**
- 3 trained models
- Production-ready inference API
- Comprehensive analysis and documentation

**Visual:** None (clean conclusion slide)

---

## SLIDE 17: Questions?

**Title:** Thank You!

**Authors:** Adi Oizerovich & Roei Michael

**Contact:**
- adi.oizerovich@gmail.com
- roeym111@gmail.com

**Repository:** [GitHub Link]

**Visual:** Optional - project logo or university logo

---

## APPENDIX SLIDES (Optional - if time permits)

### SLIDE 18: Data Preprocessing

**Title:** Data Preprocessing Pipeline

**Steps:**
1. Expand contractions (don't → do not)
2. Remove URLs and @mentions
3. Handle hashtags (keep text, remove #)
4. Remove special characters
5. Reduce repeated characters (happyyy → happy)
6. Lowercase normalization

**Example:**
- **Before:** "Im feeling soooo happy!!! 😊 #blessed @friend"
- **After:** "i am feeling so happy blessed"

---

### SLIDE 19: Model Architecture

**Title:** Transformer Architecture (Brief)

**BERT/RoBERTa/ELECTRA:**
- 12 transformer layers
- 768 hidden dimensions
- 12 attention heads
- Classification head: 768 → 6 classes

**Input:** Tweet text (max 128 tokens)
**Output:** Probability distribution over 6 emotions

**Visual:** Simple architecture diagram (optional)

---

## NOTES FOR CREATING THE PRESENTATION:

### Color Scheme Recommendations:
- **Primary:** Dark blue (#003366) for headers
- **Accent:** Orange/gold (#FF6600) for highlights
- **Background:** White or light gray
- **Charts:** Use contrasting colors (blue, orange, green)

### Font Recommendations:
- **Headers:** Arial Bold, 40-44pt
- **Body text:** Arial Regular, 20-24pt
- **Tables:** Arial, 16-18pt

### General Tips:
1. **Keep slides simple** - max 6 bullet points per slide
2. **Use visuals liberally** - you have great charts!
3. **Numbers in bold** - highlight key metrics
4. **Consistent layout** - use PowerPoint master slides
5. **Animations:** Minimal (fade in for bullets is fine)
6. **Timing:** 17 slides = ~12-15 min presentation

### Image Placement Guide:
- **Full-slide images:** Slides 7 (confusion matrix), 11 (summary table)
- **Right-side images:** Slides 5, 6, 8, 10 (charts)
- **No images:** Slides 1-4, 9, 12-17 (text/tables only)

### Most Important Slides (Don't Skip):
1. Slide 2 (Overview)
2. Slide 5 (Performance Comparison) ⭐
3. Slide 7 (Confusion Matrix) ⭐
4. Slide 10 (Compression Results) ⭐
5. Slide 13 (Contributions)
6. Slide 16 (Conclusion)

---

## QUICK BUILD ORDER:

1. Start with PowerPoint template (choose professional theme)
2. Create title slide (Slide 1)
3. Add overview slides (Slides 2-4)
4. **INSERT MAIN VISUALIZATIONS** (Slides 5-8, 10-11)
5. Add technical slides (Slides 9, 12)
6. Add summary slides (Slides 13-16)
7. End slide (Slide 17)
8. Review and adjust spacing/fonts
9. Add slide numbers
10. Practice timing!

---

Good luck with your presentation!
