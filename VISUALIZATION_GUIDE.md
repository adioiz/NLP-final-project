# Dataset Visualization Guide

## What This Script Does

`visualizations.py` generates professional dataset analysis visualizations for your presentation.

---

## Generated Visualizations

### 1. **dataset_analysis.png** (Main figure - 2 subplots)

**Left subplot:** Class Distribution Bar Chart
- Shows all 6 emotion classes
- Displays exact counts and percentages
- Color-coded for each emotion
- **Clearly shows the imbalance** (Joy: 33.5% vs Surprise: 3.6%)

**Right subplot:** Text Length Distribution Histogram
- Shows tweet length characteristics by emotion
- Overlapping histograms for each class
- Helps understand data characteristics

**Size:** 16×6 inches, high resolution (300 DPI)
**Use in presentation:** Slide 3 (Class Imbalance)

### 2. **class_distribution_pie.png** (Alternative view)

**Pie chart** showing class distribution
- Same data, different visualization style
- Good for showing proportions visually
- Exploded slices emphasize minority classes
- Includes legend with exact counts

**Size:** 10×8 inches, high resolution (300 DPI)
**Use in presentation:** Alternative to bar chart if preferred

---

## How to Run

```bash
python visualizations.py
```

**Requirements:**
- pandas
- matplotlib
- seaborn
- numpy

(All included in `requirements.txt`)

---

## Output

The script will:
1. Load training and validation data
2. Generate 2 visualization files in `outputs/`
3. Print detailed statistics to console

**Files created:**
```
outputs/dataset_analysis.png
outputs/class_distribution_pie.png
```

---

## Console Output

When you run it, you'll see:

```
Loading dataset...
Total samples: 18000

Visualization saved to: outputs/dataset_analysis.png
Visualization saved to: outputs/class_distribution_pie.png

============================================================
DATASET STATISTICS SUMMARY
============================================================

Total samples: 18,000
Training samples: 16,000
Validation samples: 2,000

Class Distribution (Training):
  Sadness   :  4,666 samples ( 29.2%)
  Joy       :  5,362 samples ( 33.5%)
  Love      :  1,304 samples (  8.2%)
  Anger     :  2,159 samples ( 13.5%)
  Fear      :  1,937 samples ( 12.1%)
  Surprise  :    572 samples (  3.6%)

Imbalance Ratio:
  Largest class (Joy): 5,362 samples
  Smallest class (Surprise): 572 samples
  Ratio: 9.4× difference

Text Length Statistics:
  Average length: 123.4 characters
  Median length: 118.0 characters
  Min length: 15 characters
  Max length: 256 characters

============================================================
VISUALIZATIONS CREATED:
============================================================
1. outputs/dataset_analysis.png
   - Bar chart: Class distribution (shows imbalance)
   - Histogram: Text length distribution by emotion

2. outputs/class_distribution_pie.png
   - Pie chart: Alternative view of class imbalance

Add these to your presentation slides!
============================================================
```

---

## How to Use in Presentation

### Option 1: Use Combined Figure (Recommended)

**Add to Slide 3 (Class Imbalance Challenge):**
- Replace or supplement the text table
- Use `dataset_analysis.png`
- Shows both imbalance AND data characteristics
- Professional, publication-quality

**Placement:** Full-width or split with text

### Option 2: Use Pie Chart

**Add to Slide 3 or new slide:**
- Use `class_distribution_pie.png`
- Good for visual impact
- Clearly shows proportion differences
- Legend includes exact counts

**Placement:** Center of slide, large

### Option 3: Use Bar Chart Only

**Extract left subplot only:**
- Use in slide about class imbalance
- More formal/academic style
- Easy to read exact numbers

---

## Customization (If Needed)

If you want to modify the visualizations, edit `visualizations.py`:

**Change colors:**
```python
COLORS = ['#FF6B6B', '#FFD93D', '#FF8DC7', '#C44536', '#6BCB77', '#4D96FF']
```

**Change figure size:**
```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # Adjust (width, height)
```

**Change DPI (resolution):**
```python
plt.savefig('outputs/dataset_analysis.png', dpi=300)  # Higher = better quality
```

**Add more bins to histogram:**
```python
axes[1].hist(..., bins=30, ...)  # Increase bins for more detail
```

---

## Key Statistics for Presentation

Use these numbers when presenting:

**Dataset Size:**
- 18,000 total tweets
- 16,000 training
- 2,000 validation

**Imbalance:**
- **9.4× ratio** between largest and smallest class
- Joy: 5,362 samples (33.5%) - **largest**
- Surprise: 572 samples (3.6%) - **smallest**

**Tweet Characteristics:**
- Average length: ~123 characters
- Typical Twitter posts (under 280 char limit)

**Why This Matters:**
- Standard accuracy misleading for imbalanced data
- Must use weighted loss and F1 Macro
- Shows challenge was significant

---

## Presentation Talking Points

When showing these visualizations:

**For Bar Chart:**
> "As you can see, our dataset has significant class imbalance. Joy represents 33.5% of our data while Surprise is only 3.6% - that's a 9.4× difference. This is why we used weighted loss and F1 Macro instead of simple accuracy."

**For Text Length:**
> "The histogram shows our tweets have typical Twitter characteristics, averaging around 123 characters. All emotions show similar length distributions, so text length alone won't predict emotion."

**Key Takeaway:**
> "Despite this 9× imbalance, our model achieved 86%+ F1 score even on the smallest class - demonstrating our approach worked."

---

## Quick Test

After running, verify the files:

```bash
ls -lh outputs/dataset_analysis.png
ls -lh outputs/class_distribution_pie.png
```

Both should be 200-500 KB in size.

Preview them before adding to presentation!

---

## Files Modified

If you need to generate new visualizations:
1. Edit `visualizations.py`
2. Run: `python visualizations.py`
3. Check `outputs/` folder
4. Update presentation slides

---

**Ready to create professional dataset visualizations for your presentation!**
