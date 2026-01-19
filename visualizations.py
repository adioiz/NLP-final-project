"""Generate dataset visualizations for presentation."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 12

LABEL_NAMES = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}

COLORS = ['#FF6B6B', '#FFD93D', '#FF8DC7', '#C44536', '#6BCB77', '#4D96FF']

print("Loading dataset...")
train_df = pd.read_csv("data/train_cleaned.csv")
val_df = pd.read_csv("data/validation_cleaned.csv")

total_df = pd.concat([train_df, val_df], ignore_index=True)
print(f"Total samples: {len(total_df)}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

counts = train_df['label'].value_counts().sort_index()
labels = [LABEL_NAMES[i] for i in range(6)]
percentages = (counts / len(train_df) * 100).values

bars = axes[0].bar(labels, counts.values, color=COLORS, alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (bar, count, pct) in enumerate(zip(bars, counts.values, percentages)):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

axes[0].set_xlabel('Emotion', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
axes[0].set_title('Class Distribution - Training Data\n(Showing Imbalance)',
                  fontsize=16, fontweight='bold', pad=20)
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim(0, max(counts.values) * 1.15)

for tick in axes[0].get_xticklabels():
    tick.set_fontsize(12)

train_df['text_length'] = train_df['text'].str.len()

for label_id in range(6):
    emotion_data = train_df[train_df['label'] == label_id]['text_length']
    axes[1].hist(emotion_data, bins=30, alpha=0.6, label=LABEL_NAMES[label_id],
                color=COLORS[label_id], edgecolor='black', linewidth=0.5)

axes[1].set_xlabel('Tweet Length (characters)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=14, fontweight='bold')
axes[1].set_title('Text Length Distribution by Emotion\n(Tweet Characteristics)',
                 fontsize=16, fontweight='bold', pad=20)
axes[1].legend(loc='upper right', fontsize=11, framealpha=0.9)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/dataset_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to: outputs/dataset_analysis.png")

plt.figure(figsize=(10, 8))

class_counts = train_df['label'].value_counts().sort_index()
emotion_labels = [LABEL_NAMES[i] for i in range(6)]

wedges, texts, autotexts = plt.pie(class_counts.values,
                                     labels=emotion_labels,
                                     colors=COLORS,
                                     autopct='%1.1f%%',
                                     startangle=90,
                                     textprops={'fontsize': 13, 'fontweight': 'bold'},
                                     explode=(0.05, 0, 0.05, 0.05, 0.05, 0.1))

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

plt.title('Training Data Distribution\n(Highlighting Imbalance)',
         fontsize=18, fontweight='bold', pad=20)

legend_labels = [f"{LABEL_NAMES[i]}: {class_counts[i]:,} samples" for i in range(6)]
plt.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1),
          fontsize=11, framealpha=0.9)

plt.tight_layout()
plt.savefig('outputs/class_distribution_pie.png', dpi=300, bbox_inches='tight')
print("Visualization saved to: outputs/class_distribution_pie.png")

print("\n" + "="*60)
print("DATASET STATISTICS SUMMARY")
print("="*60)
print(f"\nTotal samples: {len(total_df):,}")
print(f"Training samples: {len(train_df):,}")
print(f"Validation samples: {len(val_df):,}")

print("\nClass Distribution (Training):")
for label_id in range(6):
    count = (train_df['label'] == label_id).sum()
    pct = 100 * count / len(train_df)
    print(f"  {LABEL_NAMES[label_id]:10s}: {count:5,} samples ({pct:5.1f}%)")

print("\nImbalance Ratio:")
max_class = class_counts.max()
min_class = class_counts.min()
print(f"  Largest class (Joy): {max_class:,} samples")
print(f"  Smallest class (Surprise): {min_class:,} samples")
print(f"  Ratio: {max_class/min_class:.1f}× difference")

print("\nText Length Statistics:")
print(f"  Average length: {train_df['text_length'].mean():.1f} characters")
print(f"  Median length: {train_df['text_length'].median():.1f} characters")
print(f"  Min length: {train_df['text_length'].min()} characters")
print(f"  Max length: {train_df['text_length'].max()} characters")

print("\n" + "="*60)
print("VISUALIZATIONS CREATED:")
print("="*60)
print("1. outputs/dataset_analysis.png")
print("   - Bar chart: Class distribution (shows imbalance)")
print("   - Histogram: Text length distribution by emotion")
print("\n2. outputs/class_distribution_pie.png")
print("   - Pie chart: Alternative view of class imbalance")
print("\nAdd these to your presentation slides!")
print("="*60)

plt.close('all')
