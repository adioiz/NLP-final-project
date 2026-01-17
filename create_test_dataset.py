"""Create a test dataset by sampling from training data."""

import pandas as pd
import numpy as np

np.random.seed(42)

print("Creating test dataset from training data...")

train_df = pd.read_csv("data/train_cleaned.csv")
print(f"Total training samples: {len(train_df)}")

test_size = int(len(train_df) * 0.10)
print(f"Sampling {test_size} samples (10%) for testing")

test_df = train_df.sample(n=test_size, random_state=42)

print("\nClass distribution in test dataset:")
for label in range(6):
    count = (test_df['label'] == label).sum()
    pct = 100 * count / len(test_df)
    print(f"  Label {label}: {count:4d} ({pct:.1f}%)")

test_df.to_csv("data/test_sample.csv", index=False)
print(f"\nTest dataset saved to: data/test_sample.csv")
print(f"Total samples: {len(test_df)}")
