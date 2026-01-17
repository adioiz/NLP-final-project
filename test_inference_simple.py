"""Simple test script demonstrating the required run_inference function."""

from run_inference import run_inference

print("Testing run_inference function as required by project specs")
print("Function signature: run_inference(weights, csv)")
print()

weights_path = "weights/roberta_best.pt"
csv_path = "data/test_sample.csv"

print(f"Weights: {weights_path}")
print(f"Input CSV: {csv_path}")
print()

predictions = run_inference(weights_path, csv_path)

print(f"\nReturned predictions (first 10): {predictions[:10]}")
print(f"Total predictions: {len(predictions)}")
print(f"Predictions type: {type(predictions)}")
print()
print("Output saved to: predictions.csv")
