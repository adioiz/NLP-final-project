from run_inference import run_inference

weights_path = "weights/roberta_best.pt"
csv_path = "data/test_sample.csv"

print(f"Weights: {weights_path}")
print(f"Input CSV: {csv_path}")
print()
predictions = run_inference(weights_path, csv_path)

print(f"Total predictions: {len(predictions)}")
