"""
inference.py - Run inference on test data

This module provides the run_inference function:
    run_inference(weights, csv) → prediction

Functions:
    - run_inference: Core function that returns predictions
    - evaluate_predictions: Calculates metrics
    - main: CLI that runs inference and optionally evaluates

Usage:
    python inference.py --weights weights/roberta_best.pt --csv data/validation.csv
"""

import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

from config import NUM_CLASSES, MODEL_CONFIGS, MAX_LENGTH, LABEL_NAMES


def run_inference(weights_path, csv_path, model_type="roberta", output_path="predictions.csv"):
    """
    Run inference on a CSV file.
    
    Args:
        weights_path (str): Path to trained model weights (.pt file)
        csv_path (str): Path to CSV file with 'text' column
        model_type (str): Type of model ("bert", "roberta", or "electra")
        output_path (str): Path to save predictions CSV
    
    Returns:
        list: Predicted labels (0-5) for each text
    """
    print("="*60)
    print("RUNNING INFERENCE")
    print("="*60)
    
    # Determine model name from type
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: {list(MODEL_CONFIGS.keys())}")
    
    model_name = MODEL_CONFIGS[model_type]
    print(f"\nModel: {model_name}")
    print(f"Weights: {weights_path}")
    print(f"Input CSV: {csv_path}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES
    )
    
    # Load trained weights
    print(f"Loading weights from {weights_path}...")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load test data
    print(f"\nLoading data from {csv_path}...")
    test_df = pd.read_csv(csv_path)
    
    # Check if 'text' column exists
    if 'text' not in test_df.columns:
        raise ValueError("CSV file must have a 'text' column")
    
    texts = test_df["text"].tolist()
    print(f"Number of samples: {len(texts)}")
    
    # Run inference
    print("\n" + "-"*40)
    print("Running inference...")
    print("-"*40)
    predictions = []
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Predicting"):
            # Tokenize
            encoding = tokenizer(
                str(text),
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            
            # Predict
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=1).item()
            predictions.append(pred)
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        "text": texts,
        "prediction": predictions,
        "predicted_emotion": [LABEL_NAMES[p] for p in predictions]
    })
    
    # Add true labels if available
    if 'label' in test_df.columns:
        output_df["true_label"] = test_df["label"].tolist()
        output_df["true_emotion"] = [LABEL_NAMES[l] for l in test_df["label"]]
        output_df["correct"] = output_df["prediction"] == output_df["true_label"]
    
    # Save predictions
    output_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    
    # Print prediction distribution
    print("\n" + "-"*40)
    print("Prediction Distribution:")
    print("-"*40)
    for label_id, label_name in LABEL_NAMES.items():
        count = predictions.count(label_id)
        pct = 100 * count / len(predictions)
        print(f"  {label_name:10s}: {count:5d} ({pct:.1f}%)")
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE!")
    print("="*60)
    
    return predictions


def evaluate_predictions(predictions, true_labels, model_type="roberta"):
    """
    Evaluate predictions against true labels.
    
    Args:
        predictions (list): Predicted labels (0-5)
        true_labels (list): True labels (0-5)
        model_type (str): Model name for plot titles

    Returns:
        dict: Dictionary containing all metrics
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro')
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    f1_per_class = f1_score(true_labels, predictions, average=None)
    
    # Print summary metrics
    print(f"\n{'Metric':<20} {'Score':<10}")
    print("-" * 30)
    print(f"{'Accuracy':<20} {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'F1 Macro':<20} {f1_macro:.4f}")
    print(f"{'F1 Weighted':<20} {f1_weighted:.4f}")
    
    # Print per-class F1
    print("\n" + "-"*40)
    print("Per-Class F1 Scores:")
    print("-"*40)
    for i, label_name in LABEL_NAMES.items():
        print(f"  {label_name:10s}: {f1_per_class[i]:.4f}")
    
    # Print classification report
    print("\n" + "-"*40)
    print("Classification Report:")
    print("-"*40)
    label_names_list = [LABEL_NAMES[i] for i in range(NUM_CLASSES)]
    print(classification_report(true_labels, predictions, target_names=label_names_list))
    
    # Build results dictionary
    results = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_per_class": {LABEL_NAMES[i]: float(f1_per_class[i]) for i in range(NUM_CLASSES)}
    }
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    
    return results


def show_example_predictions(predictions, true_labels, texts, num_examples=3):
    """
    Display example correct and incorrect predictions.
    
    Args:
        predictions (list): Predicted labels
        true_labels (list): True labels
        texts (list): Original texts
        num_examples (int): Number of examples to show for each category
    """
    print("\n" + "-"*40)
    print("Example Predictions:")
    print("-"*40)
    
    # Find correct and incorrect predictions
    correct_indices = [i for i in range(len(predictions)) if predictions[i] == true_labels[i]]
    incorrect_indices = [i for i in range(len(predictions)) if predictions[i] != true_labels[i]]
    
    # Show correct predictions
    print("\n✓ Correct Predictions:")
    for i in correct_indices[:num_examples]:
        text_preview = texts[i][:60] + "..." if len(texts[i]) > 60 else texts[i]
        print(f"  Text: \"{text_preview}\"")
        print(f"  Predicted: {LABEL_NAMES[predictions[i]]} ✓")
        print()
    
    # Show incorrect predictions
    if incorrect_indices:
        print("✗ Incorrect Predictions:")
        for i in incorrect_indices[:num_examples]:
            text_preview = texts[i][:60] + "..." if len(texts[i]) > 60 else texts[i]
            print(f"  Text: \"{text_preview}\"")
            print(f"  Predicted: {LABEL_NAMES[predictions[i]]} | True: {LABEL_NAMES[true_labels[i]]} ✗")
            print()


def main():
    """Command-line interface for inference."""
    parser = argparse.ArgumentParser(description="Run inference on test data")
    parser.add_argument("--weights", type=str, required=True,
                       help="Path to model weights (.pt file)")
    parser.add_argument("--csv", type=str, required=True,
                       help="Path to input CSV file with 'text' column")
    parser.add_argument("--model", type=str, default="roberta",
                       choices=["bert", "roberta", "electra"],
                       help="Model type (default: roberta)")
    parser.add_argument("--output", type=str, default="predictions.csv",
                       help="Output path for predictions (default: predictions.csv)")
    
    args = parser.parse_args()
    
    # Step 1: Run inference
    predictions = run_inference(
        weights_path=args.weights,
        csv_path=args.csv,
        model_type=args.model,
        output_path=args.output
    )
    
    # Step 2: evaluate predictions
    test_df = pd.read_csv(args.csv)
    

    true_labels = test_df["label"].tolist()
    texts = test_df["text"].tolist()
    
    metrics = evaluate_predictions(
        predictions=predictions,
        true_labels=true_labels,
        model_type=args.model
    )
    
    show_example_predictions(
        predictions=predictions,
        true_labels=true_labels,
        texts=texts,
        num_examples=3
    )
        
    return {"predictions": predictions, "metrics": metrics}


if __name__ == "__main__":
    main()