"""Run inference on test data using trained models."""

import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report

from config import NUM_CLASSES, MODEL_CONFIGS, MAX_LENGTH, LABEL_NAMES


def inference(weights_path, csv_path, model_type="roberta", output_path="predictions.csv"):
    """Run inference on CSV file and return predictions."""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: {list(MODEL_CONFIGS.keys())}")

    model_name = MODEL_CONFIGS[model_type]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    test_df = pd.read_csv(csv_path)
    if 'text' not in test_df.columns:
        raise ValueError("CSV file must have a 'text' column")

    texts = test_df["text"].tolist()
    print(f"Processing {len(texts)} samples")

    predictions = []
    with torch.no_grad():
        for text in tqdm(texts, desc="Predicting"):
            encoding = tokenizer(
                str(text),
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=1).item()
            predictions.append(pred)

    output_df = pd.DataFrame({
        "text": texts,
        "prediction": predictions,
        "predicted_emotion": [LABEL_NAMES[p] for p in predictions]
    })

    if 'label' in test_df.columns:
        output_df["true_label"] = test_df["label"].tolist()
        output_df["true_emotion"] = [LABEL_NAMES[l] for l in test_df["label"]]
        output_df["correct"] = output_df["prediction"] == output_df["true_label"]

    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    return predictions


def run_inference(weights, csv):
    """Required interface: run_inference(weights, csv) -> predictions"""
    return inference(
        weights_path=weights,
        csv_path=csv,
        model_type="roberta",
        output_path="predictions.csv"
    )


def evaluate_predictions(predictions, true_labels, model_type="roberta"):
    """Evaluate predictions against true labels and return metrics."""
    accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro')
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    f1_per_class = f1_score(true_labels, predictions, average=None)

    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")

    print("\nPer-Class F1:")
    for i, label_name in LABEL_NAMES.items():
        print(f"  {label_name:10s}: {f1_per_class[i]:.4f}")

    print("\nClassification Report:")
    label_names_list = [LABEL_NAMES[i] for i in range(NUM_CLASSES)]
    print(classification_report(true_labels, predictions, target_names=label_names_list))

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_per_class": {LABEL_NAMES[i]: float(f1_per_class[i]) for i in range(NUM_CLASSES)}
    }


def show_example_predictions(predictions, true_labels, texts, num_examples=3):
    """Display example correct and incorrect predictions."""
    correct_indices = [i for i in range(len(predictions)) if predictions[i] == true_labels[i]]
    incorrect_indices = [i for i in range(len(predictions)) if predictions[i] != true_labels[i]]

    print("\nCorrect Predictions:")
    for i in correct_indices[:num_examples]:
        text_preview = texts[i][:60] + "..." if len(texts[i]) > 60 else texts[i]
        print(f"  Text: \"{text_preview}\"")
        print(f"  Predicted: {LABEL_NAMES[predictions[i]]}")
        print()

    if incorrect_indices:
        print("Incorrect Predictions:")
        for i in incorrect_indices[:num_examples]:
            text_preview = texts[i][:60] + "..." if len(texts[i]) > 60 else texts[i]
            print(f"  Text: \"{text_preview}\"")
            print(f"  Predicted: {LABEL_NAMES[predictions[i]]} | True: {LABEL_NAMES[true_labels[i]]}")
            print()


def main():
    """Command-line interface for inference."""
    parser = argparse.ArgumentParser(description="Run inference on test data")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pt file)")
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV file with 'text' column")
    parser.add_argument("--model", type=str, default="roberta", choices=["bert", "roberta", "electra"], help="Model type")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output path for predictions")

    args = parser.parse_args()

    predictions = inference(
        weights_path=args.weights,
        csv_path=args.csv,
        model_type=args.model,
        output_path=args.output
    )

    test_df = pd.read_csv(args.csv)
    true_labels = test_df["label"].tolist()
    texts = test_df["text"].tolist()

    metrics = evaluate_predictions(predictions, true_labels, args.model)
    show_example_predictions(predictions, true_labels, texts, num_examples=3)

    return {"predictions": predictions, "metrics": metrics}


if __name__ == "__main__":
    main()
