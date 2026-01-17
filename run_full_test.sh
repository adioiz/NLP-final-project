#!/bin/bash

echo "=============================================="
echo "FULL PROJECT TEST - NLP Emotion Classification"
echo "=============================================="
echo ""

echo "Step 1: Verify test dataset exists..."
if [ -f "data/test_sample.csv" ]; then
    echo "✓ Test dataset found: data/test_sample.csv"
    wc -l data/test_sample.csv
else
    echo "✗ Test dataset not found. Creating it now..."
    (head -1 data/train_cleaned.csv; tail -n +2 data/train_cleaned.csv | shuf -n 1600) > data/test_sample.csv
    echo "✓ Created test dataset"
fi
echo ""

echo "Step 2: Check for trained model weights..."
if [ -f "weights/roberta_best.pt" ]; then
    echo "✓ Model weights found: weights/roberta_best.pt"
    ls -lh weights/roberta_best.pt
else
    echo "✗ No trained weights found!"
    echo "   Please train a model first:"
    echo "   python train_roberta.py"
    exit 1
fi
echo ""

echo "Step 3: Test run_inference function (CRITICAL TEST)..."
echo "   This is what the professor will test!"
echo ""
python3 test_inference_simple.py
echo ""

echo "Step 4: Verify predictions output..."
if [ -f "predictions.csv" ]; then
    echo "✓ Predictions CSV created successfully"
    echo "   First 5 rows:"
    head -6 predictions.csv
    echo "   ..."
    echo "   Total rows: $(wc -l < predictions.csv)"
else
    echo "✗ Predictions CSV not created!"
    exit 1
fi
echo ""

echo "=============================================="
echo "TEST COMPLETE!"
echo "=============================================="
echo ""
echo "Summary:"
echo "  - Test dataset: data/test_sample.csv (1600 samples)"
echo "  - Model used: weights/roberta_best.pt"
echo "  - Predictions: predictions.csv"
echo ""
echo "Next steps:"
echo "  1. Review predictions.csv to verify correctness"
echo "  2. Test other models if needed"
echo "  3. Run model compression: python model_compression.py"
echo "  4. Compare all models: python compare_models.py"
echo ""
