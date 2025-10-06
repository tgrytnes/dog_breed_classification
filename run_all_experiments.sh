#!/bin/bash
# Run all three baseline model experiments sequentially
# This script trains ResNet50, VGG16, and EfficientNetB0 for comparison
set -euo pipefail

export PYTHONPATH=src

echo "=========================================="
echo "Dog Breed Classification - Full Pipeline"
echo "=========================================="
echo ""
echo "This will run:"
echo "  1. ResNet50 Transfer Learning (~20-25 min)"
echo "  2. VGG16 Transfer Learning (~20-25 min)"
echo "  3. EfficientNetB0 Transfer Learning (~15-20 min)"
echo "  Total: ~60-80 minutes"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

START_TIME=$(date +%s)

# Experiment 1: ResNet50
echo ""
echo "=========================================="
echo "EXPERIMENT 1/3: ResNet50 Transfer Learning"
echo "=========================================="
python3 -m dbc.train_cnn configs/exp_resnet50.yaml
EXP1_STATUS=$?

if [ $EXP1_STATUS -ne 0 ]; then
    echo "❌ ResNet50 experiment failed with exit code $EXP1_STATUS"
    exit $EXP1_STATUS
fi

# Experiment 2: VGG16
echo ""
echo "=========================================="
echo "EXPERIMENT 2/3: VGG16 Transfer Learning"
echo "=========================================="
python3 -m dbc.train_cnn configs/exp_vgg16.yaml
EXP2_STATUS=$?

if [ $EXP2_STATUS -ne 0 ]; then
    echo "❌ VGG16 experiment failed with exit code $EXP2_STATUS"
    exit $EXP2_STATUS
fi

# Experiment 3: EfficientNetB0
echo ""
echo "=========================================="
echo "EXPERIMENT 3/3: EfficientNetB0 Transfer Learning"
echo "=========================================="
python3 -m dbc.train_cnn configs/exp_efficientnet.yaml
EXP3_STATUS=$?

if [ $EXP3_STATUS -ne 0 ]; then
    echo "❌ EfficientNetB0 experiment failed with exit code $EXP3_STATUS"
    exit $EXP3_STATUS
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=========================================="
echo "Total runtime: ${MINUTES}m ${SECONDS}s"
echo ""

# Generate comparison
echo "Generating comparison report..."
python3 -m dbc.experiments compare
python3 -m dbc.experiments report

echo ""
echo "=========================================="
echo "RESULTS SUMMARY"
echo "=========================================="
echo ""
python3 -m dbc.experiments compare

echo ""
echo "✓ Full report saved to: experiments/experiments_report.md"
echo "✓ Best model checkpoints in: experiments/exp_*/checkpoints/best_model.keras"
echo ""
echo "Next steps:"
echo "  1. Review comparison above to identify best model"
echo "  2. Download best model checkpoint from experiments/"
echo "  3. (Optional) Fine-tune best model with trainable_base=true"
echo "  4. Deploy model to production"
echo ""
