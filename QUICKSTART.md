# Quick Start - Testing the CNN Training Pipeline

## Smoke Test (5-10 minutes on Mac Mini CPU)

This will verify everything works before running full training on GPU.

### What it does:
- Trains for **2 epochs only**
- Uses only **80 training images** (10 batches × 8 images)
- Uses only **40 validation images** (5 batches × 8 images)
- Should complete in **5-10 minutes** on CPU

### Run the smoke test:

```bash
cd /Users/thomasfey-grytnes/Documents/Artificial\ Intelligence\ -\ Studying/dog_breed_classification

# Run smoke test
PYTHONPATH=src python3 -m dbc.train_cnn configs/cnn_smoke_test.yaml
```

### Expected Output:

You should see:
1. ✅ Data loaders created
2. ✅ Model built (ResNet50 with ~24M parameters)
3. ⚠️  LIMITING messages (confirming smoke test mode)
4. ✅ Training progress for 2 epochs
5. ✅ Final validation metrics saved

### Results saved to:
- `artifacts/smoke_test/checkpoints/best_model.keras` - Trained model
- `artifacts/smoke_test/cnn_metrics.json` - Final metrics
- `artifacts/smoke_test/training_history.json` - Loss/accuracy per epoch
- `artifacts/smoke_test/training_log.csv` - CSV log

---

## After Smoke Test Passes

### Option 1: Full Training on CPU (SLOW - several hours)
```bash
PYTHONPATH=src python3 -m dbc.train_cnn configs/cnn_cpu_test.yaml
```

### Option 2: Full Training on Cloud GPU (FAST - recommended!)
Use Google Colab with free GPU:
- Upload your code to Colab
- Training will complete in 20-30 minutes instead of hours

---

## Troubleshooting

### SSL Certificate Error
If you see SSL errors when downloading pre-trained weights:
```bash
# Option 1: Update certificates
/Applications/Python\ 3.11/Install\ Certificates.command

# Option 2: Use Colab (has certificates pre-configured)
```

### Out of Memory
If training crashes with OOM:
- Reduce `batch_size` from 8 to 4 in config
- Close other applications

### ImportError for tensorflow
```bash
pip install tensorflow
```
