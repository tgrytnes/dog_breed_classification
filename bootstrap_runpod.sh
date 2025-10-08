#!/usr/bin/env bash
# Bootstrap script for RunPod GPU instances
# This script sets up the environment for dog breed classification training
set -euo pipefail

echo "=========================================="
echo "Dog Breed Classification - RunPod Setup"
echo "=========================================="

# Configure git
echo "Configuring git..."
git config user.name "tgrytnes"
git config user.email "web@fey-grytnes.com"
echo "✓ Git configured"

echo ""

# Check if running on GPU
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "✓ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ WARNING: No GPU detected! This may run very slowly."
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Verify Python >= 3.10
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "❌ ERROR: Python 3.10+ required, found $PYTHON_VERSION"
    exit 1
fi

echo ""
echo "Installing Python dependencies..."
echo "=================================="

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel

# Install core dependencies
echo ""
echo "Installing core ML libraries..."
pip install tensorflow>=2.15.0  # TensorFlow with GPU support
pip install pillow>=10.0.0      # Image processing (PIL/Pillow)

# Install data science libraries
echo ""
echo "Installing data science libraries..."
pip install pandas>=2.0
pip install numpy>=1.26
pip install scikit-learn>=1.3
pip install matplotlib>=3.7

# Install utilities
echo ""
echo "Installing utilities..."
pip install pyyaml>=6.0.2       # Config file parsing
pip install scipy>=1.11         # For .mat file loading

# Optional: Install IPython for better debugging
pip install ipython>=8.0

echo ""
echo "Installing Node.js (for Claude Code)..."
echo "========================================"
# Install Node.js via NodeSource repository
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
echo "✓ Node.js installed: $(node --version)"
echo "✓ npm installed: $(npm --version)"

echo ""
echo "Verifying TensorFlow GPU support..."
echo "===================================="
python3 << 'EOF'
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
if tf.config.list_physical_devices('GPU'):
    print("✓ TensorFlow can access GPU")
    for gpu in tf.config.list_physical_devices('GPU'):
        print(f"  - {gpu}")
else:
    print("⚠ WARNING: TensorFlow cannot access GPU")
EOF

echo ""
echo "Verifying all dependencies..."
echo "=============================="
python3 << 'EOF'
import sys
import importlib

required = {
    'tensorflow': '2.15.0',
    'PIL': '10.0.0',
    'pandas': '2.0',
    'numpy': '1.26',
    'sklearn': '1.3',
    'matplotlib': '3.7',
    'yaml': '6.0',
    'scipy': '1.11'
}

print("Checking required packages:")
all_ok = True
for pkg, min_ver in required.items():
    try:
        if pkg == 'PIL':
            mod = importlib.import_module('PIL')
        elif pkg == 'yaml':
            mod = importlib.import_module('yaml')
        elif pkg == 'sklearn':
            mod = importlib.import_module('sklearn')
        else:
            mod = importlib.import_module(pkg)

        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {pkg}: {version}")
    except ImportError:
        print(f"  ❌ {pkg}: NOT INSTALLED")
        all_ok = False

if all_ok:
    print("\n✓ All dependencies installed successfully!")
else:
    print("\n❌ Some dependencies are missing")
    sys.exit(1)
EOF

echo ""
echo "Setup complete!"
echo "==============="
echo ""
echo "Next steps:"
echo "  1. Download data:     PYTHONPATH=src python3 -m dbc.ingest"
echo "  2. Preprocess data:   PYTHONPATH=src python3 -m dbc.preprocess"
echo ""
echo "Phase 1: Train frozen baselines (1-2 hours):"
echo "  PYTHONPATH=src python3 -m dbc.train_cnn --config configs/exp_resnet50.yaml"
echo "  PYTHONPATH=src python3 -m dbc.train_cnn --config configs/exp_efficientnet.yaml"
echo "  PYTHONPATH=src python3 -m dbc.train_cnn --config configs/exp_efficientnetb4.yaml"
echo ""
echo "Phase 2: Fine-tune best model (3-5 hours):"
echo "  PYTHONPATH=src python3 -m dbc.finetune_from_checkpoint \\"
echo "    experiments/exp_xxx/checkpoints/best_model.h5 \\"
echo "    configs/exp_efficientnetb4_finetune.yaml \\"
echo "    --epochs 50"
echo ""
