#!/usr/bin/env bash
# bash + zsh friendly (macOS)
set -e
cd "$(dirname "$0")"

export PYTHONWARNINGS="ignore"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

echo "Installing deps (quiet)..."
python -m pip install -q datasets torch scikit-learn matplotlib pillow numpy 2>/dev/null || true

echo ""
echo "=== RBM + MLP MNIST experiment ==="
echo "Figures and a results.txt will be written to results/"
echo "(This can take a few minutes on CPU)"
python3 train.py --steps 50 --epochs 6 --device cpu --out results 2>/dev/null

echo ""
echo "Done. Open results/results.txt and the PNG files."
