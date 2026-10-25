#!/usr/bin/env bash
# bash + zsh friendly
set -e
cd "$(dirname "$0")"

export PYTHONWARNINGS="ignore"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg          # no GUI needed; always save figures to disk

echo "Installing minimal deps if needed (quiet)..."
python -m pip install -q datasets rdkit xgboost scikit-learn matplotlib numpy 2>/dev/null || true

echo ""
echo "=== Tox21 pipeline (ECFP + RDKit descriptors) ==="
echo "Results and plots will be written to results/"
python3 train_eval.py --target NR-AR --out results 2>/dev/null

echo ""
echo "Done. Check results/ for:"
echo "  - summary.csv"
echo "  - summary.txt"
echo "  - *_roc.png / *_pr.png / label_dist_*.png"
echo ""
echo "To also run MolDeBERTa embeddings (needs transformers + preferably GPU):"
echo "  python train_eval.py --target NR-AR --use-moldeberta --out results_bert"
