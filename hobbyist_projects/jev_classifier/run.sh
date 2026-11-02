#!/usr/bin/env bash
# bash + zsh friendly
set -e
cd "$(dirname "$0")"

export PYTHONWARNINGS="ignore"
export TOKENIZERS_PARALLELISM=false

echo "=== OpenJEV / d1 classifier ==="
echo ""
python3 classifier.py --backend d1 --demo both --out results

echo ""
echo "Other options:"
echo "  python classifier.py --backend auto --demo interactive"
echo "  python classifier.py --backend pytorch --demo both   # needs CUDA ideally"
