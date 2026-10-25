#!/usr/bin/env bash
# bash + zsh friendly
set -e
cd "$(dirname "$0")"

export PYTHONWARNINGS="ignore"
export TOKENIZERS_PARALLELISM=false

echo "=== OpenJEV / d1 classifier ==="
echo ""
echo "Recommended backend on Mac / CPU: LiquidAI d1-3B (decision model)"
echo "In a separate terminal start the server once:"
echo "  llama-server -hf LiquidAI/d1-3B-GGUF:Q4_K_M --port 8080"
echo "  (Q4_K_M ≈ 1.67 GB)"
echo ""
echo "Then run demos against it:"
python3 classifier.py --backend d1 --demo both --out results

echo ""
echo "Other options:"
echo "  python classifier.py --backend auto --demo interactive"
echo "  python classifier.py --backend pytorch --demo both   # needs CUDA ideally"
