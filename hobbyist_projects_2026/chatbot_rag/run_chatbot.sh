#!/usr/bin/env bash
# Works under bash and zsh (macOS default).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${1:-${LLAMA_MODEL:-models/Llama-3.2-3B-Instruct-Q4_K_M.gguf}}"
HISTORY="${2:-NONE}"

mkdir -p docs

echo "=============================================="
echo "  Local Encrypted Chatbot + RAG"
echo "=============================================="
echo "Model   : $MODEL"
echo "History : $HISTORY"
echo ""
echo "RAG: put .txt / .md files into ./docs/"
echo "     (RAG turns on automatically when files exist)"
echo "=============================================="
echo ""

# Quiet python warnings on macOS
export PYTHONWARNINGS="ignore"
export TOKENIZERS_PARALLELISM=false

python3 chatbot.py "$HISTORY" "$MODEL"
