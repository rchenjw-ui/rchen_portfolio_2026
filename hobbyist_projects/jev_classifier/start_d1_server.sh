#!/usr/bin/env bash
# Start LiquidAI d1-3B decision model server (llama.cpp)
# Requires a recent llama.cpp build with /v1/systemone support.
set -e
echo "Starting LiquidAI d1-3B (Q4_K_M) on port 8080 ..."
echo "Leave this terminal open. In another terminal run the classifier."
exec llama-server -hf LiquidAI/d1-3B-GGUF:Q4_K_M --port 8080 "$@"
