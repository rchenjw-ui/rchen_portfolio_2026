# Portfolio Code – Local ML Experiments

Four self-contained projects cleaned up from a year of evening work.
Intended to be readable, runnable on macOS (zsh + Apple Silicon) and Linux,
and free of terminal spam.

## Quick start (any project)

```bash
cd <project>
./run.sh          # works under bash and zsh
```

All projects write human-readable `results/*.txt` and PNG figures to disk
(so you do not need a live display). Warnings are suppressed.

---

### 1. chatbot_rag/
Local encrypted chatbot + optional RAG.

```bash
cd chatbot_rag
# put any .txt / .md into docs/  → RAG turns on automatically
./run_chatbot.sh /path/to/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

- Encryption via AES-256 CBC
- RAG optional; add a .txt or .md file for use
- Default model is Llama-3.2-3B-Instruct-Q4_K_M.gguf

### 2. cheminformatics/
Tox21 activity prediction pipeline.

```bash
cd cheminformatics
./run.sh
# optional MolDeBERTa embeddings:
python train_eval.py --use-moldeberta --out results_bert
```

- Random / scaffold / Butina splits
- ECFP + RDKit descriptors (+ optional MolDeBERTa)
- PCA-8, class-imbalance handling, ROC + PR curves
- Results → `results/summary.txt` + PNGs

### 3. rbm_mnist/
RBM stack vs modern MLP on MNIST (honest comparison).

```bash
cd rbm_mnist
./run.sh
```

- Plain RBM vs sparsity+L2 RBM
- MLP random init vs RBM-initialised fine-tune
- Note: modern Adam MLPs often beat classic RBM pre-training on MNIST;
  that result is expected and documented in `results/results.txt`.

### 4. jev_classifier/
Multi-label email classifier + coding-function ranking.

```bash
cd jev_classifier
./run.sh                    # auto-selects llama.cpp on Mac/CPU
```

Backends:
- `llama_cpp` (default on Mac / no CUDA) – downloads a small GGUF if needed
- `pytorch` (CUDA or MPS; bitsandbytes only on CUDA)
- `d1` (Uses .gguf) - downloads small gguf (LiquidAI d1-3B-GGUF)

```bash
python classifier.py --backend llama_cpp --demo interactive
python classifier.py --backend pytorch --demo both   # if you have CUDA
```
