# jev_classifier

Multi-label **email classifier** and **top-k coding-function ranker** using JEV-style / decision-model backends.

Two tasks share the same interface:

1. **Email** — labels such as `is_spam`, `is_science_related`, `is_legitimate_call_to_action`, …
2. **Coding project** — given a short project brief, rank which library functions are most relevant (from a small catalog).

## Backends (preference order when `--backend auto`)

| Backend     | Notes |
|-------------|--------|
| `d1`        | LiquidAI/d1-3B-GGUF via `llama-server` `/v1/systemone` (true decision model; recommended on Mac / CPU) |
| `pytorch`   | Original OpenJEV-style transformers weights (CUDA preferred; MPS often painful with bitsandbytes) |
| `llama_cpp` | Prompt-based instruct GGUF fallback |

## Requirements

```bash
# always
pip install numpy

# for d1 / llama_cpp path
# install llama.cpp and llama-server, or use a prebuilt binary

# for pytorch path
pip install torch transformers
# bitsandbytes only if you are on CUDA and want the original quant path
```

## Run

```bash
# 1) (optional but recommended) start the d1 server in another terminal
./start_d1_server.sh
# or manually:
#   llama-server -hf LiquidAI/d1-3B-GGUF:Q4_K_M --port 8080

# 2) run demos
./run.sh
python classifier.py --backend d1 --demo both
python classifier.py --backend auto --demo interactive
python classifier.py --backend llama_cpp --demo email
```

`--demo` options: `email`, `coding`, `both`, `interactive`.

Synthetic examples live in `synthetic_data.py` so the demo runs without external datasets.

## Layout

```text
jev_classifier/
├── classifier.py         # backends + CLI
├── synthetic_data.py     # email examples + function catalog + briefs
├── start_d1_server.sh    # helper to launch llama-server with d1 GGUF
└── run.sh
```

## Notes

- On Apple Silicon, prefer the **d1 / llama.cpp** path; bitsandbytes + MPS is unreliable.
- The d1 model is used as a decision / scoring backend (system-one style), not as a free-form chatbot.
- All demos write clear terminal output; no GUI required.
