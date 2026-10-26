#!/usr/bin/env python3
"""
Expanded OpenJEV-style multi-label classifier.

Backends (in preference order when --backend auto):
  1. d1          LiquidAI/d1-3B-GGUF via llama-server /v1/systemone
                 (true decision model – recommended on Mac / CPU)
  2. pytorch     Original OpenJEV transformers weights (CUDA preferred)
  3. llama_cpp   Legacy prompt-based instruct GGUF (last-resort fallback)

Usage:
  # Start d1 server in another terminal (once):
  #   llama-server -hf LiquidAI/d1-3B-GGUF:Q4_K_M --port 8080
  python classifier.py --backend d1 --demo both
  python classifier.py --backend auto --demo interactive
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Any

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from synthetic_data import (
    EMAIL_EXAMPLES,
    CODING_FUNCTIONS,
    PROJECT_BRIEFS,
    get_function_catalog_text,
)

CLASS_LABELS = [
    "is_spam",
    "is_science_related",
    "is_legitimate_call_to_action",
    "is_college_related",
    "is_work_related",
    "is_promo",
]

RULES = """
- is_spam: Unsolicited, deceptive, or low-value messages that try to sell, phish, or spread malware.
- is_science_related: Content primarily about scientific research, discoveries, experiments, academic papers.
- is_legitimate_call_to_action: A clear, honest request for a specific action that is relevant and non-deceptive.
- is_college_related: Content about university/college life, admissions, courses, campus events, deadlines.
- is_work_related: Professional or job-related content: meetings, projects, deadlines, workplace communication.
- is_promo: Transparent marketing or promotional content (distinct from pure spam).
Priority: is_college_related > is_spam > is_promo > is_legitimate_call_to_action > other labels.
"""

# ---------------------------------------------------------------------------
# d1 / System One backend (LiquidAI d1-3B-GGUF)
# ---------------------------------------------------------------------------

def score_d1_systemone(
    state: str,
    questions: Dict[str, Dict[str, Any]],
    base_url: str = "http://127.0.0.1:8080",
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """
    Call llama-server /v1/systemone.
    questions values are System One typed questions, e.g.:
      {"type": "noul", "instructions": "Is this spam?"}
    Returns the raw 'answers' dict from the response.
    """
    import urllib.request
    import urllib.error

    url = base_url.rstrip("/") + "/v1/systemone"
    payload = {"state": state, "questions": questions}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach d1 server at {url}. "
            f"Start it with:\n"
            f"  llama-server -hf LiquidAI/d1-3B-GGUF:Q4_K_M --port 8080\n"
            f"Original error: {e}"
        ) from e
    return body.get("answers", body)


def classify_email_d1(text: str, base_url: str) -> Dict[str, float]:
    state = f"Rules:\n{RULES}\n\nMessage:\n{text}"
    questions = {
        lab: {
            "type": "noul",
            "instructions": f"Does the message match the definition of {lab}?",
        }
        for lab in CLASS_LABELS
    }
    answers = score_d1_systemone(state, questions, base_url=base_url)
    out = {}
    for lab in CLASS_LABELS:
        a = answers.get(lab, {})
        # noul returns probability of "yes" under various possible keys
        if isinstance(a, dict):
            p = a.get("probability", a.get("p", a.get("yes", a.get("true", 0.5))))
            if isinstance(p, dict):
                p = p.get("true", p.get("yes", 0.5))
            out[lab] = float(p)
        else:
            out[lab] = float(a) if a is not None else 0.5
    return out


def rank_functions_d1(project_brief: str, base_url: str, top_k: int = 6):
    catalog = get_function_catalog_text()
    state = (
        f"Project brief: {project_brief}\n\n"
        f"{catalog}\n\n"
        "Decide which helper functions are necessary for this project."
    )
    questions = {
        f["name"]: {
            "type": "noul",
            "instructions": (
                f"Is the function '{f['name']}' ({f['desc']}) needed "
                f"to implement this project?"
            ),
        }
        for f in CODING_FUNCTIONS
    }
    answers = score_d1_systemone(state, questions, base_url=base_url)
    scored = []
    for f in CODING_FUNCTIONS:
        a = answers.get(f["name"], {})
        if isinstance(a, dict):
            p = a.get("probability", a.get("p", a.get("yes", a.get("true", 0.0))))
            if isinstance(p, dict):
                p = p.get("true", p.get("yes", 0.0))
            scored.append((f["name"], float(p)))
        else:
            scored.append((f["name"], float(a) if a is not None else 0.0))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def check_d1_server(base_url: str = "http://127.0.0.1:8080") -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        try:
            # some builds have no /health; try a minimal systemone
            score_d1_systemone(
                "ping",
                {"ok": {"type": "noul", "instructions": "Is this a test?"}},
                base_url=base_url,
                timeout=3,
            )
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# PyTorch OpenJEV backend (original)
# ---------------------------------------------------------------------------

def load_pytorch(subfolder: str = "qwen3.5-4b-nli-v5", quantize: bool = True):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

    print("[backend] Loading OpenJEV via PyTorch ...")
    tok = AutoTokenizer.from_pretrained("AlexWortega/openjev", subfolder=subfolder)
    kwargs = dict(
        subfolder=subfolder,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    if quantize and torch.cuda.is_available():
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs["quantization_config"] = qconf
    else:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            kwargs["torch_dtype"] = torch.float16
            print("[backend] MPS – float16, no bitsandbytes.")
        else:
            kwargs["torch_dtype"] = torch.float32
            print("[backend] CPU – float32.")
    model = AutoModelForSequenceClassification.from_pretrained(
        "AlexWortega/openjev", **kwargs
    )
    model.eval()
    return tok, model


def score_pytorch(tok, model, premise: str, hypotheses: List[str]) -> Dict[str, float]:
    import torch
    tokens = tok(
        [premise] * len(hypotheses),
        hypotheses,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )
    tokens = {k: v.to(model.device) for k, v in tokens.items()}
    with torch.no_grad():
        logits = model(**tokens).logits
        probs = torch.softmax(logits, dim=-1)
    return {h: float(probs[i, 1]) for i, h in enumerate(hypotheses)}


# ---------------------------------------------------------------------------
# Legacy prompt-based llama.cpp fallback (only if d1 server unavailable)
# ---------------------------------------------------------------------------

def load_llama_cpp_fallback(model_path: Optional[str] = None):
    try:
        from llama_cpp import Llama
    except ImportError:
        print("[llama_cpp] Install: pip install llama-cpp-python")
        sys.exit(1)
    if model_path is None:
        print(
            "[llama_cpp] No GGUF given and d1 server not running.\n"
            "Prefer starting d1:\n"
            "  llama-server -hf LiquidAI/d1-3B-GGUF:Q4_K_M --port 8080\n"
            "Or pass --gguf /path/to/instruct.gguf for the legacy fallback."
        )
        sys.exit(1)
    print(f"[backend] Legacy llama.cpp instruct: {model_path}")
    llm = Llama(model_path=model_path, n_ctx=2048, verbose=False, n_gpu_layers=-1)
    return llm


def score_llama_cpp_prompt(llm, premise: str, hypotheses: List[str]) -> Dict[str, float]:
    scores = {}
    for h in hypotheses:
        prompt = (
            f"<|im_start|>system\nAnswer only YES or NO.<|im_end|>\n"
            f"<|im_start|>user\nPremise:\n{premise[:700]}\n\n"
            f"Hypothesis: {h}\nEntailed? YES or NO.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        out = llm(prompt, max_tokens=3, temperature=0.0, stop=["<|im_end|>", "\n"])
        text = out["choices"][0]["text"].strip().upper()
        scores[h] = 0.85 if text.startswith("YES") else (0.15 if text.startswith("NO") else 0.4)
    return scores


# ---------------------------------------------------------------------------
# Unified high-level API
# ---------------------------------------------------------------------------

def classify_email(backend: str, handle, text: str, d1_url: str = "http://127.0.0.1:8080") -> Dict[str, float]:
    if backend == "d1":
        return classify_email_d1(text, d1_url)
    if backend == "pytorch":
        tok, model = handle
        premise = f"--BEGIN RULES-- {RULES} --END RULES-- --BEGIN EVIDENCE-- {text} --END EVIDENCE--"
        hyps = [f"Hypothesis: go forward with: {lab}" for lab in CLASS_LABELS]
        raw = score_pytorch(tok, model, premise, hyps)
        return {lab: raw[hyps[i]] for i, lab in enumerate(CLASS_LABELS)}
    # legacy
    premise = f"--BEGIN RULES-- {RULES} --END RULES-- --BEGIN EVIDENCE-- {text} --END EVIDENCE--"
    hyps = [f"Hypothesis: go forward with: {lab}" for lab in CLASS_LABELS]
    raw = score_llama_cpp_prompt(handle, premise, hyps)
    return {lab: raw[hyps[i]] for i, lab in enumerate(CLASS_LABELS)}


def rank_functions(backend: str, handle, project_brief: str, top_k: int = 6, d1_url: str = "http://127.0.0.1:8080"):
    if backend == "d1":
        return rank_functions_d1(project_brief, d1_url, top_k=top_k)
    catalog = get_function_catalog_text()
    premise = f"Project: {project_brief}\n\n{catalog}\nSelect necessary helpers."
    hyps = [f"Hypothesis: need '{f['name']}' ({f['desc']})" for f in CODING_FUNCTIONS]
    if backend == "pytorch":
        tok, model = handle
        scores = score_pytorch(tok, model, premise, hyps)
    else:
        scores = score_llama_cpp_prompt(handle, premise, hyps)
    ranked = sorted(
        [(CODING_FUNCTIONS[i]["name"], scores[hyps[i]]) for i in range(len(hyps))],
        key=lambda x: -x[1],
    )
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_email(backend, handle, d1_url, out_lines: list):
    print("\n===== EMAIL CLASSIFIER DEMO =====")
    out_lines.append("EMAIL CLASSIFIER DEMO")
    out_lines.append("=" * 40)
    for ex in EMAIL_EXAMPLES:
        print(f"\nTEXT: {ex['text'][:100]}...")
        out_lines.append(f"\nTEXT: {ex['text'][:100]}...")
        results = classify_email(backend, handle, ex["text"], d1_url)
        top = sorted(results.items(), key=lambda x: -x[1])[:3]
        for lab, p in top:
            line = f"  {lab:30s}  {p:.3f}"
            print(line)
            out_lines.append(line)


def demo_functions(backend, handle, d1_url, out_lines: list):
    print("\n===== FUNCTION RANKING DEMO =====")
    out_lines.append("\nFUNCTION RANKING DEMO")
    out_lines.append("=" * 40)
    for proj in PROJECT_BRIEFS:
        print(f"\nBRIEF: {proj['brief']}")
        out_lines.append(f"\nBRIEF: {proj['brief']}")
        ranked = rank_functions(backend, handle, proj["brief"], top_k=5, d1_url=d1_url)
        print("  Suggested helpers:")
        out_lines.append("  Suggested helpers:")
        for name, score in ranked:
            mark = "Y" if name in proj["needed"] else " "
            line = f"  [{mark}] {name:35s}  {score:.3f}"
            print(line)
            out_lines.append(line)


def interactive(backend, handle, d1_url):
    print("\nInteractive. Commands: email <text> | funcs <brief> | quit")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line.lower() in ("q", "quit", "exit"):
            break
        if line.lower().startswith("email "):
            res = classify_email(backend, handle, line[6:].strip(), d1_url)
            for k, v in sorted(res.items(), key=lambda x: -x[1]):
                print(f"  {k:30s} {v:.3f}")
        elif line.lower().startswith("funcs "):
            ranked = rank_functions(backend, handle, line[6:].strip(), top_k=8, d1_url=d1_url)
            for name, score in ranked:
                print(f"  {name:35s} {score:.3f}")
        else:
            print("Unknown command.")


def main():
    parser = argparse.ArgumentParser(description="OpenJEV + d1-3B classifier")
    parser.add_argument(
        "--backend",
        choices=["auto", "d1", "pytorch", "llama_cpp"],
        default="auto",
        help="auto prefers d1 (LiquidAI decision model) then pytorch then legacy",
    )
    parser.add_argument("--d1-url", default="http://127.0.0.1:8080",
                        help="llama-server base URL for d1")
    parser.add_argument("--no-quant", action="store_true")
    parser.add_argument("--gguf", default=None, help="Legacy instruct GGUF path")
    parser.add_argument("--demo", choices=["email", "funcs", "both", "interactive"], default="both")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    backend = args.backend
    handle = None

    if backend == "auto":
        if check_d1_server(args.d1_url):
            backend = "d1"
            print(f"[auto] d1 server reachable at {args.d1_url}")
        else:
            try:
                import torch
                if torch.cuda.is_available():
                    backend = "pytorch"
                    print("[auto] CUDA available → pytorch")
                else:
                    backend = "d1"  # still prefer telling user to start d1
                    print(
                        "[auto] No d1 server and no CUDA.\n"
                        "Start the recommended backend:\n"
                        "  llama-server -hf LiquidAI/d1-3B-GGUF:Q4_K_M --port 8080\n"
                        "Then re-run. Falling back to checking d1 once more..."
                    )
                    if not check_d1_server(args.d1_url):
                        if args.gguf:
                            backend = "llama_cpp"
                        else:
                            print("No usable backend. Exiting.")
                            sys.exit(1)
            except ImportError:
                if args.gguf:
                    backend = "llama_cpp"
                else:
                    print("Install torch or start d1 server / pass --gguf.")
                    sys.exit(1)

    if backend == "d1":
        if not check_d1_server(args.d1_url):
            print(
                f"d1 server not reachable at {args.d1_url}\n"
                "Start it with:\n"
                "  llama-server -hf LiquidAI/d1-3B-GGUF:Q4_K_M --port 8080\n"
                "(Q4_K_M ≈ 1.67 GB – good default for Mac)"
            )
            sys.exit(1)
        print(f"[backend] LiquidAI d1-3B via {args.d1_url}/v1/systemone")
        handle = None
    elif backend == "pytorch":
        handle = load_pytorch(quantize=not args.no_quant)
    else:
        handle = load_llama_cpp_fallback(args.gguf)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_lines: list[str] = [f"backend={backend}"]

    if args.demo in ("email", "both"):
        demo_email(backend, handle, args.d1_url, out_lines)
    if args.demo in ("funcs", "both"):
        demo_functions(backend, handle, args.d1_url, out_lines)
    if args.demo == "interactive":
        interactive(backend, handle, args.d1_url)

    if len(out_lines) > 1:
        (out_dir / "results.txt").write_text("\n".join(out_lines))
        print(f"\nWrote {out_dir}/results.txt")


if __name__ == "__main__":
    main()
