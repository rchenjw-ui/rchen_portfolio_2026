"""
RAG utilities for the local chatbot.
- Embeds documents with modernbert-embed-base (llama-cpp)
- Stores on filesystem with SHA3 hashes
- Simple cosine retrieval
"""

import os
import json
import hashlib
import base64
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def SHA3_256(text: str) -> bytes:
    h = hashlib.sha3_256()
    h.update(text.encode("utf-8"))
    return h.digest()


def byte_to_str(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")


def hash_chunk(text: str) -> str:
    """Filesystem-safe hash (no /)."""
    hashed = byte_to_str(SHA3_256(text))
    return hashed.replace("/", "|")


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

_DEFAULT_EMBED_MODEL = "models/modernbert-embed-base-Q8_0.gguf"


def get_embedder(model_path: Optional[str] = None) -> "Llama":
    if Llama is None:
        raise ImportError("llama-cpp-python is required for embeddings")
    path = model_path or _DEFAULT_EMBED_MODEL
    return Llama(model_path=path, embedding=True, verbose=False)


def embed(text: str, llm: Optional["Llama"] = None) -> np.ndarray:
    if llm is None:
        llm = get_embedder()
    emb = llm.create_embedding(text)
    return np.array(emb["data"][0]["embedding"], dtype=np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def compute_cosine_similarities(vector: np.ndarray, target_vectors: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    targets = np.asarray(target_vectors, dtype=np.float32)
    dots = np.dot(targets, v)
    v_norm = np.linalg.norm(v)
    t_norms = np.linalg.norm(targets, axis=1)
    return dots / (v_norm * t_norms + 1e-8)


# ---------------------------------------------------------------------------
# Database / filesystem
# ---------------------------------------------------------------------------

def database_gen(corpus: List[str], llm: Optional["Llama"] = None) -> Tuple[Dict, Dict]:
    if llm is None:
        llm = get_embedder()
    vector_db: Dict[str, list] = {}
    str_db: Dict[str, str] = {}
    for text in corpus:
        h = hash_chunk(text)
        emb = embed(text, llm=llm)
        vector_db[h] = emb.tolist()
        str_db[h] = text
    return vector_db, str_db


def corpus_to_filesystem(
    corpus: List[str],
    name: str = "default",
    llm: Optional["Llama"] = None,
    overwrite: bool = False,
) -> Tuple[Dict, Dict]:
    """Write a corpus to a directory of hashed text files + vector JSON."""
    if llm is None:
        llm = get_embedder()
    vector_db, str_db = database_gen(corpus, llm)

    root = Path(name)
    if root.exists() and not overwrite:
        print(f"[RAG] Directory '{name}' already exists – skipping write.")
        return vector_db, str_db

    root.mkdir(parents=True, exist_ok=True)
    for h, text in str_db.items():
        (root / f"str_{h}.txt").write_text(text, encoding="utf-8")

    with open(root / f"{name}_vector_db.json", "w", encoding="utf-8") as f:
        json.dump(vector_db, f, indent=2)

    print(f"[RAG] Wrote {len(str_db)} chunks → {root}/")
    return vector_db, str_db


def search_filesystem(
    filesystem: str,
    query: str,
    top_k: int = 3,
    llm: Optional["Llama"] = None,
) -> List[Tuple[str, float]]:
    """Return list of (text, score) sorted by descending similarity."""
    if llm is None:
        llm = get_embedder()

    root = Path(filesystem)
    vec_path = root / f"{filesystem}_vector_db.json"
    if not vec_path.exists():
        raise FileNotFoundError(f"No vector DB at {vec_path}")

    with open(vec_path, "r", encoding="utf-8") as f:
        vector_db = json.load(f)

    keys = list(vector_db.keys())
    vectors = np.array([vector_db[k] for k in keys], dtype=np.float32)
    q_vec = embed(query, llm=llm)
    sims = compute_cosine_similarities(q_vec, vectors)

    # top-k
    idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in idx:
        h = keys[i]
        text = (root / f"str_{h}.txt").read_text(encoding="utf-8")
        results.append((text, float(sims[i])))
    return results


def load_documents_from_dir(directory: str, extensions: Tuple[str, ...] = (".txt", ".md")) -> List[str]:
    """Simple document loader – one chunk per file for now."""
    docs = []
    for p in Path(directory).rglob("*"):
        if p.suffix.lower() in extensions and p.is_file():
            try:
                docs.append(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception as e:
                print(f"[RAG] Could not read {p}: {e}")
    return docs


def help_rag():
    print("Available RAG helpers:")
    print("  corpus_to_filesystem(corpus, name='default')")
    print("  search_filesystem(filesystem, query, top_k=3)")
    print("  load_documents_from_dir(directory)")
    print("  get_embedder() / embed(text)")
