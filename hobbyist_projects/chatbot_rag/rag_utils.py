"""
RAG utilities for the local chatbot.
- Embeds documents with modernbert-embed-base (llama-cpp)
- Stores on filesystem with SHA3 hashes
- Simple cosine retrieval
- Supports AES-256 encryption at rest and decryption during retrieval.
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

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding


# ---------------------------------------------------------------------------
# Hashing & Encryption helpers
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


def encrypt_text(text: str, key: str) -> tuple[bytes, bytes]:
    """Encrypt plain text using AES-256-CBC with a SHA-256 derived key."""
    data_bytes = text.encode("utf-8")
    key_bytes = hashlib.sha256(key.encode()).digest()
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv))
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(data_bytes) + padder.finalize()
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    return encrypted_data, iv


def decrypt_text(encrypted_data: bytes, key: str, iv: bytes) -> str:
    """Decrypt ciphertext bytes back to plain text."""
    key_bytes = hashlib.sha256(key.encode()).digest()
    cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv))
    decryptor = cipher.decryptor()
    decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
    unpad = padding.PKCS7(128).unpadder()
    decrypted_data = unpad.update(decrypted_data) + unpad.finalize()
    return decrypted_data.decode("utf-8")


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

_DEFAULT_EMBED_MODEL = "models/modernbert-embed-base-Q8_0.gguf"


def get_embedder(model_path: Optional[str] = None):
    if Llama is None:
        raise ImportError("llama-cpp-python is required for embeddings")
    path = model_path or _DEFAULT_EMBED_MODEL
    return Llama(model_path=path, embedding=True, verbose=False)


def embed(text: str, llm: Optional["Llama"]= None):
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
    encryption_key: Optional[str] = None,
) -> Tuple[Dict, Dict]:
    """Write a corpus to a directory of hashed files + vector DB, optionally encrypted."""
    if llm is None:
        llm = get_embedder()
    vector_db, str_db = database_gen(corpus, llm)

    root = Path(name)
    if root.exists() and not overwrite:
        print(f"[RAG] Directory '{name}' already exists – skipping write.")
        return vector_db, str_db

    root.mkdir(parents=True, exist_ok=True)
    for h, text in str_db.items():
        file_path = root / f"str_{h}.enc" if encryption_key else root / f"str_{h}.txt"
        if encryption_key:
            enc_data, iv = encrypt_text(text, encryption_key)
            file_path.write_bytes(iv + b"-8086v1-" + enc_data)
        else:
            file_path.write_text(text, encoding="utf-8")

    vec_path = root / f"{name}_vector_db.enc" if encryption_key else root / f"{name}_vector_db.json"
    vec_data = json.dumps(vector_db, indent=2)
    if encryption_key:
        enc_vec, iv = encrypt_text(vec_data, encryption_key)
        vec_path.write_bytes(iv + b"-8086v1-" + enc_vec)
    else:
        vec_path.write_text(vec_data, encoding="utf-8")

    print(f"[RAG] Wrote {len(str_db)} chunks → {root}/")
    return vector_db, str_db


def search_filesystem(
    filesystem: str,
    query: str,
    top_k: int = 3,
    llm: Optional["Llama"] = None,
    encryption_key: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """Return list of (text, score) sorted by descending similarity, supporting encrypted stores."""
    if llm is None:
        llm = get_embedder()

    root = Path(filesystem)
    vec_path_enc = root / f"{filesystem}_vector_db.enc"
    vec_path_txt = root / f"{filesystem}_vector_db.json"

    if vec_path_enc.exists():
        if not encryption_key:
            raise ValueError(f"Store '{filesystem}' is encrypted. An encryption key (--key) is required.")
        raw_bytes = vec_path_enc.read_bytes()
        iv, enc_data = raw_bytes.split(b"-8086v1-", 1)
        json_str = decrypt_text(enc_data, encryption_key, iv)
        vector_db = json.loads(json_str)
    elif vec_path_txt.exists():
        with open(vec_path_txt, "r", encoding="utf-8") as f:
            vector_db = json.load(f)
    else:
        raise FileNotFoundError(f"No vector DB found at {root}")

    keys = list(vector_db.keys())
    vectors = np.array([vector_db[k] for k in keys], dtype=np.float32)
    q_vec = embed(query, llm=llm)
    sims = compute_cosine_similarities(q_vec, vectors)

    idx = np.argsort(sims)[::-1][:top_k]
    results = []
    
    for i in idx:
        h = keys[i]
        enc_chunk_path = root / f"str_{h}.enc"
        txt_chunk_path = root / f"str_{h}.txt"
        
        if enc_chunk_path.exists():
            if not encryption_key:
                raise ValueError(f"Chunk files are encrypted. Encryption key required.")
            chunk_bytes = enc_chunk_path.read_bytes()
            iv_c, data_c = chunk_bytes.split(b"-8086v1-", 1)
            text = decrypt_text(data_c, encryption_key, iv_c)
        elif txt_chunk_path.exists():
            text = txt_chunk_path.read_text(encoding="utf-8")
        else:
            text = "(missing text file)"
            
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