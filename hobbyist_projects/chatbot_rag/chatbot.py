#!/usr/bin/env python3
"""
Local encrypted chatbot with optional encrypted RAG retrieval.
"""

from __future__ import annotations

import hashlib
import os
import sys
import datetime
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding


def encrypt(data: str, key: str):
    data_bytes = data.encode("utf-8")
    key_bytes = hashlib.sha256(key.encode()).digest()
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv))
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(data_bytes) + padder.finalize()
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    return encrypted_data, iv


def decrypt(encrypted_data: bytes, key: str, iv: bytes) -> str:
    key_bytes = hashlib.sha256(key.encode()).digest()
    cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv))
    decryptor = cipher.decryptor()
    decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
    unpad = padding.PKCS7(128).unpadder()
    decrypted_data = unpad.update(decrypted_data) + unpad.finalize()
    return decrypted_data.decode("utf-8")


def try_load_rag():
    """Return a secure retrieval callable or None."""
    try:
        from rag_utils import (
            load_documents_from_dir,
            corpus_to_filesystem,
            search_filesystem,
            get_embedder,
        )
    except ImportError:
        print("[RAG] rag_utils not found – running without retrieval.")
        return None

    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path(os.environ.get("RAG_DOCS_DIR", "")),
        script_dir / "docs",
        Path("docs"),
    ]
    docs_dir = next((p for p in candidates if p and p.is_dir()), None)
    if docs_dir is None:
        return None

    docs = load_documents_from_dir(str(docs_dir))
    if not docs:
        return None

    fs_name = os.environ.get("RAG_FS_NAME", "rag_store")
    fs_path = script_dir / fs_name
    
    # Prompt user for secure store password if they want RAG encryption enabled
    enc_key = os.environ.get("RAG_ENCRYPTION_KEY", None)
    vec_enc = fs_path / f"{fs_name}_vector_db.enc"
    vec_txt = fs_path / f"{fs_name}_vector_db.json"

    if not vec_enc.exists() and not vec_txt.exists():
        use_encrypt = input("[RAG] Encrypt this new vector store? (y/N): ").strip().lower() == 'y'
        if use_encrypt:
            enc_key = input("[RAG] Enter RAG store encryption passphrase: ").strip()
        print("[RAG] Building secure index...")
        corpus_to_filesystem(docs, name=str(fs_path), overwrite=False, encryption_key=enc_key)
    elif vec_enc.exists() and not enc_key:
        enc_key = input("[RAG] Store is encrypted. Enter decryption passphrase: ").strip()

    try:
        embedder = get_embedder()
    except Exception as e:
        print(f"[RAG] Init failed: {e}")
        return None

    def retrieve(query: str, top_k: int = 2) -> str:
        try:
            hits = search_filesystem(str(fs_path), query, top_k=top_k, llm=embedder, encryption_key=enc_key)
            if not hits:
                return ""
            context = "\n---\n".join(f"[score={s:.3f}] {t[:500]}" for t, s in hits)
            return f"\n\n[RETRIEVED CONTEXT]\n{context}\n[END CONTEXT]\n"
        except Exception as e:
            print(f"[RAG] Retrieval decryption error: {e}")
            return ""

    print("[RAG] Secure retrieval ready.")
    return retrieve


SYSTEM_PROMPT = """SYSTEM:
[You are a helpful secure assistant.]
"""


def main():
    from llama_cpp import Llama

    path = None
    llm_name = os.environ.get(
        "LLAMA_MODEL",
        "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    )

    if len(sys.argv) > 1 and sys.argv[1] != "NONE":
        path = sys.argv[1]
    if len(sys.argv) > 2:
        llm_name = sys.argv[2]

    prompt = SYSTEM_PROMPT

    if path:
        try:
            with open(path, "rb") as f:
                raw = f.read()
            iv, data = raw.split(b"-8086v1-", 1)
            key = input("Chat history encryption key: ")
            prompt = decrypt(data, key, iv)
            print("History loaded.")
        except Exception as e:
            print(f"Could not load history ({e}). Starting fresh.")
            path = None
            prompt = SYSTEM_PROMPT
    else:
        print("Starting new secure conversation.")

    retrieve = try_load_rag()

    if not Path(llm_name).exists():
        print(f"\n[!] Model not found: {llm_name}")
        sys.exit(1)

    print(f"Loading model: {llm_name}")
    model = Llama(model_path=llm_name, n_ctx=8192, verbose=False)

    print("\nChat ready. Type END to save & quit.\n")

    while True:
        user_text = input("USER: ").strip()
        if user_text == "END":
            break

        rag_ctx = retrieve(user_text) if retrieve else ""
        prompt = prompt + "USER: " + user_text + rag_ctx + "\nASSISTANT: "

        if len(prompt) < 16000:
            output = model.create_completion(
                prompt, max_tokens=784, stop=["USER:"], echo=False
            )
        else:
            output = model.create_completion(
                SYSTEM_PROMPT + prompt[-15800:],
                max_tokens=784,
                stop=["USER:"],
                echo=False,
            )

        reply = output["choices"][0]["text"]
        prompt = prompt + reply + "\nUSER: "
        print(f"\nASSISTANT: {reply}\n")

    key = input("Encryption key to save history: ")
    out_path = path or f"chat_history_{datetime.date.today()}.bin"
    encrypted_data, iv = encrypt(prompt, key)
    with open(out_path, "wb") as f:
        f.write(iv + b"-8086v1-" + encrypted_data)
    print(f"History saved → {out_path}")


if __name__ == "__main__":
    main()