#!/usr/bin/env python3
"""
RAG CLI — index documents and query a local filesystem vector store.
Supports encrypted storage and retrieval via AES-256.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rag_utils import (
    load_documents_from_dir,
    corpus_to_filesystem,
    search_filesystem,
    get_embedder,
    embed,
    hash_chunk,
    encrypt_text,
)


def _store_paths(store: str) -> tuple[Path, Path]:
    root = Path(store)
    vec_enc = root / f"{root.name}_vector_db.enc"
    return root, vec_enc


def cmd_index(args: argparse.Namespace) -> int:
    docs_dir = Path(args.docs)
    if not docs_dir.is_dir():
        print(f"[error] docs directory not found: {docs_dir}", file=sys.stderr)
        return 1

    docs = load_documents_from_dir(str(docs_dir), extensions=tuple(args.ext))
    if not docs:
        print(f"[error] no documents found in {docs_dir} (extensions: {args.ext})")
        return 1

    print(f"[rag] indexing {len(docs)} document(s) from {docs_dir} → store '{args.store}'")
    try:
        llm = get_embedder(args.model)
    except Exception as e:
        print(f"[error] could not load embedder: {e}", file=sys.stderr)
        return 1

    corpus_to_filesystem(
        docs,
        name=args.store,
        llm=llm,
        overwrite=args.overwrite,
        encryption_key=args.key,
    )
    print("[rag] done.")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    root = Path(args.store)
    vec_enc = root / f"{args.store}_vector_db.enc"
    vec_txt = root / f"{args.store}_vector_db.json"
    
    if not vec_enc.exists() and not vec_txt.exists():
        print(f"[error] no vector DB found for store '{args.store}'. Run 'index' first.", file=sys.stderr)
        return 1

    try:
        llm = get_embedder(args.model)
    except Exception as e:
        print(f"[error] could not load embedder: {e}", file=sys.stderr)
        return 1

    try:
        results = search_filesystem(
            args.store, 
            args.query, 
            top_k=args.top_k, 
            llm=llm, 
            encryption_key=args.key
        )
    except Exception as e:
        print(f"[error] search/decryption failed: {e}", file=sys.stderr)
        return 1

    if not results:
        print("[rag] no results.")
        return 0

    for rank, (text, score) in enumerate(results, 1):
        preview = text.replace("\n", " ").strip()
        if len(preview) > args.preview:
            preview = preview[: args.preview] + "…"
        print(f"--- #{rank}  score={score:.4f} ---")
        print(preview)
        print()
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.is_file():
        print(f"[error] file not found: {path}", file=sys.stderr)
        return 1

    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        print("[error] file is empty", file=sys.stderr)
        return 1

    root = Path(args.store)
    root.mkdir(parents=True, exist_ok=True)

    try:
        llm = get_embedder(args.model)
    except Exception as e:
        print(f"[error] could not load embedder: {e}", file=sys.stderr)
        return 1

    h = hash_chunk(text)
    emb = embed(text, llm=llm)

    vec_enc = root / f"{args.store}_vector_db.enc"
    vec_txt = root / f"{args.store}_vector_db.json"

    # Handle vector db load/decrypt
    vector_db = {}
    if vec_enc.exists():
        if not args.key:
            print(f"[error] store is encrypted. Provide --key.", file=sys.stderr)
            return 1
        raw = vec_enc.read_bytes()
        iv, enc_data = raw.split(b"-8086v1-", 1)
        from rag_utils import decrypt_text
        vector_db = json.loads(decrypt_text(enc_data, args.key, iv))
    elif vec_txt.exists():
        with open(vec_txt, "r", encoding="utf-8") as f:
            vector_db = json.load(f)

    if h in vector_db and not args.overwrite:
        print(f"[rag] chunk already present (hash={h[:16]}…). Use --overwrite to replace.")
        return 0

    vector_db[h] = emb.tolist()
    
    # Save chunk (encrypted if key is provided)
    if args.key:
        enc_data, iv = encrypt_text(text, args.key)
        (root / f"str_{h}.enc").write_bytes(iv + b"-8086v1-" + enc_data)
    else:
        (root / f"str_{h}.txt").write_text(text, encoding="utf-8")

    # Save vector db back
    vec_data = json.dumps(vector_db, indent=2)
    if args.key:
        enc_vec, iv_v = encrypt_text(vec_data, args.key)
        vec_enc.write_bytes(iv_v + b"-8086v1-" + enc_vec)
    else:
        vec_txt.write_text(vec_data, encoding="utf-8")

    print(f"[rag] added 1 chunk → {root}/  (total={len(vector_db)})")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    root = Path(args.store)
    vec_enc = root / f"{args.store}_vector_db.enc"
    vec_txt = root / f"{args.store}_vector_db.json"

    if not vec_enc.exists() and not vec_txt.exists():
        print(f"[error] no vector DB found in {root}", file=sys.stderr)
        return 1

    if vec_enc.exists():
        if not args.key:
            print(f"[error] store is encrypted. Provide --key.", file=sys.stderr)
            return 1
        raw = vec_enc.read_bytes()
        iv, enc_data = raw.split(b"-8086v1-", 1)
        from rag_utils import decrypt_text
        vector_db = json.loads(decrypt_text(enc_data, args.key, iv))
    else:
        with open(vec_txt, "r", encoding="utf-8") as f:
            vector_db = json.load(f)

    print(f"[rag] store '{args.store}' — {len(vector_db)} chunk(s)\n")
    for i, h in enumerate(vector_db.keys(), 1):
        enc_path = root / f"str_{h}.enc"
        txt_path = root / f"str_{h}.txt"
        preview = "(content unavailable)"
        if enc_path.exists() and args.key:
            chunk_bytes = enc_path.read_bytes()
            iv_c, data_c = chunk_bytes.split(b"-8086v1-", 1)
            from rag_utils import decrypt_text
            preview = decrypt_text(data_c, args.key, iv_c)
        elif txt_path.exists():
            preview = txt_path.read_text(encoding="utf-8", errors="ignore")
            
        preview = preview.replace("\n", " ").strip()
        if len(preview) > 120:
            preview = preview[:120] + "…"
        print(f"  {i:3d}. {h[:20]}…  {preview}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    root = Path(args.store)
    vec_enc = root / f"{args.store}_vector_db.enc"
    vec_txt = root / f"{args.store}_vector_db.json"
    exists = vec_enc.exists() or vec_txt.exists()
    
    print(f"store dir : {root.resolve()}")
    print(f"vector db : exists={exists} (encrypted: {vec_enc.exists()})")
    print(f"embed model default: modernbert-embed-base-Q8_0.gguf")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag_cli",
        description="Local secure RAG utilities with encryption support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default=None, help="Path to GGUF embedding model")
    
    sub = p.add_subparsers(dest="command", required=True)

    # index
    pi = sub.add_parser("index", help="Embed all docs into a store")
    pi.add_argument("--docs", default="docs", help="Directory of docs")
    pi.add_argument("--store", default="rag_store", help="Store name")
    pi.add_argument("--ext", nargs="+", default=[".txt", ".md"], help="Extensions")
    pi.add_argument("--overwrite", action="store_true", help="Rebuild store")
    pi.add_argument("--key", default=None, help="Encryption passphrase")
    pi.set_defaults(func=cmd_index)

    # search
    ps = sub.add_parser("search", help="Query the store")
    ps.add_argument("query", help="Search query text")
    ps.add_argument("--store", default="rag_store", help="Store name")
    ps.add_argument("--top-k", type=int, default=3, help="Top results")
    ps.add_argument("--preview", type=int, default=300, help="Max preview chars")
    ps.add_argument("--key", default=None, help="Encryption passphrase")
    ps.set_defaults(func=cmd_search)

    # add
    pa = sub.add_parser("add", help="Add a single file")
    pa.add_argument("file", help="Path to text file")
    pa.add_argument("--store", default="rag_store", help="Store name")
    pa.add_argument("--overwrite", action="store_true", help="Replace hash if exists")
    pa.add_argument("--key", default=None, help="Encryption passphrase")
    pa.set_defaults(func=cmd_add)

    # list
    pl = sub.add_parser("list", help="List chunks")
    pl.add_argument("--store", default="rag_store", help="Store name")
    pl.add_argument("--key", default=None, help="Encryption passphrase")
    pl.set_defaults(func=cmd_list)

    # info
    pn = sub.add_parser("info", help="Show store stats")
    pn.add_argument("--store", default="rag_store", help="Store name")
    pn.set_defaults(func=cmd_info)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())