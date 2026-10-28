#!/usr/bin/env python3
"""
RAG CLI — index documents and query a local filesystem vector store.

Built on rag_utils.py (ModernBERT embeddings via llama-cpp, SHA3 keys, cosine retrieval).

Examples
--------
  # Index everything under docs/ into store "rag_store"
  python rag_cli.py index --docs docs --store rag_store

  # Search
  python rag_cli.py search "how does encryption work?" --store rag_store --top-k 3

  # Add a single file without rebuilding the whole corpus
  python rag_cli.py add path/to/notes.md --store rag_store

  # List indexed chunks
  python rag_cli.py list --store rag_store

  # Show help for a subcommand
  python rag_cli.py index --help
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

# Local import
from rag_utils import (
    load_documents_from_dir,
    corpus_to_filesystem,
    search_filesystem,
    get_embedder,
    embed,
    hash_chunk,
    cosine_sim,
)


def _store_paths(store: str) -> tuple[Path, Path]:
    root = Path(store)
    vec = root / f"{root.name}_vector_db.json"
    return root, vec


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
        print("        Install llama-cpp-python and provide a GGUF embedding model.", file=sys.stderr)
        return 1

    corpus_to_filesystem(
        docs,
        name=args.store,
        llm=llm,
        overwrite=args.overwrite,
    )
    print("[rag] done.")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    root, vec = _store_paths(args.store)
    if not vec.exists():
        print(f"[error] no vector DB at {vec}. Run 'index' first.", file=sys.stderr)
        return 1

    try:
        llm = get_embedder(args.model)
    except Exception as e:
        print(f"[error] could not load embedder: {e}", file=sys.stderr)
        return 1

    results = search_filesystem(args.store, args.query, top_k=args.top_k, llm=llm)
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

    root, vec = _store_paths(args.store)
    root.mkdir(parents=True, exist_ok=True)

    try:
        llm = get_embedder(args.model)
    except Exception as e:
        print(f"[error] could not load embedder: {e}", file=sys.stderr)
        return 1

    h = hash_chunk(text)
    emb = embed(text, llm=llm)

    # Load or create vector DB
    if vec.exists():
        with open(vec, "r", encoding="utf-8") as f:
            vector_db = json.load(f)
    else:
        vector_db = {}

    if h in vector_db and not args.overwrite:
        print(f"[rag] chunk already present (hash={h[:16]}…). Use --overwrite to replace.")
        return 0

    vector_db[h] = emb.tolist()
    (root / f"str_{h}.txt").write_text(text, encoding="utf-8")
    with open(vec, "w", encoding="utf-8") as f:
        json.dump(vector_db, f, indent=2)

    print(f"[rag] added 1 chunk → {root}/  (total={len(vector_db)})")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    root, vec = _store_paths(args.store)
    if not vec.exists():
        print(f"[error] no vector DB at {vec}", file=sys.stderr)
        return 1

    with open(vec, "r", encoding="utf-8") as f:
        vector_db = json.load(f)

    print(f"[rag] store '{args.store}' — {len(vector_db)} chunk(s)\n")
    for i, h in enumerate(vector_db.keys(), 1):
        text_path = root / f"str_{h}.txt"
        if text_path.exists():
            preview = text_path.read_text(encoding="utf-8", errors="ignore")
            preview = preview.replace("\n", " ").strip()
            if len(preview) > 120:
                preview = preview[:120] + "…"
        else:
            preview = "(missing text file)"
        print(f"  {i:3d}. {h[:20]}…  {preview}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    root, vec = _store_paths(args.store)
    print(f"store dir : {root.resolve()}")
    print(f"vector db : {vec.resolve()}  exists={vec.exists()}")
    if vec.exists():
        with open(vec, "r", encoding="utf-8") as f:
            n = len(json.load(f))
        print(f"chunks    : {n}")
    print(f"embed model default: modernbert-embed-base-Q8_0.gguf")
    print(f"override with --model /path/to/model.gguf")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag_cli",
        description="Local RAG utilities: index documents and query a filesystem vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model",
        default=None,
        help="Path to GGUF embedding model (default: modernbert-embed-base-Q8_0.gguf)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # index
    pi = sub.add_parser("index", help="Embed all docs from a directory into a store")
    pi.add_argument("--docs", default="docs", help="Directory of .txt/.md files (default: docs)")
    pi.add_argument("--store", default="rag_store", help="Store name / directory (default: rag_store)")
    pi.add_argument("--ext", nargs="+", default=[".txt", ".md"], help="File extensions to include")
    pi.add_argument("--overwrite", action="store_true", help="Rebuild store even if it exists")
    pi.set_defaults(func=cmd_index)

    # search
    ps = sub.add_parser("search", help="Query the store")
    ps.add_argument("query", help="Search query text")
    ps.add_argument("--store", default="rag_store", help="Store name (default: rag_store)")
    ps.add_argument("--top-k", type=int, default=3, help="Number of results (default: 3)")
    ps.add_argument("--preview", type=int, default=300, help="Max chars to print per result")
    ps.set_defaults(func=cmd_search)

    # add
    pa = sub.add_parser("add", help="Add a single file to an existing (or new) store")
    pa.add_argument("file", help="Path to a text file")
    pa.add_argument("--store", default="rag_store", help="Store name (default: rag_store)")
    pa.add_argument("--overwrite", action="store_true", help="Replace if hash already exists")
    pa.set_defaults(func=cmd_add)

    # list
    pl = sub.add_parser("list", help="List chunks in a store")
    pl.add_argument("--store", default="rag_store", help="Store name (default: rag_store)")
    pl.set_defaults(func=cmd_list)

    # info
    pn = sub.add_parser("info", help="Show store paths and basic stats")
    pn.add_argument("--store", default="rag_store", help="Store name (default: rag_store)")
    pn.set_defaults(func=cmd_info)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
