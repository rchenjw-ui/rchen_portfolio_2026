# chatbot_rag

Local chatbot powered by a small Llama GGUF, with **encrypted chat history** and optional **retrieval-augmented generation (RAG)**.

Encryption matters because chat logs may be exposed to cloud; treating them as plaintext is a bad default.

The idea is to demonstrate encrypted chat history and RAG; this is a preliminary demo 

## Requirements

```bash
pip install llama-cpp-python cryptography numpy
# GGUF models (examples):
#   models/Llama-3.2-3B-Instruct-Q4_K_M.gguf          # chat
#   models/modernbert-embed-base-Q8_0.gguf            # embeddings for RAG
```

## Run the chatbot

```bash
# new chat, default model path (or set $LLAMA_MODEL)
./run_chatbot.sh

# explicit model
python chatbot.py NONE models/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# resume encrypted history
python chatbot.py history.bin models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

You will be prompted for a password. If you forget it, the history cannot be recovered and a new session starts.

## Enable RAG

1. Put `.txt` or `.md` files into `docs/` (next to `chatbot.py`).
2. Start the chatbot as usual. On first run it builds a local vector store (`rag_store/`).
3. Retrieved chunks are injected into the prompt for each user message.

Optional environment variables:

```bash
export RAG_DOCS_DIR=/path/to/my/docs
export RAG_FS_NAME=my_store
```

## RAG CLI

Manage the same store without starting a chat:

```bash
python rag_cli.py index --docs docs --store rag_store
python rag_cli.py search "how does the encryption work?" --top-k 3
python rag_cli.py add docs/extra.md --store rag_store
python rag_cli.py list --store rag_store
python rag_cli.py info --store rag_store
```

Full details: [RAG_CLI.md](RAG_CLI.md).

## Layout

```text
chatbot_rag/
├── chatbot.py          # main encrypted chat loop
├── rag_utils.py        # embed / index / search
├── rag_cli.py          # CLI for the vector store
├── run_chatbot.sh
├── docs/               # drop source documents here
└── rag_store/          # created on first RAG use
```

## Notes

- Core encrypt/decrypt logic is intentionally left close to the original implementation.
- Embeddings use `llama-cpp-python` in embedding mode (default model name: ModernBERT GGUF).
- No external vector database; chunks are hashed (SHA3) text files + a JSON map of vectors.
