Note: This project is entirely internal and unrestricted. 

Chatbot usage guide: 
- If you are familiar with CLI:
    - To run:
        - python chatbot.py                          # new chat, default model path
        - python chatbot.py NONE models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
        - python chatbot.py history.bin models/Llama-3.2-3B-Instruct-Q4_K_M.gguf 
        - Note: By default, the chats are encrypted. Do not forget the password, or it'll just load a new one. 
    - for RAG: 
        - Put .txt / .md files into the docs/ folder (next to this script).
        - RAG activates automatically when documents are present. 
- If not:

RAG useage guide:
- rag_cli.py is based on rag_utils.py
- Index docs into store "rag_store"
    - python rag_cli.py index --docs docs --store rag_store
- Search
    - python rag_cli.py search "how does the encryption work?" --top-k 3
- Add a file to rag_store
    - python rag_cli.py add docs/extra.md --store rag_store
- Inspect files
    - python rag_cli.py list --store rag_store
    - python rag_cli.py info --store rag_store