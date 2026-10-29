# rchen_portfolio_2026
MIT Maker Portfolio (code)

Files all located in `hobbyist_projects/` , added via copying (with necessary edits to make code runnable) from my previous and present work, either in Google Colab notebooks, my local machine, or my hobby and for-work GitHub code repositories (where you can see the checkin history)

Folders:


## Folders

- **CHLA** — Retinoblastoma tumor-fraction estimation (bioinformatics).  
  My past and current research work at CHLA. Uses DNA methylation in cell-free DNA from aqueous humor to estimate tumor fraction. Logistic-regression classifier over selected CpG probes. Public/synthetic demo data only; restricted primary data are not included.  
  → [chla/README.md](chla/README.md)

- **rbm_mnist** —  Demonstrates two deep-learning algorithms: Backpropagation and Deep Belief Network / Restricted Boltzmann Machine.  
  Both train an equivalently sized network on MNIST. Because DBNs are trained layer-by-layer, a small probe model is also trained on each layer to measure how much label signal the unsupervised features capture. Modern Adam MLPs often win when labels are plentiful; that result is documented honestly.  
  → [rbm_mnist/README.md](rbm_mnist/README.md)

- **homework** — Harvard CS50 AI problem sets.  
  Search, knowledge-based agents, constraint satisfaction, Q-learning, probabilistic reasoning, intro deep learning. Also an early challenge in data structures/algorithms. All course unit tests passed.  
  → [homework/README.md](homework/README.md)

- **vae_demo** — Variational autoencoder on CIFAR-10.  
  Chronologically the first project: my first time implementing an ML algorithm end-to-end (conv encoder/decoder, reparameterization, KL + reconstruction).  
  → [vae_demo/README.md](vae_demo/README.md)

- **cheminformatics** — Featurization and split comparison for a Tox21 activity classifier.  
  Structure-aware splits (scaffold / Butina) produced significantly worse results than a pure random split. Implication: generalization to novel structures is not always testable via classical random train/test splits.  
  → [cheminformatics/README.md](cheminformatics/README.md)

- **chatbot_rag** — Local chatbot (Llama 3.2 3B) with **encrypted** chat history and retrieval-augmented generation.  
  Encryption matters: today the store is local; tomorrow it could sit next to a synced drive. Drop `.txt`/`.md` into `docs/` to enable RAG; manage the store with `rag_cli.py`.  
  → [chatbot_rag/README.md](chatbot_rag/README.md)

- **jev_classifier** — Open-source JEV-style / “System One” models as a multi-label classification and retrieval system (email labels + top-k coding-function ranking).  
  Preferred path on Mac/CPU: LiquidAI d1-3B GGUF via `llama-server`; PyTorch and prompt-based llama.cpp as fallbacks.  
  → [jev_classifier/README.md](jev_classifier/README.md)

## Notes

- **chatbot_rag** needs two GGUF files: one chat model and one embedding model for retrieval.  
  Examples used here: Llama 3.2 3B Instruct and ModernBERT-Embed; any compatible GGUFs work. More may be added later. 

- Each subproject has its own README with requirements, how to run, and layout. Prefer `./run.sh` (bash/zsh) where present; plots and metrics are written under `results/` so a display is not required.

- macOS (Apple Silicon) and Linux are both supported; GPU is optional except where noted.
