"""
Synthetic examples for the expanded OpenJEV multi-label classifier
and for the coding-function ranking demo.
"""

# ---------------------------------------------------------------------------
# Email / message examples (for the original 6-class head)
# ---------------------------------------------------------------------------

EMAIL_EXAMPLES = [
    {
        "text": "URGENT: Your account has been locked! Click http://bit.ly/totally-legit to verify within 2 hours or lose access forever!!!",
        "labels": {"is_spam": 0.95, "is_promo": 0.1, "is_legitimate_call_to_action": 0.05},
    },
    {
        "text": "Hi team, can we schedule a 30-min sync tomorrow to review the Q3 roadmap? Let me know what times work.",
        "labels": {"is_work_related": 0.9, "is_legitimate_call_to_action": 0.7},
    },
    {
        "text": "New paper on arXiv: 'Efficient Sparse Attention for Long-Context LLMs' – interesting results on the 128k benchmark.",
        "labels": {"is_science_related": 0.92},
    },
    {
        "text": "Reminder: Fall registration opens Monday. Don't forget to submit your course preferences by Friday.",
        "labels": {"is_college_related": 0.9, "is_legitimate_call_to_action": 0.6},
    },
    {
        "text": "20% off all cloud GPUs this weekend only – use code FASTTRAIN at checkout.",
        "labels": {"is_promo": 0.85, "is_spam": 0.2},
    },
    {
        "text": "Can you review the PR for the new embedding cache? It should cut latency by ~30%.",
        "labels": {"is_work_related": 0.88, "is_legitimate_call_to_action": 0.75},
    },
]

# ---------------------------------------------------------------------------
# Coding-project function ranking examples
# ---------------------------------------------------------------------------

# Each item is a "project brief". The model should rank which helper
# functions are most needed.

CODING_FUNCTIONS = [
    {"name": "load_and_clean_csv", "desc": "Read a CSV, drop nulls, coerce types, return DataFrame"},
    {"name": "train_test_split_stratified", "desc": "Stratified train/test split preserving class ratios"},
    {"name": "compute_roc_pr_auc", "desc": "Compute ROC-AUC and PR-AUC given y_true and scores"},
    {"name": "plot_confusion_matrix", "desc": "Draw and save a confusion matrix heatmap"},
    {"name": "embed_texts_with_sentence_transformer", "desc": "Encode a list of texts into dense vectors"},
    {"name": "cosine_topk_search", "desc": "Given a query vector and a matrix, return top-k nearest"},
    {"name": "encrypt_aes_cbc", "desc": "AES-CBC encrypt a string with a password-derived key"},
    {"name": "decrypt_aes_cbc", "desc": "AES-CBC decrypt ciphertext given key and IV"},
    {"name": "build_morgan_fingerprint", "desc": "Compute ECFP/Morgan fingerprint from SMILES"},
    {"name": "butina_cluster_split", "desc": "Cluster molecules with Butina and produce train/test split"},
    {"name": "train_xgboost_classifier", "desc": "Fit an XGBClassifier with early stopping and scale_pos_weight"},
    {"name": "apply_pca", "desc": "Fit PCA on train, transform train and test to n components"},
    {"name": "oversample_minority_class", "desc": "Random oversample the minority class to balance labels"},
    {"name": "rbm_cd_k_update", "desc": "Perform one CD-k step and return gradients for an RBM"},
    {"name": "stack_rbm_layers", "desc": "Greedy layer-wise training of a deep RBM / DBN"},
    {"name": "fine_tune_mlp_from_rbm", "desc": "Initialise an MLP from RBM weights and fine-tune with CE loss"},
    {"name": "render_reconstruction_grid", "desc": "Save a grid of original vs reconstructed images"},
    {"name": "load_gguf_llama", "desc": "Load a GGUF model with llama-cpp for local inference"},
    {"name": "rag_retrieve_topk", "desc": "Embed query and return top-k passages from a vector store"},
    {"name": "multi_label_nli_classify", "desc": "Score multiple hypotheses against a premise with an NLI model"},
]

PROJECT_BRIEFS = [
    {
        "brief": "I need a local encrypted chatbot that can answer questions about a folder of research PDFs.",
        "needed": [
            "load_gguf_llama",
            "rag_retrieve_topk",
            "encrypt_aes_cbc",
            "decrypt_aes_cbc",
            "embed_texts_with_sentence_transformer",
            "cosine_topk_search",
        ],
    },
    {
        "brief": "Build a Tox21 activity predictor that handles class imbalance and reports PR-AUC.",
        "needed": [
            "build_morgan_fingerprint",
            "butina_cluster_split",
            "oversample_minority_class",
            "train_xgboost_classifier",
            "compute_roc_pr_auc",
            "apply_pca",
        ],
    },
    {
        "brief": "Compare a randomly-initialised MLP against an RBM-pretrained stack on MNIST, with sparsity.",
        "needed": [
            "rbm_cd_k_update",
            "stack_rbm_layers",
            "fine_tune_mlp_from_rbm",
            "render_reconstruction_grid",
        ],
    },
    {
        "brief": "Make a multi-label email classifier that also ranks the most useful coding helpers for a new project.",
        "needed": [
            "multi_label_nli_classify",
            "load_and_clean_csv",
            "train_test_split_stratified",
            "plot_confusion_matrix",
        ],
    },
]


def get_function_catalog_text() -> str:
    lines = ["Available helper functions:"]
    for f in CODING_FUNCTIONS:
        lines.append(f"- {f['name']}: {f['desc']}")
    return "\n".join(lines)
