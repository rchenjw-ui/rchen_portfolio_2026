#!/usr/bin/env python3
"""
Tox21 NR-AR pipeline:
- Random / Scaffold / Butina splits
- ECFP + RDKit descriptors (+ optional MolDeBERTa)
- PCA (8 dims)
- Class imbalance handling (oversample + scale_pos_weight)
- PR-AUC + ROC-AUC
- Plots
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
import xgboost as xgb

from data import load_tox21_target, class_stats, oversample_minority, scale_features
from features import morgan_fingerprint, rdkit_descriptors, moldeberta_embeddings
from splits import random_train_test_split, scaffold_train_test_split, butina_train_test_split


# ---------------------------------------------------------------------------
# Feature matrices
# ---------------------------------------------------------------------------

def build_feature_matrix(
    smiles: List[str],
    method: str,
    moldeberta_cache: Dict[str, np.ndarray] | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """Returns (X, kept_smiles)."""
    X, kept = [], []
    for s in smiles:
        if method == "ecfp":
            vec = morgan_fingerprint(s)
        elif method == "rdkit":
            vec = rdkit_descriptors(s)
        elif method == "moldeberta":
            if moldeberta_cache is None or s not in moldeberta_cache:
                continue
            vec = moldeberta_cache[s]
        else:
            raise ValueError(method)
        if vec is not None:
            X.append(vec)
            kept.append(s)
    return np.asarray(X, dtype=np.float32), kept


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def run_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_scale_pos: bool = True,
    seed: int = 42,
) -> Tuple[float, float, np.ndarray]:
    """Returns ROC-AUC, PR-AUC, predicted probabilities."""
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    spw = (neg / pos) if (use_scale_pos and pos > 0) else 1.0

    clf = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=4,
        use_label_encoder=False,
    )
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, probs)
    pr = average_precision_score(y_test, probs)
    return roc, pr, probs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray, float, float]],
    out_dir: Path,
    title_prefix: str,
):
    """results[name] = (y_true, probs, roc, pr)"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # ROC
    plt.figure(figsize=(8, 6))
    for name, (y, p, roc, pr) in results.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{title_prefix} – ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{title_prefix}_roc.png", dpi=120)
    plt.close()

    # PR
    plt.figure(figsize=(8, 6))
    for name, (y, p, roc, pr) in results.items():
        prec, rec, _ = precision_recall_curve(y, p)
        plt.plot(rec, prec, label=f"{name} (AP={pr:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix} – Precision-Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{title_prefix}_pr.png", dpi=120)
    plt.close()


def plot_label_dist(y: np.ndarray, out_path: Path, title: str):
    plt.figure(figsize=(5, 3))
    counts = [ (y == 0).sum(), (y == 1).sum() ]
    plt.bar(["Negative", "Positive"], counts, color=["#4C72B0", "#C44E52"])
    plt.title(title)
    for i, c in enumerate(counts):
        plt.text(i, c + max(counts)*0.01, str(c), ha="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="NR-AR")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--use-moldeberta", action="store_true")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Tox21 / {args.target} ...")
    label_map = load_tox21_target(args.target)
    all_smiles = list(label_map.keys())
    print(f"Total valid molecules: {len(all_smiles)}")
    print("Overall label distribution:", class_stats(list(label_map.values())))

    # Optional MolDeBERTa pre-compute
    moldeberta_cache = None
    if args.use_moldeberta:
        print("Computing MolDeBERTa embeddings (this can take a while)...")
        moldeberta_cache = moldeberta_embeddings(all_smiles)

    splitters = {
        "random": random_train_test_split,
        "scaffold": scaffold_train_test_split,
        "butina": butina_train_test_split,
    }

    methods = ["ecfp", "rdkit"]
    if args.use_moldeberta:
        methods.append("moldeberta")

    summary_rows = []

    for split_name, split_fn in splitters.items():
        print(f"\n===== Split: {split_name} =====")
        train_smi, test_smi = split_fn(all_smiles, test_size=args.test_size)
        y_train_raw = np.array([label_map[s] for s in train_smi])
        y_test_raw = np.array([label_map[s] for s in test_smi])
        print(f"Train: {class_stats(y_train_raw.tolist())}")
        print(f"Test : {class_stats(y_test_raw.tolist())}")

        plot_label_dist(
            y_train_raw,
            out_dir / f"label_dist_{split_name}_train.png",
            f"{split_name} train labels",
        )

        results_for_plot = {}

        for method in methods:
            print(f"  --- Featurizer: {method} ---")
            X_tr, kept_tr = build_feature_matrix(train_smi, method, moldeberta_cache)
            X_te, kept_te = build_feature_matrix(test_smi, method, moldeberta_cache)

            y_tr = np.array([label_map[s] for s in kept_tr])
            y_te = np.array([label_map[s] for s in kept_te])

            # Oversample minority on train
            X_tr_bal, y_tr_bal = oversample_minority(X_tr, y_tr)
            X_tr_s, X_te_s = scale_features(X_tr_bal, X_te)

            # --- No PCA ---
            roc, pr, probs = run_xgb(X_tr_s, y_tr_bal, X_te_s, y_te)
            print(f"    no-PCA   ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}")
            results_for_plot[f"{method}_raw"] = (y_te, probs, roc, pr)
            summary_rows.append(
                dict(split=split_name, method=method, pca=False, roc=roc, pr=pr)
            )

            # --- PCA 8 dims ---
            n_comp = min(8, X_tr_s.shape[1], X_tr_s.shape[0] - 1)
            pca = PCA(n_components=n_comp, random_state=42)
            X_tr_p = pca.fit_transform(X_tr_s)
            X_te_p = pca.transform(X_te_s)
            roc_p, pr_p, probs_p = run_xgb(X_tr_p, y_tr_bal, X_te_p, y_te)
            print(f"    PCA-8    ROC-AUC={roc_p:.4f}  PR-AUC={pr_p:.4f}")
            results_for_plot[f"{method}_pca8"] = (y_te, probs_p, roc_p, pr_p)
            summary_rows.append(
                dict(split=split_name, method=method, pca=True, roc=roc_p, pr=pr_p)
            )

        plot_curves(results_for_plot, out_dir, title_prefix=split_name)

    # Summary table
    print("\n========== SUMMARY ==========")
    print(f"{'split':<10} {'method':<12} {'pca':<6} {'ROC-AUC':>8} {'PR-AUC':>8}")
    for r in summary_rows:
        print(
            f"{r['split']:<10} {r['method']:<12} {str(r['pca']):<6} "
            f"{r['roc']:>8.4f} {r['pr']:>8.4f}"
        )

    # Save CSV
    import csv
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "method", "pca", "roc", "pr"])
        w.writeheader()
        w.writerows(summary_rows)

    # Human-readable summary for portfolio / terminal-less viewing
    lines = [
        "Tox21 NR-AR experiment summary",
        "=" * 40,
        f"{'split':<10} {'method':<12} {'pca':<6} {'ROC-AUC':>8} {'PR-AUC':>8}",
    ]
    for r in summary_rows:
        lines.append(
            f"{r['split']:<10} {r['method']:<12} {str(r['pca']):<6} "
            f"{r['roc']:>8.4f} {r['pr']:>8.4f}"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- pca=True means features reduced to 8 principal components")
    lines.append("- Class imbalance handled via oversampling + scale_pos_weight")
    lines.append("- PR-AUC is the primary metric under heavy imbalance")
    (out_dir / "summary.txt").write_text("\n".join(lines))
    print(f"\nResults + plots written to {out_dir}/")
    print(f"Also wrote {out_dir}/summary.txt")


if __name__ == "__main__":
    main()
