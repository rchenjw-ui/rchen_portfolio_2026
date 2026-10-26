"""
Load Tox21 and prepare balanced / stratified views.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from datasets import load_dataset
from rdkit import Chem
from collections import Counter


def load_tox21_target(target: str = "NR-AR") -> Dict[str, int]:
    """
    Returns {smiles: label} for a single Tox21 assay.
    Labels are 0/1; NAs are dropped. Invalid SMILES dropped.
    """
    ds = load_dataset("HUBiodatalab/tox21")["train"]
    if target not in ds.features:
        raise ValueError(f"Unknown target {target}. Available: {list(ds.features.keys())}")

    smiles = ds["smiles"]
    labels = ds[target]

    out = {}
    for s, y in zip(smiles, labels):
        if y is None:
            continue
        if Chem.MolFromSmiles(s) is None:
            continue
        out[s] = int(y)
    return out


def class_stats(labels: List[int]) -> str:
    c = Counter(labels)
    n0 = c.get(0, 0)
    n1 = c.get(1, 0)
    total = n0 + n1
    pct = 100.0 * n1 / total if total else 0.0
    return f"{n0} neg / {n1} pos  ({pct:.2f}% positive)"


def oversample_minority(
    X: np.ndarray, y: np.ndarray, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Naive random oversampling of the minority class."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y
    maj = classes[np.argmax(counts)]
    min_c = classes[np.argmin(counts)]
    n_maj = counts.max()
    n_min = counts.min()
    if n_maj == n_min:
        return X, y

    idx_min = np.where(y == min_c)[0]
    n_extra = n_maj - n_min
    extra = rng.choice(idx_min, size=n_extra, replace=True)
    X_new = np.concatenate([X, X[extra]], axis=0)
    y_new = np.concatenate([y, y[extra]], axis=0)
    # shuffle
    perm = rng.permutation(len(y_new))
    return X_new[perm], y_new[perm]


def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Simple z-score using train statistics. Safe for fingerprints too."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std
