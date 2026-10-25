"""
Train / test splitters for molecular data.
- Random
- Scaffold (Murcko)
- Butina (fingerprint clustering)
"""

from __future__ import annotations
import random
from typing import List, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from rdkit.Chem.Scaffolds import MurckoScaffold


def _valid_smiles(smiles_list: List[str]) -> List[str]:
    return [s for s in smiles_list if Chem.MolFromSmiles(s) is not None]


def random_train_test_split(
    smiles_list: List[str], test_size: float = 0.2, seed: int = 42
) -> Tuple[List[str], List[str]]:
    valid = _valid_smiles(smiles_list)
    rng = random.Random(seed)
    rng.shuffle(valid)
    n_test = int(round(test_size * len(valid)))
    return valid[n_test:], valid[:n_test]


def scaffold_train_test_split(
    smiles_list: List[str], test_size: float = 0.2
) -> Tuple[List[str], List[str]]:
    mols = {}
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        mols[s] = scaf

    clusters: dict[str, list[str]] = {}
    for smi, scaf in mols.items():
        clusters.setdefault(scaf, []).append(smi)

    test_target = int(len(mols) * test_size)
    train, test = [], []
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

    for cluster in reversed(sorted_clusters):  # smallest first
        if len(test) + len(cluster) <= test_target:
            test.extend(cluster)
        else:
            train.extend(cluster)

    # any leftovers
    for cluster in sorted_clusters:
        for s in cluster:
            if s not in train and s not in test:
                train.append(s)

    return train, test


def butina_train_test_split(
    smiles_list: List[str],
    test_size: float = 0.2,
    dist_cutoff: float = 0.65,
    radius: int = 2,
    n_bits: int = 2048,
) -> Tuple[List[str], List[str]]:
    valid = _valid_smiles(smiles_list)
    if len(valid) != len(smiles_list):
        print(f"Warning: {len(smiles_list) - len(valid)} invalid SMILES skipped.")

    generator = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    mols = []
    for s in valid:
        mol = Chem.MolFromSmiles(s)
        fps.append(generator.GetFingerprint(mol))
        mols.append(s)

    num_mols = len(fps)
    dist_matrix = []
    for i in range(1, num_mols):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i], returnDistance=True)
        dist_matrix.extend(sims)

    clusters = Butina.ClusterData(dist_matrix, num_mols, dist_cutoff, isDistData=True)
    sorted_clusters = sorted(clusters, key=len, reverse=True)

    test_target = int(num_mols * test_size)
    train_idx, test_idx = [], []

    for cluster in reversed(sorted_clusters):
        if len(test_idx) + len(cluster) <= test_target:
            test_idx.extend(cluster)
        else:
            train_idx.extend(cluster)

    for cluster in sorted_clusters:
        for idx in cluster:
            if idx not in train_idx and idx not in test_idx:
                train_idx.append(idx)

    return [mols[i] for i in train_idx], [mols[i] for i in test_idx]
