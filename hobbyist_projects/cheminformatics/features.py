"""
Molecular featurizers:
- Morgan / ECFP fingerprints
- RDKit 2-D descriptors
- MolDeBERTa embeddings (optional, heavy)
"""

from __future__ import annotations
from typing import List, Optional, Dict
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator


# ---------------------------------------------------------------------------
# Classic fingerprints / descriptors
# ---------------------------------------------------------------------------

def morgan_fingerprint(
    smiles: str, radius: int = 2, n_bits: int = 2048
) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    return np.array(fp, dtype=np.float32)


def rdkit_descriptors(smiles: str) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    vals = []
    for _, fn in Descriptors.descList:
        try:
            v = fn(mol)
            if v is None or np.isnan(v) or np.isinf(v):
                return None
            # clamp extreme values that break float32
            if abs(v) > np.finfo(np.float32).max:
                return None
            vals.append(float(v))
        except Exception:
            return None
    return np.array(vals, dtype=np.float32)


# ---------------------------------------------------------------------------
# MolDeBERTa (lazy – only loads when asked)
# ---------------------------------------------------------------------------

_moldeberta_model = None
_moldeberta_tok = None


def _load_moldeberta(model_name: str = "SaeedLab/MolDeBERTa-base-10M-mtr"):
    global _moldeberta_model, _moldeberta_tok
    if _moldeberta_model is not None:
        return _moldeberta_model, _moldeberta_tok

    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"[MolDeBERTa] Loading {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    _moldeberta_model = model
    _moldeberta_tok = tok
    return model, tok


def moldeberta_embeddings(
    smiles_list: List[str],
    batch_size: int = 64,
    device: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Returns dict {smiles: embedding vector}.
    Uses CLS token (first position) of last_hidden_state.
    """
    import torch

    model, tok = _load_moldeberta()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    out: Dict[str, np.ndarray] = {}
    for i in range(0, len(smiles_list), batch_size):
        chunk = smiles_list[i : i + batch_size]
        inputs = tok(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # CLS token
        emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
        for smi, e in zip(chunk, emb):
            out[smi] = e
    return out
