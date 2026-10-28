# cheminformatics

Tox21 (NR-AR) activity prediction pipeline.  
Compares **featurization methods** and **train/test split strategies**, with proper class imbalance handling. Metrics reported are ROC-AUC and PR-AUC.

## Main finding

Split method matters. Structure-aware splits (scaffold / Butina) are harder than a pure random split; good random-split numbers do **not** guarantee generalization to novel chemotypes. 

Across the board, regardless of featurization methods, random-forest methods rapidly underperformed under the Butina split condition. 

Statistical significance between methodologies will be evaluated at a future date. 

## Requirements

```bash
pip install numpy pandas scikit-learn xgboost rdkit-pypi matplotlib
# optional:
#   pip install torch transformers   # for --use-moldeberta
```

## Run

```bash
./run.sh
# or
python train_eval.py --out results

# with MolDeBERTa embeddings (slower, needs GPU/MPS-friendly transformers)
python train_eval.py --use-moldeberta --out results_bert
```

Results land in `results/` (or the directory you pass to `--out`):

- `summary.txt` — tables of ROC-AUC / PR-AUC by feature set and split
- PNG curves (ROC, PR) per configuration

## What is compared

| Piece | Options |
|-------|---------|
| Features | Morgan/ECFP fingerprints, RDKit descriptors, optional MolDeBERTa |
| Dimensionality | Raw vs PCA to 8 components |
| Splits | Random, Bemis–Murcko scaffold, Butina clustering |
| Imbalance | Oversampling + XGBoost `scale_pos_weight` |
| Metrics | ROC-AUC and **PR-AUC** (more informative when positives are rare) |

## Layout

```text
cheminformatics/
├── data.py           # load Tox21 target, oversample, scale
├── features.py       # ECFP, RDKit descriptors, MolDeBERTa
├── splits.py         # random / scaffold / Butina
├── train_eval.py     # main pipeline + plots
└── run.sh
```

## Notes

- PR-AUC is emphasized because the positive class is sparse; accuracy and ROC alone can look optimistic.
- PCA-8 is included as a simple check of whether a low-dimensional projection retains signal.
- The pipeline is intentionally modular so you can swap in other targets or feature sets.

## Next Steps
- Combinations of featurization methods may be tested
- We may test deep-learning and otherwise methods
- Graph neural network features may be utilized as well
- We may try self-normalizing neural networks; these generally perform better on structured data than standard deep nets
- Feature importance may be explored 
