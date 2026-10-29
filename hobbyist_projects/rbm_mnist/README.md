# rbm_mnist

Side-by-side comparison of a **Restricted Boltzmann Machine (RBM) stack** and a modern supervised **MLP** on MNIST.

Classic deep-belief-network style pre-training is reimplemented (CD-*k*, optional sparsity + L2), then used both as a feature extractor and as weight initialization for an MLP. A randomly initialized MLP trained end-to-end with Adam is the baseline.

## Expected result (and why it is kept)

On MNIST with abundant labels, a modern Adam-trained MLP usually **outperforms** classic RBM pre-training. That is not a bug in the code; it reflects the post-2010s reality that good random init + ReLU + Adam removed much of the need for unsupervised pre-training on simple vision tasks. The experiment documents that fact instead of forcing the RBM to look better with unfair settings.

Unsupervised (or self-supervised) pre-training still matters when labels are scarce. MNIST is simply not that regime.

## Requirements

```bash
pip install torch numpy matplotlib pillow scikit-learn
# MNIST is loaded via the existing data_loaders (or torchvision if you adapt the loader)
```

## Run

```bash
./run.sh
# or
python train.py --out results
```

Outputs under `results/`:

- `results.txt` — accuracies for each configuration
- reconstruction PNGs
- training-history plots

## Configurations compared

1. Pure supervised MLP (random init, Adam)
2. RBM stack (greedy layer-wise CD) + linear / RF probes on hidden units
3. RBM stack used as **initialization** of an MLP, then fine-tuned
4. Same RBM stack **with** vs **without** sparsity constraint + L2 on hidden units

## Layout

```text
rbm_mnist/
├── rbm.py            # CD-k RBM (Adam, optional sparsity + L2)
├── mlp.py            # small MLP used for supervised baselines
├── data_loaders.py   # MNIST batch helpers
├── train.py          # full comparison + plots
└── run.sh
```

## Notes

- Plots use a non-interactive backend (`Agg`); nothing requires a display.
- RBM hidden units can be probed with a cheap classifier to see how much label signal the unsupervised layers captured.
- The write-up in `results/results.txt` is intentionally honest about the MLP winning under full supervision.
