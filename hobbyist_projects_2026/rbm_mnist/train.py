#!/usr/bin/env python3
"""
Compare:
1. Pure supervised MLP (random init)
2. RBM stack (greedy layer-wise) → linear probes
3. RBM stack as initialisation of MLP + fine-tune
4. Same RBM stack with / without sparsity + L2

Why an RBM stack can underperform a modern MLP
----------------------------------------------
Classic RBMs were important for pre-training deep nets before good
random initialisation + ReLU + batch-norm + Adam existed. With modern
optimisers a randomly-initialised MLP often wins on MNIST. The RBM
comparison is still scientifically interesting and is kept honest.
"""

from __future__ import annotations
import argparse
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

from data_loaders import load_batch_imgs, load_test_set, flatten_batch
from rbm import RBM
from mlp import MLPModel


def eval_linear_probe(rbm: RBM, train_x, train_y, test_x, test_y) -> float:
    with torch.no_grad():
        h_tr = rbm.forward(train_x).cpu().numpy()
        h_te = rbm.forward(test_x).cpu().numpy()
    clf = RandomForestClassifier(
        n_estimators=80, max_features="sqrt", random_state=42, n_jobs=2
    )
    y_tr = train_y.numpy() if torch.is_tensor(train_y) else train_y
    clf.fit(h_tr, y_tr)
    preds = clf.predict(h_te)
    return float((preds == test_y).mean())


def save_reconstruction(rbm: RBM, img: torch.Tensor, path: Path):
    with torch.no_grad():
        h = rbm.forward(img.unsqueeze(0))
        rec = rbm.backwards(h).squeeze(0).cpu().numpy()
    rec = (rec.reshape(28, 28) * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(rec).save(path)


def plot_history(history: dict, out_dir: Path, title: str):
    plt.figure(figsize=(8, 4))
    for k, v in history.items():
        plt.plot(v, label=k)
    plt.xlabel("checkpoint")
    plt.ylabel("value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{title.replace(' ', '_')}.png", dpi=120)
    plt.close()


def train_rbm_stack(
    train_imgs: torch.Tensor,
    train_labels: torch.Tensor,
    test_imgs: torch.Tensor,
    test_labels: np.ndarray,
    sizes: list[tuple[int, int]],
    steps: int = 80,
    k: int = 3,
    sparsity: bool = False,
    l2: float = 0.0,
    device: str = "cpu",
    out_dir: Path | None = None,
) -> list[RBM]:
    rbms = []
    data = train_imgs.to(device)
    test_data = test_imgs.to(device)

    for layer_idx, (in_s, out_s) in enumerate(sizes):
        print(f"\n=== RBM layer {layer_idx}: {in_s} → {out_s} "
              f"(sparsity={sparsity}, l2={l2}) ===")
        rbm = RBM(
            in_s, out_s,
            device=device,
            sparsity_target=0.05,
            sparsity_weight=0.1 if sparsity else 0.0,
            l2_weight=l2,
        )
        lr = 0.01
        hist_loss, hist_acc = [], []

        for step in range(steps):
            loss, gw, gv, gh = rbm.cd_k(data, k=k)
            rbm.adam_optimizer(step + 1, gw, gv, gh, lr=lr)

            if step % 10 == 0:
                acc = eval_linear_probe(
                    rbm, data.cpu(), train_labels, test_data.cpu(), test_labels
                )
                print(f"  step {step:3d}  loss={loss:.4f}  probe_acc={acc:.3f}")
                hist_loss.append(loss)
                hist_acc.append(acc)
                if out_dir and layer_idx == 0:
                    save_reconstruction(
                        rbm, test_imgs[0],
                        out_dir / f"rec_L0_step{step:03d}.png",
                    )
            lr *= 0.97

        with torch.no_grad():
            data = rbm.forward(data)
            test_data = rbm.forward(test_data)
        print(f"  hidden mean={data.mean().item():.3f}  std={data.std().item():.3f}")
        rbms.append(rbm)

        if out_dir:
            plot_history(
                {"loss": hist_loss, "probe_acc": hist_acc},
                out_dir,
                f"rbm_L{layer_idx}_{'sparse' if sparsity else 'plain'}",
            )
    return rbms


def train_mlp(
    model: MLPModel,
    batches,
    labels,
    test_images,
    test_labels,
    epochs: int = 8,
    device: str = "cpu",
) -> list[float]:
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    acc_hist = []

    for ep in range(epochs):
        model.train()
        correct = total = 0
        for bx, by in zip(batches, labels):
            bx = bx.to(device)
            by = by.to(device)
            optim.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optim.step()
            pred = logits.argmax(dim=1)
            correct += (pred == by).sum().item()
            total += by.size(0)
        train_acc = 100.0 * correct / total

        model.eval()
        t_correct = t_total = 0
        with torch.no_grad():
            for ti, tl in zip(test_images, test_labels):
                ti = ti.to(device)
                out = model(ti)
                pred = out.argmax(dim=1).cpu().numpy()
                t_correct += (pred == tl).sum()
                t_total += len(tl)
        test_acc = 100.0 * t_correct / t_total
        print(f"  epoch {ep+1}  train={train_acc:.1f}%  test={test_acc:.1f}%")
        acc_hist.append(test_acc)
    return acc_hist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    print("Loading MNIST ...")
    batches, img_labels = load_batch_imgs(100)
    test_images, test_labels = load_test_set(100)

    # Use several batches for RBM so it sees more than 100 examples
    train_x = torch.cat([flatten_batch(b) for b in batches[:5]], dim=0)  # 500 imgs
    train_y = torch.cat(img_labels[:5], dim=0)
    test_x = flatten_batch(test_images[0])
    test_y = test_labels[0]

    sizes = [(784, 256), (256, 64), (64, 32)]
    log_lines = []

    print("\n########## Plain RBM stack ##########")
    rbms_plain = train_rbm_stack(
        train_x, train_y, test_x, test_y,
        sizes, steps=args.steps, sparsity=False, l2=0.0,
        device=device, out_dir=out_dir,
    )

    print("\n########## Sparse + L2 RBM stack ##########")
    rbms_sparse = train_rbm_stack(
        train_x, train_y, test_x, test_y,
        sizes, steps=args.steps, sparsity=True, l2=1e-4,
        device=device, out_dir=out_dir,
    )

    print("\n########## Supervised MLP (random) ##########")
    mlp_rand = MLPModel(random_init=True)
    acc_rand = train_mlp(
        mlp_rand, batches, img_labels, test_images, test_labels,
        epochs=args.epochs, device=device,
    )
    log_lines.append(f"MLP random init – final test acc: {acc_rand[-1]:.2f}%")

    print("\n########## MLP init from plain RBMs ##########")
    mlp_rbm = MLPModel(random_init=False)
    mlp_rbm.load_rbm_weights(rbms_plain)
    acc_rbm = train_mlp(
        mlp_rbm, batches, img_labels, test_images, test_labels,
        epochs=args.epochs, device=device,
    )
    log_lines.append(f"MLP ← plain RBM   – final test acc: {acc_rbm[-1]:.2f}%")

    print("\n########## MLP init from sparse RBMs ##########")
    mlp_sp = MLPModel(random_init=False)
    mlp_sp.load_rbm_weights(rbms_sparse)
    acc_sp = train_mlp(
        mlp_sp, batches, img_labels, test_images, test_labels,
        epochs=args.epochs, device=device,
    )
    log_lines.append(f"MLP ← sparse RBM  – final test acc: {acc_sp[-1]:.2f}%")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(acc_rand, label="MLP random")
    plt.plot(acc_rbm, label="MLP ← plain RBM")
    plt.plot(acc_sp, label="MLP ← sparse RBM")
    plt.xlabel("epoch")
    plt.ylabel("test accuracy %")
    plt.title("Fine-tuning comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "finetune_comparison.png", dpi=120)
    plt.close()

    # Write human-readable results
    note = """
Note on RBM vs modern MLP
-------------------------
It is common for a well-optimised randomly-initialised MLP (Adam + good
init) to beat a classic RBM stack on MNIST. RBMs were historically
important for making deep nets trainable; with modern optimisers the
unsupervised pre-training advantage often shrinks or disappears on
simple datasets. The comparison is kept intentional and honest.
"""
    (out_dir / "results.txt").write_text(
        "RBM vs MLP on MNIST\n" + "=" * 40 + "\n" +
        "\n".join(log_lines) + "\n" + note
    )
    print("\nResults written to", out_dir / "results.txt")
    print("Done.")


if __name__ == "__main__":
    main()
