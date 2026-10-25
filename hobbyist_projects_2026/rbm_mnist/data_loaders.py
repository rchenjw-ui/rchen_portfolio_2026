"""
MNIST data helpers (torch tensors, batched).
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
from datasets import load_dataset


def load_imgs_and_labels() -> Tuple[torch.Tensor, List[int]]:
    data = load_dataset("ylecun/mnist")
    train_images = torch.Tensor(
        [[np.array(x["image"]) / 255.0] for x in data["train"]]
    )  # (N, 1, 28, 28)
    train_labels = [int(x["label"]) for x in data["train"]]
    return train_images, train_labels


def load_batch_imgs(batch_size: int = 100) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    train_images, train_labels = load_imgs_and_labels()
    batched_images = list(torch.split(train_images, batch_size))
    batched_labels = [
        torch.tensor(
            np.array(train_labels[x : x + batch_size]), dtype=torch.long
        )
        for x in range(0, len(train_images), batch_size)
    ]
    return batched_images, batched_labels


def load_test_set(batch_size: int = 100) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
    data = load_dataset("ylecun/mnist")
    test_labels = [x["label"] for x in data["test"]]
    test_images = [[np.array(x["image"]) / 255.0] for x in data["test"]]
    images = list(torch.split(torch.Tensor(test_images), batch_size))
    labels = [
        np.array(test_labels[x : x + batch_size])
        for x in range(0, len(test_images), batch_size)
    ]
    return images, labels


def flatten_batch(batch: torch.Tensor) -> torch.Tensor:
    """(B, 1, 28, 28) -> (B, 784)"""
    return batch.view(batch.size(0), -1)
