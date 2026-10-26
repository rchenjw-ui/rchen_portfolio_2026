"""
Simple MLP used both as a pure supervised baseline and as a fine-tuned
stack that can be initialised from trained RBMs.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, seed: int = 42, random_init: bool = True):
        super().__init__()
        torch.manual_seed(seed)

        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 10)
        self.activation = nn.Sigmoid()

        if random_init:
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
            self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.layer4(x)  # logits
        return x

    def load_rbm_weights(self, rbm_list: list):
        """Copy RBM weights into the first three linear layers."""
        layers = [self.layer1, self.layer2, self.layer3]
        with torch.no_grad():
            for layer, rbm in zip(layers, rbm_list):
                # RBM stores (in, out); Linear expects (out, in)
                layer.weight.copy_(rbm.weights.T.cpu())
                layer.bias.copy_(rbm.b_h.cpu())
        print("[MLP] Loaded RBM weights into first 3 layers.")
