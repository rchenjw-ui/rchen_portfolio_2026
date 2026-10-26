"""
Restricted Boltzmann Machine with optional sparsity + L2 regularisation.
"""

from __future__ import annotations
import copy
from typing import Tuple
import torch
import torch.nn.functional as F


class RBM:
    def __init__(
        self,
        in_size: int,
        out_size: int,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        device: str = "cpu",
        sparsity_target: float = 0.05,
        sparsity_weight: float = 0.0,
        l2_weight: float = 0.0,
    ):
        self.device = device
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        self.l2_weight = l2_weight

        self.weights = (torch.rand(in_size, out_size, device=device) - 0.5) * 0.01
        self.b_h = torch.zeros(out_size, device=device)
        self.b_v = torch.zeros(in_size, device=device)

        self.momentum_w = torch.zeros_like(self.weights)
        self.variance_w = torch.zeros_like(self.weights)
        self.momentum_h = torch.zeros_like(self.b_h)
        self.variance_h = torch.zeros_like(self.b_h)
        self.momentum_v = torch.zeros_like(self.b_v)
        self.variance_v = torch.zeros_like(self.b_v)

        # running average of hidden activations for sparsity
        self.q_h = torch.zeros(out_size, device=device)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(v @ self.weights + self.b_h)

    def backwards(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(h @ self.weights.T + self.b_v)

    def cd_k(
        self, data: torch.Tensor, k: int = 1
    ) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
        data = data.to(self.device)
        p_hd = self.forward(data)
        h_data = torch.bernoulli(p_hd)

        h_run = h_data.clone()
        for _ in range(k):
            p_vm = self.backwards(h_run)
            v_model = torch.bernoulli(p_vm)
            p_hm = self.forward(v_model)
            h_run = torch.bernoulli(p_hm)

        # gradients (positive - negative)
        grad_w = (data.T @ p_hd - v_model.T @ p_hm) / len(data)
        grad_b_v = (data - v_model).mean(dim=0)
        grad_b_h = (p_hd - p_hm).mean(dim=0)

        # L2 decay on weights
        if self.l2_weight > 0:
            grad_w = grad_w - self.l2_weight * self.weights

        # Sparsity penalty (encourage mean activation ≈ target)
        if self.sparsity_weight > 0:
            with torch.no_grad():
                self.q_h = 0.9 * self.q_h + 0.1 * p_hd.mean(dim=0)
            # gradient of (q - target)^2 w.r.t. bias roughly
            sparsity_grad = self.sparsity_weight * (self.q_h - self.sparsity_target)
            grad_b_h = grad_b_h - sparsity_grad

        loss = float(F.binary_cross_entropy(p_vm, data))
        return loss, grad_w, grad_b_v, grad_b_h

    def adam_optimizer(
        self,
        t: int,
        grad_w: torch.Tensor,
        grad_v: torch.Tensor,
        grad_h: torch.Tensor,
        lr: float = 1e-3,
        eps: float = 1e-8,
    ):
        grad_w = grad_w.to(self.device)
        grad_v = grad_v.to(self.device)
        grad_h = grad_h.to(self.device)

        self.momentum_w = self.beta_1 * self.momentum_w + (1 - self.beta_1) * grad_w
        self.momentum_h = self.beta_1 * self.momentum_h + (1 - self.beta_1) * grad_h
        self.momentum_v = self.beta_1 * self.momentum_v + (1 - self.beta_1) * grad_v

        self.variance_w = self.beta_2 * self.variance_w + (1 - self.beta_2) * (grad_w ** 2)
        self.variance_h = self.beta_2 * self.variance_h + (1 - self.beta_2) * (grad_h ** 2)
        self.variance_v = self.beta_2 * self.variance_v + (1 - self.beta_2) * (grad_v ** 2)

        m_w = self.momentum_w / (1 - self.beta_1 ** t)
        v_w = self.variance_w / (1 - self.beta_2 ** t)
        m_h = self.momentum_h / (1 - self.beta_1 ** t)
        v_h = self.variance_h / (1 - self.beta_2 ** t)
        m_v = self.momentum_v / (1 - self.beta_1 ** t)
        v_v = self.variance_v / (1 - self.beta_2 ** t)

        self.weights = self.weights + lr * m_w / (torch.sqrt(v_w) + eps)
        self.b_h = self.b_h + lr * m_h / (torch.sqrt(v_h) + eps)
        self.b_v = self.b_v + lr * m_v / (torch.sqrt(v_v) + eps)
