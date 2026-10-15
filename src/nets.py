"""Neural networks: a RealNVP normalizing flow and the deterministic classifier."""

import math

import torch
import torch.nn as nn


def mlp(in_dim: int, hidden: list[int], out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LeakyReLU(0.01)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class CouplingLayer(nn.Module):
    """Affine coupling: the masked half of x is passed through and conditions the rest."""

    def __init__(self, dim: int, hidden: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = mlp(dim, [hidden, hidden], 2 * dim)
        nn.init.zeros_(self.net[-1].weight)  # start as the identity map
        nn.init.zeros_(self.net[-1].bias)

    def _scale_shift(self, x_masked: torch.Tensor):
        s, t = self.net(x_masked).chunk(2, dim=-1)
        s = 2.0 * torch.tanh(s)  # bound the log-scale to keep training stable
        return s * (1 - self.mask), t * (1 - self.mask)

    def forward(self, x: torch.Tensor):
        """Data -> latent. Returns (z, log|det Jacobian|)."""
        x_masked = x * self.mask
        s, t = self._scale_shift(x_masked)
        z = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        return z, s.sum(-1)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Latent -> data."""
        z_masked = z * self.mask
        s, t = self._scale_shift(z_masked)
        return z_masked + (1 - self.mask) * (z - t) * torch.exp(-s)


class RealNVP(nn.Module):
    """RealNVP with a standard normal base distribution (paper: 5 blocks, 64 hidden)."""

    def __init__(self, dim: int = 2, n_blocks: int = 5, hidden: int = 64):
        super().__init__()
        self.dim = dim
        masks = [((torch.arange(dim) + i) % 2).float() for i in range(n_blocks)]  # alternating
        self.blocks = nn.ModuleList([CouplingLayer(dim, hidden, m) for m in masks])

    def forward(self, x: torch.Tensor):
        log_det = torch.zeros(x.shape[0], device=x.device)
        for block in self.blocks:
            x, ld = block(x)
            log_det = log_det + ld
        return x, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        for block in reversed(self.blocks):
            z = block.inverse(z)
        return z

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det = self(x)
        base = -0.5 * (z**2).sum(-1) - 0.5 * self.dim * math.log(2 * math.pi)
        return base + log_det

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        device = next(self.parameters()).device
        return self.inverse(torch.randn(n, self.dim, device=device))


class Classifier(nn.Module):
    """Feed-forward classifier. `features` is the last hidden layer (the NLM basis)."""

    def __init__(
        self, in_dim: int, n_classes: int, hidden=(64, 64, 64, 1024), dropout: float = 0.1
    ):
        """`hidden` is the width of each hidden layer; the last one is the feature layer."""
        super().__init__()
        self.body = mlp(in_dim, list(hidden[:-1]), hidden[-1], dropout=dropout)
        self.act = nn.LeakyReLU(0.01)
        self.head = nn.Linear(hidden[-1], n_classes)
        self.feature_dim = hidden[-1]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
