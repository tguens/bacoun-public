"""Datasets and boundary-point generation.

Two toy datasets from the BaCOUn paper (Appendix A.1) plus the latent-shell
sampler used to create the extra "boundary" class (Appendix D).
"""

import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Class means of the Gaussian mixture: an equilateral triangle of side 2*sqrt(3).
GMM_MEANS = np.array([[0.0, 2.0], [-np.sqrt(3), -1.0], [np.sqrt(3), -1.0]])


def gaussian_mixture(n_per_class: int = 2000, sigma: float = 1.5, seed: int = 0):
    """3-component isotropic Gaussian mixture, one class per component.

    `sigma` is the covariance scale: each component has covariance sigma * I,
    i.e. a per-axis standard deviation of sqrt(sigma).
    """
    rng = np.random.default_rng(seed)
    std = np.sqrt(sigma)
    X = np.concatenate([m + std * rng.standard_normal((n_per_class, 2)) for m in GMM_MEANS])
    y = np.repeat(np.arange(len(GMM_MEANS)), n_per_class)
    return X.astype(np.float32), y.astype(np.int64)


def two_moons(n_per_class: int = 2000, noise: float = 0.05, seed: int = 0):
    """sklearn's two-moons, K = 2 classes."""
    X, y = make_moons(n_samples=2 * n_per_class, noise=noise, random_state=seed)
    return X.astype(np.float32), y.astype(np.int64)


def bayes_accuracy_gmm(sigma: float = None) -> float:
    """Accuracy of the optimal classifier on the mixture (Monte-Carlo, 200k points).

    Used to check that a chosen `sigma` leaves the classes learnable but overlapping.
    """
    kw = {} if sigma is None else dict(sigma=sigma)
    X, y = gaussian_mixture(n_per_class=200_000, seed=123, **kw)
    d2 = ((X[:, None, :] - GMM_MEANS[None]) ** 2).sum(-1)  # equal priors + equal covariances
    return float((d2.argmin(1) == y).mean())


DATASETS = {
    "gmm": dict(fn=gaussian_mixture, K=3, plot_range=(-9.0, 9.0), n_boundary=6000),
    "moons": dict(fn=two_moons, K=2, plot_range=(-4.0, 4.0), n_boundary=3000),
}


def get_dataset(name: str, seed: int = 0, test_frac: float = 0.2, n_boundary: int = None, **kwargs):
    """Load a dataset and split it into train/test. Extra kwargs go to the generator."""
    spec = DATASETS[name]
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    X, y = spec["fn"](seed=seed, **kwargs)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_frac, random_state=seed, stratify=y
    )
    return dict(
        name=name,
        K=spec["K"],
        plot_range=spec["plot_range"],
        n_boundary=n_boundary or spec["n_boundary"],
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
    )


def sample_latent_shell(n: int, dim: int, inner: float, outer: float, device) -> torch.Tensor:
    """Uniform directions with radius uniform in [inner, outer] (Appendix D)."""
    direction = torch.randn(n, dim, device=device)
    direction = direction / direction.norm(dim=1, keepdim=True)
    radius = inner + (outer - inner) * torch.rand(n, 1, device=device)
    return direction * radius


@torch.no_grad()
def boundary_points(flow, n: int, inner: float, outer: float) -> torch.Tensor:
    """Sample the latent shell and push it through the flow into data space."""
    flow.eval()
    device = next(flow.parameters()).device
    z = sample_latent_shell(n, flow.dim, inner, outer, device)
    return flow.inverse(z)


def far_field_points(n: int, radius: float, center, seed: int = 0) -> np.ndarray:
    """Points on a circle of the given radius: unambiguously out-of-distribution.

    Used only at evaluation time, never for training.
    """
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * np.pi, n)
    return (np.asarray(center) + radius * np.stack([np.cos(angle), np.sin(angle)], 1)).astype(
        np.float32
    )
