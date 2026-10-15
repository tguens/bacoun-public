"""Bayesian logistic regression on fixed features (the 'Bayesian layer' of an NLM).

The features come from the last hidden layer of the deterministic classifier and
are held fixed; only the last linear layer is treated as random. Inference is
either mean-field variational (BBVI) or NUTS/HMC, both via Pyro.
"""

from functools import partial

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, MCMC, NUTS, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam


def _augment(phi: torch.Tensor) -> torch.Tensor:
    """Append a constant 1 so the weight matrix carries the bias."""
    return torch.cat([phi, torch.ones(phi.shape[0], 1, device=phi.device, dtype=phi.dtype)], 1)


def model(phi: torch.Tensor, y: torch.Tensor, K: int, prior_std: float = 1.0):
    """y | x ~ Cat(softmax(W' phi(x))),  W ~ N(0, prior_std^2)."""
    phi = _augment(phi)
    W = pyro.sample("W", dist.Normal(phi.new_zeros(phi.shape[1], K), prior_std).to_event(2))
    logits = phi @ W
    with pyro.plate("data", phi.shape[0]):
        pyro.sample("obs", dist.Categorical(logits=logits), obs=y)


def fit_bbvi(
    phi, y, K, n_steps=10_000, lr=1e-2, n_samples=200, seed=0, prior_std=1.0, log_every=2000
):
    """Mean-field Gaussian variational inference. Returns weight samples [S, L+1, K]."""
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    conditioned = partial(model, prior_std=prior_std)
    guide = AutoNormal(conditioned)
    svi = SVI(conditioned, guide, Adam({"lr": lr}), loss=Trace_ELBO())
    losses = []
    for step in range(n_steps):
        losses.append(svi.step(phi, y, K))
        if log_every and step % log_every == 0:
            print(f"    bbvi step {step:6d}  elbo_loss {np.mean(losses[-100:]):.2f}", flush=True)
    with torch.no_grad():
        W = torch.stack([guide(phi, y, K)["W"] for _ in range(n_samples)])
    return W, losses


def fit_hmc(phi, y, K, n_samples=1000, warmup=100, seed=0, prior_std=1.0):
    """NUTS. Returns weight samples [S, L+1, K]."""
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    conditioned = partial(model, prior_std=prior_std)
    mcmc = MCMC(
        NUTS(conditioned), num_samples=n_samples, warmup_steps=warmup, disable_progbar=False
    )
    mcmc.run(phi, y, K)
    return mcmc.get_samples()["W"], []


@torch.no_grad()
def posterior_predictive(W: torch.Tensor, phi: torch.Tensor, batch: int = 4096) -> np.ndarray:
    """Class probabilities under every posterior sample: [S, N, K]."""
    phi = _augment(phi)
    out = []
    for i in range(0, phi.shape[0], batch):
        out.append(torch.softmax(phi[i : i + batch] @ W, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=1)


@torch.no_grad()
def mc_dropout_predict(clf, X: torch.Tensor, n_samples: int = 200) -> np.ndarray:
    """MC-Dropout: keep dropout on at test time and treat the passes as an ensemble.

    Returns class probabilities of shape [S, N, K], the same layout as
    `posterior_predictive`, so the uncertainty decomposition applies unchanged.
    """
    clf.train()  # the only stochastic layers are Dropout, so this just re-enables them
    probs = torch.stack([torch.softmax(clf(X), dim=-1) for _ in range(n_samples)])
    clf.eval()
    return probs.cpu().numpy()
