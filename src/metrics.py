"""Predictive metrics and the aleatoric / epistemic uncertainty decomposition."""

import numpy as np
from sklearn.metrics import roc_auc_score


def _entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis)


def decompose_uncertainty(probs: np.ndarray) -> dict:
    """Split predictive uncertainty (Appendix C, Eq. 4).

    `probs` has shape [S, N, K]: K class probabilities for N points under each of
    S posterior samples.

      total      = H[ mean_S p(y|x, W) ]                (entropy of the mean prediction)
      aleatoric  = mean_S H[ p(y|x, W) ]                (average entropy of each sample)
      epistemic  = total - aleatoric                    (mutual information between y and W)
    """
    mean = probs.mean(0)
    total = _entropy(mean)
    aleatoric = _entropy(probs).mean(0)
    return dict(mean_probs=mean, total=total, aleatoric=aleatoric, epistemic=total - aleatoric)


def accuracy(mean_probs: np.ndarray, y: np.ndarray) -> float:
    return float((mean_probs.argmax(1) == y).mean())


def auc(mean_probs: np.ndarray, y: np.ndarray) -> float:
    """One-vs-rest macro AUC; plain binary AUC when K = 2."""
    if mean_probs.shape[1] == 2:
        return float(roc_auc_score(y, mean_probs[:, 1]))
    return float(roc_auc_score(y, mean_probs, multi_class="ovr", average="macro"))


def separation_auroc(score_in: np.ndarray, score_ood: np.ndarray) -> float:
    """How well a score separates in-distribution from OOD points. 1.0 is perfect."""
    labels = np.concatenate([np.zeros(len(score_in)), np.ones(len(score_ood))])
    return float(roc_auc_score(labels, np.concatenate([score_in, score_ood])))


def radial_profile(
    points: np.ndarray, values: np.ndarray, center, n_bins: int = 40, max_radius: float = None
):
    """Mean of `values` binned by distance from `center`.

    Used to check the shape that matters here: epistemic uncertainty should rise
    as soon as you leave the data, not only very far away.
    """
    r = np.linalg.norm(points - np.asarray(center), axis=1)
    max_radius = max_radius or r.max()
    edges = np.linspace(0, max_radius, n_bins + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.array(
        [values[idx == b].mean() if (idx == b).any() else np.nan for b in range(n_bins)]
    )
    return centers, means
