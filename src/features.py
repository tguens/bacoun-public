"""What the boundary class does to the learned features.

Two quantities decide whether a Bayesian last layer reports uncertainty at a point x.

1. How much the posterior over the last-layer weights disagrees there. With an
   N(0, I) prior the variance of a logit is

       v(x) = phi(x)' (Phi' Phi + I)^-1 phi(x)

   where phi(x) is the feature vector and Phi stacks the training features. v(x)
   is small along directions the training data already pinned down.

2. How far apart the K real-class logits already are, on average -- the margin
   between the top two. Disagreement only becomes uncertainty if it can flip the
   prediction, so what matters is the spread of the top-two logit gap relative to
   that gap. Saturate the softmax and every posterior sample agrees, whatever v(x)
   is, and the mutual information collapses to zero.

This script measures both, for the same network trained with and without the extra
boundary class, and finds that (1) is not the difference -- (2) is.

    uv run python src/features.py --dataset gmm
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import data as data_mod
import metrics as metrics_mod
import pipeline
import plot
import utils

RESULTS = Path(__file__).resolve().parents[1] / "results"


@torch.no_grad()
def features_of(clf, X, device):
    phi = clf.features(torch.as_tensor(np.asarray(X, dtype=np.float32), device=device))
    return phi.cpu().double()  # float64 on CPU: MPS has no float64, and the solve wants it


def leverage(phi_train: torch.Tensor, phi: torch.Tensor, lam: float = 1.0) -> np.ndarray:
    """v(x) = phi(x)' (Phi' Phi + lam I)^-1 phi(x), the last layer's logit variance."""
    M = phi_train.T @ phi_train + lam * torch.eye(phi_train.shape[1], dtype=torch.float64)
    L = torch.linalg.cholesky(M)
    return (phi * torch.cholesky_solve(phi.T, L).T).sum(1).numpy()


def logit_margin(clf, X, K, device) -> np.ndarray:
    """Mean gap between the top two of the K real-class logits (boundary logit dropped)."""
    with torch.no_grad():
        logits = clf(torch.as_tensor(np.asarray(X, dtype=np.float32), device=device))
    top2 = logits[:, :K].cpu().numpy()
    top2.sort(axis=1)
    return top2[:, -1] - top2[:, -2]


def project(phi_train: torch.Tensor, phi: torch.Tensor):
    """Project onto the first two principal directions of the *training* features."""
    mean = phi_train.mean(0)
    _, _, V = torch.linalg.svd(phi_train - mean, full_matrices=False)
    return ((phi - mean) @ V[:2].T).numpy()


def analyse(dataset: str, with_boundary: bool, seed: int, grid_n: int, refresh: bool):
    out = pipeline.fit(
        dataset, mode="boundary" if with_boundary else "plain", seed=seed, refresh=refresh
    )
    ds, clf, device = out["ds"], out["clf"], out["device"]
    X_tr, y_tr, rng = ds["X_train"], ds["y_train"], ds["plot_range"]

    center = X_tr.mean(0)
    data_radius = float(np.linalg.norm(X_tr - center, axis=1).max())
    X_near = data_mod.far_field_points(1000, 1.2 * data_radius, center, seed=seed + 1)
    X_far = data_mod.far_field_points(1000, 12.0, center, seed=seed)
    grid, xx, yy = utils.make_grid(rng, rng, grid_n)

    # The Bayesian layer is fitted on the real classes only, so the reference set of
    # features is the training data -- boundary points are shown but never included.
    phi = {
        k: features_of(clf, v, device)
        for k, v in dict(
            train=X_tr, near=X_near, far=X_far, grid=grid, boundary=out["X_boundary"]
        ).items()
    }
    v = {k: leverage(phi["train"], p) for k, p in phi.items()}
    margin = {
        k: logit_margin(clf, X, ds["K"], device)
        for k, X in dict(train=X_tr, near=X_near, far=X_far, grid=grid).items()
    }
    pca = {k: project(phi["train"], p) for k, p in phi.items() if k != "grid"}
    # Weights are independent across classes with the same covariance, so the gap
    # between two logits has variance 2 v(x).
    sd_gap = {k: np.sqrt(2 * val) for k, val in v.items()}

    keys = ("train", "near", "far")
    stats = dict(
        model="with boundary class" if with_boundary else "no boundary class",
        median_logit_variance={k: float(np.median(v[k])) for k in keys},
        mean_feature_norm={k: float(phi[k].norm(dim=1).mean()) for k in keys},
        mean_logit_margin={k: float(margin[k].mean()) for k in keys},
        # > 1 means the posterior spread can flip the prediction, so the samples
        # disagree and the mutual information is high.
        spread_over_margin={k: float(np.median(sd_gap[k]) / margin[k].mean()) for k in keys},
        logit_variance_auroc_far=metrics_mod.separation_auroc(v["train"], v["far"]),
        logit_variance_auroc_near=metrics_mod.separation_auroc(v["train"], v["near"]),
    )
    if with_boundary:
        stats["median_logit_variance"]["boundary"] = float(np.median(v["boundary"]))
        stats["mean_feature_norm"]["boundary"] = float(phi["boundary"].norm(dim=1).mean())

    radii, margin_profile = metrics_mod.radial_profile(grid, margin["grid"], center)
    _, sd_profile = metrics_mod.radial_profile(grid, sd_gap["grid"], center)
    panel = dict(
        label=stats["model"],
        X=X_tr,
        y=y_tr,
        xx=xx,
        yy=yy,
        log_variance=np.log10(np.clip(v["grid"], 1e-12, None)),
        margin=margin["grid"],
        pca=pca,
        radii=radii,
        margin_profile=margin_profile,
        sd_profile=sd_profile,
        data_radius=data_radius,
        center=center,
    )
    return panel, stats


def main(args):
    panels, report = [], []
    for with_boundary in (False, True):
        panel, stats = analyse(args.dataset, with_boundary, args.seed, args.grid, args.refresh)
        panels.append(panel)
        report.append(stats)
        print(f"[{stats['model']}]")
        for name, fmt in [
            ("median_logit_variance", "{:.2e}"),
            ("mean_feature_norm", "{:.1f}"),
            ("mean_logit_margin", "{:.2f}"),
            ("spread_over_margin", "{:.3f}"),
        ]:
            print(
                f"  {name:22s} "
                + "  ".join(f"{k} " + fmt.format(val) for k, val in stats[name].items())
            )
        print(
            f"  {'logit variance AUROC':22s} far {stats['logit_variance_auroc_far']:.3f}  "
            f"near {stats['logit_variance_auroc_near']:.3f}"
        )

    rng = data_mod.DATASETS[args.dataset]["plot_range"]
    fig = plot.plot_feature_effect(
        panels,
        rng,
        RESULTS / f"features_{args.dataset}.png",
        title=f"{args.dataset}: where out-of-distribution inputs land in feature space",
    )
    utils.save_json(
        dict(
            dataset=args.dataset,
            seed=args.seed,
            commit=utils.git_commit(),
            models=report,
            figure=str(Path(fig).relative_to(RESULTS.parent)),
        ),
        RESULTS / f"features_{args.dataset}.json",
    )
    print(f"  -> {fig}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=list(data_mod.DATASETS), default="gmm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid", type=int, default=200)
    p.add_argument("--refresh", action="store_true", help="retrain instead of using results/cache/")
    main(p.parse_args())
