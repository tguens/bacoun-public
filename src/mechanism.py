"""Test the mechanism, rather than just describing it.

`features.py` finds that the boundary class does not stop the features from growing
away from the data -- it stops them from growing in the directions that separate the
K real classes. The gap between the top two class logits stays flat off the manifold
instead of running away, so the softmax never saturates and the posterior over the
last layer can still flip the prediction.

If that is the operative ingredient, then a model that is *told* to flatten those
logits should behave like BaCOUn without an extra class at all. This script trains
three models on the same data and compares them:

    plain     K classes, training data only (the NLM baseline)
    flatten   K classes, plus a penalty pushing the logits flat at the boundary points
    boundary  K+1 classes, the boundary points being the extra class (BaCOUn)

`flatten` uses the same flow and the same boundary points as `boundary`, so the only
difference between them is what those points are asked to do.

    uv run python src/mechanism.py --dataset gmm
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import bayes
import data as data_mod
import metrics as metrics_mod
import pipeline
import plot
import utils

RESULTS = Path(__file__).resolve().parents[1] / "results"
MODES = ["plain", "flatten", "boundary"]
LABELS = {
    "plain": "plain: K classes, data only (NLM)",
    "flatten": "flatten: K classes + flat-logit penalty",
    "boundary": "boundary: K+1 classes (BaCOUn)",
}


def ray_points(center, radii, n_dirs=64):
    """`n_dirs` evenly spaced directions at each radius. Returns [n_radii, n_dirs, 2]."""
    angle = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    unit = np.stack([np.cos(angle), np.sin(angle)], 1)
    return (np.asarray(center) + radii[:, None, None] * unit[None]).astype(np.float32)


def top2_gap(logits: np.ndarray) -> np.ndarray:
    """Posterior samples of the gap between the two classes the mean prediction ranks
    first and second. `logits` is [S, N, K]; returns [S, N]. The gap crossing zero is
    exactly a posterior sample changing the prediction."""
    order = logits.mean(0).argsort(1)
    first, second = order[:, -1], order[:, -2]
    n = np.arange(logits.shape[1])
    return logits[:, n, first] - logits[:, n, second]


def fit_posterior(phi_tr, y_tr, K, inference, seed, bbvi_steps, n_samples):
    if inference == "bbvi":
        W, _ = bayes.fit_bbvi(
            phi_tr, y_tr, K, n_steps=bbvi_steps, n_samples=n_samples, seed=seed, log_every=0
        )
        return W
    W, _ = bayes.fit_hmc(phi_tr, y_tr, K, n_samples=300, warmup=100, seed=seed)
    return W[:: max(1, len(W) // n_samples)]


def study(dataset, mode, seed, grid_n, radii, bbvi_steps, n_samples, refresh, inference="bbvi"):
    art = pipeline.fit(dataset, mode=mode, seed=seed, refresh=refresh)
    ds, clf, device, K = art["ds"], art["clf"], art["device"], art["ds"]["K"]
    X_tr, y_tr, rng = ds["X_train"], ds["y_train"], ds["plot_range"]
    center = X_tr.mean(0)

    grid, xx, yy = utils.make_grid(rng, rng, grid_n)
    rays = ray_points(center, radii)
    data_radius = float(np.linalg.norm(X_tr - center, axis=1).max())
    X_far = data_mod.far_field_points(1000, 12.0, center, seed=seed)
    X_near = data_mod.far_field_points(1000, 1.2 * data_radius, center, seed=seed + 1)

    @torch.no_grad()
    def phi(X):
        return clf.features(
            torch.as_tensor(np.asarray(X, dtype=np.float32).reshape(-1, 2), device=device)
        ).cpu()

    print(f"  [{mode}] fitting the Bayesian last layer ({inference})")
    W = fit_posterior(phi(X_tr), torch.as_tensor(y_tr), K, inference, seed, bbvi_steps, n_samples)

    probs_grid = bayes.posterior_predictive(W, phi(grid))
    unc_grid = metrics_mod.decompose_uncertainty(probs_grid)
    unc_te = metrics_mod.decompose_uncertainty(bayes.posterior_predictive(W, phi(ds["X_test"])))
    unc_far = metrics_mod.decompose_uncertainty(bayes.posterior_predictive(W, phi(X_far)))
    unc_near = metrics_mod.decompose_uncertainty(bayes.posterior_predictive(W, phi(X_near)))

    # Logits along the rays, so the gap and its posterior spread can be read off
    # against distance from the data.
    with torch.no_grad():
        logits = (bayes._augment(phi(rays)) @ W).numpy()  # [S, n_radii*n_dirs, K]
    gap = top2_gap(logits).reshape(len(W), len(radii), -1)
    unc_rays = metrics_mod.decompose_uncertainty(bayes.posterior_predictive(W, phi(rays)))

    stats = dict(
        mode=mode,
        label=LABELS[mode],
        test_accuracy=metrics_mod.accuracy(unc_te["mean_probs"], ds["y_test"]),
        epistemic_ood_auroc=metrics_mod.separation_auroc(unc_te["epistemic"], unc_far["epistemic"]),
        epistemic_ood_auroc_near=metrics_mod.separation_auroc(
            unc_te["epistemic"], unc_near["epistemic"]
        ),
        epistemic_in=float(unc_te["epistemic"].mean()),
        epistemic_far=float(unc_far["epistemic"].mean()),
    )
    panel = dict(
        label=LABELS[mode],
        xx=xx,
        yy=yy,
        X=X_tr,
        y=y_tr,
        K=K,
        epistemic=unc_grid["epistemic"],
        sample_pred=probs_grid.argmax(-1),
        radii=radii,
        gap_mean=gap.mean((0, 2)),
        gap_sd=gap.std(0).mean(1),
        epistemic_ray=unc_rays["epistemic"].reshape(len(radii), -1).mean(1),
        auroc=stats["epistemic_ood_auroc"],
        auroc_near=stats["epistemic_ood_auroc_near"],
        accuracy=stats["test_accuracy"],
    )
    print(
        f"  [{mode}] accuracy {stats['test_accuracy']:.3f} | epistemic OOD AUROC far/near "
        f"{stats['epistemic_ood_auroc']:.3f}/{stats['epistemic_ood_auroc_near']:.3f} "
        f"| epistemic in/far "
        f"{stats['epistemic_in']:.3f}/{stats['epistemic_far']:.3f}"
    )
    return panel, stats


def sweep(args):
    """The comparison across seeds, without the figures. Answers whether a gap between
    two modes on one seed is real."""
    rows = []
    for seed in args.seeds:
        for mode in MODES:
            _, stats = study(
                args.dataset,
                mode,
                seed,
                40,
                np.linspace(0.2, 13.0, 4),
                args.bbvi_steps,
                args.posterior_samples,
                args.refresh,
                args.inference,
            )
            rows.append(dict(seed=seed, **stats))
    print(f"\n  {args.dataset}, {args.inference}, far-ring epistemic AUROC by seed")
    for mode in MODES:
        got = [r for r in rows if r["mode"] == mode]
        print(
            f"    {mode:9s} "
            + "  ".join(f"{r['epistemic_ood_auroc']:.3f}" for r in got)
            + "   accuracy "
            + "  ".join(f"{r['test_accuracy']:.3f}" for r in got)
        )
    utils.save_json(
        dict(
            dataset=args.dataset,
            inference=args.inference,
            seeds=args.seeds,
            commit=utils.git_commit(),
            runs=rows,
        ),
        RESULTS / f"mechanism_{args.dataset}_seeds.json",
    )
    return rows


def main(args):
    if args.seeds:
        return sweep(args)
    radii = np.linspace(0.2, 13.0, 40)
    panels, report = [], []
    for mode in MODES:
        panel, stats = study(
            args.dataset,
            mode,
            args.seed,
            args.grid,
            radii,
            args.bbvi_steps,
            args.posterior_samples,
            args.refresh,
            args.inference,
        )
        panels.append(panel)
        report.append(stats)

    rng = data_mod.DATASETS[args.dataset]["plot_range"]
    data_radius = float(np.linalg.norm(panels[0]["X"] - panels[0]["X"].mean(0), axis=1).max())
    figs = [
        plot.plot_posterior_samples(
            panels,
            rng,
            RESULTS / f"mechanism_{args.dataset}_samples.png",
            n_samples=5,
            title=f"{args.dataset}: do the posterior samples disagree away from the data?",
        ),
        plot.plot_margin_vs_spread(
            panels,
            data_radius,
            RESULTS / f"mechanism_{args.dataset}_margin.png",
            title=f"{args.dataset}: the posterior can only flip the prediction where the "
            "band reaches zero",
        ),
    ]
    utils.save_json(
        dict(
            dataset=args.dataset,
            seed=args.seed,
            commit=utils.git_commit(),
            inference=args.inference,
            models=report,
            figures=[str(Path(f).relative_to(RESULTS.parent)) for f in figs],
        ),
        RESULTS / f"mechanism_{args.dataset}.json",
    )
    for f in figs:
        print(f"  -> {f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=list(data_mod.DATASETS), default="gmm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid", type=int, default=160)
    p.add_argument("--bbvi-steps", type=int, default=10_000)
    p.add_argument("--posterior-samples", type=int, default=200)
    p.add_argument("--inference", choices=["bbvi", "hmc"], default="bbvi")
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="run the comparison at several seeds and skip the figures",
    )
    p.add_argument("--refresh", action="store_true")
    main(p.parse_args())
