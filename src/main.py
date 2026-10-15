"""BaCOUn on the two toy datasets: flow -> boundary points -> K+1 classifier -> Bayesian layer.

    uv run python src/main.py --dataset gmm
    uv run python src/main.py --dataset moons --method nlm

Everything a run produces (config, metrics, figures) lands in results/<tag>/.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import bayes
import data as data_mod
import metrics as metrics_mod
import plot
import utils
from nets import Classifier, RealNVP

RESULTS = Path(__file__).resolve().parents[1] / "results"

# Hyperparameters follow Appendix B of the paper unless noted in the README.
FLOW_EPOCHS = {"gmm": 200, "moons": 100}
# Appendix B: BaCOUn/NLM use 64,64,64,1024; MC-Dropout uses 4 layers of 512 at rate 0.2.
ARCHITECTURE = {
    "bacoun": dict(hidden=(64, 64, 64, 1024), dropout=0.1),
    "nlm": dict(hidden=(64, 64, 64, 1024), dropout=0.1),
    "mcdropout": dict(hidden=(512, 512, 512, 512), dropout=0.2),
}
SHELL = {"gmm": (3.0, 3.1), "moons": (3.0, 3.1)}  # inner / outer latent radius (Appendix A, D)


def train_flow(X, epochs, lr, weight_decay, batch_size, device, log_every=50):
    """Fit a RealNVP by maximum likelihood. Returns the flow and the loss curve."""
    flow = RealNVP(dim=X.shape[1]).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)
    X = torch.as_tensor(X, device=device)
    curve = []
    for epoch in range(epochs):
        perm = torch.randperm(len(X), device=device)
        epoch_loss = 0.0
        for i in range(0, len(X), batch_size):
            xb = X[perm[i : i + batch_size]]
            loss = -flow.log_prob(xb).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)
        curve.append(epoch_loss / len(X))
        if log_every and epoch % log_every == 0:
            print(f"    flow epoch {epoch:4d}  nll {curve[-1]:.3f}", flush=True)
    print(f"    flow final nll {curve[-1]:.3f}", flush=True)
    return flow, curve


def train_classifier(
    X, y, n_classes, epochs, lr, weight_decay, batch_size, dropout, hidden, device, log_every=100
):
    """Train the deterministic classifier whose last hidden layer becomes the NLM basis."""
    clf = Classifier(X.shape[1], n_classes, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    X = torch.as_tensor(X, device=device)
    y = torch.as_tensor(y, device=device)
    curve = []
    for epoch in range(epochs):
        clf.train()
        perm = torch.randperm(len(X), device=device)
        epoch_loss = 0.0
        for i in range(0, len(X), batch_size):
            idx = perm[i : i + batch_size]
            loss = loss_fn(clf(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        curve.append(epoch_loss / len(X))
        if log_every and epoch % log_every == 0:
            print(f"    clf epoch {epoch:4d}  loss {curve[-1]:.4f}", flush=True)
    clf.eval()
    with torch.no_grad():
        acc = (clf(X).argmax(1) == y).float().mean().item()
    print(f"    clf final loss {curve[-1]:.4f}  train acc (all classes) {acc:.3f}", flush=True)
    return clf, curve


@torch.no_grad()
def features_of(clf, X, device):
    clf.eval()
    return clf.features(torch.as_tensor(X, device=device)).cpu()


def uncertainties(predict, X):
    """`predict` maps raw inputs to class probabilities of shape [S, N, K]."""
    return metrics_mod.decompose_uncertainty(predict(X))


def main(args):
    t0 = time.time()
    utils.set_seed(args.seed)
    device = utils.get_device(args.device)
    inference_label = "mcdropout" if args.method == "mcdropout" else args.inference
    tag = args.tag or f"{args.dataset}_{args.method}_{inference_label}_seed{args.seed}"
    out = RESULTS / tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{tag}] device={device} commit={utils.git_commit()}")

    ds = data_mod.get_dataset(
        args.dataset,
        seed=args.seed,
        n_boundary=args.n_boundary,
        **(
            {"sigma": args.gmm_sigma, "n_per_class": args.gmm_n_per_class}
            if args.dataset == "gmm"
            else {}
        ),
    )
    K, rng = ds["K"], ds["plot_range"]
    X_tr, y_tr, X_te, y_te = ds["X_train"], ds["y_train"], ds["X_test"], ds["y_test"]
    figs = {}

    # --- Step 1: generative model of the inputs (RealNVP) ---------------------
    X_boundary = np.zeros((0, X_tr.shape[1]), dtype=np.float32)
    if args.method == "bacoun":
        print("  [1/4] training RealNVP")
        flow, flow_curve = train_flow(
            X_tr,
            epochs=args.flow_epochs or FLOW_EPOCHS[args.dataset],
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            device=device,
        )
        figs["flow_loss"] = plot.plot_loss(
            flow_curve, out / "flow_loss.png", "Flow negative log-likelihood"
        )
        figs["flow_samples"] = plot.plot_flow_samples(
            X_tr, flow.sample(len(X_tr)).cpu().numpy(), rng, out / "flow_samples.png"
        )

        # --- Step 2: boundary points = a latent shell mapped back to data space
        print("  [2/4] sampling boundary points from the latent shell")
        inner = args.shell_inner or SHELL[args.dataset][0]
        outer = args.shell_outer or SHELL[args.dataset][1]
        X_boundary = data_mod.boundary_points(flow, ds["n_boundary"], inner, outer).cpu().numpy()
        figs["data"] = plot.plot_data(X_tr, y_tr, X_boundary, rng, out / "data.png")

    # --- Step 3: deterministic classifier over K (+1) classes ----------------
    arch = dict(ARCHITECTURE[args.method])
    if args.hidden:
        arch["hidden"] = tuple(args.hidden)
    dropout = args.dropout if args.dropout is not None else arch["dropout"]
    n_classes = K + 1 if args.method == "bacoun" else K
    print(
        f"  [3/4] training the deterministic classifier on {n_classes} classes "
        f"(hidden {arch['hidden']}, dropout {dropout})"
    )
    X_all = np.concatenate([X_tr, X_boundary])
    y_all = np.concatenate([y_tr, np.full(len(X_boundary), K, dtype=np.int64)])
    clf, clf_curve = train_classifier(
        X_all,
        y_all,
        n_classes,
        epochs=args.clf_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        dropout=dropout,
        hidden=arch["hidden"],
        device=device,
    )

    grid, xx, yy = utils.make_grid(rng, rng, args.grid)
    with torch.no_grad():
        grid_pred = clf(torch.as_tensor(grid, device=device)).argmax(1).cpu().numpy()
    figs["decision"] = plot.plot_decision_boundaries(
        xx,
        yy,
        grid_pred,
        X_tr,
        y_tr,
        n_classes,
        rng,
        out / "decision_boundaries.png",
        title=f"{args.method}: decision regions of the deterministic classifier",
    )

    # --- Step 4: drop the boundary points, put a Bayesian layer on the features
    if args.method == "mcdropout":
        # No Bayesian layer: the dropout masks themselves are the ensemble.
        print(f"  [4/4] MC-Dropout, {args.posterior_samples} stochastic forward passes")
        inference_name = "mcdropout"

        def predict(X):
            return bayes.mc_dropout_predict(
                clf, torch.as_tensor(X, device=device), n_samples=args.posterior_samples
            )

    else:
        inference_name = args.inference
        print(
            f"  [4/4] {args.inference} over the last layer "
            f"(K = {K} classes, boundary points dropped)"
        )
        phi_tr = features_of(clf, X_tr, device)
        y_tr_t = torch.as_tensor(y_tr)
        if args.inference == "bbvi":
            W, elbo = bayes.fit_bbvi(
                phi_tr,
                y_tr_t,
                K,
                n_steps=args.bbvi_steps,
                lr=args.bbvi_lr,
                n_samples=args.posterior_samples,
                seed=args.seed,
                prior_std=args.prior_std,
            )
            figs["elbo"] = plot.plot_loss(elbo, out / "elbo.png")
        else:
            W, _ = bayes.fit_hmc(
                phi_tr,
                y_tr_t,
                K,
                n_samples=args.hmc_samples,
                warmup=args.hmc_warmup,
                seed=args.seed,
                prior_std=args.prior_std,
            )
            W = W[:: max(1, len(W) // args.posterior_samples)]

        def predict(X):
            return bayes.posterior_predictive(W, features_of(clf, X, device))

    # --- Evaluation ----------------------------------------------------------
    # Two rings of OOD points. The far ring is a fixed absolute radius; the near
    # ring sits just outside the data, which is the harder test.
    center = X_tr.mean(0)
    data_radius = float(np.linalg.norm(X_tr - center, axis=1).max())
    near_radius = args.near_radius_factor * data_radius
    X_far = data_mod.far_field_points(1000, radius=args.far_radius, center=center, seed=args.seed)
    X_near = data_mod.far_field_points(1000, radius=near_radius, center=center, seed=args.seed + 1)
    unc_te = uncertainties(predict, X_te)
    unc_far = uncertainties(predict, X_far)
    unc_near = uncertainties(predict, X_near)
    unc_grid = uncertainties(predict, grid)

    res = dict(
        tag=tag,
        commit=utils.git_commit(),
        config=vars(args),
        device=str(device),
        method=args.method,
        inference=inference_name,
        test_accuracy=metrics_mod.accuracy(unc_te["mean_probs"], y_te),
        test_auc=metrics_mod.auc(unc_te["mean_probs"], y_te),
        epistemic_ood_auroc=metrics_mod.separation_auroc(unc_te["epistemic"], unc_far["epistemic"]),
        epistemic_ood_auroc_near=metrics_mod.separation_auroc(
            unc_te["epistemic"], unc_near["epistemic"]
        ),
        data_radius=data_radius,
        near_radius=near_radius,
        epistemic_near=float(unc_near["epistemic"].mean()),
        epistemic_in=float(unc_te["epistemic"].mean()),
        epistemic_far=float(unc_far["epistemic"].mean()),
        aleatoric_in=float(unc_te["aleatoric"].mean()),
        aleatoric_far=float(unc_far["aleatoric"].mean()),
        runtime_s=round(time.time() - t0, 1),
    )
    if args.dataset == "gmm":
        res["bayes_optimal_accuracy"] = data_mod.bayes_accuracy_gmm(args.gmm_sigma)

    radii, epis_profile = metrics_mod.radial_profile(grid, unc_grid["epistemic"], center)
    _, alea_profile = metrics_mod.radial_profile(grid, unc_grid["aleatoric"], center)
    res["epistemic_at_1.1x_data_radius"] = float(np.interp(1.1 * data_radius, radii, epis_profile))
    res["epistemic_peak"] = float(np.nanmax(epis_profile))
    # How much of the far-field epistemic uncertainty is already reached at the edge
    # of the data. 1.0 means it rises immediately; near 0 means it only grows far away.
    res["epistemic_rise_ratio"] = res["epistemic_at_1.1x_data_radius"] / max(
        res["epistemic_peak"], 1e-9
    )

    label = f"{args.method} / {inference_name} on {args.dataset}"
    figs["uncertainty"] = plot.plot_uncertainty(
        xx, yy, unc_grid, X_tr, y_tr, rng, out / "uncertainty.png", title=label
    )
    figs["profile"] = plot.plot_radial_profile(
        radii,
        {"epistemic": epis_profile, "aleatoric": alea_profile},
        data_radius,
        out / "radial_profile.png",
        title=f"{label}: uncertainty vs distance",
    )
    figs["summary"] = plot.plot_summary(
        xx, yy, grid_pred, unc_grid, X_tr, y_tr, n_classes, rng, out / "summary.png", title=label
    )
    res["figures"] = {k: str(Path(v).relative_to(RESULTS.parent)) for k, v in figs.items()}

    utils.save_json(res, out / "metrics.json")
    utils.write_report(RESULTS)
    # Everything the cross-method comparison figure needs, so it never retrains.
    np.savez_compressed(
        out / "grid_uncertainty.npz",
        xx=xx,
        yy=yy,
        pred=grid_pred,
        X_train=X_tr,
        y_train=y_tr,
        n_classes=n_classes,
        plot_range=np.asarray(rng),
        **unc_grid,
    )
    print(
        f"  accuracy {res['test_accuracy']:.3f} | auc {res['test_auc']:.3f} | "
        f"epistemic OOD auroc far/near {res['epistemic_ood_auroc']:.3f}/"
        f"{res['epistemic_ood_auroc_near']:.3f} | "
        f"epistemic in/far {res['epistemic_in']:.3f}/{res['epistemic_far']:.3f} | "
        f"aleatoric in/far {res['aleatoric_in']:.3f}/{res['aleatoric_far']:.3f}"
    )
    print(f"  -> {out} ({res['runtime_s']}s)")
    return res


# Greedy staged search: each stage varies one group of knobs, the winner is carried
# into the next stage. Ordered by how directly a knob controls the boundary the
# features can express.
SEARCH_STAGES = [
    (
        "boundary",
        [
            dict(shell_inner=3.0, shell_outer=3.1),
            dict(shell_inner=2.5, shell_outer=2.6),
            dict(shell_inner=2.75, shell_outer=2.85),
            dict(shell_inner=3.5, shell_outer=3.6),
            dict(shell_inner=2.5, shell_outer=3.5),
            dict(shell_inner=3.0, shell_outer=3.1, n_boundary=12000),
            dict(shell_inner=3.0, shell_outer=3.1, n_boundary=3000),
        ],
    ),
    (
        "architecture",
        [
            dict(hidden=[64, 64, 64, 1024]),
            dict(hidden=[128, 128, 128, 1024]),
            dict(hidden=[64, 64, 64, 256]),
            dict(hidden=[256, 256, 256, 1024]),
        ],
    ),
    (
        "regularisation",
        [
            dict(weight_decay=1e-4, prior_std=1.0),
            dict(weight_decay=1e-3, prior_std=1.0),
            dict(weight_decay=1e-2, prior_std=1.0),
            dict(weight_decay=1e-4, prior_std=3.0),
            dict(weight_decay=1e-3, prior_std=3.0),
            dict(weight_decay=1e-4, prior_std=10.0),
        ],
    ),
    ("dropout", [dict(dropout=0.0), dict(dropout=0.1), dict(dropout=0.2)]),
]


# The published result set. `--reproduce` runs all of it in order; every entry is a
# list of command-line flags, so any single row can also be run by hand.
TUNED = [
    "--hidden",
    "128",
    "128",
    "128",
    "1024",
    "--shell-inner",
    "3.5",
    "--shell-outer",
    "3.6",
    "--dropout",
    "0.0",
]
SEEDS = (0, 1, 2)
EXPERIMENTS = (
    # GMM. Best configuration is the wider trunk with the default prior under NUTS;
    # baselines get the same trunk and prior so the comparison isolates the
    # boundary class rather than the tuning.
    [
        [
            "--dataset",
            "gmm",
            "--method",
            "bacoun",
            "--inference",
            "hmc",
            *TUNED,
            "--hmc-samples",
            "300",
            "--seed",
            str(s),
            "--tag",
            f"gmm_bacoun_hmc_seed{s}",
        ]
        for s in SEEDS
    ]
    + [
        [
            "--dataset",
            "gmm",
            "--method",
            "nlm",
            "--inference",
            "hmc",
            *TUNED,
            "--hmc-samples",
            "300",
            "--seed",
            str(s),
            "--tag",
            f"gmm_nlm_hmc_seed{s}",
        ]
        for s in SEEDS
    ]
    + [
        [
            "--dataset",
            "gmm",
            "--method",
            "mcdropout",
            "--seed",
            str(s),
            "--tag",
            f"gmm_mcdropout_seed{s}",
        ]
        for s in SEEDS
    ]
    # Moons, at the paper's settings throughout.
    + [
        ["--dataset", "moons", "--method", "bacoun", "--inference", "hmc", "--seed", str(s)]
        for s in SEEDS
    ]
    + [
        ["--dataset", "moons", "--method", "nlm", "--inference", "hmc", "--seed", str(s)]
        for s in SEEDS
    ]
    + [["--dataset", "moons", "--method", "mcdropout", "--seed", str(s)] for s in SEEDS]
    # BBVI on the GMM: where mean-field costs you, and where it does not.
    + [
        [
            "--dataset",
            "gmm",
            "--method",
            "bacoun",
            "--inference",
            "bbvi",
            *TUNED,
            "--tag",
            "gmm_bacoun_bbvi_tunedarch",
        ],
        [
            "--dataset",
            "gmm",
            "--method",
            "bacoun",
            "--inference",
            "bbvi",
            "--tag",
            "gmm_bacoun_bbvi_paperarch",
        ],
    ]
)


def reproduce(base_args):
    """Run every published experiment, then rebuild the comparison figures."""
    for i, flags in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] {' '.join(flags)}", flush=True)
        args = parse_args(flags + ["--device", base_args.device])
        try:
            main(args)
        except Exception as exc:
            print(f"  crash: {type(exc).__name__}: {exc}", flush=True)
    summarize()
    print(f"\nreport: {RESULTS / 'report.md'}")


def grid_search(base_args):
    """Greedy staged search maximising near-ring epistemic AUROC subject to fit.

    Fast settings are used during the search; re-run the winner at full settings.
    """
    import copy

    best = dict(shell_inner=None, shell_outer=None)
    history, run_id = [], 0
    for stage, options in SEARCH_STAGES:
        scored = []
        for option in options:
            cfg = {**best, **option}
            args = copy.deepcopy(base_args)
            for k, v in cfg.items():
                setattr(args, k, v)
            args.tag = f"search_{run_id:02d}_{stage}"
            run_id += 1
            print(f"\n[search] {stage}: {option}")
            try:
                res = main(args)
            except Exception as exc:  # a bad config should not kill the search
                print(f"  crash: {exc}")
                history.append(dict(stage=stage, config=cfg, status="crash"))
                continue
            fit_ok = res["test_accuracy"] >= 0.95 * res.get("bayes_optimal_accuracy", 0.98)
            score = res["epistemic_ood_auroc_near"] if fit_ok else -1.0
            scored.append((score, cfg, res))
            history.append(
                dict(
                    stage=stage,
                    config=cfg,
                    score=score,
                    fit_ok=fit_ok,
                    accuracy=res["test_accuracy"],
                    auroc_near=res["epistemic_ood_auroc_near"],
                    auroc_far=res["epistemic_ood_auroc"],
                    tag=args.tag,
                )
            )
        if scored:
            score, best, _ = max(scored, key=lambda t: t[0])
            print(f"\n[search] best after {stage}: {best} -> near AUROC {score:.3f}\n")

    lines = [
        "# Grid search",
        "",
        f"Objective: near-ring epistemic AUROC at {base_args.near_radius_factor}x the data "
        "radius, subject to accuracy >= 0.95x Bayes-optimal (configs failing the fit "
        "constraint score -1).",
        "",
        "| stage | config | accuracy | AUROC near | AUROC far | fit ok |",
        "|---|---|---|---|---|---|",
    ]
    for h in history:
        if h.get("status") == "crash":
            lines.append(f"| {h['stage']} | `{h['config']}` | crash | | | |")
            continue
        lines.append(
            f"| {h['stage']} | `{h['config']}` | {h['accuracy']:.3f} | "
            f"{h['auroc_near']:.3f} | {h['auroc_far']:.3f} | {h['fit_ok']} |"
        )
    lines += ["", f"**Winner:** `{best}`"]
    path = RESULTS / "gridsearch.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"\n[search] winner: {best}\n  -> {path}")
    return best


def summarize():
    """Build one cross-method comparison figure per dataset from saved grids."""
    order = ["bacoun", "nlm", "mcdropout"]
    made = []
    for dataset in data_mod.DATASETS:
        runs = []
        for method in order:
            # Prefer NUTS runs, then seed-0 runs, so the comparison is like-for-like.
            grids = sorted(
                RESULTS.glob(f"{dataset}_{method}_*/grid_uncertainty.npz"),
                key=lambda p: (
                    0 if "_hmc" in p.parent.name else 1,
                    0 if not p.parent.name.endswith(("seed1", "seed2")) else 1,
                    p.parent.name,
                ),
            )
            if not grids:
                continue
            d = np.load(grids[0])
            if "pred" not in d:  # grid saved by an older version of this script
                print(f"  skipping {grids[0].parent.name}: no decision grid saved, re-run it")
                continue
            runs.append(
                dict(
                    label=grids[0].parent.name.replace(f"{dataset}_", ""),
                    xx=d["xx"],
                    yy=d["yy"],
                    pred=d["pred"],
                    X=d["X_train"],
                    y=d["y_train"],
                    n_classes=int(d["n_classes"]),
                    unc={k: d[k] for k in ("total", "aleatoric", "epistemic")},
                    plot_range=d["plot_range"],
                )
            )
        if not runs:
            continue
        rng = tuple(float(v) for v in runs[0]["plot_range"])
        for key in ("epistemic", "aleatoric"):
            made.append(
                plot.plot_method_comparison(
                    runs,
                    rng,
                    RESULTS / f"summary_{dataset}_{key}.png",
                    key=key,
                    title=f"{dataset}: decision regions and {key} uncertainty by method",
                )
            )
    for path in made:
        print(f"  -> {path}")
    return made


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=list(data_mod.DATASETS), default="gmm")
    p.add_argument(
        "--method",
        choices=["bacoun", "nlm", "mcdropout"],
        default="bacoun",
        help="baselines: 'nlm' drops the boundary class, 'mcdropout' replaces the "
        "Bayesian layer with test-time dropout",
    )
    p.add_argument("--inference", choices=["bbvi", "hmc"], default="bbvi")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--tag", default=None)
    p.add_argument(
        "--reproduce",
        action="store_true",
        help="run every published experiment, then rebuild the figures",
    )
    p.add_argument(
        "--grid-search",
        action="store_true",
        help="run the staged hyperparameter search and write results/gridsearch.md",
    )
    p.add_argument(
        "--summarize",
        action="store_true",
        help="skip training; rebuild the cross-method figures from saved grids",
    )
    p.add_argument("--grid", type=int, default=200, help="resolution of the evaluation grid")
    p.add_argument("--gmm-sigma", type=float, default=None, help="GMM covariance scale")
    p.add_argument("--gmm-n-per-class", type=int, default=None)
    p.add_argument("--n-boundary", type=int, default=None, help="number of boundary points")
    p.add_argument("--shell-inner", type=float, default=None)
    p.add_argument("--shell-outer", type=float, default=None)
    p.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=None,
        help="hidden layer widths; the last one is the feature layer",
    )
    p.add_argument(
        "--far-radius", type=float, default=12.0, help="radius of the far-field OOD ring"
    )
    p.add_argument(
        "--near-radius-factor",
        type=float,
        default=1.2,
        help="radius of the near OOD ring, as a multiple of the training-data radius",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="overrides the per-method default in ARCHITECTURE",
    )
    p.add_argument("--flow-epochs", type=int, default=None)
    p.add_argument("--clf-epochs", type=int, default=500)
    p.add_argument("--bbvi-steps", type=int, default=10_000)
    p.add_argument("--bbvi-lr", type=float, default=1e-2)
    p.add_argument("--posterior-samples", type=int, default=200)
    p.add_argument("--prior-std", type=float, default=1.0, help="std of the N(0, s^2) prior on W")
    p.add_argument("--hmc-samples", type=int, default=1000)
    p.add_argument("--hmc-warmup", type=int, default=100)
    return p.parse_args(argv)


if __name__ == "__main__":
    _args = parse_args()
    if _args.summarize:
        summarize()
    elif _args.reproduce:
        reproduce(_args)
    elif _args.grid_search:
        grid_search(_args)
    else:
        main(_args)
