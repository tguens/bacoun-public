"""Plotting. Pure matplotlib; every function takes numpy arrays and saves a figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASS_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _scatter_classes(ax, X, y, s=4, alpha=0.5):
    for k in np.unique(y):
        ax.scatter(
            *X[y == k].T,
            s=s,
            alpha=alpha,
            color=CLASS_COLORS[k % len(CLASS_COLORS)],
            label=f"class {k}",
            linewidths=0,
        )


def plot_data(X, y, X_boundary, rng, path, title="Data and generated boundary points"):
    fig, ax = plt.subplots(figsize=(5, 5))
    _scatter_classes(ax, X, y)
    ax.scatter(*X_boundary.T, s=4, alpha=0.35, color="black", label="boundary (OOD)", linewidths=0)
    ax.set(xlim=rng, ylim=rng, title=title)
    ax.legend(markerscale=3, fontsize=8, loc="upper right")
    return _save(fig, path)


def plot_flow_samples(X, X_flow, rng, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(*X.T, s=4, alpha=0.4, color="tab:blue", label="data", linewidths=0)
    ax.scatter(*X_flow.T, s=4, alpha=0.4, color="tab:red", label="flow samples", linewidths=0)
    ax.set(xlim=rng, ylim=rng, title="RealNVP samples vs data")
    ax.legend(markerscale=3, fontsize=8)
    return _save(fig, path)


def plot_decision_boundaries(
    xx, yy, pred, X, y, n_classes, rng, path, title="K+1 classifier decision regions"
):
    """`pred` is the argmax class on the grid; class n_classes-1 is the boundary class."""
    fig, ax = plt.subplots(figsize=(5, 5))
    cmap = matplotlib.colors.ListedColormap(CLASS_COLORS[: n_classes - 1] + ["dimgray"])
    ax.contourf(
        xx, yy, pred.reshape(xx.shape), levels=np.arange(n_classes + 1) - 0.5, cmap=cmap, alpha=0.35
    )
    _scatter_classes(ax, X, y, s=3, alpha=0.5)
    ax.set(xlim=rng, ylim=rng, title=title)
    return _save(fig, path)


def plot_uncertainty(xx, yy, unc, X, y, rng, path, title=""):
    """Heatmaps of total / aleatoric / epistemic uncertainty over the input grid."""
    keys = ["total", "aleatoric", "epistemic"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, key in zip(axes, keys):
        z = unc[key].reshape(xx.shape)
        im = ax.pcolormesh(xx, yy, z, cmap="viridis", shading="auto")
        fig.colorbar(im, ax=ax, fraction=0.046)
        _scatter_classes(ax, X, y, s=2, alpha=0.25)
        ax.set(xlim=rng, ylim=rng, title=f"{key.capitalize()} uncertainty")
    fig.suptitle(title)
    return _save(fig, path)


def plot_loss(losses, path, title="ELBO loss"):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(losses, lw=0.6)
    ax.set(xlabel="step", ylabel="loss", title=title, yscale="symlog")
    return _save(fig, path)


def plot_summary(xx, yy, pred, unc, X, y, n_classes, rng, path, title=""):
    """The paper's Figure 2 for one run: decision regions plus the three uncertainties."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    cmap = matplotlib.colors.ListedColormap(
        CLASS_COLORS[: n_classes - 1] + ["dimgray"]
        if n_classes > len(np.unique(y))
        else CLASS_COLORS[:n_classes]
    )
    ax = axes[0, 0]
    ax.contourf(
        xx, yy, pred.reshape(xx.shape), levels=np.arange(n_classes + 1) - 0.5, cmap=cmap, alpha=0.35
    )
    _scatter_classes(ax, X, y, s=3, alpha=0.5)
    ax.set(xlim=rng, ylim=rng, title="(a) Decision regions")

    for ax, key, label in zip(
        axes.ravel()[1:],
        ["total", "aleatoric", "epistemic"],
        ["(b) Total", "(c) Aleatoric", "(d) Epistemic"],
    ):
        im = ax.pcolormesh(xx, yy, unc[key].reshape(xx.shape), cmap="viridis", shading="auto")
        fig.colorbar(im, ax=ax, fraction=0.046)
        _scatter_classes(ax, X, y, s=2, alpha=0.2)
        ax.set(xlim=rng, ylim=rng, title=f"{label} uncertainty")
    fig.suptitle(title, fontsize=13)
    return _save(fig, path)


def plot_method_comparison(runs, rng, path, key="epistemic", title=""):
    """One column per method: decision regions on top, the chosen uncertainty below.

    `runs` is a list of dicts with keys: label, xx, yy, pred, unc, X, y, n_classes.
    """
    n = len(runs)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9.5), squeeze=False)
    for col, r in enumerate(runs):
        n_classes = r["n_classes"]
        cmap = matplotlib.colors.ListedColormap(
            CLASS_COLORS[: n_classes - 1] + ["dimgray"]
            if n_classes > len(np.unique(r["y"]))
            else CLASS_COLORS[:n_classes]
        )
        ax = axes[0][col]
        ax.contourf(
            r["xx"],
            r["yy"],
            r["pred"].reshape(r["xx"].shape),
            levels=np.arange(n_classes + 1) - 0.5,
            cmap=cmap,
            alpha=0.35,
        )
        _scatter_classes(ax, r["X"], r["y"], s=3, alpha=0.5)
        ax.set(xlim=rng, ylim=rng, title=f"{r['label']}\ndecision regions")

        ax = axes[1][col]
        im = ax.pcolormesh(
            r["xx"], r["yy"], r["unc"][key].reshape(r["xx"].shape), cmap="viridis", shading="auto"
        )
        fig.colorbar(im, ax=ax, fraction=0.046)
        _scatter_classes(ax, r["X"], r["y"], s=2, alpha=0.2)
        ax.set(xlim=rng, ylim=rng, title=f"{key} uncertainty")
    fig.suptitle(title, fontsize=14)
    return _save(fig, path)


def plot_radial_profile(radii, curves, data_radius, path, title="Uncertainty vs distance"):
    """Mean uncertainty against distance from the data centre; the data edge is marked."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, values in curves.items():
        ax.plot(radii, values, label=label, lw=2)
    ax.axvline(data_radius, color="k", ls="--", lw=1, label="edge of training data")
    ax.set(xlabel="distance from data centre", ylabel="mean uncertainty", title=title)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_feature_effect(panels, rng, path, title=""):
    """One row per model, for the same architecture trained with and without boundary points.

    Columns: where inputs land in feature space, how much the last layer's posterior
    disagrees there, how far apart the real-class logits are, and the two against each
    other as a function of distance. `panels` come from features.py.
    """
    fig, axes = plt.subplots(
        len(panels), 4, figsize=(21, 5.2 * len(panels)), squeeze=False, constrained_layout=True
    )
    for row, p in enumerate(panels):
        ax = axes[row][0]
        if len(p["pca"].get("boundary", [])):
            ax.scatter(
                *p["pca"]["boundary"].T,
                s=4,
                alpha=0.3,
                color="dimgray",
                label="boundary class",
                linewidths=0,
            )
        ax.scatter(
            *p["pca"]["train"].T,
            s=4,
            alpha=0.3,
            color="tab:blue",
            label="training data",
            linewidths=0,
        )
        for key, color in [("near", "tab:orange"), ("far", "tab:red")]:
            ax.scatter(
                *p["pca"][key].T, s=8, alpha=0.6, color=color, label=f"{key} ring", linewidths=0
            )
        ax.set(
            xlabel="PC 1",
            ylabel="PC 2",
            title="(a) features, top two principal directions of the data",
        )
        ax.text(
            -0.22,
            0.5,
            p["label"],
            transform=ax.transAxes,
            rotation=90,
            fontsize=13,
            va="center",
            ha="center",
            fontweight="bold",
        )
        ax.legend(markerscale=3, fontsize=8)

        for col, (values, cmap, label) in enumerate(
            [
                (p["log_variance"], "magma", "(b) log10 logit variance v(x)"),
                (p["margin"], "cividis", "(c) gap between the top two real-class logits"),
            ],
            1,
        ):
            ax = axes[row][col]
            im = ax.pcolormesh(
                p["xx"], p["yy"], values.reshape(p["xx"].shape), cmap=cmap, shading="auto"
            )
            fig.colorbar(im, ax=ax, fraction=0.046)
            _scatter_classes(ax, p["X"], p["y"], s=2, alpha=0.2)
            ax.add_patch(
                plt.Circle(p["center"], p["data_radius"], fill=False, color="w", lw=1, ls="--")
            )
            ax.set(xlim=rng, ylim=rng, title=label)

        ax = axes[row][3]
        ax.plot(p["radii"], p["margin_profile"], color="tab:purple", lw=2, label="mean logit gap")
        ax.plot(
            p["radii"], p["sd_profile"], color="tab:green", lw=2, label="posterior sd of that gap"
        )
        ax.axvline(p["data_radius"], color="k", ls="--", lw=1, label="edge of the data")
        ax.set(
            xlabel="distance from the data centre",
            ylabel="logits",
            yscale="log",
            title="(d) how close the posterior comes to flipping the prediction",
        )
        ax.legend(fontsize=8)
    fig.suptitle(
        f"{title}\ndashed line: edge of the training data. "
        "The prediction stops being pinned down where the two curves in (d) meet.",
        fontsize=13,
    )
    return _save(fig, path)


def plot_posterior_samples(panels, rng, path, n_samples=5, title=""):
    """One row per model: individual posterior predictions, then the epistemic map.

    Each of the first `n_samples` columns is the class prediction under a single
    draw from the last layer's posterior. Where the draws disagree, the model is
    uncertain; the last column is that disagreement measured as mutual information.
    """
    fig, axes = plt.subplots(
        len(panels),
        n_samples + 1,
        squeeze=False,
        figsize=(3.1 * (n_samples + 1), 3.4 * len(panels)),
        constrained_layout=True,
    )
    for row, p in enumerate(panels):
        cmap = matplotlib.colors.ListedColormap(CLASS_COLORS[: p["K"]])
        for col in range(n_samples):
            ax = axes[row][col]
            pred = p["sample_pred"][col * (len(p["sample_pred"]) // n_samples)]
            ax.pcolormesh(
                p["xx"],
                p["yy"],
                pred.reshape(p["xx"].shape),
                cmap=cmap,
                vmin=-0.5,
                vmax=p["K"] - 0.5,
                shading="auto",
                alpha=0.45,
            )
            _scatter_classes(ax, p["X"], p["y"], s=2, alpha=0.35)
            ax.set(
                xlim=rng,
                ylim=rng,
                xticks=[],
                yticks=[],
                title=f"posterior sample {col + 1}" if row == 0 else "",
            )
        ax = axes[row][n_samples]
        im = ax.pcolormesh(
            p["xx"], p["yy"], p["epistemic"].reshape(p["xx"].shape), cmap="viridis", shading="auto"
        )
        fig.colorbar(im, ax=ax, fraction=0.046)
        _scatter_classes(ax, p["X"], p["y"], s=2, alpha=0.25)
        ax.set(
            xlim=rng,
            ylim=rng,
            xticks=[],
            yticks=[],
            title=f"epistemic\nOOD AUROC {p['auroc']:.3f} far, {p['auroc_near']:.3f} near",
        )
        name, detail = (p["label"].split(":") + [""])[:2]
        axes[row][0].set_ylabel(
            f"{name}\n{detail.strip()}\naccuracy {p['accuracy']:.3f}", fontsize=9
        )
    fig.suptitle(
        f"{title}\nEach panel is one draw from the last layer's posterior. "
        "Where the draws differ, the model is uncertain.",
        fontsize=13,
    )
    return _save(fig, path)


def plot_margin_vs_spread(panels, data_radius, path, title=""):
    """Top: the top-two logit gap and its posterior spread against distance.
    Bottom: the epistemic uncertainty that results. Note the y scales differ by panel;
    what matters is whether the band reaches zero, not how wide it is."""
    fig, axes = plt.subplots(
        2,
        len(panels),
        figsize=(5.2 * len(panels), 8),
        squeeze=False,
        constrained_layout=True,
        sharex=True,
    )
    for col, p in enumerate(panels):
        ax = axes[0][col]
        lo, hi = p["gap_mean"] - 2 * p["gap_sd"], p["gap_mean"] + 2 * p["gap_sd"]
        ax.fill_between(
            p["radii"], lo, hi, color="tab:green", alpha=0.3, label="+/- 2 posterior sd"
        )
        ax.plot(p["radii"], p["gap_mean"], color="tab:purple", lw=2, label="mean logit gap")
        ax.axhline(0, color="k", lw=1)
        ax.axvline(data_radius, color="k", ls="--", lw=1, label="edge of the data")
        ax.set(title=p["label"], ylabel="gap between the top two logits")
        ax.legend(fontsize=8)

        ax = axes[1][col]
        ax.plot(p["radii"], p["epistemic_ray"], color="tab:red", lw=2)
        ax.axvline(data_radius, color="k", ls="--", lw=1)
        ax.set(
            xlabel="distance from the data centre",
            ylabel="epistemic uncertainty",
            title=f"OOD AUROC {p['auroc']:.3f} far, {p['auroc_near']:.3f} near",
        )
    for ax in axes[1]:
        ax.set_ylim(0, max(p["epistemic_ray"].max() for p in panels) * 1.05)
    fig.suptitle(
        f"{title}\n(y scales differ by panel: what matters is whether the band "
        "reaches zero, not how wide it is)",
        fontsize=13,
    )
    return _save(fig, path)
