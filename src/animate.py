"""The whole pipeline as one animation: data -> flow -> shell -> boundary points ->
K+1 classifier -> Bayesian last layer -> uncertainty split.

    uv run python src/animate.py --dataset gmm

Every step is the real trained model, not an illustration: the morphing frames
interpolate between the outputs of the flow's coupling blocks, so what you see is
the transform the network actually learned. Writes results/workflow_<dataset>.gif.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import PillowWriter
from matplotlib.patches import Circle, Wedge

import bayes
import data as data_mod
import metrics as metrics_mod
import pipeline
import plot
import utils

RESULTS = Path(__file__).resolve().parents[1] / "results"
LATENT_RANGE = (-4.5, 4.5)
BOUNDARY_COLOR = "0.25"


def smoothstep(u: float) -> float:
    """Ease-in-out, so the morphs start and end at rest."""
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3 - 2 * u)


def lerp(a, b, u):
    return a + (b - a) * u


@torch.no_grad()
def block_states(flow, points, forward=True):
    """Point positions after each coupling block: [n_blocks + 1, N, 2]."""
    x = torch.as_tensor(np.asarray(points, dtype=np.float32), device=next(flow.parameters()).device)
    states = [x]
    blocks = flow.blocks if forward else reversed(flow.blocks)
    for block in blocks:
        x = block(x)[0] if forward else block.inverse(x)
        states.append(x)
    return np.stack([s.cpu().numpy() for s in states])


def morph(states, t: float):
    """Position at time t in [0, n_blocks], interpolating between block outputs."""
    i = int(np.clip(int(t), 0, len(states) - 2))
    return lerp(states[i], states[i + 1], smoothstep(t - i))


def scatter_classes(ax, X, y, alpha=0.55, s=5):
    for k in np.unique(y):
        ax.scatter(
            *X[y == k].T,
            s=s,
            alpha=alpha,
            linewidths=0,
            color=plot.CLASS_COLORS[k % len(plot.CLASS_COLORS)],
        )


def decision_cmap(n_classes, boundary_class=True):
    """Colours for a decision map. The boundary class, when present, is the last one."""
    if boundary_class:
        return matplotlib.colors.ListedColormap(plot.CLASS_COLORS[: n_classes - 1] + ["dimgray"])
    return matplotlib.colors.ListedColormap(plot.CLASS_COLORS[:n_classes])


def build(dataset, seed, grid_n, n_points, posterior_samples, bbvi_steps):
    """Everything the animation needs, computed once."""
    art = pipeline.fit(dataset, mode="boundary", seed=seed)
    ds, flow, clf, device = art["ds"], art["flow"], art["clf"], art["device"]
    K, rng = ds["K"], ds["plot_range"]
    X_tr, y_tr = ds["X_train"], ds["y_train"]

    rs = np.random.default_rng(seed)
    keep = rs.choice(len(X_tr), size=min(n_points, len(X_tr)), replace=False)
    X_show, y_show = X_tr[keep], y_tr[keep]
    keep_b = rs.choice(
        len(art["X_boundary"]), size=min(n_points, len(art["X_boundary"])), replace=False
    )
    X_bnd = art["X_boundary"][keep_b]

    # The two morphs: data forward to the latent, and the shell back to data space.
    to_latent = block_states(flow, X_show, forward=True)
    z_shell = data_mod.sample_latent_shell(len(X_bnd), 2, *art["config"]["shell"], device)
    from_latent = block_states(flow, z_shell.cpu().numpy(), forward=False)

    grid, xx, yy = utils.make_grid(rng, rng, grid_n)
    with torch.no_grad():
        grid_t = torch.as_tensor(grid, device=device)
        det_pred = clf(grid_t).argmax(1).cpu().numpy()
        phi_grid = clf.features(grid_t).cpu()
        phi_tr = clf.features(torch.as_tensor(X_tr, device=device)).cpu()

    print("  fitting the Bayesian last layer for the animation")
    W, _ = bayes.fit_bbvi(
        phi_tr,
        torch.as_tensor(y_tr),
        K,
        n_steps=bbvi_steps,
        n_samples=posterior_samples,
        seed=seed,
        log_every=0,
    )
    probs = bayes.posterior_predictive(W, phi_grid)  # [S, N, K]
    unc = metrics_mod.decompose_uncertainty(probs)
    sample_pred = probs.argmax(-1)  # one decision map per sample

    return dict(
        rng=rng,
        K=K,
        X=X_show,
        y=y_show,
        X_bnd=X_bnd,
        shell=art["config"]["shell"],
        to_latent=to_latent,
        from_latent=from_latent,
        xx=xx,
        yy=yy,
        det_pred=det_pred,
        sample_pred=sample_pred,
        unc=unc,
    )


def render(a, path, fps, dpi):
    rng, xx, yy, K = a["rng"], a["xx"], a["yy"], a["K"]
    n_blocks = len(a["to_latent"]) - 1
    shell_in, shell_out = a["shell"]
    fig, ax = plt.subplots(figsize=(6.4, 6.9))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.86, bottom=0.04)
    writer = PillowWriter(fps=fps)

    def frame(title, sub, limits=rng):
        ax.set(xlim=limits, ylim=limits, xticks=[], yticks=[], aspect="equal")
        ax.set_title(f"{title}\n{sub}", fontsize=11, linespacing=1.5)
        writer.grab_frame()

    def heat(values, alpha=1.0):
        ax.pcolormesh(
            xx,
            yy,
            values.reshape(xx.shape),
            cmap="viridis",
            shading="auto",
            alpha=alpha,
            vmin=0.0,
            vmax=float(values.max()),
        )

    def regions(pred, n_classes, alpha, boundary_class=True):
        ax.contourf(
            xx,
            yy,
            pred.reshape(xx.shape),
            levels=np.arange(n_classes + 1) - 0.5,
            cmap=decision_cmap(n_classes, boundary_class),
            alpha=alpha,
        )

    with writer.saving(fig, str(path), dpi):
        # 1. the data
        for _ in range(18):
            ax.clear()
            scatter_classes(ax, a["X"], a["y"])
            frame("1. The data", f"{K} classes, and nothing else to go on")

        # 2. the flow maps the data to a standard normal
        for i in range(36):
            u = smoothstep(i / 35)
            ax.clear()
            scatter_classes(ax, morph(a["to_latent"], u * n_blocks), a["y"])
            frame(
                "2. Fit a RealNVP to the inputs",
                "it learns a map from the data to a standard normal",
                limits=tuple(lerp(np.array(rng), np.array(LATENT_RANGE), u)),
            )

        # 3. in the latent, "outside the data" is just "far from the origin"
        for i in range(14):
            ax.clear()
            scatter_classes(ax, a["to_latent"][-1], a["y"])
            for r in (1, 2, 3):
                ax.add_patch(Circle((0, 0), r, fill=False, color="0.6", lw=0.8, ls=":"))
            frame(
                "3. In the latent space",
                "the data is a unit Gaussian: outside it means " "far from the origin",
                limits=LATENT_RANGE,
            )

        # 4. sample the shell
        for i in range(20):
            u = smoothstep(i / 19)
            ax.clear()
            scatter_classes(ax, a["to_latent"][-1], a["y"], alpha=0.3)
            ax.add_patch(
                Wedge(
                    (0, 0), shell_out, 0, 360, width=shell_out - shell_in, color="red", alpha=0.25
                )
            )
            shown = int(u * len(a["from_latent"][0]))
            ax.scatter(
                *a["from_latent"][0][:shown].T, s=5, alpha=0.6, linewidths=0, color=BOUNDARY_COLOR
            )
            frame(
                "4. Sample a shell around it",
                f"radius {shell_in} to {shell_out}: just outside the mass, in every direction",
                limits=LATENT_RANGE,
            )

        # 5. push the shell back through the flow
        for i in range(36):
            u = smoothstep(i / 35)
            ax.clear()
            scatter_classes(ax, morph(a["to_latent"][::-1], u * n_blocks), a["y"])
            ax.scatter(
                *morph(a["from_latent"], u * n_blocks).T,
                s=5,
                alpha=0.6,
                linewidths=0,
                color=BOUNDARY_COLOR,
            )
            frame(
                "5. Run the flow backwards",
                "the shell becomes a wrapper around the data: these are the boundary points",
                limits=tuple(lerp(np.array(LATENT_RANGE), np.array(rng), u)),
            )

        # 6. train the K+1 classifier
        for i in range(22):
            u = smoothstep(i / 21)
            ax.clear()
            regions(a["det_pred"], K + 1, 0.4 * u)
            scatter_classes(ax, a["X"], a["y"])
            ax.scatter(*a["X_bnd"].T, s=5, alpha=0.6, linewidths=0, color=BOUNDARY_COLOR)
            frame(
                "6. Train a classifier on K+1 classes",
                "the boundary points are the extra class (grey)",
            )

        # 7. drop the boundary points, put a posterior on the last layer
        for i in range(10):
            u = smoothstep(i / 9)
            ax.clear()
            regions(a["det_pred"], K + 1, 0.4 * (1 - u))
            scatter_classes(ax, a["X"], a["y"])
            ax.scatter(*a["X_bnd"].T, s=5, alpha=0.6 * (1 - u), linewidths=0, color=BOUNDARY_COLOR)
            frame(
                "7. Throw the boundary points away",
                "they have done their job: they shaped the features",
            )
        for i in range(24):
            ax.clear()
            regions(a["sample_pred"][i % len(a["sample_pred"])], K, 0.4, boundary_class=False)
            scatter_classes(ax, a["X"], a["y"])
            frame(
                "8. Sample the last layer's posterior",
                "each frame is one posterior sample; they agree on the data, " "not away from it",
            )

        # 8. the uncertainty split
        for key, name, note in [
            (
                "aleatoric",
                "9. Aleatoric uncertainty",
                "noise in the labels: highest where the classes overlap",
            ),
            (
                "epistemic",
                "10. Epistemic uncertainty",
                "disagreement between posterior samples: highest off the data",
            ),
        ]:
            for i in range(34):
                ax.clear()
                heat(a["unc"][key], alpha=min(1.0, i / 8))
                scatter_classes(ax, a["X"], a["y"], alpha=0.25, s=3)
                frame(name, note)
    plt.close(fig)
    return str(path)


def main(args):
    utils.set_seed(args.seed)
    a = build(
        args.dataset, args.seed, args.grid, args.points, args.posterior_samples, args.bbvi_steps
    )
    path = RESULTS / f"workflow_{args.dataset}.gif"
    render(a, path, args.fps, args.dpi)
    print(f"  -> {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=list(data_mod.DATASETS), default="gmm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid", type=int, default=120, help="resolution of the heatmap grid")
    p.add_argument("--points", type=int, default=1200, help="points shown in the morphs")
    p.add_argument("--posterior-samples", type=int, default=200)
    p.add_argument("--bbvi-steps", type=int, default=10_000)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--dpi", type=int, default=90)
    main(p.parse_args())
