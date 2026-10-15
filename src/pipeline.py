"""Train the flow and classifier that the explanatory figures need, with a disk cache.

`animate.py`, `features.py` and `mechanism.py` all want the same pieces and all get
re-run while a figure is being adjusted, so trained weights are cached under
results/cache/. The `boundary` and `plain` modes are main.py's pipeline with the
plumbing removed; `flatten` is the ablation described in mechanism.py.
"""

from pathlib import Path

import numpy as np
import torch

import data as data_mod
import main as main_mod
import utils
from nets import Classifier, RealNVP

CACHE = Path(__file__).resolve().parents[1] / "results" / "cache"

# Per-dataset settings, matching the published runs: the GMM uses the tuned trunk
# and shell (see README), the moons uses the paper's defaults.
CONFIG = {
    "gmm": dict(hidden=(128, 128, 128, 1024), dropout=0.0, shell=(3.5, 3.6)),
    "moons": dict(hidden=(64, 64, 64, 1024), dropout=0.1, shell=(3.0, 3.1)),
}
TRAIN = dict(lr=1e-3, weight_decay=1e-4, batch_size=256, clf_epochs=500)

# How the classifier is trained. All three see the same data; two also see the
# boundary points, and differ in what they are asked to do with them.
MODES = {
    "boundary": "K+1 classes: the boundary points are an extra class",
    "plain": "K classes, training data only",
    "flatten": "K classes, plus a penalty pushing the logits flat at the boundary points",
}
CACHE_SUFFIX = {"boundary": "kplus1", "plain": "k", "flatten": "flat"}


def train_flattened(
    X, y, X_boundary, n_classes, hidden, dropout, device, epochs, weight: float = 1.0
):
    """K-class classifier with an extra term that flattens the logits at the boundary.

    The boundary points get no label. They are only asked to make the K real-class
    logits equal, which is cross-entropy against the uniform distribution. If the
    boundary class works because it flattens the margin off the data manifold, this
    should reproduce it without adding a class.
    """
    clf = Classifier(X.shape[1], n_classes, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
    loss_fn = torch.nn.CrossEntropyLoss()
    X = torch.as_tensor(X, device=device)
    y = torch.as_tensor(y, device=device)
    B = torch.as_tensor(X_boundary, device=device)
    bs = TRAIN["batch_size"]
    for _ in range(epochs):
        clf.train()
        perm = torch.randperm(len(X), device=device)
        for i in range(0, len(X), bs):
            idx = perm[i : i + bs]
            b = B[torch.randint(len(B), (len(idx),), device=device)]
            flat = -torch.log_softmax(clf(b), dim=-1).mean()
            loss = loss_fn(clf(X[idx]), y[idx]) + weight * flat
            opt.zero_grad()
            loss.backward()
            opt.step()
    clf.eval()
    with torch.no_grad():
        acc = (clf(X).argmax(1) == y).float().mean().item()
    print(f"    flattened classifier, train acc {acc:.3f}", flush=True)
    return clf


def fit(dataset: str, mode: str = "boundary", seed: int = 0, device=None, refresh=False):
    """Flow + classifier for one dataset, trained one of the three ways in MODES.

    The classifier seed is reset just before training in every mode, so the models
    differ only by what the boundary points are used for -- `plain` does not use
    them at all, and is the NLM baseline.
    """
    device = device or utils.get_device("auto")
    cfg = CONFIG[dataset]
    path = CACHE / f"{dataset}_seed{seed}_{CACHE_SUFFIX[mode]}.pt"

    utils.set_seed(seed)
    ds = data_mod.get_dataset(dataset, seed=seed)
    K, X_tr = ds["K"], ds["X_train"]
    n_classes = K + 1 if mode == "boundary" else K

    cached = None
    if path.exists() and not refresh:
        cached = torch.load(path, map_location=device, weights_only=False)
        if cached["config"] != cfg:  # settings changed since the cache was written
            cached = None

    flow, X_boundary = None, np.zeros((0, X_tr.shape[1]), dtype=np.float32)
    if mode != "plain":
        flow = RealNVP(dim=X_tr.shape[1]).to(device)
        if cached:
            flow.load_state_dict(cached["flow"])
            X_boundary = cached["X_boundary"]
        else:
            flow, _ = main_mod.train_flow(
                X_tr,
                epochs=main_mod.FLOW_EPOCHS[dataset],
                lr=TRAIN["lr"],
                weight_decay=TRAIN["weight_decay"],
                batch_size=TRAIN["batch_size"],
                device=device,
                log_every=0,
            )
            X_boundary = (
                data_mod.boundary_points(flow, ds["n_boundary"], *cfg["shell"]).cpu().numpy()
            )

    utils.set_seed(seed)
    clf = Classifier(X_tr.shape[1], n_classes, hidden=cfg["hidden"], dropout=cfg["dropout"]).to(
        device
    )
    if cached:
        clf.load_state_dict(cached["clf"])
    elif mode == "flatten":
        clf = train_flattened(
            X_tr,
            ds["y_train"],
            X_boundary,
            n_classes,
            cfg["hidden"],
            cfg["dropout"],
            device,
            TRAIN["clf_epochs"],
        )
    else:
        X_all = np.concatenate([X_tr, X_boundary])
        y_all = np.concatenate([ds["y_train"], np.full(len(X_boundary), K, dtype=np.int64)])
        clf, _ = main_mod.train_classifier(
            X_all,
            y_all,
            n_classes,
            epochs=TRAIN["clf_epochs"],
            lr=TRAIN["lr"],
            weight_decay=TRAIN["weight_decay"],
            batch_size=TRAIN["batch_size"],
            dropout=cfg["dropout"],
            hidden=cfg["hidden"],
            device=device,
            log_every=0,
        )
    if not cached:
        CACHE.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                config=cfg,
                clf=clf.state_dict(),
                X_boundary=X_boundary,
                flow=flow.state_dict() if flow is not None else None,
            ),
            path,
        )

    clf.eval()
    return dict(
        ds=ds,
        flow=flow,
        clf=clf,
        X_boundary=X_boundary,
        config=cfg,
        device=device,
        n_classes=n_classes,
        mode=mode,
    )
