"""Shared helpers: device selection, seeding, evaluation grids, run bookkeeping."""

import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def get_device(name: str = "auto") -> torch.device:
    """Pick a device. 'auto' prefers CUDA, then Apple MPS, then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_grid(x_range, y_range, n: int = 200):
    """Regular 2-D grid. Returns (points [n*n, 2], xx [n, n], yy [n, n])."""
    xs = np.linspace(*x_range, n)
    ys = np.linspace(*y_range, n)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    return pts, xx, yy


# Files a result must be reproducible from. Uncommitted output under results/
# does not make a run untraceable, so it is deliberately excluded.
SOURCE_PATHS = ["src", "pyproject.toml", "uv.lock"]


def git_commit() -> str:
    """Short commit hash, with '-dirty' if any *source* file has uncommitted changes."""
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--"] + SOURCE_PATHS, text=True
        ).strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=float)


# --- run bookkeeping ---------------------------------------------------------

# Success criteria, fixed before the experiments were run. Each takes the metrics
# dict of one run and returns (passed, human-readable value).
CRITERIA = {
    "fit: accuracy >= 0.95 x reference": lambda r: (
        r["test_accuracy"] >= 0.95 * r.get("bayes_optimal_accuracy", 0.98),
        f"{r['test_accuracy']:.3f} vs {0.95 * r.get('bayes_optimal_accuracy', 0.98):.3f}",
    ),
    "OOD (far ring): epistemic AUROC >= 0.95": lambda r: (
        r["epistemic_ood_auroc"] >= 0.95,
        f"{r['epistemic_ood_auroc']:.3f}",
    ),
    "OOD (near ring): epistemic AUROC >= 0.95": lambda r: (
        r.get("epistemic_ood_auroc_near", 0.0) >= 0.95,
        f"{r.get('epistemic_ood_auroc_near', float('nan')):.3f}",
    ),
    "epistemic higher far from data": lambda r: (
        r["epistemic_far"] > r["epistemic_in"],
        f"{r['epistemic_far']:.3f} > {r['epistemic_in']:.3f}",
    ),
    "aleatoric higher on the data": lambda r: (
        r["aleatoric_in"] > r["aleatoric_far"],
        f"{r['aleatoric_in']:.3f} > {r['aleatoric_far']:.3f}",
    ),
}


def check_criteria(res: dict) -> dict:
    return {name: fn(res) for name, fn in CRITERIA.items()}


def write_report(results_dir) -> str:
    """Rebuild results/report.md from every metrics.json found. One row per run."""
    results_dir = Path(results_dir)
    runs = []
    for path in sorted(results_dir.glob("*/metrics.json")):
        try:
            runs.append(json.load(open(path)))
        except Exception:
            runs.append({"tag": path.parent.name, "status": "crash"})

    cols = [
        "tag",
        "status",
        "accuracy",
        "AUC",
        "OOD AUROC far",
        "OOD AUROC near",
        "rise",
        "epistemic in/far",
        "aleatoric in/far",
        "commit",
        "s",
    ]
    lines = [
        "# Run report",
        "",
        "Regenerated from `results/*/metrics.json` at the end of every run.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for r in runs:
        if r.get("status") == "crash":
            lines.append(f"| {r['tag']} | crash |" + " |" * (len(cols) - 2))
            continue
        checks = check_criteria(r)
        status = "pass" if all(ok for ok, _ in checks.values()) else "partial"
        lines.append(
            f"| {r['tag']} | {status} | {r['test_accuracy']:.3f} | {r['test_auc']:.3f} | "
            f"{r['epistemic_ood_auroc']:.3f} | {r.get('epistemic_ood_auroc_near', float('nan')):.3f} | "
            f"{r.get('epistemic_rise_ratio', float('nan')):.3f} | "
            f"{r['epistemic_in']:.3f} / {r['epistemic_far']:.3f} | "
            f"{r['aleatoric_in']:.3f} / {r['aleatoric_far']:.3f} | {r['commit']} | {r['runtime_s']} |"
        )

    lines += ["", "## Success criteria per run", ""]
    for r in runs:
        if r.get("status") == "crash":
            continue
        lines.append(f"**{r['tag']}**")
        for name, (ok, value) in check_criteria(r).items():
            lines.append(f"- [{'x' if ok else ' '}] {name} — {value}")
        lines.append("")

    path = results_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return str(path)
