"""Generate visualization images for README.md using SciencePlots."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import scienceplots  # type: ignore[import-not-found]  # noqa: F401
import torch

import pytorch_scheduler as ps
from pytorch_scheduler import WarmupScheduler

warnings.filterwarnings("ignore")

TOTAL_STEPS = 10_000
ASSETS = "assets"
STYLE = ["science", "nature"]

# 4 linestyles cycled to distinguish overlapping curves
LINESTYLES = ["-", "--", "-.", ":"]


def make_opt(lr: float = 0.1) -> torch.optim.SGD:
    return torch.optim.SGD([torch.randn(2, requires_grad=True)], lr=lr)


def _collect_lrs(scheduler, total_steps: int) -> tuple[list[int], list[float]]:
    """Step a scheduler and collect LRs."""
    steps, lrs = [], []
    for step in range(total_steps):
        lrs.append(scheduler.get_last_lr()[0])
        steps.append(step)
        scheduler.step()
    return steps, lrs


# ── 1. All Schedulers Overview ──────────────────────────────────────────
def plot_all_schedulers() -> None:
    configs = [
        ("Rex", ps.RexScheduler(make_opt(), total_steps=TOTAL_STEPS)),
        ("InverseSqrt", ps.InverseSqrtScheduler(make_opt(), warmup_steps=1000)),
        (
            "LinearDecay",
            ps.LinearDecayScheduler(make_opt(), total_steps=TOTAL_STEPS, warmup_steps=500),
        ),
        (
            "Trapezoidal",
            ps.TrapezoidalScheduler(
                make_opt(),
                total_steps=TOTAL_STEPS,
                warmup_steps=500,
                decay_steps=3000,
            ),
        ),
        (
            "WSD",
            ps.WSDScheduler(
                make_opt(),
                total_steps=TOTAL_STEPS,
                warmup_steps=500,
                stable_steps=5000,
            ),
        ),
        ("K-Decay", ps.KDecayScheduler(make_opt(), total_steps=TOTAL_STEPS, k=3.0)),
        ("TanhDecay", ps.TanhDecayScheduler(make_opt(), total_steps=TOTAL_STEPS, steepness=3.0)),
        ("SlantedTri", ps.SlantedTriangularScheduler(make_opt(), total_steps=TOTAL_STEPS)),
        (
            "FlatCosine",
            ps.FlatCosineScheduler(make_opt(), total_steps=TOTAL_STEPS, flat_fraction=0.7),
        ),
        ("Polynomial", ps.PolynomialScheduler(make_opt(), total_steps=TOTAL_STEPS, power=2.0)),
        (
            "PowerDecay",
            ps.PowerDecayScheduler(
                make_opt(),
                total_steps=TOTAL_STEPS,
                warmup_steps=500,
                alpha=0.5,
            ),
        ),
        (
            "HyperbolicLR",
            ps.HyperbolicLRScheduler(make_opt(), total_steps=TOTAL_STEPS, upper_bound=25000),
        ),
        (
            "ExpHyperbolicLR",
            ps.ExpHyperbolicLRScheduler(make_opt(), total_steps=TOTAL_STEPS, upper_bound=25000),
        ),
    ]

    cmap = plt.get_cmap("tab20")
    n = len(configs)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        for i, (name, sched) in enumerate(configs):
            steps, lrs = _collect_lrs(sched, TOTAL_STEPS)
            ax.plot(
                steps,
                lrs,
                label=name,
                linewidth=1.0,
                color=cmap(i / n),
                linestyle=LINESTYLES[i % len(LINESTYLES)],
            )
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("All Schedulers Comparison")
        ax.legend(fontsize=5.5, ncol=3, loc="upper right")
        fig.tight_layout()
        fig.savefig(f"{ASSETS}/all_schedulers.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {ASSETS}/all_schedulers.png")


# ── 2. Cosine Annealing Warm Restarts ───────────────────────────────────
def plot_cosine_restarts() -> None:
    configs = [
        (
            "gamma=1.0 (no decay)",
            ps.CosineAnnealingWarmupRestarts(
                make_opt(),
                first_cycle_steps=2000,
                warmup_steps=200,
                max_lr=0.1,
                min_lr=0.001,
                gamma=1.0,
            ),
        ),
        (
            "gamma=0.8 (max-LR decay)",
            ps.CosineAnnealingWarmupRestarts(
                make_opt(),
                first_cycle_steps=2000,
                warmup_steps=200,
                max_lr=0.1,
                min_lr=0.001,
                gamma=0.8,
            ),
        ),
        (
            "gamma=0.8, cycle x1.5",
            ps.CosineAnnealingWarmupRestarts(
                make_opt(),
                first_cycle_steps=1500,
                warmup_steps=150,
                max_lr=0.1,
                min_lr=0.001,
                gamma=0.8,
                cycle_mult=1.5,
            ),
        ),
    ]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
        for i, (name, sched) in enumerate(configs):
            steps, lrs = _collect_lrs(sched, TOTAL_STEPS)
            ax.plot(
                steps,
                lrs,
                label=name,
                linewidth=1.0,
                linestyle=LINESTYLES[i % len(LINESTYLES)],
            )
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Cosine Annealing with Warm Restarts")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(f"{ASSETS}/cosine_restarts.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {ASSETS}/cosine_restarts.png")


# ── 3. Warmup Composition ──────────────────────────────────────────────
def plot_warmup_types() -> None:
    base_kwargs: dict = {"total_steps": TOTAL_STEPS, "k": 2.0}
    configs = [
        ("No warmup", ps.KDecayScheduler(make_opt(), **base_kwargs)),
        (
            "Linear warmup",
            WarmupScheduler(
                make_opt(),
                ps.KDecayScheduler(make_opt(), **base_kwargs),
                warmup_steps=1000,
                warmup_type="linear",
            ),
        ),
        (
            "Cosine warmup",
            WarmupScheduler(
                make_opt(),
                ps.KDecayScheduler(make_opt(), **base_kwargs),
                warmup_steps=1000,
                warmup_type="cosine",
            ),
        ),
        (
            "Exponential warmup",
            WarmupScheduler(
                make_opt(),
                ps.KDecayScheduler(make_opt(), **base_kwargs),
                warmup_steps=1000,
                warmup_type="exponential",
            ),
        ),
    ]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
        for name, sched in configs:
            steps, lrs = _collect_lrs(sched, TOTAL_STEPS)
            ax.plot(steps, lrs, label=name, linewidth=1.0)
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Warmup Composition (KDecay base)")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(f"{ASSETS}/warmup_types.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {ASSETS}/warmup_types.png")


# ── 4. Visualization example (matches README code snippet) ──────────────
def plot_viz_example() -> None:
    from pytorch_scheduler.visualization import compare_schedules

    fig = compare_schedules(
        {
            "Rex": ps.RexScheduler(make_opt(), total_steps=TOTAL_STEPS),
            "CosineAnnealing": ps.CosineAnnealingWarmupRestarts(
                make_opt(),
                first_cycle_steps=2000,
                warmup_steps=200,
                max_lr=0.1,
                min_lr=0.001,
                gamma=0.9,
            ),
            "WSD": ps.WSDScheduler(
                make_opt(),
                total_steps=TOTAL_STEPS,
                warmup_steps=500,
                stable_steps=5000,
            ),
        },
        total_steps=TOTAL_STEPS,
    )
    fig.savefig(f"{ASSETS}/viz_example.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {ASSETS}/viz_example.png")


if __name__ == "__main__":
    plot_all_schedulers()
    plot_cosine_restarts()
    plot_warmup_types()
    plot_viz_example()
    print("Done!")
