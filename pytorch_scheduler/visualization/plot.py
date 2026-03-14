from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from matplotlib.figure import Figure
    from torch.optim.lr_scheduler import LRScheduler

# Default style applied to all plots when SciencePlots is available.
_STYLE = ["science", "nature"]


def _import_matplotlib():
    """Lazy import matplotlib, raising helpful error if not installed."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. Install it with: pip install pytorch-scheduler[viz]"
        ) from None


@contextmanager
def _style_context() -> Generator[None]:
    """Apply SciencePlots style if available, otherwise fall back to default."""
    plt = _import_matplotlib()
    try:
        import scienceplots  # type: ignore[import-not-found]  # noqa: F401

        with plt.style.context(_STYLE):
            yield
    except ImportError:
        yield


def plot_schedule(
    scheduler: LRScheduler,
    total_steps: int,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 4),
    ylabel: str = "Learning Rate",
    xlabel: str = "Step",
) -> Figure:
    """Plot the learning rate schedule for a given scheduler.

    Uses SciencePlots ``['science', 'nature']`` style when available.

    Note: This will advance the scheduler by `total_steps` calls to step().
    Create a fresh scheduler instance for this function.

    Args:
        scheduler: An LRScheduler instance.
        total_steps: Number of steps to simulate.
        title: Plot title (defaults to class name).
        figsize: Figure size.
        ylabel: Y-axis label.
        xlabel: X-axis label.

    Returns:
        The matplotlib Figure.
    """
    plt = _import_matplotlib()

    lrs = []
    steps = []
    for step in range(total_steps):
        lrs.append(scheduler.get_last_lr()[0])
        steps.append(step)
        scheduler.step()

    with _style_context():
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(steps, lrs, linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title or type(scheduler).__name__)
        fig.tight_layout()
    return fig


def compare_schedules(
    schedulers: Mapping[str, LRScheduler],
    total_steps: int,
    title: str = "LR Schedule Comparison",
    figsize: tuple[float, float] = (12, 5),
    ylabel: str = "Learning Rate",
    xlabel: str = "Step",
) -> Figure:
    """Compare multiple LR schedules on the same plot.

    Uses SciencePlots ``['science', 'nature']`` style when available.

    Note: This will advance each scheduler by `total_steps` calls to step().
    Create fresh scheduler instances.

    Args:
        schedulers: Dict of {label: scheduler_instance}.
        total_steps: Number of steps to simulate.
        title: Plot title.
        figsize: Figure size.
        ylabel: Y-axis label.
        xlabel: X-axis label.

    Returns:
        The matplotlib Figure.
    """
    plt = _import_matplotlib()

    all_data: list[tuple[str, list[int], list[float]]] = []
    for name, scheduler in schedulers.items():
        lrs = []
        steps = []
        for step in range(total_steps):
            lrs.append(scheduler.get_last_lr()[0])
            steps.append(step)
            scheduler.step()
        all_data.append((name, steps, lrs))

    with _style_context():
        fig, ax = plt.subplots(figsize=figsize)
        for name, steps, lrs in all_data:
            ax.plot(steps, lrs, label=name, linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
    return fig
