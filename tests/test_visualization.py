"""Tests for visualization helpers: plot_schedule and compare_schedules."""

from __future__ import annotations

import warnings

import pytest
import torch

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Conditional import: skip all tests if matplotlib is not available
# ---------------------------------------------------------------------------
matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")

import pytorch_scheduler as ps  # noqa: E402
from pytorch_scheduler.visualization import compare_schedules, plot_schedule  # noqa: E402


def _make_optimizer(lr: float = 0.1) -> torch.optim.SGD:
    param = torch.randn(2, requires_grad=True)
    return torch.optim.SGD([param], lr=lr)


# ---------------------------------------------------------------------------
# plot_schedule
# ---------------------------------------------------------------------------


def test_plot_schedule_returns_figure():
    """plot_schedule should return a matplotlib Figure."""
    from matplotlib.figure import Figure

    opt = _make_optimizer()
    scheduler = ps.RexScheduler(opt, total_steps=50)
    fig = plot_schedule(scheduler, total_steps=50)
    assert isinstance(fig, Figure)


def test_plot_schedule_figure_has_axes():
    """The returned Figure should contain at least one Axes."""
    opt = _make_optimizer()
    scheduler = ps.RexScheduler(opt, total_steps=50)
    fig = plot_schedule(scheduler, total_steps=50)
    assert len(fig.get_axes()) >= 1


def test_plot_schedule_axes_has_data():
    """The Axes should contain at least one plotted line."""
    opt = _make_optimizer()
    scheduler = ps.RexScheduler(opt, total_steps=50)
    fig = plot_schedule(scheduler, total_steps=50)
    ax = fig.get_axes()[0]
    assert len(ax.get_lines()) >= 1


def test_plot_schedule_custom_title():
    """Custom title should be applied to the Axes."""
    opt = _make_optimizer()
    scheduler = ps.RexScheduler(opt, total_steps=50)
    fig = plot_schedule(scheduler, total_steps=50, title="My Custom Title")
    ax = fig.get_axes()[0]
    assert ax.get_title() == "My Custom Title"


def test_plot_schedule_default_title_is_class_name():
    """Default title should be the scheduler's class name."""
    opt = _make_optimizer()
    scheduler = ps.RexScheduler(opt, total_steps=50)
    fig = plot_schedule(scheduler, total_steps=50)
    ax = fig.get_axes()[0]
    assert ax.get_title() == "RexScheduler"


def test_plot_schedule_custom_figsize():
    """Custom figsize should be reflected in the Figure dimensions."""
    opt = _make_optimizer()
    scheduler = ps.RexScheduler(opt, total_steps=50)
    fig = plot_schedule(scheduler, total_steps=50, figsize=(8, 3))
    w, h = fig.get_size_inches()
    assert abs(w - 8) < 0.1
    assert abs(h - 3) < 0.1


def test_plot_schedule_axis_labels():
    """Default axis labels should be 'Step' and 'Learning Rate'."""
    opt = _make_optimizer()
    scheduler = ps.RexScheduler(opt, total_steps=50)
    fig = plot_schedule(scheduler, total_steps=50)
    ax = fig.get_axes()[0]
    assert ax.get_xlabel() == "Step"
    assert ax.get_ylabel() == "Learning Rate"


def test_plot_schedule_custom_labels():
    """Custom axis labels should override defaults."""
    opt = _make_optimizer()
    scheduler = ps.RexScheduler(opt, total_steps=50)
    fig = plot_schedule(scheduler, total_steps=50, xlabel="Iteration", ylabel="LR Value")
    ax = fig.get_axes()[0]
    assert ax.get_xlabel() == "Iteration"
    assert ax.get_ylabel() == "LR Value"


def test_plot_schedule_line_data_length():
    """The plotted line should have exactly total_steps data points."""
    opt = _make_optimizer()
    scheduler = ps.KDecayScheduler(opt, total_steps=80)
    total_steps = 80
    fig = plot_schedule(scheduler, total_steps=total_steps)
    ax = fig.get_axes()[0]
    line = ax.get_lines()[0]
    xdata = line.get_xdata()
    assert len(xdata) == total_steps  # type: ignore[arg-type]


def test_plot_schedule_different_schedulers():
    """plot_schedule should work for a variety of schedulers."""
    schedulers_to_test = [
        ps.RexScheduler(_make_optimizer(), total_steps=50),
        ps.KDecayScheduler(_make_optimizer(), total_steps=50),
        ps.FlatCosineScheduler(_make_optimizer(), total_steps=50),
        ps.ChebyshevScheduler(_make_optimizer(), total_steps=50),
        ps.LinearDecayScheduler(_make_optimizer(), total_steps=50, warmup_steps=5),
        ps.CosineAnnealingWarmupRestarts(_make_optimizer(), first_cycle_steps=50, max_lr=0.1, min_lr=0.001),
    ]
    for scheduler in schedulers_to_test:
        from matplotlib.figure import Figure

        fig = plot_schedule(scheduler, total_steps=50)
        assert isinstance(fig, Figure), f"plot_schedule failed for {type(scheduler).__name__}"


# ---------------------------------------------------------------------------
# compare_schedules
# ---------------------------------------------------------------------------


def test_compare_schedules_returns_figure():
    """compare_schedules should return a matplotlib Figure."""
    from matplotlib.figure import Figure

    schedulers = {
        "Rex": ps.RexScheduler(_make_optimizer(), total_steps=50),
        "KDecay": ps.KDecayScheduler(_make_optimizer(), total_steps=50),
    }
    fig = compare_schedules(schedulers, total_steps=50)
    assert isinstance(fig, Figure)


def test_compare_schedules_figure_has_axes():
    """The returned Figure should have at least one Axes."""
    schedulers = {
        "Rex": ps.RexScheduler(_make_optimizer(), total_steps=50),
    }
    fig = compare_schedules(schedulers, total_steps=50)
    assert len(fig.get_axes()) >= 1


def test_compare_schedules_correct_number_of_lines():
    """The Axes should contain one line per scheduler."""
    schedulers = {
        "Rex": ps.RexScheduler(_make_optimizer(), total_steps=50),
        "KDecay": ps.KDecayScheduler(_make_optimizer(), total_steps=50),
        "Cosine": ps.FlatCosineScheduler(_make_optimizer(), total_steps=50),
    }
    fig = compare_schedules(schedulers, total_steps=50)
    ax = fig.get_axes()[0]
    # Each scheduler produces one line
    assert len(ax.get_lines()) == 3


def test_compare_schedules_has_legend():
    """compare_schedules should produce a legend on the Axes."""
    schedulers = {
        "Rex": ps.RexScheduler(_make_optimizer(), total_steps=50),
        "KDecay": ps.KDecayScheduler(_make_optimizer(), total_steps=50),
    }
    fig = compare_schedules(schedulers, total_steps=50)
    ax = fig.get_axes()[0]
    legend = ax.get_legend()
    assert legend is not None


def test_compare_schedules_legend_labels():
    """Legend should contain the scheduler names passed as keys."""
    labels = ["SchedulerA", "SchedulerB"]
    schedulers = {
        "SchedulerA": ps.RexScheduler(_make_optimizer(), total_steps=50),
        "SchedulerB": ps.KDecayScheduler(_make_optimizer(), total_steps=50),
    }
    fig = compare_schedules(schedulers, total_steps=50)
    ax = fig.get_axes()[0]
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [t.get_text() for t in legend.get_texts()]
    for label in labels:
        assert label in legend_texts


def test_compare_schedules_custom_title():
    """Custom title should appear on the Axes."""
    schedulers = {
        "Rex": ps.RexScheduler(_make_optimizer(), total_steps=50),
    }
    fig = compare_schedules(schedulers, total_steps=50, title="Comparison Plot")
    ax = fig.get_axes()[0]
    assert ax.get_title() == "Comparison Plot"


def test_compare_schedules_default_title():
    """Default title should be 'LR Schedule Comparison'."""
    schedulers = {
        "Rex": ps.RexScheduler(_make_optimizer(), total_steps=50),
    }
    fig = compare_schedules(schedulers, total_steps=50)
    ax = fig.get_axes()[0]
    assert ax.get_title() == "LR Schedule Comparison"


def test_compare_schedules_axis_labels():
    """Default axis labels should be 'Step' and 'Learning Rate'."""
    schedulers = {
        "Rex": ps.RexScheduler(_make_optimizer(), total_steps=50),
    }
    fig = compare_schedules(schedulers, total_steps=50)
    ax = fig.get_axes()[0]
    assert ax.get_xlabel() == "Step"
    assert ax.get_ylabel() == "Learning Rate"


def test_compare_schedules_single_scheduler():
    """compare_schedules with a single scheduler should work without error."""
    from matplotlib.figure import Figure

    schedulers = {"Single": ps.ChebyshevScheduler(_make_optimizer(), total_steps=50)}
    fig = compare_schedules(schedulers, total_steps=50)
    assert isinstance(fig, Figure)


def test_compare_schedules_custom_figsize():
    """Custom figsize should be reflected in the Figure dimensions."""
    schedulers = {"Rex": ps.RexScheduler(_make_optimizer(), total_steps=50)}
    fig = compare_schedules(schedulers, total_steps=50, figsize=(14, 6))
    w, h = fig.get_size_inches()
    assert abs(w - 14) < 0.1
    assert abs(h - 6) < 0.1


# ---------------------------------------------------------------------------
# Import error path (when matplotlib is not installed)
# ---------------------------------------------------------------------------


def test_import_matplotlib_raises_on_missing(monkeypatch):
    """_import_matplotlib should raise ImportError if matplotlib is absent."""

    from pytorch_scheduler.visualization import plot as viz_module

    # Temporarily remove matplotlib from sys.modules to simulate absence
    # Patch builtins.__import__ via monkeypatching the function inside the module

    def mock_import():
        raise ImportError("matplotlib not installed")

    monkeypatch.setattr(viz_module, "_import_matplotlib", mock_import)

    opt = _make_optimizer()
    scheduler = ps.RexScheduler(opt, total_steps=50)
    with pytest.raises(ImportError):
        viz_module.plot_schedule(scheduler, total_steps=50)
