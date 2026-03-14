"""Tests for parameter validation and edge cases."""

from __future__ import annotations

import warnings

import pytest
import torch

import pytorch_scheduler as ps
from pytorch_scheduler import BaseScheduler

warnings.filterwarnings("ignore")


def _make_optimizer(lr: float = 0.1) -> torch.optim.SGD:
    param = torch.randn(2, requires_grad=True)
    return torch.optim.SGD([param], lr=lr)


# ---------------------------------------------------------------------------
# BaseScheduler: abstract — cannot be instantiated directly
# ---------------------------------------------------------------------------


def test_base_scheduler_is_abstract():
    """BaseScheduler should raise TypeError when instantiated directly."""
    opt = _make_optimizer()
    with pytest.raises(TypeError):
        BaseScheduler(opt)  # type: ignore[abstract]


def test_base_scheduler_subclass_must_implement_get_lr():
    """A subclass that doesn't implement get_lr() should raise TypeError."""

    class IncompleteScheduler(BaseScheduler):
        pass  # missing get_lr

    opt = _make_optimizer()
    with pytest.raises(TypeError):
        IncompleteScheduler(opt)  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# RexScheduler
# ---------------------------------------------------------------------------


def test_rex_negative_total_steps():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="total_steps must be positive"):
        ps.RexScheduler(opt, total_steps=0)


def test_rex_negative_total_steps_negative():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="total_steps must be positive"):
        ps.RexScheduler(opt, total_steps=-10)


# ---------------------------------------------------------------------------
# InverseSqrtScheduler
# ---------------------------------------------------------------------------


def test_inverse_sqrt_zero_warmup_steps():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="warmup_steps must be positive"):
        ps.InverseSqrtScheduler(opt, warmup_steps=0)


def test_inverse_sqrt_negative_warmup_steps():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="warmup_steps must be positive"):
        ps.InverseSqrtScheduler(opt, warmup_steps=-5)


# ---------------------------------------------------------------------------
# LinearDecayScheduler
# ---------------------------------------------------------------------------


def test_linear_decay_zero_total_steps():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="total_steps must be positive"):
        ps.LinearDecayScheduler(opt, total_steps=0)


def test_linear_decay_negative_min_lr():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="min_lr must be non-negative"):
        ps.LinearDecayScheduler(opt, total_steps=100, min_lr=-0.01)


def test_linear_decay_warmup_exceeds_total():
    opt = _make_optimizer()
    with pytest.raises(ValueError):
        ps.LinearDecayScheduler(opt, total_steps=50, warmup_steps=50)


def test_linear_decay_warmup_equals_total():
    opt = _make_optimizer()
    with pytest.raises(ValueError):
        ps.LinearDecayScheduler(opt, total_steps=100, warmup_steps=100)


def test_linear_decay_warmup_zero_valid():
    """warmup_steps=0 should be valid (no warmup)."""
    opt = _make_optimizer()
    s = ps.LinearDecayScheduler(opt, total_steps=100, warmup_steps=0)
    assert s is not None


# ---------------------------------------------------------------------------
# TrapezoidalScheduler
# ---------------------------------------------------------------------------


def test_trapezoidal_zero_total_steps():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="total_steps must be positive"):
        ps.TrapezoidalScheduler(opt, total_steps=0, warmup_steps=0, decay_steps=0)


def test_trapezoidal_negative_warmup():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="warmup_steps must be non-negative"):
        ps.TrapezoidalScheduler(opt, total_steps=100, warmup_steps=-1, decay_steps=10)


def test_trapezoidal_negative_decay():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="decay_steps must be non-negative"):
        ps.TrapezoidalScheduler(opt, total_steps=100, warmup_steps=10, decay_steps=-1)


def test_trapezoidal_warmup_plus_decay_exceeds_total():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="must be <= total_steps"):
        ps.TrapezoidalScheduler(opt, total_steps=100, warmup_steps=60, decay_steps=60)


def test_trapezoidal_warmup_plus_decay_equals_total_valid():
    """warmup + decay == total_steps should be valid (no flat phase)."""
    opt = _make_optimizer()
    s = ps.TrapezoidalScheduler(opt, total_steps=100, warmup_steps=40, decay_steps=60)
    assert s is not None


# ---------------------------------------------------------------------------
# CosineAnnealingWarmupRestarts
# ---------------------------------------------------------------------------


def test_cosine_zero_first_cycle_steps():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="first_cycle_steps must be positive"):
        ps.CosineAnnealingWarmupRestarts(opt, first_cycle_steps=0)


def test_cosine_negative_warmup_steps():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="warmup_steps must be non-negative"):
        ps.CosineAnnealingWarmupRestarts(opt, first_cycle_steps=100, warmup_steps=-1)


def test_cosine_warmup_exceeds_cycle():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="must be less than"):
        ps.CosineAnnealingWarmupRestarts(opt, first_cycle_steps=50, warmup_steps=50)


def test_cosine_max_lr_less_than_min_lr():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match=r"max_lr.*must be >= min_lr"):
        ps.CosineAnnealingWarmupRestarts(opt, first_cycle_steps=100, max_lr=0.001, min_lr=0.1)


def test_cosine_negative_max_lr():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="max_lr must be non-negative"):
        ps.CosineAnnealingWarmupRestarts(opt, first_cycle_steps=100, max_lr=-0.1)


def test_cosine_negative_min_lr():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="min_lr must be non-negative"):
        ps.CosineAnnealingWarmupRestarts(opt, first_cycle_steps=100, min_lr=-0.001)


def test_cosine_non_positive_cycle_mult():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="cycle_mult must be positive"):
        ps.CosineAnnealingWarmupRestarts(opt, first_cycle_steps=100, cycle_mult=0.0)


def test_cosine_non_positive_gamma():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="gamma must be positive"):
        ps.CosineAnnealingWarmupRestarts(opt, first_cycle_steps=100, gamma=0.0)


# ---------------------------------------------------------------------------
# WSDScheduler
# ---------------------------------------------------------------------------


def test_wsd_zero_total_steps():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="total_steps must be positive"):
        ps.WSDScheduler(opt, total_steps=0, warmup_steps=0, stable_steps=0)


def test_wsd_negative_warmup():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="warmup_steps must be non-negative"):
        ps.WSDScheduler(opt, total_steps=100, warmup_steps=-1, stable_steps=50)


def test_wsd_negative_stable():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="stable_steps must be non-negative"):
        ps.WSDScheduler(opt, total_steps=100, warmup_steps=10, stable_steps=-1)


def test_wsd_warmup_plus_stable_equals_total():
    """warmup + stable == total_steps is invalid (need at least one decay step)."""
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="must be < total_steps"):
        ps.WSDScheduler(opt, total_steps=100, warmup_steps=50, stable_steps=50)


def test_wsd_warmup_plus_stable_exceeds_total():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="must be < total_steps"):
        ps.WSDScheduler(opt, total_steps=100, warmup_steps=60, stable_steps=50)


def test_wsd_negative_min_lr():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="min_lr must be non-negative"):
        ps.WSDScheduler(opt, total_steps=100, warmup_steps=10, stable_steps=50, min_lr=-0.001)


@pytest.mark.parametrize("decay_type", ["cosine", "linear", "sqrt"])
def test_wsd_all_decay_types(decay_type):
    """All three decay types should be constructable and step without error."""
    opt = _make_optimizer()
    s = ps.WSDScheduler(opt, total_steps=100, warmup_steps=10, stable_steps=50, decay_type=decay_type)
    for _ in range(101):
        s.step()
    assert s.get_last_lr()[0] >= 0.0


# ---------------------------------------------------------------------------
# KDecayScheduler
# ---------------------------------------------------------------------------


def test_kdecay_zero_k():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="k must be positive"):
        ps.KDecayScheduler(opt, total_steps=100, k=0.0)


def test_kdecay_negative_k():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="k must be positive"):
        ps.KDecayScheduler(opt, total_steps=100, k=-1.0)


def test_kdecay_negative_min_lr():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="min_lr must be non-negative"):
        ps.KDecayScheduler(opt, total_steps=100, min_lr=-0.001)


def test_kdecay_zero_total_steps():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="total_steps must be positive"):
        ps.KDecayScheduler(opt, total_steps=0)


# ---------------------------------------------------------------------------
# TanhDecayScheduler
# ---------------------------------------------------------------------------


def test_tanh_zero_steepness():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="steepness must be positive"):
        ps.TanhDecayScheduler(opt, total_steps=100, steepness=0.0)


def test_tanh_negative_steepness():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="steepness must be positive"):
        ps.TanhDecayScheduler(opt, total_steps=100, steepness=-1.0)


def test_tanh_negative_min_lr():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="min_lr must be non-negative"):
        ps.TanhDecayScheduler(opt, total_steps=100, min_lr=-0.001)


# ---------------------------------------------------------------------------
# SlantedTriangularScheduler
# ---------------------------------------------------------------------------


def test_slanted_zero_cut_frac():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="cut_frac must be in"):
        ps.SlantedTriangularScheduler(opt, total_steps=100, cut_frac=0.0)


def test_slanted_cut_frac_one():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="cut_frac must be in"):
        ps.SlantedTriangularScheduler(opt, total_steps=100, cut_frac=1.0)


def test_slanted_negative_ratio():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="ratio must be positive"):
        ps.SlantedTriangularScheduler(opt, total_steps=100, ratio=-1.0)


def test_slanted_zero_ratio():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="ratio must be positive"):
        ps.SlantedTriangularScheduler(opt, total_steps=100, ratio=0.0)


def test_slanted_zero_total_steps():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="total_steps must be positive"):
        ps.SlantedTriangularScheduler(opt, total_steps=0)


# ---------------------------------------------------------------------------
# FlatCosineScheduler
# ---------------------------------------------------------------------------


def test_flat_cosine_zero_total_steps():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="total_steps must be positive"):
        ps.FlatCosineScheduler(opt, total_steps=0)


def test_flat_cosine_flat_fraction_equals_one():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="flat_fraction must be in"):
        ps.FlatCosineScheduler(opt, total_steps=100, flat_fraction=1.0)


def test_flat_cosine_flat_fraction_negative():
    """flat_fraction < 0 should raise ValueError."""
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="flat_fraction must be in"):
        ps.FlatCosineScheduler(opt, total_steps=100, flat_fraction=-0.1)


def test_flat_cosine_flat_fraction_zero_valid():
    """flat_fraction=0.0 is valid (no flat phase, pure cosine)."""
    opt = _make_optimizer()
    s = ps.FlatCosineScheduler(opt, total_steps=100, flat_fraction=0.0)
    assert s is not None


def test_flat_cosine_negative_min_lr():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="min_lr must be non-negative"):
        ps.FlatCosineScheduler(opt, total_steps=100, min_lr=-0.001)


# ---------------------------------------------------------------------------
# PolynomialScheduler
# ---------------------------------------------------------------------------


def test_polynomial_zero_total_steps():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="total_steps must be positive"):
        ps.PolynomialScheduler(opt, total_steps=0)


def test_polynomial_zero_power():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="power must be positive"):
        ps.PolynomialScheduler(opt, total_steps=100, power=0.0)


def test_polynomial_negative_power():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="power must be positive"):
        ps.PolynomialScheduler(opt, total_steps=100, power=-1.0)


def test_polynomial_negative_min_lr():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="min_lr must be non-negative"):
        ps.PolynomialScheduler(opt, total_steps=100, min_lr=-0.001)


def test_polynomial_warmup_negative():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="warmup_steps must be non-negative"):
        ps.PolynomialScheduler(opt, total_steps=100, warmup_steps=-1)


def test_polynomial_warmup_equals_total():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match=r"warmup_steps.*must be less than"):
        ps.PolynomialScheduler(opt, total_steps=100, warmup_steps=100)


def test_polynomial_cycle_continues_beyond_total():
    """cycle=True should allow stepping past total_steps without clamping to min_lr."""
    opt = _make_optimizer(lr=0.1)
    s = ps.PolynomialScheduler(opt, total_steps=50, cycle=True)
    for _ in range(150):
        s.step()
    # After cycling, LR should still be positive (not stuck at 0)
    lr = s.get_last_lr()[0]
    assert lr >= 0.0


# ---------------------------------------------------------------------------
# ChebyshevScheduler
# ---------------------------------------------------------------------------


def test_chebyshev_zero_total_steps():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="total_steps must be positive"):
        ps.ChebyshevScheduler(opt, total_steps=0)


def test_chebyshev_negative_min_lr():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="min_lr must be non-negative"):
        ps.ChebyshevScheduler(opt, total_steps=100, min_lr=-0.001)


def test_chebyshev_node_formula():
    """ChebyshevScheduler should follow the Chebyshev node formula exactly."""
    import math

    opt = _make_optimizer(lr=0.1)
    total_steps = 20
    s = ps.ChebyshevScheduler(opt, total_steps=total_steps)
    for step in range(total_steps):
        lr = s.get_last_lr()[0]
        j = step  # j == step for step in [0, total_steps-1]
        x_j = math.cos(math.pi * (2 * j + 1) / (2 * total_steps))
        expected = 0.1 * (0.5 * (1.0 + x_j))
        assert abs(lr - expected) < 1e-9, f"step={step}: LR={lr}, expected={expected}"
        s.step()


def test_chebyshev_lr_stays_in_range():
    """ChebyshevScheduler LRs should stay within [min_lr, base_lr]."""
    opt = _make_optimizer(lr=0.1)
    s = ps.ChebyshevScheduler(opt, total_steps=50, min_lr=0.01)
    for _ in range(60):
        for lr in s.get_last_lr():
            assert lr >= 0.01 - 1e-9, f"LR {lr} below min_lr=0.01"
            assert lr <= 0.1 + 1e-9, f"LR {lr} above base_lr=0.1"
        s.step()


def test_chebyshev_clamps_past_total_steps():
    """After total_steps, ChebyshevScheduler should reuse the final node."""
    opt = _make_optimizer(lr=0.1)
    s = ps.ChebyshevScheduler(opt, total_steps=10)
    for _ in range(10):
        s.step()
    lr_at_total = s.get_last_lr()[0]
    s.step()
    lr_past_total = s.get_last_lr()[0]
    assert abs(lr_at_total - lr_past_total) < 1e-9, "LR should be clamped after total_steps"


# ---------------------------------------------------------------------------
# PowerDecayScheduler
# ---------------------------------------------------------------------------


def test_power_decay_zero_total_steps():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="total_steps must be positive"):
        ps.PowerDecayScheduler(opt, total_steps=0)


def test_power_decay_zero_alpha():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="alpha must be positive"):
        ps.PowerDecayScheduler(opt, total_steps=100, alpha=0.0)


def test_power_decay_negative_alpha():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="alpha must be positive"):
        ps.PowerDecayScheduler(opt, total_steps=100, alpha=-0.5)


def test_power_decay_negative_min_lr():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="min_lr must be non-negative"):
        ps.PowerDecayScheduler(opt, total_steps=100, min_lr=-0.001)


def test_power_decay_warmup_equals_total():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match=r"warmup_steps.*must be less than"):
        ps.PowerDecayScheduler(opt, total_steps=100, warmup_steps=100)


def test_power_decay_no_warmup_starts_at_base_lr():
    """With warmup_steps=0, LR at step 0 should equal base_lr."""
    opt = _make_optimizer(lr=0.1)
    s = ps.PowerDecayScheduler(opt, total_steps=100, warmup_steps=0)
    assert abs(s.get_last_lr()[0] - 0.1) < 1e-6


def test_power_decay_min_lr_floor():
    """LR should never drop below min_lr even for large alpha."""
    opt = _make_optimizer(lr=0.1)
    s = ps.PowerDecayScheduler(opt, total_steps=100, alpha=5.0, min_lr=0.001)
    for _ in range(200):
        for lr in s.get_last_lr():
            assert lr >= 0.001 - 1e-9, f"LR {lr} below min_lr=0.001"
        s.step()


# ---------------------------------------------------------------------------
# HyperbolicLRScheduler
# ---------------------------------------------------------------------------


def test_hyperbolic_zero_total_steps():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="total_steps must be positive"):
        ps.HyperbolicLRScheduler(opt, total_steps=0, upper_bound=100)


def test_hyperbolic_upper_bound_less_than_total():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match=r"upper_bound.*must be >= total_steps"):
        ps.HyperbolicLRScheduler(opt, total_steps=100, upper_bound=50)


def test_hyperbolic_negative_min_lr():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="min_lr must be non-negative"):
        ps.HyperbolicLRScheduler(opt, total_steps=100, upper_bound=250, min_lr=-0.001)


def test_hyperbolic_warmup_exceeds_total():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match=r"warmup_steps.*must be less than"):
        ps.HyperbolicLRScheduler(opt, total_steps=100, upper_bound=250, warmup_steps=100)


def test_hyperbolic_negative_warmup():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="warmup_steps must be non-negative"):
        ps.HyperbolicLRScheduler(opt, total_steps=100, upper_bound=250, warmup_steps=-1)


def test_hyperbolic_no_warmup_starts_at_base_lr():
    opt = _make_optimizer(lr=0.1)
    s = ps.HyperbolicLRScheduler(opt, total_steps=100, upper_bound=250, warmup_steps=0)
    assert abs(s.get_last_lr()[0] - 0.1) < 1e-6


def test_hyperbolic_lr_stays_positive():
    """LR should remain positive throughout training."""
    opt = _make_optimizer(lr=0.1)
    s = ps.HyperbolicLRScheduler(opt, total_steps=100, upper_bound=250, min_lr=1e-6)
    for _ in range(110):
        for lr in s.get_last_lr():
            assert lr >= 0.0, f"Negative LR: {lr}"
        s.step()


# ---------------------------------------------------------------------------
# ExpHyperbolicLRScheduler
# ---------------------------------------------------------------------------


def test_exp_hyperbolic_zero_total_steps():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="total_steps must be positive"):
        ps.ExpHyperbolicLRScheduler(opt, total_steps=0, upper_bound=100)


def test_exp_hyperbolic_upper_bound_less_than_total():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match=r"upper_bound.*must be >= total_steps"):
        ps.ExpHyperbolicLRScheduler(opt, total_steps=100, upper_bound=50)


def test_exp_hyperbolic_zero_min_lr():
    """ExpHyperbolicLR requires min_lr > 0 (used as divisor)."""
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="min_lr must be positive"):
        ps.ExpHyperbolicLRScheduler(opt, total_steps=100, upper_bound=250, min_lr=0.0)


def test_exp_hyperbolic_negative_min_lr():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="min_lr must be positive"):
        ps.ExpHyperbolicLRScheduler(opt, total_steps=100, upper_bound=250, min_lr=-0.001)


def test_exp_hyperbolic_warmup_exceeds_total():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match=r"warmup_steps.*must be less than"):
        ps.ExpHyperbolicLRScheduler(opt, total_steps=100, upper_bound=250, warmup_steps=100)


def test_exp_hyperbolic_no_warmup_starts_at_base_lr():
    opt = _make_optimizer(lr=0.1)
    s = ps.ExpHyperbolicLRScheduler(opt, total_steps=100, upper_bound=250, warmup_steps=0)
    assert abs(s.get_last_lr()[0] - 0.1) < 1e-6


def test_exp_hyperbolic_lr_stays_positive():
    """LR should remain positive throughout training."""
    opt = _make_optimizer(lr=0.1)
    s = ps.ExpHyperbolicLRScheduler(opt, total_steps=100, upper_bound=250, min_lr=1e-6)
    for _ in range(110):
        for lr in s.get_last_lr():
            assert lr > 0.0, f"Non-positive LR: {lr}"
        s.step()
