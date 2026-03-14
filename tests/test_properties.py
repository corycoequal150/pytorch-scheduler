"""Property-based tests: boundary fuzzing with Hypothesis."""

from __future__ import annotations

import math
import warnings

import pytest
import torch
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import pytorch_scheduler as ps

warnings.filterwarnings("ignore")


def make_optimizer(lr: float = 0.1) -> torch.optim.SGD:
    model = torch.nn.Linear(2, 1)
    return torch.optim.SGD(model.parameters(), lr=lr)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# General step-based strategy components
s_total_steps = st.integers(min_value=10, max_value=2000)
s_base_lr = st.floats(min_value=1e-6, max_value=1.0)
s_min_lr = st.floats(min_value=0.0, max_value=1e-4)


# ---------------------------------------------------------------------------
# Property 1: No NaN/Inf for step-based schedulers
# ---------------------------------------------------------------------------


@pytest.mark.slow
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    total_steps=st.integers(min_value=20, max_value=2000),
    base_lr=st.floats(min_value=1e-5, max_value=1.0),
    step=st.integers(min_value=0, max_value=2000),
)
def test_property_rex_no_nan_inf(total_steps, base_lr, step):
    """RexScheduler always returns finite LR values."""
    assume(step <= total_steps)
    opt = make_optimizer(lr=base_lr)
    sched = ps.RexScheduler(opt, total_steps=total_steps)
    result = sched._lr_at(step, [base_lr])
    assert math.isfinite(result[0]), f"NaN/Inf at step={step}, total={total_steps}"


@pytest.mark.slow
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    total_steps=st.integers(min_value=20, max_value=2000),
    k=st.floats(min_value=0.1, max_value=5.0),
    base_lr=st.floats(min_value=1e-5, max_value=1.0),
    step=st.integers(min_value=0, max_value=2000),
)
def test_property_kdecay_no_nan_inf(total_steps, k, base_lr, step):
    """KDecayScheduler always returns finite LR values."""
    assume(step <= total_steps)
    opt = make_optimizer(lr=base_lr)
    sched = ps.KDecayScheduler(opt, total_steps=total_steps, k=k, min_lr=0.0)
    result = sched._lr_at(step, [base_lr])
    assert math.isfinite(result[0])


@pytest.mark.slow
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    total_steps=st.integers(min_value=20, max_value=2000),
    warmup_steps=st.integers(min_value=1, max_value=500),
    base_lr=st.floats(min_value=1e-5, max_value=1.0),
    step=st.integers(min_value=0, max_value=2000),
)
def test_property_linear_decay_no_nan_inf(total_steps, warmup_steps, base_lr, step):
    """LinearDecayScheduler always returns finite LR values."""
    assume(warmup_steps < total_steps)
    assume(step <= total_steps)
    opt = make_optimizer(lr=base_lr)
    sched = ps.LinearDecayScheduler(opt, total_steps=total_steps, warmup_steps=warmup_steps, min_lr=0.0)
    result = sched._lr_at(step, [base_lr])
    assert math.isfinite(result[0])


@pytest.mark.slow
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    total_steps=st.integers(min_value=20, max_value=2000),
    warmup_steps=st.integers(min_value=1, max_value=400),
    step=st.integers(min_value=0, max_value=2000),
    base_lr=st.floats(min_value=1e-5, max_value=1.0),
)
def test_property_cosine_with_warmup_no_nan_inf(total_steps, warmup_steps, step, base_lr):
    """CosineWithWarmupScheduler always returns finite LR values."""
    assume(warmup_steps < total_steps)
    assume(step <= total_steps)
    opt = make_optimizer(lr=base_lr)
    sched = ps.CosineWithWarmupScheduler(opt, total_steps=total_steps, warmup_steps=warmup_steps, min_lr=0.0)
    result = sched._lr_at(step, [base_lr])
    assert math.isfinite(result[0])


@pytest.mark.slow
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    total_steps=st.integers(min_value=20, max_value=2000),
    warmup_steps=st.integers(min_value=0, max_value=300),
    hold_steps=st.integers(min_value=0, max_value=300),
    step=st.integers(min_value=0, max_value=2000),
    base_lr=st.floats(min_value=1e-5, max_value=1.0),
)
def test_property_warmup_hold_cosine_no_nan_inf(total_steps, warmup_steps, hold_steps, step, base_lr):
    """WarmupHoldCosineScheduler always returns finite LR values."""
    assume(warmup_steps + hold_steps < total_steps)
    assume(step <= total_steps)
    opt = make_optimizer(lr=base_lr)
    sched = ps.WarmupHoldCosineScheduler(
        opt,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        hold_steps=hold_steps,
        min_lr=0.0,
    )
    result = sched._lr_at(step, [base_lr])
    assert math.isfinite(result[0])


@pytest.mark.slow
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    total_steps=st.integers(min_value=20, max_value=2000),
    warmup_steps=st.integers(min_value=1, max_value=400),
    stable_steps=st.integers(min_value=1, max_value=800),
    step=st.integers(min_value=0, max_value=2000),
    base_lr=st.floats(min_value=1e-5, max_value=1.0),
)
def test_property_wsd_no_nan_inf(total_steps, warmup_steps, stable_steps, step, base_lr):
    """WSDScheduler always returns finite LR values."""
    assume(warmup_steps + stable_steps < total_steps)
    assume(step <= total_steps)
    opt = make_optimizer(lr=base_lr)
    sched = ps.WSDScheduler(
        opt,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        min_lr=0.0,
    )
    result = sched._lr_at(step, [base_lr])
    assert math.isfinite(result[0])


@pytest.mark.slow
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    total_steps=st.integers(min_value=20, max_value=2000),
    warmup_steps=st.integers(min_value=1, max_value=400),
    alpha=st.floats(min_value=0.1, max_value=3.0),
    step=st.integers(min_value=0, max_value=2000),
    base_lr=st.floats(min_value=1e-5, max_value=1.0),
)
def test_property_power_decay_no_nan_inf(total_steps, warmup_steps, alpha, step, base_lr):
    """PowerDecayScheduler always returns finite LR values."""
    assume(warmup_steps < total_steps)
    assume(step <= total_steps)
    opt = make_optimizer(lr=base_lr)
    sched = ps.PowerDecayScheduler(opt, total_steps=total_steps, warmup_steps=warmup_steps, alpha=alpha, min_lr=0.0)
    result = sched._lr_at(step, [base_lr])
    assert math.isfinite(result[0])


# ---------------------------------------------------------------------------
# Property 2: Bounded output — LR >= 0 for all configurations
# (ChebyshevScheduler is excluded since it can produce very small values
#  near min_lr=0 but never negative)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    total_steps=st.integers(min_value=20, max_value=2000),
    step=st.integers(min_value=0, max_value=2000),
    base_lr=st.floats(min_value=1e-5, max_value=1.0),
)
def test_property_rex_non_negative(total_steps, step, base_lr):
    """RexScheduler LR is always non-negative."""
    assume(step <= total_steps)
    opt = make_optimizer(lr=base_lr)
    sched = ps.RexScheduler(opt, total_steps=total_steps)
    result = sched._lr_at(step, [base_lr])
    assert result[0] >= -1e-9


@pytest.mark.slow
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    total_steps=st.integers(min_value=20, max_value=2000),
    warmup_steps=st.integers(min_value=1, max_value=400),
    step=st.integers(min_value=0, max_value=2000),
    base_lr=st.floats(min_value=1e-5, max_value=1.0),
)
def test_property_linear_decay_non_negative(total_steps, warmup_steps, step, base_lr):
    """LinearDecayScheduler LR is always non-negative."""
    assume(warmup_steps < total_steps)
    assume(step <= total_steps)
    opt = make_optimizer(lr=base_lr)
    sched = ps.LinearDecayScheduler(opt, total_steps=total_steps, warmup_steps=warmup_steps, min_lr=0.0)
    result = sched._lr_at(step, [base_lr])
    assert result[0] >= -1e-9


@pytest.mark.slow
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    total_steps=st.integers(min_value=20, max_value=2000),
    warmup_steps=st.integers(min_value=1, max_value=400),
    step=st.integers(min_value=0, max_value=2000),
    base_lr=st.floats(min_value=1e-5, max_value=1.0),
)
def test_property_cosine_with_warmup_non_negative(total_steps, warmup_steps, step, base_lr):
    """CosineWithWarmupScheduler LR is always non-negative."""
    assume(warmup_steps < total_steps)
    assume(step <= total_steps)
    opt = make_optimizer(lr=base_lr)
    sched = ps.CosineWithWarmupScheduler(opt, total_steps=total_steps, warmup_steps=warmup_steps, min_lr=0.0)
    result = sched._lr_at(step, [base_lr])
    assert result[0] >= -1e-9


# ---------------------------------------------------------------------------
# Property 3: Warmup monotonicity — LR is non-decreasing during linear warmup
# ---------------------------------------------------------------------------


@pytest.mark.slow
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    total_steps=st.integers(min_value=20, max_value=2000),
    warmup_steps=st.integers(min_value=2, max_value=400),
    base_lr=st.floats(min_value=1e-5, max_value=1.0),
)
def test_property_linear_decay_warmup_monotone(total_steps, warmup_steps, base_lr):
    """LR must be non-decreasing during the linear warmup phase."""
    assume(warmup_steps < total_steps)
    opt = make_optimizer(lr=base_lr)
    sched = ps.LinearDecayScheduler(opt, total_steps=total_steps, warmup_steps=warmup_steps, min_lr=0.0)
    prev_lr = sched._lr_at(0, [base_lr])[0]
    for step in range(1, warmup_steps + 1):
        curr_lr = sched._lr_at(step, [base_lr])[0]
        assert curr_lr >= prev_lr - 1e-9, f"Warmup not monotone at step={step}: {prev_lr} -> {curr_lr}"
        prev_lr = curr_lr


@pytest.mark.slow
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    total_steps=st.integers(min_value=20, max_value=2000),
    warmup_steps=st.integers(min_value=2, max_value=400),
    base_lr=st.floats(min_value=1e-5, max_value=1.0),
)
def test_property_cosine_warmup_monotone(total_steps, warmup_steps, base_lr):
    """CosineWithWarmupScheduler LR must be non-decreasing during warmup."""
    assume(warmup_steps < total_steps)
    opt = make_optimizer(lr=base_lr)
    sched = ps.CosineWithWarmupScheduler(opt, total_steps=total_steps, warmup_steps=warmup_steps, min_lr=0.0)
    prev_lr = sched._lr_at(0, [base_lr])[0]
    for step in range(1, warmup_steps + 1):
        curr_lr = sched._lr_at(step, [base_lr])[0]
        assert curr_lr >= prev_lr - 1e-9, f"Cosine warmup not monotone at step={step}: {prev_lr} -> {curr_lr}"
        prev_lr = curr_lr


@pytest.mark.slow
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    total_steps=st.integers(min_value=20, max_value=2000),
    warmup_steps=st.integers(min_value=2, max_value=400),
    hold_steps=st.integers(min_value=0, max_value=200),
    base_lr=st.floats(min_value=1e-5, max_value=1.0),
)
def test_property_warmup_hold_cosine_warmup_monotone(total_steps, warmup_steps, hold_steps, base_lr):
    """WarmupHoldCosineScheduler LR must be non-decreasing during warmup."""
    assume(warmup_steps + hold_steps < total_steps)
    opt = make_optimizer(lr=base_lr)
    sched = ps.WarmupHoldCosineScheduler(
        opt,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        hold_steps=hold_steps,
        min_lr=0.0,
    )
    prev_lr = sched._lr_at(0, [base_lr])[0]
    for step in range(1, warmup_steps + 1):
        curr_lr = sched._lr_at(step, [base_lr])[0]
        assert curr_lr >= prev_lr - 1e-9, f"WarmupHold warmup not monotone at step={step}: {prev_lr} -> {curr_lr}"
        prev_lr = curr_lr


# ---------------------------------------------------------------------------
# Property 4: Boundary values — extreme inputs do not crash or produce NaN
# ---------------------------------------------------------------------------


@pytest.mark.slow
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(base_lr=st.floats(min_value=1e-8, max_value=1.0))
def test_property_boundary_rex_minimal_total_steps(base_lr):
    """RexScheduler with total_steps=2 should not crash or produce NaN."""
    opt = make_optimizer(lr=base_lr)
    sched = ps.RexScheduler(opt, total_steps=2)
    for step in [0, 1, 2]:
        result = sched._lr_at(step, [base_lr])
        assert math.isfinite(result[0])
        assert result[0] >= -1e-9


@pytest.mark.slow
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(base_lr=st.floats(min_value=1e-8, max_value=1.0))
def test_property_boundary_linear_decay_minimal(base_lr):
    """LinearDecayScheduler with minimal total_steps/warmup_steps should not crash."""
    opt = make_optimizer(lr=base_lr)
    # warmup_steps=1, total_steps=2 is the minimum valid config
    sched = ps.LinearDecayScheduler(opt, total_steps=2, warmup_steps=1, min_lr=0.0)
    for step in [0, 1, 2]:
        result = sched._lr_at(step, [base_lr])
        assert math.isfinite(result[0])


@pytest.mark.slow
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(base_lr=st.floats(min_value=1e-8, max_value=1.0))
def test_property_boundary_cosine_with_warmup_minimal(base_lr):
    """CosineWithWarmupScheduler with warmup_steps=1, total_steps=2 should not crash."""
    opt = make_optimizer(lr=base_lr)
    sched = ps.CosineWithWarmupScheduler(opt, total_steps=2, warmup_steps=1, min_lr=0.0)
    for step in [0, 1, 2]:
        result = sched._lr_at(step, [base_lr])
        assert math.isfinite(result[0])


@pytest.mark.slow
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(base_lr=st.floats(min_value=1e-8, max_value=1.0))
def test_property_boundary_kdecay_large_k(base_lr):
    """KDecayScheduler with large k should not crash."""
    opt = make_optimizer(lr=base_lr)
    sched = ps.KDecayScheduler(opt, total_steps=100, k=10.0, min_lr=0.0)
    for step in [0, 1, 50, 99, 100]:
        result = sched._lr_at(step, [base_lr])
        assert math.isfinite(result[0])
        assert result[0] >= -1e-9


@pytest.mark.slow
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(base_lr=st.floats(min_value=1e-8, max_value=1.0))
def test_property_boundary_wsd_minimal_decay(base_lr):
    """WSDScheduler with single decay step should not crash."""
    opt = make_optimizer(lr=base_lr)
    # warmup=1, stable=1, total=3 → decay_steps=1
    sched = ps.WSDScheduler(opt, total_steps=3, warmup_steps=1, stable_steps=1, min_lr=0.0)
    for step in [0, 1, 2, 3]:
        result = sched._lr_at(step, [base_lr])
        assert math.isfinite(result[0])


# ---------------------------------------------------------------------------
# Property 5: Continuity at phase boundaries — no large jumps at transitions
# ---------------------------------------------------------------------------


@pytest.mark.slow
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    total_steps=st.integers(min_value=20, max_value=2000),
    warmup_steps=st.integers(min_value=2, max_value=400),
    base_lr=st.floats(min_value=1e-4, max_value=1.0),
)
def test_property_linear_decay_warmup_boundary_continuity(total_steps, warmup_steps, base_lr):
    """LR at warmup_steps and warmup_steps-1 should be close (no large jump)."""
    assume(warmup_steps < total_steps)
    opt = make_optimizer(lr=base_lr)
    sched = ps.LinearDecayScheduler(opt, total_steps=total_steps, warmup_steps=warmup_steps, min_lr=0.0)
    lr_before = sched._lr_at(warmup_steps - 1, [base_lr])[0]
    lr_at = sched._lr_at(warmup_steps, [base_lr])[0]
    # Jump should be at most one warmup step's worth: base_lr / warmup_steps
    max_step = base_lr / warmup_steps + 1e-6
    assert abs(lr_at - lr_before) <= max_step + 1e-9, (
        f"Large jump at warmup boundary: {lr_before} -> {lr_at}, max_step={max_step}"
    )


@pytest.mark.slow
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    total_steps=st.integers(min_value=20, max_value=2000),
    warmup_steps=st.integers(min_value=2, max_value=400),
    base_lr=st.floats(min_value=1e-4, max_value=1.0),
)
def test_property_cosine_warmup_boundary_continuity(total_steps, warmup_steps, base_lr):
    """CosineWithWarmupScheduler has small LR jump at warmup boundary."""
    assume(warmup_steps < total_steps)
    opt = make_optimizer(lr=base_lr)
    sched = ps.CosineWithWarmupScheduler(opt, total_steps=total_steps, warmup_steps=warmup_steps, min_lr=0.0)
    lr_before = sched._lr_at(warmup_steps - 1, [base_lr])[0]
    lr_at = sched._lr_at(warmup_steps, [base_lr])[0]
    # Jump should be at most one step's worth: base_lr / warmup_steps
    max_step = base_lr / warmup_steps + 1e-6
    assert abs(lr_at - lr_before) <= max_step + 1e-9


@pytest.mark.slow
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    total_steps=st.integers(min_value=20, max_value=2000),
    warmup_steps=st.integers(min_value=2, max_value=400),
    hold_steps=st.integers(min_value=1, max_value=400),
    base_lr=st.floats(min_value=1e-4, max_value=1.0),
)
def test_property_warmup_hold_cosine_hold_boundary_continuity(total_steps, warmup_steps, hold_steps, base_lr):
    """WarmupHoldCosineScheduler: LR at start/end of hold phase must be base_lr."""
    assume(warmup_steps + hold_steps < total_steps)
    opt = make_optimizer(lr=base_lr)
    sched = ps.WarmupHoldCosineScheduler(
        opt,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        hold_steps=hold_steps,
        min_lr=0.0,
    )
    # At warmup_steps, LR should be base_lr (start of hold)
    lr_hold_start = sched._lr_at(warmup_steps, [base_lr])[0]
    assert abs(lr_hold_start - base_lr) < 1e-9, f"LR at hold start: expected {base_lr}, got {lr_hold_start}"
    # At warmup_steps + hold_steps - 1, LR should still be base_lr
    hold_end = warmup_steps + hold_steps
    lr_hold_end = sched._lr_at(hold_end - 1, [base_lr])[0]
    assert abs(lr_hold_end - base_lr) < 1e-9, f"LR just before hold end: expected {base_lr}, got {lr_hold_end}"
