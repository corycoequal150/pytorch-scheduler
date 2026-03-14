"""Tests for WarmupScheduler."""

from __future__ import annotations

import copy
import math
import warnings

import pytest
import torch

import pytorch_scheduler as ps
from pytorch_scheduler import WarmupScheduler

warnings.filterwarnings("ignore")

WARMUP_STEPS = 20
TOTAL_STEPS = 100


def _make_optimizer(lr: float = 0.1) -> torch.optim.SGD:
    param = torch.randn(2, requires_grad=True)
    return torch.optim.SGD([param], lr=lr)


def _make_base_scheduler(optimizer, total_steps=TOTAL_STEPS):
    """Return a LinearDecayScheduler with no internal warmup (warmup_steps=0)."""
    return ps.LinearDecayScheduler(optimizer, total_steps=total_steps)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("warmup_type", ["linear", "cosine", "exponential"])
def test_warmup_scheduler_constructable(warmup_type):
    opt = _make_optimizer()
    base = _make_base_scheduler(opt)
    scheduler = WarmupScheduler(opt, base, warmup_steps=WARMUP_STEPS, warmup_type=warmup_type)
    assert scheduler is not None


def test_warmup_scheduler_invalid_type():
    opt = _make_optimizer()
    base = _make_base_scheduler(opt)
    scheduler = WarmupScheduler(opt, base, warmup_steps=WARMUP_STEPS, warmup_type="linear")
    # Manually set invalid type and force get_lr
    scheduler.warmup_type = "invalid"
    scheduler.last_epoch = 5  # inside warmup
    with pytest.raises(ValueError, match="Unknown warmup type"):
        scheduler.get_lr()


# ---------------------------------------------------------------------------
# LR starts at 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("warmup_type", ["linear", "cosine", "exponential"])
def test_warmup_lr_starts_at_zero(warmup_type):
    """LR at step 0 should be 0 for all warmup types."""
    opt = _make_optimizer()
    base = _make_base_scheduler(opt)
    scheduler = WarmupScheduler(opt, base, warmup_steps=WARMUP_STEPS, warmup_type=warmup_type)
    lr0 = scheduler.get_last_lr()[0]
    assert lr0 == 0.0, f"warmup_type={warmup_type}: expected LR=0 at step 0, got {lr0}"


# ---------------------------------------------------------------------------
# LR reaches base_lr at warmup_steps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("warmup_type", ["linear", "cosine", "exponential"])
def test_warmup_lr_reaches_base_lr(warmup_type):
    """After exactly warmup_steps steps, LR should equal base_lr."""
    opt = _make_optimizer(lr=0.1)
    base = _make_base_scheduler(opt)
    scheduler = WarmupScheduler(opt, base, warmup_steps=WARMUP_STEPS, warmup_type=warmup_type)

    for _ in range(WARMUP_STEPS):
        scheduler.step()

    lr_at_warmup_end = scheduler.get_last_lr()[0]
    assert abs(lr_at_warmup_end - 0.1) < 1e-5, (
        f"warmup_type={warmup_type}: LR={lr_at_warmup_end} at warmup end, expected ~0.1"
    )


# ---------------------------------------------------------------------------
# LR is monotonically increasing during linear warmup
# ---------------------------------------------------------------------------


def test_linear_warmup_monotone_increasing():
    """Linear warmup LR should increase monotonically from 0 to base_lr."""
    opt = _make_optimizer(lr=0.1)
    base = _make_base_scheduler(opt)
    scheduler = WarmupScheduler(opt, base, warmup_steps=WARMUP_STEPS, warmup_type="linear")

    lrs = [scheduler.get_last_lr()[0]]
    for _ in range(WARMUP_STEPS):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    for i in range(1, len(lrs)):
        assert lrs[i] >= lrs[i - 1] - 1e-9, f"LR not monotone at step {i}: {lrs[i - 1]} -> {lrs[i]}"


# ---------------------------------------------------------------------------
# Cosine warmup: factor shape check
# ---------------------------------------------------------------------------


def test_cosine_warmup_factor_shape():
    """Cosine warmup should follow 0.5*(1-cos(pi*progress)) curve."""
    opt = _make_optimizer(lr=0.1)
    base = _make_base_scheduler(opt)
    scheduler = WarmupScheduler(opt, base, warmup_steps=WARMUP_STEPS, warmup_type="cosine")

    for step in range(1, WARMUP_STEPS):
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        progress = step / WARMUP_STEPS
        expected = 0.1 * 0.5 * (1.0 - math.cos(math.pi * progress))
        assert abs(lr - expected) < 1e-6, f"step={step}: cosine LR={lr}, expected={expected}"


# ---------------------------------------------------------------------------
# Exponential warmup: factor shape check
# ---------------------------------------------------------------------------


def test_exponential_warmup_factor_shape():
    """Exponential warmup should follow progress^2 curve."""
    opt = _make_optimizer(lr=0.1)
    base = _make_base_scheduler(opt)
    scheduler = WarmupScheduler(opt, base, warmup_steps=WARMUP_STEPS, warmup_type="exponential")

    for step in range(1, WARMUP_STEPS):
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        progress = step / WARMUP_STEPS
        expected = 0.1 * (progress**2)
        assert abs(lr - expected) < 1e-6, f"step={step}: exponential LR={lr}, expected={expected}"


# ---------------------------------------------------------------------------
# After warmup, base scheduler governs LR
# ---------------------------------------------------------------------------


def test_after_warmup_base_scheduler_used():
    """After warmup completes, the WarmupScheduler should delegate to base_scheduler."""
    opt = _make_optimizer(lr=0.1)
    base = _make_base_scheduler(opt, total_steps=TOTAL_STEPS)
    scheduler = WarmupScheduler(opt, base, warmup_steps=WARMUP_STEPS, warmup_type="linear")

    # Advance past warmup
    for _ in range(WARMUP_STEPS + 5):
        scheduler.step()

    # LR should be strictly between 0 and 0.1 and governed by linear decay
    lr = scheduler.get_last_lr()[0]
    assert 0.0 <= lr <= 0.1 + 1e-6, f"Post-warmup LR {lr} out of expected range"


def test_after_warmup_lr_is_decaying():
    """Post-warmup LR should not increase (base scheduler is LinearDecay)."""
    opt = _make_optimizer(lr=0.1)
    base = _make_base_scheduler(opt, total_steps=TOTAL_STEPS)
    scheduler = WarmupScheduler(opt, base, warmup_steps=WARMUP_STEPS, warmup_type="linear")

    # Advance to just past warmup
    for _ in range(WARMUP_STEPS):
        scheduler.step()

    # Collect a few post-warmup LRs
    post_lrs = []
    for _ in range(10):
        scheduler.step()
        post_lrs.append(scheduler.get_last_lr()[0])

    for i in range(1, len(post_lrs)):
        assert post_lrs[i] <= post_lrs[i - 1] + 1e-9, (
            f"Post-warmup LR increased at step {WARMUP_STEPS + i}: {post_lrs[i - 1]} -> {post_lrs[i]}"
        )


# ---------------------------------------------------------------------------
# state_dict / load_state_dict round-trip
# ---------------------------------------------------------------------------


def test_warmup_state_dict_round_trip():
    """state_dict/load_state_dict should preserve scheduler state exactly."""
    opt = _make_optimizer(lr=0.1)
    base = _make_base_scheduler(opt)
    scheduler = WarmupScheduler(opt, base, warmup_steps=WARMUP_STEPS, warmup_type="linear")

    # Advance a few steps
    for _ in range(15):
        scheduler.step()

    lr_before = scheduler.get_last_lr()[0]
    sd = scheduler.state_dict()

    # Verify state_dict contains base_scheduler key
    assert "base_scheduler" in sd

    # Create a fresh scheduler and load state
    opt2 = _make_optimizer(lr=0.1)
    base2 = _make_base_scheduler(opt2)
    scheduler2 = WarmupScheduler(opt2, base2, warmup_steps=WARMUP_STEPS, warmup_type="linear")
    scheduler2.load_state_dict(copy.deepcopy(sd))

    lr_after = scheduler2.get_last_lr()[0]
    assert abs(lr_before - lr_after) < 1e-9, f"LR mismatch after state_dict round-trip: {lr_before} vs {lr_after}"


def test_warmup_state_dict_contains_base():
    """state_dict should embed the base scheduler's state."""
    opt = _make_optimizer(lr=0.1)
    base = _make_base_scheduler(opt)
    scheduler = WarmupScheduler(opt, base, warmup_steps=WARMUP_STEPS)

    sd = scheduler.state_dict()
    assert "base_scheduler" in sd
    assert isinstance(sd["base_scheduler"], dict)


# ---------------------------------------------------------------------------
# LR values non-negative throughout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("warmup_type", ["linear", "cosine", "exponential"])
def test_warmup_all_lrs_non_negative(warmup_type):
    """All LRs should be non-negative for the entire schedule."""
    opt = _make_optimizer(lr=0.1)
    base = _make_base_scheduler(opt, total_steps=TOTAL_STEPS)
    scheduler = WarmupScheduler(opt, base, warmup_steps=WARMUP_STEPS, warmup_type=warmup_type)

    for _ in range(TOTAL_STEPS + 1):
        for lr in scheduler.get_last_lr():
            assert lr >= 0.0, f"Negative LR {lr} at step {scheduler.last_epoch}"
        scheduler.step()
