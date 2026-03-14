"""Tests for experimental module: SequentialComposer and ScheduleFreeWrapper."""

from __future__ import annotations

import warnings

import pytest
import torch

from pytorch_scheduler.experimental import ScheduleFreeWrapper, SequentialComposer
from pytorch_scheduler.scheduler import LinearDecayScheduler, RexScheduler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_optimizer(lr: float = 0.1) -> torch.optim.SGD:
    param = torch.randn(2, requires_grad=True)
    return torch.optim.SGD([param], lr=lr)


# ===========================================================================
# SequentialComposer
# ===========================================================================


class TestSequentialComposer:
    """Tests for SequentialComposer."""

    def test_construct(self):
        """Should construct with valid inputs."""
        opt = _make_optimizer()
        s1 = RexScheduler(opt, total_steps=100)
        s2 = LinearDecayScheduler(opt, total_steps=100)
        composer = SequentialComposer(opt, schedulers=[s1, s2], milestones=[50])
        assert composer is not None

    def test_step_through(self):
        """Should step through without exceptions."""
        opt = _make_optimizer()
        s1 = RexScheduler(opt, total_steps=50)
        s2 = LinearDecayScheduler(opt, total_steps=50)
        composer = SequentialComposer(opt, schedulers=[s1, s2], milestones=[50])
        for _ in range(100):
            composer.step()

    def test_milestone_transition(self):
        """LR should change appropriately at milestone boundaries."""
        opt = _make_optimizer(lr=0.1)
        s1 = RexScheduler(opt, total_steps=50)
        s2 = LinearDecayScheduler(opt, total_steps=50)
        composer = SequentialComposer(opt, schedulers=[s1, s2], milestones=[50])

        lrs = []
        for _ in range(100):
            lrs.append(composer.get_last_lr()[0])
            composer.step()
        # LR values should change over time (not all identical)
        assert len(set(f"{lr:.8f}" for lr in lrs)) > 1

    def test_single_scheduler_no_milestones(self):
        """Single scheduler with empty milestones should work."""
        opt = _make_optimizer()
        s1 = RexScheduler(opt, total_steps=100)
        composer = SequentialComposer(opt, schedulers=[s1], milestones=[])
        for _ in range(100):
            composer.step()
        # Should have completed without error

    def test_state_dict_round_trip(self):
        """state_dict save/load should reproduce LRs."""
        opt1 = _make_optimizer()
        s1 = RexScheduler(opt1, total_steps=50)
        s2 = LinearDecayScheduler(opt1, total_steps=50)
        composer1 = SequentialComposer(opt1, schedulers=[s1, s2], milestones=[50])

        # Step to midpoint
        for _ in range(30):
            composer1.step()
        saved = composer1.state_dict()
        lr_at_save = composer1.get_last_lr()

        # Create new and load
        opt2 = _make_optimizer()
        s1b = RexScheduler(opt2, total_steps=50)
        s2b = LinearDecayScheduler(opt2, total_steps=50)
        composer2 = SequentialComposer(opt2, schedulers=[s1b, s2b], milestones=[50])
        composer2.load_state_dict(saved)

        lr_loaded = composer2.get_last_lr()
        for a, b in zip(lr_at_save, lr_loaded, strict=True):
            assert abs(a - b) < 1e-7

    # --- Validation tests ---

    def test_milestones_count_mismatch(self):
        """Should raise when milestones count != len(schedulers) - 1."""
        opt = _make_optimizer()
        s1 = RexScheduler(opt, total_steps=100)
        s2 = LinearDecayScheduler(opt, total_steps=100)
        with pytest.raises(ValueError, match="milestones"):
            SequentialComposer(opt, schedulers=[s1, s2], milestones=[50, 70])

    def test_milestones_unsorted(self):
        """Should raise when milestones are not sorted."""
        opt = _make_optimizer()
        s1 = RexScheduler(opt, total_steps=100)
        s2 = LinearDecayScheduler(opt, total_steps=100)
        s3 = RexScheduler(opt, total_steps=100)
        with pytest.raises(ValueError, match="sorted"):
            SequentialComposer(opt, schedulers=[s1, s2, s3], milestones=[70, 50])

    def test_milestones_negative(self):
        """Should raise when milestones contain negative values."""
        opt = _make_optimizer()
        s1 = RexScheduler(opt, total_steps=100)
        s2 = LinearDecayScheduler(opt, total_steps=100)
        with pytest.raises(ValueError, match="non-negative"):
            SequentialComposer(opt, schedulers=[s1, s2], milestones=[-1])


# ===========================================================================
# ScheduleFreeWrapper
# ===========================================================================


class TestScheduleFreeWrapper:
    """Tests for ScheduleFreeWrapper."""

    def test_construct(self):
        """Should construct with valid inputs."""
        opt = _make_optimizer()
        wrapper = ScheduleFreeWrapper(opt, warmup_steps=10, beta=0.9)
        assert wrapper is not None

    def test_step(self):
        """Should step without exceptions."""
        opt = _make_optimizer()
        wrapper = ScheduleFreeWrapper(opt, warmup_steps=5, beta=0.9)
        for _ in range(20):
            wrapper.step()

    def test_eval_train_mode(self):
        """Should switch between eval and train modes."""
        opt = _make_optimizer()
        wrapper = ScheduleFreeWrapper(opt, warmup_steps=5, beta=0.9)
        for _ in range(10):
            wrapper.step()

        # Switch to eval - parameters should change to averaged
        wrapper.eval()
        assert not wrapper._training

        # Switch back to train
        wrapper.train()
        assert wrapper._training

    def test_eval_idempotent(self):
        """Calling eval() twice should be safe."""
        opt = _make_optimizer()
        wrapper = ScheduleFreeWrapper(opt, warmup_steps=5, beta=0.9)
        for _ in range(5):
            wrapper.step()
        wrapper.eval()
        wrapper.eval()  # Should not error
        assert not wrapper._training

    def test_train_idempotent(self):
        """Calling train() twice should be safe."""
        opt = _make_optimizer()
        wrapper = ScheduleFreeWrapper(opt, warmup_steps=5, beta=0.9)
        wrapper.train()
        wrapper.train()  # Should not error
        assert wrapper._training

    def test_warmup_factor(self):
        """Warmup factor should ramp from 0 to 1."""
        opt = _make_optimizer()
        wrapper = ScheduleFreeWrapper(opt, warmup_steps=10, beta=0.9)

        # At step 0, warmup factor is 0
        assert wrapper._warmup_factor() == 0.0

        # Step halfway through warmup
        for _ in range(5):
            wrapper.step()
        assert abs(wrapper._warmup_factor() - 0.5) < 1e-6

        # Step to end of warmup
        for _ in range(5):
            wrapper.step()
        assert abs(wrapper._warmup_factor() - 1.0) < 1e-6

    def test_warmup_zero_steps(self):
        """With warmup_steps=0, warmup factor should always be 1."""
        opt = _make_optimizer()
        wrapper = ScheduleFreeWrapper(opt, warmup_steps=0, beta=0.9)
        assert wrapper._warmup_factor() == 1.0
        wrapper.step()
        assert wrapper._warmup_factor() == 1.0

    def test_state_dict_round_trip(self):
        """state_dict save/load should preserve state."""
        opt1 = _make_optimizer()
        wrapper1 = ScheduleFreeWrapper(opt1, warmup_steps=5, beta=0.9)
        for _ in range(10):
            wrapper1.step()

        saved = wrapper1.state_dict()
        assert saved["step_count"] == 10

        # Create new and load
        opt2 = _make_optimizer()
        wrapper2 = ScheduleFreeWrapper(opt2, warmup_steps=5, beta=0.9)
        wrapper2.load_state_dict(saved)
        assert wrapper2._step_count == 10

    # --- Validation tests ---

    def test_negative_warmup_steps(self):
        """Should raise on negative warmup_steps."""
        opt = _make_optimizer()
        with pytest.raises(ValueError, match="warmup_steps"):
            ScheduleFreeWrapper(opt, warmup_steps=-1)

    def test_beta_too_high(self):
        """Should raise when beta >= 1."""
        opt = _make_optimizer()
        with pytest.raises(ValueError, match="beta"):
            ScheduleFreeWrapper(opt, beta=1.0)

    def test_beta_negative(self):
        """Should raise when beta < 0."""
        opt = _make_optimizer()
        with pytest.raises(ValueError, match="beta"):
            ScheduleFreeWrapper(opt, beta=-0.1)
