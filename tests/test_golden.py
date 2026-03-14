"""Golden tests: verify scheduler outputs against hand-computed values from source papers."""

from __future__ import annotations

import math
import warnings

import pytest
import torch

import pytorch_scheduler as ps

warnings.filterwarnings("ignore")


def make_optimizer(lr: float = 0.1) -> torch.optim.SGD:
    model = torch.nn.Linear(2, 1)
    return torch.optim.SGD(model.parameters(), lr=lr)


# ---------------------------------------------------------------------------
# 1. RexScheduler
# Formula: lr = base_lr * (1 - t) / (1 - t/2)  where t = step / total_steps
# base_lr=0.1, total_steps=1000
# ---------------------------------------------------------------------------


class TestRexSchedulerGolden:
    def setup_method(self):
        self.opt = make_optimizer(lr=0.1)
        self.sched = ps.RexScheduler(self.opt, total_steps=1000)
        self.base_lrs = [0.1]

    def test_step_0(self):
        # t=0 → factor = 1.0 → lr = 0.1
        result = self.sched._lr_at(0, self.base_lrs)
        assert result[0] == pytest.approx(0.1, abs=1e-6)

    def test_step_500(self):
        # t=0.5 → factor = 0.5 / 0.75 = 0.6667 → lr ≈ 0.06667
        t = 0.5
        expected = 0.1 * (1 - t) / (1 - t / 2)  # = 0.06666...
        result = self.sched._lr_at(500, self.base_lrs)
        assert result[0] == pytest.approx(expected, abs=1e-6)

    def test_step_999(self):
        # t=0.999 → factor = 0.001 / 0.5005 ≈ 0.001998 → lr ≈ 0.0001998
        t = 0.999
        expected = 0.1 * (1 - t) / (1 - t / 2)
        result = self.sched._lr_at(999, self.base_lrs)
        assert result[0] == pytest.approx(expected, abs=1e-6)

    def test_step_1000_is_zero(self):
        # At total_steps, LR reaches 0
        result = self.sched._lr_at(1000, self.base_lrs)
        assert result[0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. KDecayScheduler
# Formula: lr = min_lr + 0.5*(base_lr - min_lr)*(1 + cos(pi*(step/total)^k))
# base_lr=0.1, total_steps=1000, k=2.0, min_lr=0.0
# ---------------------------------------------------------------------------


class TestKDecaySchedulerGolden:
    def setup_method(self):
        self.opt = make_optimizer(lr=0.1)
        self.sched = ps.KDecayScheduler(self.opt, total_steps=1000, k=2.0, min_lr=0.0)
        self.base_lrs = [0.1]

    def test_step_0(self):
        # At step 0, returns base_lr
        result = self.sched._lr_at(0, self.base_lrs)
        assert result[0] == pytest.approx(0.1, abs=1e-6)

    def test_step_500(self):
        # t = (500/1000)^2 = 0.25
        # cosine_factor = 0.5*(1 + cos(pi*0.25)) = 0.5*(1 + cos(45°))
        # cos(45°) = sqrt(2)/2 ≈ 0.7071
        # lr = 0.5 * 0.1 * (1 + 0.7071) = 0.08536
        t = (500 / 1000) ** 2.0
        expected = 0.5 * 0.1 * (1.0 + math.cos(math.pi * t))
        result = self.sched._lr_at(500, self.base_lrs)
        assert result[0] == pytest.approx(expected, abs=1e-6)

    def test_step_1000_is_min_lr(self):
        # At total_steps, returns min_lr = 0.0
        result = self.sched._lr_at(1000, self.base_lrs)
        assert result[0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. InverseSqrtScheduler
# Formula: lr = base_lr * min(step^{-0.5}, step * warmup_steps^{-1.5}) * sqrt(warmup_steps)
# base_lr=0.1, warmup_steps=100
# ---------------------------------------------------------------------------


class TestInverseSqrtSchedulerGolden:
    def setup_method(self):
        self.opt = make_optimizer(lr=0.1)
        self.sched = ps.InverseSqrtScheduler(self.opt, warmup_steps=100)
        self.base_lrs = [0.1]

    def test_step_0(self):
        # step=0 → avoid div by zero → return 0.0
        result = self.sched._lr_at(0, self.base_lrs)
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_step_50_warmup_phase(self):
        # step=50, warmup=100: linear_warmup = 50 * 100^{-1.5} = 50/1000 = 0.05
        # inv_sqrt = 50^{-0.5} ≈ 0.1414
        # factor = min(0.1414, 0.05) = 0.05 → norm = 0.05 * sqrt(100) = 0.5
        # lr = 0.1 * 0.5 = 0.05
        result = self.sched._lr_at(50, self.base_lrs)
        assert result[0] == pytest.approx(0.05, abs=1e-6)

    def test_step_100_at_warmup_boundary(self):
        # Both branches meet: inv_sqrt(100) = 0.1, linear(100) = 100/1000 = 0.1
        # norm = 0.1 * 10 = 1.0 → lr = 0.1 * 1.0 = 0.1
        result = self.sched._lr_at(100, self.base_lrs)
        assert result[0] == pytest.approx(0.1, abs=1e-6)

    def test_step_400_decay_phase(self):
        # step=400: inv_sqrt(400) = 1/20 = 0.05, linear = 400/1000=0.4 → min=0.05
        # norm = 0.05 * sqrt(100) = 0.05 * 10 = 0.5 → lr = 0.1 * 0.5 = 0.05
        result = self.sched._lr_at(400, self.base_lrs)
        assert result[0] == pytest.approx(0.05, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. LinearDecayScheduler
# Linear warmup then linear decay
# base_lr=0.1, total_steps=1000, warmup_steps=100, min_lr=0.0
# ---------------------------------------------------------------------------


class TestLinearDecaySchedulerGolden:
    def setup_method(self):
        self.opt = make_optimizer(lr=0.1)
        self.sched = ps.LinearDecayScheduler(self.opt, total_steps=1000, warmup_steps=100, min_lr=0.0)
        self.base_lrs = [0.1]

    def test_step_0(self):
        # Warmup start → 0.0
        result = self.sched._lr_at(0, self.base_lrs)
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_step_50_warmup(self):
        # Warmup: 0.1 * 50/100 = 0.05
        result = self.sched._lr_at(50, self.base_lrs)
        assert result[0] == pytest.approx(0.05, abs=1e-6)

    def test_step_100_peak(self):
        # End of warmup / start of decay: base_lr
        # progress = (100-100)/900 = 0.0 → lr = 0.1
        result = self.sched._lr_at(100, self.base_lrs)
        assert result[0] == pytest.approx(0.1, abs=1e-6)

    def test_step_550_midpoint(self):
        # Decay: progress = (550-100)/900 = 0.5 → lr = 0.1 + (0-0.1)*0.5 = 0.05
        result = self.sched._lr_at(550, self.base_lrs)
        assert result[0] == pytest.approx(0.05, abs=1e-6)

    def test_step_1000_min_lr(self):
        result = self.sched._lr_at(1000, self.base_lrs)
        assert result[0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 5. TanhDecayScheduler
# Formula: lr = min_lr + 0.5*(base_lr-min_lr)*(1 - tanh(steepness*(2*t-1)))
# base_lr=0.1, total_steps=1000, steepness=3.0, min_lr=0.0
# ---------------------------------------------------------------------------


class TestTanhDecaySchedulerGolden:
    def setup_method(self):
        self.opt = make_optimizer(lr=0.1)
        self.sched = ps.TanhDecayScheduler(self.opt, total_steps=1000, steepness=3.0, min_lr=0.0)
        self.base_lrs = [0.1]

    def test_step_0_near_base_lr(self):
        # t=0 → factor = 0.5*(1 - tanh(3*(0-1))) = 0.5*(1 - tanh(-3)) ≈ 0.9975
        # lr ≈ 0.09975
        t = 0.0
        expected = 0.0 + (0.1 - 0.0) * 0.5 * (1.0 - math.tanh(3.0 * (2.0 * t - 1.0)))
        result = self.sched._lr_at(0, self.base_lrs)
        assert result[0] == pytest.approx(expected, abs=1e-6)

    def test_step_500_midpoint(self):
        # t=0.5 → factor = 0.5*(1 - tanh(0)) = 0.5*(1-0) = 0.5 → lr = 0.05
        result = self.sched._lr_at(500, self.base_lrs)
        assert result[0] == pytest.approx(0.05, abs=1e-6)

    def test_step_1000_min_lr(self):
        result = self.sched._lr_at(1000, self.base_lrs)
        assert result[0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 6. SlantedTriangularScheduler
# cut_frac=0.1 → cut_point=100, ratio=32.0, total=1000
# lr = base_lr * (1 + p*(ratio-1)) / ratio
# ---------------------------------------------------------------------------


class TestSlantedTriangularSchedulerGolden:
    def setup_method(self):
        self.opt = make_optimizer(lr=0.1)
        self.sched = ps.SlantedTriangularScheduler(self.opt, total_steps=1000, cut_frac=0.1, ratio=32.0)
        self.base_lrs = [0.1]

    def test_step_0_start_of_warmup(self):
        # p=0 → factor = 1/32 = 0.03125 → lr = 0.003125
        result = self.sched._lr_at(0, self.base_lrs)
        expected = 0.1 * (1.0 + 0.0 * (32.0 - 1.0)) / 32.0  # = 0.003125
        assert result[0] == pytest.approx(expected, abs=1e-6)

    def test_step_50_mid_warmup(self):
        # p = 50/100 = 0.5 → factor = (1 + 0.5*31)/32 = 16.5/32 ≈ 0.515625
        p = 50 / 100
        expected = 0.1 * (1.0 + p * (32.0 - 1.0)) / 32.0
        result = self.sched._lr_at(50, self.base_lrs)
        assert result[0] == pytest.approx(expected, abs=1e-6)

    def test_step_100_peak(self):
        # p=1.0 (at cut_point) → factor = 32/32 = 1.0 → lr = 0.1
        result = self.sched._lr_at(100, self.base_lrs)
        assert result[0] == pytest.approx(0.1, abs=1e-6)

    def test_step_1000_end(self):
        # p at total_steps: 1 - (1000-100)/900 = 0.0 → factor = 1/32
        result = self.sched._lr_at(1000, self.base_lrs)
        expected = 0.1 * (1.0 + 0.0 * (32.0 - 1.0)) / 32.0  # = 0.003125
        assert result[0] == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# 7. WSDScheduler
# Warmup-Stable-Decay: warmup=100, stable=500, total=1000, cosine decay
# stable_end=600, decay_steps=400
# ---------------------------------------------------------------------------


class TestWSDSchedulerGolden:
    def setup_method(self):
        self.opt = make_optimizer(lr=0.1)
        self.sched = ps.WSDScheduler(
            self.opt,
            total_steps=1000,
            warmup_steps=100,
            stable_steps=500,
            decay_type="cosine",
            min_lr=0.0,
        )
        self.base_lrs = [0.1]

    def test_step_0(self):
        result = self.sched._lr_at(0, self.base_lrs)
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_step_50_warmup(self):
        # progress = 50/100 = 0.5 → lr = 0.05
        result = self.sched._lr_at(50, self.base_lrs)
        assert result[0] == pytest.approx(0.05, abs=1e-6)

    def test_step_300_stable(self):
        # In stable phase (100 ≤ 300 < 600) → lr = base_lr = 0.1
        result = self.sched._lr_at(300, self.base_lrs)
        assert result[0] == pytest.approx(0.1, abs=1e-6)

    def test_step_800_decay(self):
        # progress = (800-600)/400 = 0.5
        # cosine: 0.5*(1+cos(pi*0.5)) = 0.5*(1+0) = 0.5 → lr = 0.05
        progress = (800 - 600) / 400
        expected = 0.0 + (0.1 - 0.0) * 0.5 * (1.0 + math.cos(math.pi * progress))
        result = self.sched._lr_at(800, self.base_lrs)
        assert result[0] == pytest.approx(expected, abs=1e-6)

    def test_step_1000_min_lr(self):
        result = self.sched._lr_at(1000, self.base_lrs)
        assert result[0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 8. PowerDecayScheduler
# Power-law decay: warmup=100, alpha=0.5, min_lr=0.001
# After warmup: lr = max(base_lr * (step/warmup_steps)^(-alpha), min_lr)
# ---------------------------------------------------------------------------


class TestPowerDecaySchedulerGolden:
    def setup_method(self):
        self.opt = make_optimizer(lr=0.1)
        self.sched = ps.PowerDecayScheduler(self.opt, total_steps=1000, warmup_steps=100, alpha=0.5, min_lr=0.001)
        self.base_lrs = [0.1]

    def test_step_0(self):
        # Warmup start → 0.0
        result = self.sched._lr_at(0, self.base_lrs)
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_step_50_warmup(self):
        # progress = 50/100 = 0.5 → lr = 0.05
        result = self.sched._lr_at(50, self.base_lrs)
        assert result[0] == pytest.approx(0.05, abs=1e-6)

    def test_step_100_at_warmup_end(self):
        # (100/100)^(-0.5) = 1.0 → lr = max(0.1 * 1.0, 0.001) = 0.1
        result = self.sched._lr_at(100, self.base_lrs)
        assert result[0] == pytest.approx(0.1, abs=1e-6)

    def test_step_400_decay(self):
        # (400/100)^(-0.5) = 4^(-0.5) = 0.5 → lr = max(0.1*0.5, 0.001) = 0.05
        result = self.sched._lr_at(400, self.base_lrs)
        assert result[0] == pytest.approx(0.05, abs=1e-6)


# ---------------------------------------------------------------------------
# 9. CosineAnnealingWarmupRestarts
# first_cycle_steps=500, warmup_steps=50, max_lr=0.1, min_lr=0.001, gamma=1.0
# ---------------------------------------------------------------------------


class TestCosineAnnealingWarmupRestartsGolden:
    def setup_method(self):
        self.opt = make_optimizer(lr=0.1)
        self.sched = ps.CosineAnnealingWarmupRestarts(
            self.opt,
            first_cycle_steps=500,
            warmup_steps=50,
            max_lr=0.1,
            min_lr=0.001,
            gamma=1.0,
            cycle_mult=1.0,
        )

    def test_step_0_starts_at_min_lr(self):
        # step_in_cycle=0 → min_lr
        result = self.sched._lr_at(0, [0.1])
        assert result[0] == pytest.approx(0.001, abs=1e-6)

    def test_step_25_warmup(self):
        # progress = 25/50 = 0.5 → lr = 0.001 + (0.1-0.001)*0.5 = 0.0505
        expected = 0.001 + (0.1 - 0.001) * 0.5
        result = self.sched._lr_at(25, [0.1])
        assert result[0] == pytest.approx(expected, abs=1e-6)

    def test_step_50_at_max_lr(self):
        # End of warmup → max_lr
        # cosine_steps = 500-50 = 450, progress = (50-50)/450 = 0.0 → cosf = 1.0
        # lr = 0.001 + (0.1-0.001)*1.0 = 0.1
        result = self.sched._lr_at(50, [0.1])
        assert result[0] == pytest.approx(0.1, abs=1e-6)

    def test_step_275_midpoint(self):
        # cosine_steps = 450, progress = (275-50)/450 = 225/450 = 0.5
        # cosine_factor = 0.5*(1+cos(pi*0.5)) = 0.5*(1+0) = 0.5
        # lr = 0.001 + (0.1-0.001)*0.5 = 0.0505
        expected = 0.001 + (0.1 - 0.001) * 0.5 * (1.0 + math.cos(math.pi * 0.5))
        result = self.sched._lr_at(275, [0.1])
        assert result[0] == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# 10. HyperbolicLRScheduler
# Formula after warmup: lr = base_lr + (base_lr - min_lr) * (f(x) - f(0))
# total_steps=100, upper_bound=250, warmup_steps=10, min_lr=1e-6
# decay_steps N=90, U=250
# ---------------------------------------------------------------------------


class TestHyperbolicLRSchedulerGolden:
    def setup_method(self):
        self.opt = make_optimizer(lr=0.1)
        self.sched = ps.HyperbolicLRScheduler(self.opt, total_steps=100, upper_bound=250, min_lr=1e-6, warmup_steps=10)
        self.base_lrs = [0.1]
        # Precompute f(0)
        n, u = 90, 250
        self.f0 = math.sqrt(n / u * (2.0 - n / u))

    def test_step_0_starts_at_min_lr(self):
        result = self.sched._lr_at(0, self.base_lrs)
        assert result[0] == pytest.approx(1e-6, abs=1e-9)

    def test_step_5_warmup(self):
        # progress = 5/10 = 0.5 → lr = 1e-6 + (0.1-1e-6)*0.5 ≈ 0.0500005
        expected = 1e-6 + (0.1 - 1e-6) * 0.5
        result = self.sched._lr_at(5, self.base_lrs)
        assert result[0] == pytest.approx(expected, abs=1e-6)

    def test_step_10_at_warmup_end(self):
        # End of warmup → base_lr
        result = self.sched._lr_at(10, self.base_lrs)
        assert result[0] == pytest.approx(0.1, abs=1e-6)

    def test_step_100_finite_and_positive(self):
        # Fully decayed — should be finite and positive
        result = self.sched._lr_at(100, self.base_lrs)
        assert math.isfinite(result[0])
        assert result[0] >= 0.0


# ---------------------------------------------------------------------------
# 11. ExpHyperbolicLRScheduler
# Formula after warmup: lr = base_lr * (base_lr/min_lr)^(f(x) - f(0))
# total_steps=100, upper_bound=250, warmup_steps=10, min_lr=1e-6
# ---------------------------------------------------------------------------


class TestExpHyperbolicLRSchedulerGolden:
    def setup_method(self):
        self.opt = make_optimizer(lr=0.1)
        self.sched = ps.ExpHyperbolicLRScheduler(
            self.opt, total_steps=100, upper_bound=250, min_lr=1e-6, warmup_steps=10
        )
        self.base_lrs = [0.1]

    def test_step_0_starts_at_min_lr(self):
        result = self.sched._lr_at(0, self.base_lrs)
        assert result[0] == pytest.approx(1e-6, abs=1e-9)

    def test_step_5_warmup(self):
        expected = 1e-6 + (0.1 - 1e-6) * 0.5
        result = self.sched._lr_at(5, self.base_lrs)
        assert result[0] == pytest.approx(expected, abs=1e-6)

    def test_step_10_at_warmup_end(self):
        result = self.sched._lr_at(10, self.base_lrs)
        assert result[0] == pytest.approx(0.1, abs=1e-6)

    def test_step_100_finite_and_positive(self):
        result = self.sched._lr_at(100, self.base_lrs)
        assert math.isfinite(result[0])
        assert result[0] >= 0.0


# ---------------------------------------------------------------------------
# 12. CosineWithWarmupScheduler
# Warmup + cosine decay
# base_lr=0.1, total_steps=1000, warmup_steps=100, min_lr=0.0
# ---------------------------------------------------------------------------


class TestCosineWithWarmupSchedulerGolden:
    def setup_method(self):
        self.opt = make_optimizer(lr=0.1)
        self.sched = ps.CosineWithWarmupScheduler(self.opt, total_steps=1000, warmup_steps=100, min_lr=0.0)
        self.base_lrs = [0.1]

    def test_step_0(self):
        result = self.sched._lr_at(0, self.base_lrs)
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_step_50_warmup(self):
        # lr = 0.1 * 50/100 = 0.05
        result = self.sched._lr_at(50, self.base_lrs)
        assert result[0] == pytest.approx(0.05, abs=1e-6)

    def test_step_100_peak(self):
        # End of warmup: cosine_progress = 0.0 → cosf = 1.0 → lr = 0.1
        result = self.sched._lr_at(100, self.base_lrs)
        assert result[0] == pytest.approx(0.1, abs=1e-6)

    def test_step_550_midpoint(self):
        # progress = (550-100)/900 = 0.5
        # cosine_factor = 0.5*(1+cos(pi*0.5)) = 0.5*(1+0) = 0.5 → lr = 0.05
        progress = (550 - 100) / 900
        expected = 0.0 + (0.1 - 0.0) * 0.5 * (1.0 + math.cos(math.pi * progress))
        result = self.sched._lr_at(550, self.base_lrs)
        assert result[0] == pytest.approx(expected, abs=1e-6)

    def test_step_1000_min_lr(self):
        result = self.sched._lr_at(1000, self.base_lrs)
        assert result[0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 13. WarmupHoldCosineScheduler
# Warmup + hold + cosine
# base_lr=0.1, total_steps=1000, warmup_steps=100, hold_steps=400, min_lr=0.0
# hold_end = 500, decay_steps = 500
# ---------------------------------------------------------------------------


class TestWarmupHoldCosineSchedulerGolden:
    def setup_method(self):
        self.opt = make_optimizer(lr=0.1)
        self.sched = ps.WarmupHoldCosineScheduler(
            self.opt,
            total_steps=1000,
            warmup_steps=100,
            hold_steps=400,
            min_lr=0.0,
        )
        self.base_lrs = [0.1]

    def test_step_0(self):
        result = self.sched._lr_at(0, self.base_lrs)
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_step_50_warmup(self):
        # lr = 0.1 * 50/100 = 0.05
        result = self.sched._lr_at(50, self.base_lrs)
        assert result[0] == pytest.approx(0.05, abs=1e-6)

    def test_step_100_peak(self):
        result = self.sched._lr_at(100, self.base_lrs)
        assert result[0] == pytest.approx(0.1, abs=1e-6)

    def test_step_300_hold_phase(self):
        # In hold phase (100 ≤ 300 < 500) → lr = base_lr = 0.1
        result = self.sched._lr_at(300, self.base_lrs)
        assert result[0] == pytest.approx(0.1, abs=1e-6)

    def test_step_750_cosine_decay(self):
        # progress = (750-500)/500 = 0.5
        # cosine_factor = 0.5*(1+cos(pi*0.5)) = 0.5*(1+0) = 0.5 → lr = 0.05
        progress = (750 - 500) / 500
        expected = 0.0 + (0.1 - 0.0) * 0.5 * (1.0 + math.cos(math.pi * progress))
        result = self.sched._lr_at(750, self.base_lrs)
        assert result[0] == pytest.approx(expected, abs=1e-6)

    def test_step_1000_min_lr(self):
        result = self.sched._lr_at(1000, self.base_lrs)
        assert result[0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 14. FlatCosineScheduler
# Flat phase then cosine decay
# base_lr=0.1, total_steps=1000, flat_fraction=0.7, min_lr=0.0
# flat_end = 700
# ---------------------------------------------------------------------------


class TestFlatCosineSchedulerGolden:
    def setup_method(self):
        self.opt = make_optimizer(lr=0.1)
        self.sched = ps.FlatCosineScheduler(self.opt, total_steps=1000, flat_fraction=0.7, min_lr=0.0)
        self.base_lrs = [0.1]

    def test_step_0_flat_phase(self):
        # Flat phase → base_lr
        result = self.sched._lr_at(0, self.base_lrs)
        assert result[0] == pytest.approx(0.1, abs=1e-6)

    def test_step_350_flat_phase(self):
        result = self.sched._lr_at(350, self.base_lrs)
        assert result[0] == pytest.approx(0.1, abs=1e-6)

    def test_step_700_flat_boundary(self):
        # flat_end=700, step<=flat_end → still base_lr
        result = self.sched._lr_at(700, self.base_lrs)
        assert result[0] == pytest.approx(0.1, abs=1e-6)

    def test_step_850_cosine(self):
        # cosine_steps = 1000-700 = 300, progress = (850-700)/300 = 0.5
        # cosine_factor = 0.5*(1+cos(pi*0.5)) = 0.5 → lr = 0.05
        progress = (850 - 700) / 300
        expected = 0.0 + (0.1 - 0.0) * 0.5 * (1.0 + math.cos(math.pi * progress))
        result = self.sched._lr_at(850, self.base_lrs)
        assert result[0] == pytest.approx(expected, abs=1e-6)

    def test_step_1000_min_lr(self):
        result = self.sched._lr_at(1000, self.base_lrs)
        assert result[0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 15. PolynomialScheduler
# Formula: lr = (base_lr - min_lr) * (1 - progress)^power + min_lr
# base_lr=0.1, total_steps=100, power=2.0, min_lr=0.0, warmup_steps=10
# decay_steps = 90
# ---------------------------------------------------------------------------


class TestPolynomialSchedulerGolden:
    def setup_method(self):
        self.opt = make_optimizer(lr=0.1)
        self.sched = ps.PolynomialScheduler(self.opt, total_steps=100, power=2.0, min_lr=0.0, warmup_steps=10)
        self.base_lrs = [0.1]

    def test_step_0(self):
        result = self.sched._lr_at(0, self.base_lrs)
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_step_5_warmup(self):
        # lr = 0.1 * 5/10 = 0.05
        result = self.sched._lr_at(5, self.base_lrs)
        assert result[0] == pytest.approx(0.05, abs=1e-6)

    def test_step_10_peak(self):
        # progress = (10-10)/90 = 0.0 → lr = 0.1 * 1.0 = 0.1
        result = self.sched._lr_at(10, self.base_lrs)
        assert result[0] == pytest.approx(0.1, abs=1e-6)

    def test_step_55_decay(self):
        # progress = (55-10)/90 = 45/90 = 0.5
        # lr = 0.1 * (1-0.5)^2 = 0.1 * 0.25 = 0.025
        progress = (55 - 10) / 90
        expected = (0.1 - 0.0) * ((1.0 - progress) ** 2.0) + 0.0
        result = self.sched._lr_at(55, self.base_lrs)
        assert result[0] == pytest.approx(expected, abs=1e-6)

    def test_step_100_min_lr(self):
        result = self.sched._lr_at(100, self.base_lrs)
        assert result[0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 16. ChebyshevScheduler
# Formula: x_j = cos(pi*(2j+1)/(2*total_steps))
# lr = min_lr + 0.5*(base_lr - min_lr)*(1 + x_j)
# base_lr=0.1, total_steps=100, min_lr=0.0
# ---------------------------------------------------------------------------


class TestChebyshevSchedulerGolden:
    def setup_method(self):
        self.opt = make_optimizer(lr=0.1)
        self.sched = ps.ChebyshevScheduler(self.opt, total_steps=100, min_lr=0.0)
        self.base_lrs = [0.1]

    def test_step_0_near_base_lr(self):
        # j=0: x_0 = cos(pi*1/200) ≈ 0.999877 → scale ≈ 0.9999 → lr ≈ 0.09999
        j = 0
        xj = math.cos(math.pi * (2 * j + 1) / (2 * 100))
        expected = 0.0 + (0.1 - 0.0) * 0.5 * (1.0 + xj)
        result = self.sched._lr_at(0, self.base_lrs)
        assert result[0] == pytest.approx(expected, abs=1e-6)

    def test_step_50_near_midpoint(self):
        # j=50: x_50 = cos(pi*101/200) ≈ -0.01571 → lr ≈ 0.04921
        j = 50
        xj = math.cos(math.pi * (2 * j + 1) / (2 * 100))
        expected = 0.0 + (0.1 - 0.0) * 0.5 * (1.0 + xj)
        result = self.sched._lr_at(50, self.base_lrs)
        assert result[0] == pytest.approx(expected, abs=1e-6)

    def test_step_99_near_min_lr(self):
        # j=99: x_99 = cos(pi*199/200) ≈ -0.999877 → lr ≈ 0.0
        j = 99
        xj = math.cos(math.pi * (2 * j + 1) / (2 * 100))
        expected = 0.0 + (0.1 - 0.0) * 0.5 * (1.0 + xj)
        result = self.sched._lr_at(99, self.base_lrs)
        assert result[0] == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# 17. TrapezoidalScheduler
# Three phases: linear warmup → constant → linear decay
# base_lr=0.1, total_steps=1000, warmup_steps=100, decay_steps=200, min_lr=0.0
# decay_start = 800
# ---------------------------------------------------------------------------


class TestTrapezoidalSchedulerGolden:
    def setup_method(self):
        self.opt = make_optimizer(lr=0.1)
        self.sched = ps.TrapezoidalScheduler(self.opt, total_steps=1000, warmup_steps=100, decay_steps=200, min_lr=0.0)
        self.base_lrs = [0.1]

    def test_step_0(self):
        result = self.sched._lr_at(0, self.base_lrs)
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_step_50_warmup(self):
        # lr = 0.1 * 50/100 = 0.05
        result = self.sched._lr_at(50, self.base_lrs)
        assert result[0] == pytest.approx(0.05, abs=1e-6)

    def test_step_300_constant(self):
        # In stable phase (100 ≤ 300 < 800) → lr = 0.1
        result = self.sched._lr_at(300, self.base_lrs)
        assert result[0] == pytest.approx(0.1, abs=1e-6)

    def test_step_900_decay(self):
        # progress = (900-800)/200 = 0.5 → lr = 0.1 + (0-0.1)*0.5 = 0.05
        progress = (900 - 800) / 200
        expected = 0.1 + (0.0 - 0.1) * progress
        result = self.sched._lr_at(900, self.base_lrs)
        assert result[0] == pytest.approx(expected, abs=1e-6)

    def test_step_1000_min_lr(self):
        result = self.sched._lr_at(1000, self.base_lrs)
        assert result[0] == pytest.approx(0.0, abs=1e-6)
