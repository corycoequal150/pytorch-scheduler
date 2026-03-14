"""Visualization examples for pytorch_scheduler."""

import torch

from pytorch_scheduler import (
    FlatCosineScheduler,
    KDecayScheduler,
    RexScheduler,
    WSDScheduler,
)
from pytorch_scheduler.visualization import compare_schedules, plot_schedule

TOTAL_STEPS = 10000


def make_optimizer():
    return torch.optim.SGD([torch.randn(2, requires_grad=True)], lr=0.1)


# 1. Plot a single schedule
scheduler = WSDScheduler(
    make_optimizer(),
    total_steps=TOTAL_STEPS,
    warmup_steps=500,
    stable_steps=5000,
)
fig = plot_schedule(scheduler, total_steps=TOTAL_STEPS, title="WSD Schedule")
fig.savefig("wsd_schedule.png", dpi=150)
print("Saved wsd_schedule.png")

# 2. Compare multiple schedules
schedulers = {
    "REX": RexScheduler(make_optimizer(), total_steps=TOTAL_STEPS),
    "WSD (cosine)": WSDScheduler(
        make_optimizer(),
        total_steps=TOTAL_STEPS,
        warmup_steps=500,
        stable_steps=5000,
    ),
    "Flat+Cosine": FlatCosineScheduler(
        make_optimizer(),
        total_steps=TOTAL_STEPS,
    ),
    "K-Decay (k=3)": KDecayScheduler(
        make_optimizer(),
        total_steps=TOTAL_STEPS,
        k=3.0,
    ),
}
fig = compare_schedules(schedulers, total_steps=TOTAL_STEPS)
fig.savefig("schedule_comparison.png", dpi=150)
print("Saved schedule_comparison.png")
