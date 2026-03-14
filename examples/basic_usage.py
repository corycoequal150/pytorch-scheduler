"""Basic usage examples for pytorch_scheduler."""

import torch

from pytorch_scheduler import (
    CosineAnnealingWarmupRestarts,
    WSDScheduler,
    create_scheduler,
)

# 1. Direct usage
model = torch.nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = WSDScheduler(
    optimizer,
    total_steps=10000,
    warmup_steps=500,
    stable_steps=5000,
    decay_type="cosine",
)

# Training loop
for _step in range(100):
    optimizer.zero_grad()
    loss = model(torch.randn(4, 10)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")

# 2. Factory usage
optimizer2 = torch.optim.SGD(model.parameters(), lr=0.01)
scheduler2 = create_scheduler(optimizer2, "rexscheduler", total_steps=5000)
print(f"REX LR at start: {scheduler2.get_last_lr()[0]:.6f}")

# 3. Cosine with warm restarts
optimizer3 = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler3 = CosineAnnealingWarmupRestarts(
    optimizer3,
    first_cycle_steps=1000,
    warmup_steps=100,
    max_lr=1e-3,
    min_lr=1e-5,
    gamma=0.9,
)

print("Basic usage examples completed.")
