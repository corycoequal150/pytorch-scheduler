"""Composable warmup examples."""

import torch

from pytorch_scheduler import LinearDecayScheduler, WarmupScheduler

model = torch.nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Any scheduler can be wrapped with warmup
base = LinearDecayScheduler(optimizer, total_steps=900)
scheduler = WarmupScheduler(
    optimizer,
    base_scheduler=base,
    warmup_steps=100,
    warmup_type="cosine",  # 'linear', 'cosine', 'exponential'
)

# Step through training
lrs = []
for step in range(1000):
    optimizer.zero_grad()
    loss = model(torch.randn(4, 10)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    if step % 200 == 0:
        print(f"Step {step:4d}: LR = {scheduler.get_last_lr()[0]:.6f}")

# State can be saved/loaded
state = scheduler.state_dict()
scheduler.load_state_dict(state)
print("State dict round-trip successful.")
