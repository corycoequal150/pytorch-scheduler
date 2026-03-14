from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.optim import Optimizer


class ScheduleFreeWrapper:
    """Schedule-Free optimizer wrapper.

    Wraps any optimizer to make it schedule-free. Instead of using a
    learning rate schedule, this wrapper applies online-to-batch conversion
    to automatically find good learning rates.

    The key idea: maintain two sequences of parameters:
    - z_t (iterate sequence): used for gradient computation
    - x_t (evaluation sequence): the weighted average for evaluation

    During training: use z_t parameters
    During evaluation: switch to x_t parameters

    Args:
        optimizer: The base optimizer to wrap.
        warmup_steps: Number of warmup steps (default 0).
        beta: Momentum/averaging parameter (default 0.9).

    Reference:
        Paper: "The Road Less Scheduled" (Defazio et al., 2024)
        URL: https://arxiv.org/abs/2405.15682
    """

    paper_title = "The Road Less Scheduled"
    paper_url = "https://arxiv.org/abs/2405.15682"
    paper_year = 2024

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 0,
        beta: float = 0.9,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")

        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.beta = beta
        self._step_count = 0
        self._training = True

        # Store initial parameters for averaging
        self._z_params: list[torch.Tensor] = []  # iterate sequence
        self._x_params: list[torch.Tensor] = []  # average sequence

        for group in self.optimizer.param_groups:
            for p in group["params"]:
                self._z_params.append(p.data.clone())
                self._x_params.append(p.data.clone())

    def _warmup_factor(self) -> float:
        if self.warmup_steps == 0:
            return 1.0
        return min(1.0, self._step_count / self.warmup_steps)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        self._step_count += 1
        warmup = self._warmup_factor()

        # Apply warmup to learning rate
        for group in self.optimizer.param_groups:
            if "initial_lr" not in group:
                group["initial_lr"] = group["lr"]
            group["lr"] = group["initial_lr"] * warmup

        # Step the base optimizer (updates z_t)
        loss = self.optimizer.step(closure)

        # Update the average (x_t) using exponential moving average
        # x_t = beta * x_{t-1} + (1 - beta) * z_t
        idx = 0
        ck = 1.0 / self._step_count  # 1/k weighting for Polyak averaging
        beta = max(self.beta, 1.0 - ck)  # adapt beta based on step count

        for group in self.optimizer.param_groups:
            for p in group["params"]:
                z = p.data
                self._z_params[idx].copy_(z)
                self._x_params[idx].lerp_(z, 1.0 - beta)
                idx += 1

        return loss

    @torch.no_grad()
    def eval(self):
        """Switch to evaluation parameters (averaged)."""
        if not self._training:
            return
        self._training = False
        idx = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                # Save current z, load x for evaluation
                self._z_params[idx].copy_(p.data)
                p.data.copy_(self._x_params[idx])
                idx += 1

    @torch.no_grad()
    def train(self):
        """Switch back to training parameters (iterates)."""
        if self._training:
            return
        self._training = True
        idx = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                p.data.copy_(self._z_params[idx])
                idx += 1

    def state_dict(self) -> dict:
        return {
            "optimizer": self.optimizer.state_dict(),
            "step_count": self._step_count,
            "z_params": [z.clone() for z in self._z_params],
            "x_params": [x.clone() for x in self._x_params],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self._step_count = state_dict["step_count"]
        for i, z in enumerate(state_dict["z_params"]):
            self._z_params[i].copy_(z)
        for i, x in enumerate(state_dict["x_params"]):
            self._x_params[i].copy_(x)
