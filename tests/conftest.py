from __future__ import annotations

import warnings

import pytest
import torch

warnings.filterwarnings("ignore")


@pytest.fixture
def optimizer():
    """Create a simple SGD optimizer for testing."""
    param = torch.randn(2, requires_grad=True)
    return torch.optim.SGD([param], lr=0.1)


@pytest.fixture
def multi_param_optimizer():
    """Optimizer with multiple param groups."""
    p1 = torch.randn(2, requires_grad=True)
    p2 = torch.randn(3, requires_grad=True)
    return torch.optim.SGD(
        [
            {"params": [p1], "lr": 0.1},
            {"params": [p2], "lr": 0.01},
        ]
    )
