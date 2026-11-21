"""Hardware admission gates shared by Grug GPU tests."""

import pytest
import torch

from tests.gpu.utils import get_available_gpus


def require_hoppers(count: int) -> None:
    """Skip unless at least ``count`` Hopper-or-newer GPUs are available."""

    if len(get_available_gpus()) < count:
        pytest.skip(f"Grug GPU tests require {count} Hopper-or-newer GPUs")
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).major < 9:
        pytest.skip("Grug GPU tests require Hopper-or-newer GPUs")
