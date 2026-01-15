# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for HuggingFace model streaming.

These tests verify that models can be loaded from HuggingFace Hub using
the hf:// fsspec streaming protocol.
"""

import pytest

from levanter.models.gpt2 import Gpt2Config
from test_utils import use_test_mesh


# Test models with their expected approximate parameter counts
HF_TEST_MODELS = [
    pytest.param(
        "hf-internal-testing/tiny-random-gpt2",
        1000,  # expected min params (tiny model)
        id="tiny-gpt2",
    ),
    pytest.param(
        "sshleifer/tiny-gpt2",
        1000,
        id="sshleifer-tiny-gpt2",
    ),
]


def _count_params(state_dict: dict) -> int:
    """Count total parameters in a state dict."""
    total = 0
    for value in state_dict.values():
        if hasattr(value, "size"):
            total += value.size
        elif hasattr(value, "shape"):
            import numpy as np

            total += int(np.prod(value.shape))
    return total


@pytest.mark.slow
@pytest.mark.parametrize("model_id,min_params", HF_TEST_MODELS)
def test_load_hf_model_streaming(model_id: str, min_params: int):
    """Test loading various HF models via streaming.

    This test verifies that:
    1. The model loads successfully from HuggingFace Hub
    2. The state dict contains the expected minimum number of parameters
    """
    gpt2_config = Gpt2Config(
        num_layers=2,
        num_heads=2,
        hidden_dim=32,
        use_flash_attention=False,
    )
    converter = gpt2_config.hf_checkpoint_converter()

    with use_test_mesh():
        state_dict = converter.load_state_dict(model_id)

    assert isinstance(state_dict, dict)
    assert len(state_dict) > 0

    param_count = _count_params(state_dict)
    assert param_count >= min_params, f"Expected at least {min_params} params, got {param_count}"
