# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for HuggingFace model streaming.

These tests verify that models can be loaded from HuggingFace Hub using
the hf:// fsspec streaming protocol, without hitting the local HF cache.
"""

import os
import tempfile
from unittest import mock

import numpy as np
import pytest

from levanter.models.gpt2 import Gpt2Config
from levanter.models.llama import LlamaConfig
from test_utils import use_test_mesh


# Small models for fast tests
SMALL_HF_MODELS = [
    pytest.param(
        "hf-internal-testing/tiny-random-gpt2",
        Gpt2Config(num_layers=2, num_heads=2, hidden_dim=32, use_flash_attention=False),
        1_000,
        id="tiny-gpt2",
    ),
    pytest.param(
        "sshleifer/tiny-gpt2",
        Gpt2Config(num_layers=2, num_heads=2, hidden_dim=32, use_flash_attention=False),
        1_000,
        id="sshleifer-tiny-gpt2",
    ),
]

# Larger models for more comprehensive testing
LARGE_HF_MODELS = [
    pytest.param(
        "openai-community/gpt2",
        Gpt2Config(num_layers=12, num_heads=12, hidden_dim=768, use_flash_attention=False),
        100_000_000,  # ~124M params
        id="gpt2-124m",
    ),
    pytest.param(
        "HuggingFaceTB/SmolLM2-135M",
        LlamaConfig(num_layers=30, num_heads=9, num_kv_heads=3, hidden_dim=576),
        100_000_000,  # ~135M params
        id="smollm2-135m",
    ),
]


def _count_params(state_dict: dict) -> int:
    """Count total parameters in a state dict."""
    total = 0
    for value in state_dict.values():
        if hasattr(value, "size"):
            total += value.size
        elif hasattr(value, "shape"):
            total += int(np.prod(value.shape))
    return total


def _get_cached_model_files(cache_dir: str) -> list[str]:
    """Find all model weight files in the HF cache directory."""
    model_files = []
    if os.path.exists(cache_dir):
        for root, _, files in os.walk(cache_dir):
            for f in files:
                if f.endswith((".safetensors", ".bin", ".pt", ".pth")):
                    model_files.append(os.path.join(root, f))
    return model_files


@pytest.mark.slow
@pytest.mark.parametrize("model_id,config,min_params", SMALL_HF_MODELS)
def test_load_hf_model_streaming(model_id: str, config, min_params: int):
    """Test loading HF models via streaming."""
    converter = config.hf_checkpoint_converter()

    with use_test_mesh():
        state_dict = converter.load_state_dict(model_id)

    assert isinstance(state_dict, dict)
    assert len(state_dict) > 0

    param_count = _count_params(state_dict)
    assert param_count >= min_params, f"Expected at least {min_params} params, got {param_count}"


@pytest.mark.slow
@pytest.mark.parametrize("model_id,config,min_params", SMALL_HF_MODELS)
def test_hf_streaming_does_not_use_cache(model_id: str, config, min_params: int):
    """Verify that streaming loads do not populate the HF cache.

    This test sets HF_HOME to a temp directory and verifies that no model
    weight files are downloaded there when using hf:// streaming.
    """
    with tempfile.TemporaryDirectory() as tmp_cache:
        with mock.patch.dict(os.environ, {"HF_HOME": tmp_cache}):
            converter = config.hf_checkpoint_converter()

            with use_test_mesh():
                state_dict = converter.load_state_dict(model_id)

            assert isinstance(state_dict, dict)
            assert len(state_dict) > 0

            # Verify no model files were cached
            cached_files = _get_cached_model_files(tmp_cache)
            assert len(cached_files) == 0, f"Expected no cached model files, found: {cached_files}"


@pytest.mark.slow
@pytest.mark.parametrize("model_id,config,min_params", LARGE_HF_MODELS)
def test_load_larger_hf_models(model_id: str, config, min_params: int):
    """Test loading larger HF models to verify streaming works at scale.

    These tests use real models with 100M+ parameters to ensure the streaming
    approach works correctly for production-sized models.
    """
    with tempfile.TemporaryDirectory() as tmp_cache:
        with mock.patch.dict(os.environ, {"HF_HOME": tmp_cache}):
            converter = config.hf_checkpoint_converter()

            with use_test_mesh():
                state_dict = converter.load_state_dict(model_id)

            assert isinstance(state_dict, dict)
            assert len(state_dict) > 0

            param_count = _count_params(state_dict)
            assert param_count >= min_params, f"Expected at least {min_params} params, got {param_count}"

            # Verify no model files were cached
            cached_files = _get_cached_model_files(tmp_cache)
            assert len(cached_files) == 0, f"Expected no cached model files, found: {cached_files}"
