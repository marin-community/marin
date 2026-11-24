"""Megatron configuration remains independent of GPU package imports."""

import sys

import pytest

from skyrl_train.config.utils import get_default_config
from skyrl_train.config.driver_validation import validate_cfg


def test_flash_attention_config_validates_without_fa2(monkeypatch):
    config = get_default_config()
    config.trainer.strategy = "megatron"
    config.trainer.flash_attn = True
    config.trainer.policy.model.path = "Qwen/Qwen3-0.6B"
    config.trainer.logger = "console"
    monkeypatch.setitem(sys.modules, "flash_attn", None)

    # A valid FA3 configuration must not require FA2's separate Python module.
    assert validate_cfg(config) is None


def test_chunked_sizing_accepts_ppo_and_rejects_incompatible_execution():
    config = get_default_config()
    config.trainer.strategy = "megatron"
    config.trainer.policy.model.path = "Qwen/Qwen3-0.6B"
    config.trainer.logger = "console"
    megatron = config.trainer.policy.megatron_config
    megatron.output_projection_chunk_size = 1024
    megatron.memory_profile.enabled = True
    megatron.memory_profile.max_sequence_length = 1024
    megatron.memory_profile.valid_token_counts = [512]
    assert validate_cfg(config) is None

    megatron.tensor_model_parallel_size = 2
    with pytest.raises(ValueError, match="requires TP1"):
        validate_cfg(config)
    megatron.tensor_model_parallel_size = 1
    config.trainer.policy.optimizer_config.optimizer = "muonh"
    with pytest.raises(ValueError, match="sizing currently requires Adam"):
        validate_cfg(config)
