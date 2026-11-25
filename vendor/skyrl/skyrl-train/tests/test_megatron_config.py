"""Megatron configuration remains independent of GPU package imports."""

import sys

import pytest
from omegaconf import open_dict

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
    with open_dict(megatron):
        megatron.output_projection_chunk_size = 1024
        megatron.memory_profile = {
            "enabled": True,
            "max_sequence_length": 2048,
            "valid_token_counts": [512],
            "minimum_headroom_fraction": 0.1,
        }
    assert validate_cfg(config) is None

    # The default trained bound is 512 prompt plus 1024 generated tokens; sizing must cover it.
    config.generator.sampling_params.max_generate_length = 2048
    with pytest.raises(ValueError, match="below the trained sequence bound"):
        validate_cfg(config)
    config.generator.sampling_params.max_generate_length = 1024
    assert validate_cfg(config) is None

    megatron.tensor_model_parallel_size = 2
    with pytest.raises(ValueError, match="requires TP1"):
        validate_cfg(config)
    megatron.tensor_model_parallel_size = 1
    config.trainer.policy.optimizer_config.optimizer = "muonh"
    config.trainer.policy.optimizer_config.weight_decay = 0.0
    assert validate_cfg(config) is None
