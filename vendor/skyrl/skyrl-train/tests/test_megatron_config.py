"""Megatron configuration remains independent of GPU package imports."""

import sys

import pytest

from skyrl_train.config.utils import get_default_config
from skyrl_train.utils.utils import validate_megatron_cfg


def test_flash_attention_config_validates_without_fa2(monkeypatch):
    config = get_default_config()
    config.trainer.strategy = "megatron"
    config.trainer.flash_attn = True
    config.trainer.policy.model.path = "Qwen/Qwen3-0.6B"
    config.trainer.logger = "console"
    monkeypatch.setitem(sys.modules, "flash_attn", None)

    # A valid FA3 configuration must not require FA2's separate Python module.
    assert validate_megatron_cfg(config) is None


def test_fp32_lm_head_accepts_snowball_pp2_cp5_ep8_tp1_and_rejects_unsupported_configs():
    config = get_default_config()
    config.trainer.strategy = "megatron"
    config.trainer.use_sample_packing = True
    config.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
    config.trainer.policy.megatron_config.context_parallel_size = 5
    config.trainer.policy.megatron_config.expert_model_parallel_size = 8
    config.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    config.trainer.policy.megatron_config.fp32_lm_head = True
    config.trainer.ref.megatron_config.fp32_lm_head = True
    assert validate_megatron_cfg(config) is None

    bad_tp = get_default_config()
    bad_tp.trainer.strategy = "megatron"
    bad_tp.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    bad_tp.trainer.policy.megatron_config.fp32_lm_head = True
    with pytest.raises(ValueError, match="tensor_model_parallel_size == 1"):
        validate_megatron_cfg(bad_tp)

    bad_deferred = get_default_config()
    bad_deferred.trainer.strategy = "megatron"
    bad_deferred.trainer.ref.megatron_config.fp32_lm_head = True
    bad_deferred.trainer.ref.megatron_config.transformer_config_kwargs = {"defer_embedding_wgrad_compute": True}
    with pytest.raises(ValueError, match="defer_embedding_wgrad_compute"):
        validate_megatron_cfg(bad_deferred)
