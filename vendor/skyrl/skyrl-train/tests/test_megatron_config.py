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


def _topk_distillation_config():
    """A compiling sparse_forward_kl plan: top-K evidence the wrapper only guards at batch time."""
    config = get_default_config()
    config.trainer.strategy = "megatron"
    config.trainer.policy.model.path = "Qwen/Qwen3-0.6B"
    config.trainer.logger = "console"
    with open_dict(config):
        config.trainer.use_sample_packing = False
        config.trainer.algorithm.distillation = {
            "objective": "sparse_forward_kl",
            "routing_plan": "opd",
            "coefficient": 0.5,
            "reward_mode": "add",
        }
        config.teachers = {
            "primary": {
                "source": "local_inference",
                "placement": "pinned",
                "model": {"path": "Qwen/teacher", "revision": "teacher-revision"},
                "backend": "vllm",
                "evidence": "topk_distribution",
                "top_k": 8,
                "resources": {
                    "num_nodes": 1,
                    "gpus_per_node": 1,
                    "tensor_parallel_size": 1,
                    "colocation_group": "teacher",
                },
            }
        }
        config.teacher_routing = {
            "opd": {"revision": "route-revision", "routes": {"default": {"teacher": "primary", "weight": 1.0}}}
        }
    return config


def test_topk_distillation_rejected_with_chunking_or_sizing():
    config = _topk_distillation_config()
    megatron = config.trainer.policy.megatron_config
    # The plan itself is valid; only the opt-in memory options conflict with it.
    assert validate_cfg(config) is None

    with open_dict(megatron):
        megatron.output_projection_chunk_size = 1024
    with pytest.raises(ValueError, match="without top-K distillation"):
        validate_cfg(config)
    with open_dict(megatron):
        del megatron.output_projection_chunk_size

    with open_dict(megatron):
        megatron.memory_profile = {
            "enabled": True,
            "max_sequence_length": 2048,
            "valid_token_counts": [512],
            "minimum_headroom_fraction": 0.1,
        }
    with pytest.raises(ValueError, match="teacher workload"):
        validate_cfg(config)


def test_sizing_rejects_maximum_below_trained_generation_bound():
    config = get_default_config()
    config.trainer.strategy = "megatron"
    config.trainer.policy.model.path = "Qwen/Qwen3-0.6B"
    config.trainer.logger = "console"
    megatron = config.trainer.policy.megatron_config
    with open_dict(megatron):
        megatron.memory_profile = {
            "enabled": True,
            "max_sequence_length": 2048,
            "valid_token_counts": [512],
            "minimum_headroom_fraction": 0.1,
        }
    # The default trained bound is 512 prompt plus 1024 generated tokens; sizing must cover it.
    config.generator.sampling_params.max_generate_length = 2048
    with pytest.raises(ValueError):
        validate_cfg(config)
