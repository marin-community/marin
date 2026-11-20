# utility functions used for CPU tests

import importlib.machinery
import importlib.util
import os
import sys
import types
from contextlib import contextmanager
from datetime import timedelta
from unittest.mock import patch

import torch.distributed as dist
from skyrl_train.config.utils import get_default_config
from omegaconf import OmegaConf

# The `+algorithm_recipe=dpo` selection plus the pairing constraints every DPO test needs;
# mirror the recipe file so the tested gates and the shipped recipe stay one block.
DPO_OVERRIDES = {
    "environment": {"env_class": "preference_pair"},
    "trainer": {
        "algorithm": {
            "policy_loss_type": "dpo",
            "advantage_estimator": "uniform",
            "loss_reduction": "pair_mean",
            "advantage_batch_normalize": False,
            "use_kl_loss": False,
            "use_kl_in_reward": False,
            "use_entropy_loss": False,
            "off_policy_correction": "none",
            "dynamic_sampling": {"type": None},
            "dpo": {"beta": 0.1, "label_smoothing": 0.0},
        },
        "placement": {"colocate_all": False, "colocate_policy_ref": True},
        "use_sample_packing": False,
        "micro_train_batch_size_per_gpu": 2,
    },
    "generator": {"n_samples_per_prompt": 2},
}


def dpo_test_config(**overrides):
    """Compose the default config with the DPO recipe plus test-local overrides."""
    cfg = get_default_config()
    cfg = OmegaConf.merge(cfg, OmegaConf.create(DPO_OVERRIDES))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    cfg.trainer.logger = "console"
    return cfg


def example_dummy_config():
    cfg = get_default_config()
    # TODO (sumanthrh): Some of these overrides are no longer needed after reading from the config file
    trainer_overrides = {
        "project_name": "unit-test",
        "run_name": "test-run",
        "logger": "tensorboard",
        "micro_train_batch_size_per_gpu": 2,
        "train_batch_size": 2,
        "eval_batch_size": 2,
        "update_epochs_per_batch": 1,
        "epochs": 1,
        "max_prompt_length": 20,
        "use_sample_packing": False,
        "seed": 42,
        "resume_mode": "none",
        "algorithm": {
            "advantage_estimator": "grpo",
            "kl_estimator_type": "k1",
            "use_kl_loss": True,
            "kl_loss_coef": 0.0,
            "loss_reduction": "token_mean",
            "grpo_norm_by_std": True,
        },
    }
    generator_overrides = {
        "sampling_params": {"max_generate_length": 20},
        "n_samples_per_prompt": 1,
        "max_turns": 1,
        "enable_http_endpoint": False,
        "http_endpoint_host": "127.0.0.1",
        "http_endpoint_port": 8000,
    }
    OmegaConf.update(cfg, "trainer", trainer_overrides)
    OmegaConf.update(cfg, "generator", generator_overrides)

    return cfg


@contextmanager
def gloo_process_group(rank: int, world_size: int, port: int, *, timeout_seconds: int | None = None):
    """Initialize and reliably tear down a local Gloo process group."""
    rendezvous_env = {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "RANK": str(rank),
        "WORLD_SIZE": str(world_size),
    }
    with patch.dict(os.environ, rendezvous_env):
        kwargs = {"timeout": timedelta(seconds=timeout_seconds)} if timeout_seconds is not None else {}
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, **kwargs)
        try:
            yield
        finally:
            dist.barrier()
            dist.destroy_process_group()


def stub_megatron_modules() -> None:
    """Install stub `megatron` submodules when megatron is not installed.

    Several skyrl_train megatron modules import `megatron.core` names at module
    load, but the functions the CPU tests exercise never call into them. The CPU
    CI environment has no megatron, so the stubs let those modules import. No-op
    when the real megatron is importable, so a real-megatron env is untouched.
    Idempotent: missing attributes are added to already-installed stubs.
    """
    try:
        if importlib.util.find_spec("megatron") is not None:
            return
    except ValueError:
        # A spec-less stub from an earlier call/test is already installed.
        pass

    stub_attrs = {
        "megatron": {},
        "megatron.core": {},
        "megatron.core.parallel_state": {},
        "megatron.core.pipeline_parallel": {"get_forward_backward_func": lambda: None},
        "megatron.core.distributed": {
            "DistributedDataParallel": type("DistributedDataParallel", (), {}),
            "finalize_model_grads": lambda *args, **kwargs: None,
        },
        "megatron.core.transformer": {},
        "megatron.core.transformer.module": {"Float16Module": type("Float16Module", (), {})},
        "megatron.core.optimizer": {
            "ChainedOptimizer": type("ChainedOptimizer", (), {}),
            "DistributedOptimizer": type("DistributedOptimizer", (), {}),
        },
        "megatron.core.optimizer.cpu_offloading": {
            "HybridDeviceOptimizer": type("HybridDeviceOptimizer", (), {}),
        },
        "megatron.core.optimizer.clip_grads": {"clip_grad_by_total_norm_fp32": lambda *args, **kwargs: None},
        "megatron.core.utils": {"get_attr_wrapped_model": lambda *args, **kwargs: None},
        "megatron.core.packed_seq_params": {
            "PackedSeqParams": type(
                "PackedSeqParams", (), {"__init__": lambda self, **kwargs: self.__dict__.update(kwargs)}
            )
        },
        # The checkpoint strategy module imports these at load; tests replace what they call.
        "megatron.core.dist_checkpointing": {
            "load": lambda *args, **kwargs: {},
            "save": lambda *args, **kwargs: None,
            "load_common_state_dict": lambda *args, **kwargs: {},
        },
        "megatron.core.dist_checkpointing.dict_utils": {"nested_values": lambda value: value.values()},
        "megatron.core.dist_checkpointing.mapping": {
            "ShardedStateDict": dict,
            "ShardedTensor": type("ShardedTensor", (), {}),
        },
        "megatron.core.dist_checkpointing.serialization": {
            "get_default_load_sharded_strategy": lambda *args, **kwargs: None,
            "get_default_save_sharded_strategy": lambda *args, **kwargs: None,
        },
        "megatron.core.dist_checkpointing.strategies": {},
        "megatron.core.dist_checkpointing.strategies.base": {},
        "megatron.core.dist_checkpointing.strategies.async_utils": {
            "AsyncCallsQueue": type("AsyncCallsQueue", (), {"__init__": lambda self, **kwargs: None})
        },
        "megatron.core.dist_checkpointing.strategies.fully_parallel": {
            "FullyParallelLoadStrategyWrapper": type("FullyParallelLoadStrategyWrapper", (), {}),
            "FullyParallelSaveStrategyWrapper": type("FullyParallelSaveStrategyWrapper", (), {}),
        },
        "megatron.core.dist_checkpointing.strategies.torch": {
            "TorchDistLoadShardedStrategy": type("TorchDistLoadShardedStrategy", (), {}),
            "TorchDistSaveShardedStrategy": type("TorchDistSaveShardedStrategy", (), {}),
            "MCoreLoadPlanner": type("MCoreLoadPlanner", (), {}),
            "MCoreSavePlanner": type("MCoreSavePlanner", (), {}),
            "_replace_sharded_keys_with_state_dict_keys": lambda *args, **kwargs: None,
            "_replace_state_dict_keys_with_sharded_keys": lambda *args, **kwargs: None,
            "_restore_dict_types": lambda *args, **kwargs: None,
            "_unwrap_pyt_sharded_tensor": lambda value: value,
            "mcore_to_pyt_state_dict": lambda *args, **kwargs: None,
        },
        "megatron.core.optimizer_param_scheduler": {"OptimizerParamScheduler": type("OptimizerParamScheduler", (), {})},
    }
    for name, members in stub_attrs.items():
        module = sys.modules.setdefault(name, types.ModuleType(name))
        # Give stubs a spec so later find_spec("megatron") calls see a normal
        # module instead of raising ValueError on a spec-less one.
        if getattr(module, "__spec__", None) is None:
            module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        for attr, value in members.items():
            setattr(module, attr, value)
