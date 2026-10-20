"""Config validation of the ``expert_block`` weight-sync transport."""

import subprocess
import sys

import pytest

from marinskyrl.inference_placement import validate_expert_block_transport
from skyrl_train import objective  # noqa: F401 - register losses as normal training startup does
from skyrl_train.utils.utils import validate_cfg
from tests.cpu.util import example_dummy_config


def expert_block_config():
    cfg = example_dummy_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 8
    cfg.generator.update(
        backend="vllm",
        run_engines_locally=True,
        weight_sync_backend="nccl",
        weight_sync_transport="expert_block",
        inference_engine_tensor_parallel_size=1,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=8,
        inference_engine_expert_parallel_size=8,
    )
    return cfg


def test_pipeline_parallel_engines_on_the_mp_backend_are_refused():
    cfg = expert_block_config()
    cfg.generator.inference_engine_pipeline_parallel_size = 2
    validate_expert_block_transport(cfg)
    cfg.generator.inference_engine_mp_backend = True
    with pytest.raises(ValueError, match="Ray executor"):
        validate_expert_block_transport(cfg)


def test_validate_cfg_runs_the_transport_check():
    cfg = example_dummy_config()
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 4
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.generator.weight_sync_transport = "expert_block"
    with pytest.raises(ValueError, match="weight_sync_transport=expert_block requires"):
        validate_cfg(cfg)


def test_training_channels_cannot_change_the_expert_block_broadcast_layout():
    cfg = expert_block_config()
    cfg.trainer.policy.megatron_config.nccl_max_channels = 32
    # Mismatched NCCL channel limits can break a cross-world broadcast.
    with pytest.raises(ValueError, match="nccl_max_channels"):
        validate_expert_block_transport(cfg)
    cfg.generator.weight_sync_transport = "broadcast"
    validate_expert_block_transport(cfg)


def test_the_model_package_imports_before_the_trainer_utilities():
    # The frozen-runtime bootstrap imports the Grug model first. That import reaches
    # skyrl_train.utils and this validator, and must not be circular.
    subprocess.run(
        [sys.executable, "-c", "from skyrl_train.models.grug_moe import GRUG_MOE_ARCHITECTURE"],
        check=True,
        capture_output=True,
        text=True,
    )
