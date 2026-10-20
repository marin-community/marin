"""
Integration test for full trainer checkpointing functionality.

Exercise a policy update, full trainer save, restore into a fresh trainer and
a second save with optimizer sharding metadata and the latest-step file.

Run on Iris with an s3:// MARIN_TEMP_PREFIX or MARIN_PREFIX:
uv run --group dev --extra vllm --extra megatron pytest skyrl-train/tests/gpu/gpu_ci/test_trainer_full_checkpointing.py
"""

import asyncio
from types import SimpleNamespace

import ray
import pytest
import hydra
import torch
import os
import posixpath
from uuid import uuid4
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from unittest.mock import AsyncMock, MagicMock
from transformers import AutoTokenizer

from skyrl_train.rollouts.context import TrainingContextState
from skyrl_train.rollouts.loader import PromptLoaderState
from skyrl_train.utils.tracking import Tracking
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.io import io
from skyrl_train.distributed.megatron.checkpoint_metadata import remote_checkpoint_metadata
from tests.gpu.utils import import_worker, ray_init_for_tests
from skyrl_train.entrypoints.main_base import config_dir

MODEL_NAME = "Qwen/Qwen3-0.6B"
NUM_GPUS = 2
ROLLOUT_STATE = TrainingContextState(
    loader=PromptLoaderState(order={"epoch": 0, "position": 2}, retries=[]), ready=[], object_store_root=None
)


class DummyDataset(Dataset):
    """Minimal dataset for testing"""

    def __init__(self, size=10):
        self.data = [([{"role": "user", "content": f"Question {i}"}], None) for i in range(size)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        return batch


def get_test_trainer_config(strategy: str, optimizer_checkpoint_sharding_type: str | None = None) -> DictConfig:
    """Create minimal trainer config for testing"""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.strategy = strategy

    # Use minimal settings for faster testing
    cfg.trainer.placement.policy_num_gpus_per_node = NUM_GPUS
    cfg.trainer.placement.critic_num_gpus_per_node = NUM_GPUS
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.critic_num_nodes = 1
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.placement.colocate_all = False  # Disable colocation for simpler testing
    cfg.trainer.train_batch_size = NUM_GPUS
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.epochs = 1
    cfg.trainer.logger = "console"
    cfg.generator.n_samples_per_prompt = 1
    cfg.generator.num_inference_engines = NUM_GPUS // 2
    cfg.generator.inference_engine_tensor_parallel_size = 2

    # Megatron-specific
    if strategy == "megatron":
        OmegaConf.update(cfg, "trainer.algorithm.max_seq_len", 128, force_add=True)
        cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
        cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
        cfg.trainer.policy.megatron_config.optimizer_checkpoint_sharding_type = optimizer_checkpoint_sharding_type
        cfg.trainer.placement.policy_num_gpus_per_node = 4
        cfg.trainer.train_batch_size = 4
        cfg.trainer.policy_mini_batch_size = 4

    checkpoint_prefix = os.environ.get("MARIN_TEMP_PREFIX", os.environ.get("MARIN_PREFIX", ""))
    if not checkpoint_prefix.startswith("s3://"):
        raise ValueError("Run the trainer checkpoint test on Iris with CoreWeave object storage configured")
    cfg.trainer.export_path = posixpath.join(checkpoint_prefix, "tests", "trainer-checkpoint", uuid4().hex)
    cfg.trainer.ckpt_path = cfg.trainer.export_path

    # Enable checkpointing with correct config names
    cfg.trainer.ckpt_interval = 1  # Save every step
    cfg.trainer.resume_mode = "none"  # Initially false, will be set to True for resume

    return cfg


def create_minimal_trainer(cfg: DictConfig):
    """Create a minimal trainer setup for testing"""
    # Create minimal tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Create dummy dataset
    train_dataset = DummyDataset(size=4)  # Small dataset for quick testing

    mock_trajectory_runner = MagicMock()

    # Create tracker
    tracker = Tracking(
        project_name=cfg.trainer.project_name,
        experiment_name=cfg.trainer.run_name,
        backends=cfg.trainer.logger,
        config=cfg,
    )

    # Create trainer (no inference engine needed for checkpointing tests)
    trainer = RayPPOTrainer(
        cfg=cfg,
        tracker=tracker,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        inference_engine_client=None,
        trajectory_runner=mock_trajectory_runner,
        context=SimpleNamespace(
            config=SimpleNamespace(max_staleness_steps=0, batch_size=cfg.trainer.train_batch_size),
            state_dict=AsyncMock(return_value=ROLLOUT_STATE),
        ),
    )

    return trainer


def saved_optimizer_format(checkpoint_dir: str) -> str:
    from megatron.core import dist_checkpointing
    from skyrl_train.distributed.megatron.megatron_strategy import _saved_optimizer_sharding_type

    with remote_checkpoint_metadata(posixpath.join(checkpoint_dir, "policy")) as metadata_dir:
        common_state = dist_checkpointing.load_common_state_dict(metadata_dir)
    return _saved_optimizer_sharding_type(common_state)


@pytest.mark.parametrize(
    ("initial_sharding_type, resumed_sharding_type"),
    [
        ("fully_reshardable", "dp_reshardable"),
        ("dp_reshardable", "dp_reshardable"),
    ],
)
def test_trainer_full_checkpointing(ray_init_fixture, initial_sharding_type, resumed_sharding_type):
    """Update, save, restore into a fresh trainer and save with the selected sharding format."""
    strategy = "megatron"
    cfg = get_test_trainer_config(strategy, initial_sharding_type)

    try:
        # ============= PHASE 1: Initial Training and Save =============
        print("Phase 1: Initial training and checkpoint save")

        trainer1 = create_minimal_trainer(cfg)
        PolicyWorker = import_worker(strategy, "policy")
        CriticWorker = import_worker(strategy, "critic")
        RefWorker = import_worker(strategy, "ref")

        # Build models
        trainer1.build_models(PolicyWorker, CriticWorker, RefWorker)

        if strategy == "megatron":
            # A real optimizer step initializes Adam moments, which the checkpoint must preserve.
            # Keep this Megatron-only fixture out of non-Megatron test collection.
            from tests.gpu.test_megatron_worker import get_test_training_batch

            batch = get_test_training_batch(batch_size=4)
            batch.metadata["global_step"] = 1
            ray.get(trainer1.policy_model.async_run_ray_method("mesh", "ppo_train", batch))

        # Set initial global step as if 2 steps were completed
        trainer1.global_step = 2

        # Save checkpoint
        asyncio.run(trainer1.save_checkpoints())

        # Capture state before teardown
        saved_global_step = trainer1.global_step
        checkpoint_dir = posixpath.join(cfg.trainer.export_path, f"global_step_{trainer1.global_step}")

        # Verify checkpoint structure was created
        expected_files = [
            posixpath.join(checkpoint_dir, "policy"),
            posixpath.join(checkpoint_dir, "trainer_state.pt"),
            posixpath.join(checkpoint_dir, "data.pt"),
        ]
        for expected_file in expected_files:
            assert io.exists(expected_file), f"Expected checkpoint file/dir not found: {expected_file}"
        if strategy == "megatron":
            assert saved_optimizer_format(checkpoint_dir) == initial_sharding_type

        # Verify atomic tracking file
        latest_ckpt_file = posixpath.join(cfg.trainer.ckpt_path, "latest_ckpt_global_step.txt")
        assert io.exists(latest_ckpt_file)
        with io.open_file(latest_ckpt_file, "r") as f:
            latest_step = int(f.read())
        assert latest_step == trainer1.global_step, "Atomic tracking file has incorrect step after first save"

        # Verify trainer state content
        print("Verifying checkpoint content...")
        with io.open_file(posixpath.join(checkpoint_dir, "trainer_state.pt"), "rb") as file:
            loaded_trainer_state = torch.load(file, map_location="cpu", weights_only=False)

        # Check key configuration values are preserved
        assert loaded_trainer_state["config"]["trainer"]["train_batch_size"] == cfg.trainer.train_batch_size, (
            "train_batch_size not preserved in checkpoint"
        )
        assert loaded_trainer_state["config"]["trainer"]["strategy"] == strategy, "strategy not preserved in checkpoint"
        assert loaded_trainer_state["global_step"] == saved_global_step, "global_step not preserved in checkpoint"

        # Cleanup first trainer
        del trainer1
        ray.shutdown()

        # ============= PHASE 2: Resume from Checkpoint =============
        print("Phase 2: Resume from checkpoint")
        ray_init_for_tests()
        # Create new config with resume enabled
        cfg_resume = get_test_trainer_config(strategy, resumed_sharding_type)
        cfg_resume.trainer.resume_mode = "from_path"  # Enable resume
        cfg_resume.trainer.resume_path = checkpoint_dir  # Set resume path
        cfg_resume.trainer.export_path = cfg.trainer.export_path  # Use same export path
        cfg_resume.trainer.ckpt_path = cfg.trainer.ckpt_path

        trainer2 = create_minimal_trainer(cfg_resume)
        trainer2.build_models(PolicyWorker, CriticWorker, RefWorker)

        # Load checkpoints
        loaded_global_step, loaded_checkpoint_dir = trainer2.load_checkpoints()
        assert loaded_global_step == saved_global_step, (
            f"Expected global_step={saved_global_step}, got {loaded_global_step}"
        )
        assert loaded_checkpoint_dir == checkpoint_dir, "Checkpoint path mismatch"
        assert trainer2._restored_rollout_state == ROLLOUT_STATE

        print("Phase 3: Second checkpoint save")

        trainer2.global_step = 3
        asyncio.run(trainer2.save_checkpoints())

        next_checkpoint_dir = posixpath.join(cfg.trainer.export_path, f"global_step_{trainer2.global_step}")
        assert io.exists(next_checkpoint_dir), "Could not save checkpoint after resume"
        if strategy == "megatron":
            assert saved_optimizer_format(next_checkpoint_dir) == resumed_sharding_type

        # Verify atomic tracking file is updated
        latest_ckpt_file = posixpath.join(cfg.trainer.ckpt_path, "latest_ckpt_global_step.txt")
        assert io.exists(latest_ckpt_file)
        with io.open_file(latest_ckpt_file, "r") as f:
            latest_step = int(f.read())
        assert latest_step == trainer2.global_step, "Atomic tracking file was not updated after second save"

    finally:
        if io.exists(cfg.trainer.ckpt_path):
            io.remove(cfg.trainer.ckpt_path)
