"""Qwen3-0.6B training and model-only restores from compact Megatron S3 checkpoints."""

import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import uuid
from unittest.mock import patch

import hydra
from megatron.core.optimizer import ChainedOptimizer
import pytest
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from rigging.filesystem.cluster_config import marin_temp_bucket
import torch
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from marinskyrl.remote_io import create_s3_filesystem
from marinskyrl.resource_locator import join_resource_path
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.distributed.megatron.megatron_utils import materialize_megatron_params
from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.io.checkpoint_reader import RecordCheckpointReader
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils.utils import validate_cfg
from skyrl_train.workers.megatron.megatron_worker import MegatronPolicyWorkerBase
from skyrl_train.workers.worker import PPORayActorGroup
from tests.gpu.grug_gpu_gates import require_hoppers


MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def state_fingerprint(value):
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        return (str(value.dtype), tuple(value.shape), hashlib.sha256(data).hexdigest())
    if isinstance(value, dict):
        return {key: state_fingerprint(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [state_fingerprint(item) for item in value]
    return value


class CheckpointPolicyWorker(MegatronPolicyWorkerBase):
    def load_checkpoint(self, *args, **kwargs):
        read_bytes = 0
        read_record = RecordCheckpointReader._slice_file

        def counted_read(reader, file, storage):
            nonlocal read_bytes
            buffer = read_record(reader, file, storage)
            read_bytes += buffer.getbuffer().nbytes
            return buffer

        with patch.object(RecordCheckpointReader, "_slice_file", counted_read):
            state = super().load_checkpoint(*args, **kwargs)
        return {"state": state, "read_bytes": read_bytes}

    def checkpoint_snapshot(self):
        materialize_megatron_params(self.actor_module)
        optimizers = (
            self.optimizer.chained_optimizers if isinstance(self.optimizer, ChainedOptimizer) else [self.optimizer]
        )
        return {
            "model": [
                state_fingerprint({**dict(chunk.named_parameters()), **dict(chunk.named_buffers())})
                for chunk in self.actor_module
            ],
            "masters": [
                state_fingerprint([group["params"] for group in optimizer.optimizer.param_groups])
                for optimizer in optimizers
            ],
            "optimizer": [state_fingerprint(optimizer.optimizer.state_dict()) for optimizer in optimizers],
            "scheduler": self.scheduler.state_dict(),
        }


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_qwen_checkpoint_dedup_restores_model_and_optimizer_continuation(ray_init_fixture, tmp_path: Path):
    require_hoppers(2)
    path = marin_temp_bucket(1, f"skyrl/checkpoint-dedup-smoke/{uuid.uuid4().hex}")
    assert path.startswith("s3://"), "run this smoke on a CoreWeave GPU with regional S3 storage"
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")
    cfg.trainer.strategy = "megatron"
    cfg.trainer.logger = "console"
    cfg.trainer.flash_attn = False
    cfg.trainer.use_sample_packing = False
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 4
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.policy.model.path = MODEL
    cfg.trainer.policy.model.revision = REVISION
    cfg.trainer.policy.optimizer_config.lr = 1e-4
    cfg.trainer.policy.optimizer_config.offload_after_step = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.placement.colocate_all = False
    cfg.generator.n_samples_per_prompt = 1
    cfg.trainer.policy.megatron_config.optimizer_checkpoint_sharding_type = "fully_reshardable"
    validate_cfg(cfg)

    batch = TrainingInputBatch(
        {
            "sequences": torch.arange(1, 129).reshape(4, 32),
            "attention_mask": torch.ones((4, 32), dtype=torch.long),
            "action_log_probs": torch.full((4, 8), -5.0),
            "base_action_log_probs": torch.full((4, 8), -5.0),
            "rollout_logprobs": torch.full((4, 8), -5.0),
            "values": torch.zeros((4, 8)),
            "returns": torch.ones((4, 8)),
            "advantages": torch.ones((4, 8)),
            "loss_mask": torch.ones((4, 8)),
            "response_mask": torch.ones((4, 8)),
        }
    )
    batch.metadata = {"response_length": 8, "global_step": 0}
    pg = placement_group([{"GPU": 1, "CPU": 1}] * 2, strategy="PACK")
    ray.get(pg.ready(), timeout=120)
    policy = PPORayActorGroup(
        cfg,
        num_nodes=1,
        num_gpus_per_node=2,
        ray_actor_type=ray.remote(num_gpus=1)(CheckpointPolicyWorker),
        pg=pg,
        num_gpus_per_actor=1,
        colocate_all=False,
    )

    def call(method, *args, **kwargs):
        return ray.get(policy.async_run_ray_method("pass_through", method, *args, **kwargs))

    def train(step):
        batch.metadata["global_step"] = step
        forward = concatenate_outputs_after_mesh_dispatch(
            policy.actor_infos, ray.get(policy.async_run_ray_method("mesh", "forward", data=batch))
        )
        batch["action_log_probs"] = forward["output"]
        batch["rollout_logprobs"] = forward["output"]
        results = ray.get(policy.async_run_ray_method("mesh", "ppo_train", data=batch))
        assert all(result.metadata["train_status"]["policy_update_steps"] == 1 for result in results)

    try:
        ray.get(policy.async_init_model(MODEL, num_training_steps=2))
        train(0)
        saved = call("checkpoint_snapshot")
        assert saved[0]["model"] == saved[1]["model"]
        call("save_checkpoint", path)

        fs = create_s3_filesystem(default_cache_type="none")
        with fs.open(join_resource_path(path, ".metadata"), "rb") as source:
            metadata = pickle.load(source)
        fp32_records = {
            (record.relative_path, record.offset, record.length)
            for index, record in metadata.storage_data.items()
            if isinstance(metadata.state_dict_metadata[index.fqn], TensorStorageMetadata)
            and metadata.state_dict_metadata[index.fqn].properties.dtype == torch.float32
        }
        shared = [
            metadata.state_dict_metadata[index.fqn].chunks[index.index or 0]
            for index, record in metadata.storage_data.items()
            if isinstance(metadata.state_dict_metadata[index.fqn], TensorStorageMetadata)
            and metadata.state_dict_metadata[index.fqn].properties.dtype == torch.bfloat16
            and (record.relative_path, record.offset, record.length) in fp32_records
        ]
        shared_bytes = sum(math.prod(chunk.sizes) * 2 for chunk in shared)
        logical_bytes = sum(
            math.prod(value.size) * torch.empty((), dtype=value.properties.dtype).element_size()
            for value in metadata.state_dict_metadata.values()
            if isinstance(value, TensorStorageMetadata)
        )
        stored_bytes = sum(item["size"] for item in fs.ls(path, detail=True) if item["name"].endswith(".distcp"))
        assert shared_bytes > 0
        assert stored_bytes < logical_bytes - shared_bytes * 0.95

        train(1)
        continued = call("checkpoint_snapshot")
        assert any(current["model"] != old["model"] for current, old in zip(continued, saved, strict=True))
        model_reads = call("load_checkpoint", path, load_training_state=False)
        model_only = call("checkpoint_snapshot")
        for loaded, old, current in zip(model_only, saved, continued, strict=True):
            assert loaded["model"] == old["model"]
            assert loaded["optimizer"] == current["optimizer"]
        training_reads = call("load_checkpoint", path)
        assert call("checkpoint_snapshot") == saved
        model_read_bytes = sum(result["read_bytes"] for result in model_reads)
        training_read_bytes = sum(result["read_bytes"] for result in training_reads)
        assert model_read_bytes <= 2 * shared_bytes + 1024 * 1024
        assert training_read_bytes <= stored_bytes + 1024 * 1024
        train(1)
        assert call("checkpoint_snapshot") == continued

        report = {
            "model": MODEL,
            "revision": REVISION,
            "logical_tensor_bytes": logical_bytes,
            "stored_bytes": stored_bytes,
            "shared_bf16_bytes": shared_bytes,
            "data_parallel_size": 2,
            "model_read_bytes": model_read_bytes,
            "training_read_bytes": training_read_bytes,
            "model_only_restore": "bitwise_equal",
            "training_restore": "bitwise_equal",
            "optimizer_continuation": "bitwise_equal",
        }
        print("CHECKPOINT_SMOKE " + json.dumps(report), flush=True)
        output = Path(os.environ.get("IRIS_OUTPUT_DIR", str(tmp_path))) / "checkpoint-smoke.json"
        output.write_text(json.dumps(report, indent=2) + "\n")
        fs.rm(path, recursive=True)
    finally:
        policy.kill_actors()
        remove_placement_group(pg)
