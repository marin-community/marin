"""Megatron policy setup shared by the Grug GPU tests."""

import math

import ray

from skyrl_train.training_batch import TrainingInputBatch
from tests.gpu.utils import get_test_actor_config, init_worker_with_type

ROLLOUT_WORLD_SIZE = 2


def policy_config(model_path: str, *, world_size: int, pp: int, ep: int):
    cfg = get_test_actor_config()
    cfg.trainer.policy.model.path = model_path
    cfg.trainer.critic.model.path = None
    cfg.trainer.strategy = "megatron"
    cfg.trainer.flash_attn = False
    cfg.trainer.bf16 = True
    cfg.trainer.gradient_checkpointing = True
    cfg.trainer.use_sample_packing = False
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 4
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = world_size
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = pp
    cfg.trainer.policy.megatron_config.context_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = ep
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
    # Large enough that one Adam step moves bf16 norm weights sitting at 1.0 (ulp 2^-7).
    cfg.trainer.policy.optimizer_config.lr = 2.0e-2
    cfg.trainer.policy.optimizer_config.max_grad_norm = 0.0
    cfg.generator.backend = "vllm"
    cfg.generator.weight_sync_backend = "nccl"
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.inference_engine_data_parallel_size = ROLLOUT_WORLD_SIZE
    cfg.generator.inference_engine_expert_parallel_size = ROLLOUT_WORLD_SIZE
    cfg.generator.num_inference_engines = 1
    cfg.generator.n_samples_per_prompt = 1
    cfg.generator.gpu_memory_utilization = 0.35
    cfg.generator.sampling_params.temperature = 1.0
    cfg.generator.sampling_params.top_p = 1.0
    cfg.generator.sampling_params.top_k = -1
    cfg.generator.sampling_params.max_generate_length = 4
    return cfg


def init_policy(cfg, world_size: int):
    return init_worker_with_type(
        "policy", shared_pg=None, colocate_all=False, num_gpus_per_node=world_size, num_nodes=1, cfg=cfg
    )


def train_step(policy, batch: TrainingInputBatch) -> dict[str, float]:
    train_output = ray.get(policy.async_run_ray_method("mesh", "ppo_train", batch))[0]
    status = train_output.metadata["train_status"]
    assert math.isfinite(status["policy_loss"])
    return status
