"""Grug router telemetry on the Megatron trainer.

One forward under the telemetry observer must record loads that match a
recomputation from the observation's own router inputs, and the frozen-bias
snapshot must survive the forward untouched.
"""

from __future__ import annotations

import ray

import skyrl_train.models.grug_megatron_bridge  # noqa: F401  # registers the Grug bridge
from skyrl_train.utils import initialize_ray
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_tiny_checkpoint import TOY_SHAPE, write_tiny_checkpoint
from tests.gpu.utils import get_test_actor_config, init_worker_with_type

ROLLOUT_WORLD_SIZE = 2


def _config(model_path: str) -> object:
    cfg = get_test_actor_config()
    cfg.trainer.policy.model.path = model_path
    cfg.trainer.critic.model.path = None
    cfg.trainer.strategy = "megatron"
    cfg.trainer.flash_attn = False
    cfg.trainer.bf16 = True
    cfg.trainer.gradient_checkpointing = False
    cfg.trainer.use_sample_packing = False
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 4
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.context_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
    cfg.trainer.router_telemetry.enabled = True
    return cfg


def test_router_observations_match_a_recomputation_from_router_inputs(tmp_path):
    require_hoppers(1)
    model_path = tmp_path / "model"
    model_path.mkdir()
    write_tiny_checkpoint(model_path, shape=TOY_SHAPE)
    cfg = _config(str(model_path))
    initialize_ray(cfg)
    try:
        policy = init_worker_with_type("policy", cfg=cfg, num_gpus_per_node=1, num_nodes=1)
        try:
            mismatches = ray.get(
                policy.async_run_ray_method("pass_through", "verify_router_observations", str(model_path))
            )[0]
        finally:
            policy.kill_actors()
        assert mismatches == 0
    finally:
        ray.shutdown()
