"""Shared Grug policy setup and batches for GPU qualification."""

import math

import ray
import torch

from skyrl_train.training_batch import TrainingInputBatch
from tests.gpu.grug_serving import (
    GRUG_PARAMETER_NAMES as PARAMETER_NAMES,
)
from tests.gpu.utils import get_test_actor_config, init_worker_with_type

ROLLOUT_WORLD_SIZE = 2

RESPONSE_LENGTH = 8

PROMPT_LENGTH = 12


def _spot_check_name(suffix: str) -> str:
    """First broad-coverage parameter name with the given suffix."""
    return next(name for name in PARAMETER_NAMES if name.endswith(suffix))


ATTN_GATE_NAME = _spot_check_name(".self_attn.attn_gate.weight")

GATED_NORM_NAME = _spot_check_name(".attn_gated_norm.up_proj.weight")


def _config(model_path: str, *, world_size: int, pp: int, ep: int):
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


def _padded_batch(
    pad_token_id: int,
    batch_size: int = 4,
    prompt_length: int = PROMPT_LENGTH,
    response_length: int = RESPONSE_LENGTH,
    variable_lengths: bool = False,
) -> TrainingInputBatch:
    """Return a bf16-friendly batch with left and right padding and fixed content.

    With ``variable_lengths`` each row's body is shorter than the previous one, as rollouts
    are, so consecutive micro-batches have different widths.
    """
    generator = torch.Generator().manual_seed(5)
    full_body_length = prompt_length + response_length
    total_length = full_body_length + 6
    pad_before = [3, 0, 5, 1]
    sequences = []
    masks = []
    # Each row loses this many tokens relative to the previous one; the last row still
    # keeps some response tokens once total padding is accounted for.
    shrink = response_length // (2 * batch_size) if variable_lengths else 0
    for row in range(batch_size):
        body_length = full_body_length - shrink * row
        body = torch.randint(10, 500, (body_length,), generator=generator).tolist()
        before = pad_before[row % len(pad_before)]
        after = total_length - body_length - before
        assert after < response_length, (row, after, response_length)
        sequences.append([pad_token_id] * before + body + [pad_token_id] * after)
        masks.append([0] * before + [1] * body_length + [0] * after)
    sequences = torch.tensor(sequences, dtype=torch.long)
    attention_mask = torch.tensor(masks, dtype=torch.long)
    response_mask = attention_mask[:, -response_length:]
    zeros = torch.zeros(batch_size, response_length, dtype=torch.float32)
    advantages = torch.linspace(-1.0, 1.0, response_length).unsqueeze(0).repeat(batch_size, 1)
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "action_log_probs": zeros.clone(),
            "base_action_log_probs": zeros.clone(),
            "rollout_logprobs": zeros.clone(),
            "values": zeros.clone(),
            "returns": zeros.clone(),
            "advantages": advantages,
            "loss_mask": response_mask.clone(),
            "response_mask": response_mask.clone(),
        }
    )
    batch.metadata = {"response_length": response_length, "global_step": 0}
    return batch


def _init_policy(cfg, world_size: int):
    return init_worker_with_type(
        "policy", shared_pg=None, colocate_all=False, num_gpus_per_node=world_size, num_nodes=1, cfg=cfg
    )


def _train_step(policy, batch: TrainingInputBatch) -> dict[str, float]:
    train_output = ray.get(policy.async_run_ray_method("mesh", "ppo_train", batch))[0]
    status = train_output.metadata["train_status"]
    assert math.isfinite(status["policy_loss"])
    return status
