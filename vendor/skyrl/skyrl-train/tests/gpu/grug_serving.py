"""Marin vLLM serving helpers shared by the Grug GPU cycle tests."""

import math

import ray
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
from skyrl_train.training_batch import TrainingInputBatch
from tests.gpu.grug_tiny_checkpoint import TOY_SHAPE, router_bias_names

MAX_MODEL_LEN = 128
STACKED_EXPERT_NAME = "model.layers.0.mlp.experts.gate_proj.weight"
LM_HEAD_NAME = "lm_head.weight"
ROUTER_NAME = "model.layers.0.mlp.router.weight"
# Broad HF-named parameter coverage for training-vs-serving readback: attention, gated
# norms, embeddings, stacked and per-expert MoE weights, shared expert, output head,
# router, and the frozen router biases. The bias list derives from the toy fixture shape
# so it cannot drift from the checkpoint the serving tests load.
GRUG_BIAS_NAMES = router_bias_names(TOY_SHAPE["num_hidden_layers"])
GRUG_PARAMETER_NAMES = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.3.self_attn.q_proj.weight",
    "model.layers.1.self_attn.attn_gate.weight",
    "model.layers.1.attn_gated_norm.up_proj.weight",
    "model.embed_gated_norm.down_proj.weight",
    "model.norm.weight",
    STACKED_EXPERT_NAME,
    "model.layers.7.mlp.experts.down_proj.weight",
    "model.layers.2.shared_expert.up_proj.weight",
    LM_HEAD_NAME,
    ROUTER_NAME,
]
GRUG_SERVING_EXPERT_INDEX_BY_NAME = {
    "model.layers.0.mlp.experts.0.gate_proj.weight": 0,
    "model.layers.0.mlp.experts.5.gate_proj.weight": 5,
}
# Alternating per-row advantages so one PPO step pushes sampled tokens in both directions.
ADVANTAGE_PATTERN = torch.tensor([[-1.0, -0.25, 0.5, 1.0], [1.0, 0.5, -0.25, -1.0]])


def grug_engine_client(
    cfg,
    model_path: str,
    *,
    shared_pg=None,
    inference_engine_enable_sleep: bool = False,
    moe_backend: str | None = None,
) -> InferenceEngineClient:
    """Start vLLM engines for a tiny Grug checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    engine_init_kwargs = {
        "max_model_len": MAX_MODEL_LEN,
        **OmegaConf.to_container(cfg.generator.engine_init_kwargs, resolve=True),
    }
    if moe_backend is not None:
        engine_init_kwargs["kernel_config"] = {"moe_backend": moe_backend}
    engines = create_ray_wrapped_inference_engines(
        num_inference_engines=cfg.generator.num_inference_engines,
        tensor_parallel_size=1,
        data_parallel_size=cfg.generator.inference_engine_data_parallel_size,
        expert_parallel_size=cfg.generator.inference_engine_expert_parallel_size,
        model_dtype="bfloat16",
        pretrain=model_path,
        seed=23,
        vllm_v1_disable_multiproc=True,
        enable_prefix_caching=False,
        enforce_eager=cfg.generator.enforce_eager,
        engine_init_timeout_seconds=cfg.generator.engine_init_timeout_seconds,
        shared_pg=shared_pg,
        gpu_memory_utilization=cfg.generator.gpu_memory_utilization,
        inference_engine_enable_sleep=inference_engine_enable_sleep,
        max_num_batched_tokens=MAX_MODEL_LEN * cfg.generator.inference_engine_data_parallel_size,
        max_num_seqs=cfg.trainer.train_batch_size,
        tokenizer=tokenizer,
        backend="vllm",
        engine_init_kwargs=engine_init_kwargs,
    )
    return InferenceEngineClient(engines, tokenizer, cfg)


def rollout_training_batch(prompts: list[list[int]], rollout) -> TrainingInputBatch:
    """Build a PPO training batch from a fixed-length vLLM rollout of the given prompts."""
    assert rollout["response_logprobs"] is not None
    sequences = torch.tensor(
        [prompt + response for prompt, response in zip(prompts, rollout["response_ids"])], dtype=torch.long
    )
    action_log_probs = torch.tensor(rollout["response_logprobs"], dtype=torch.float32)
    response_length = action_log_probs.shape[1]
    advantages = ADVANTAGE_PATTERN.repeat(
        math.ceil(sequences.shape[0] / 2), math.ceil(response_length / ADVANTAGE_PATTERN.shape[1])
    )[: sequences.shape[0], :response_length]
    ones = torch.ones_like(action_log_probs)
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": torch.ones_like(sequences),
            "action_log_probs": action_log_probs,
            "base_action_log_probs": action_log_probs.clone(),
            "rollout_logprobs": action_log_probs.clone(),
            "values": torch.zeros_like(ones),
            "returns": torch.zeros_like(ones),
            "advantages": advantages,
            "loss_mask": ones.long(),
            "response_mask": ones.long(),
        }
    )
    batch.metadata = {"response_length": action_log_probs.shape[1], "global_step": 0}
    return batch


def rank0_validation_snapshot(policy, names=()) -> dict[str, torch.Tensor]:
    """Gather the named full parameters from the policy workers and return rank 0's copy."""
    snapshots = ray.get(policy.async_run_ray_method("pass_through", "grug_validation_snapshot", names))
    return next(snapshot for snapshot in snapshots if snapshot.rank == 0).weights


def assert_engine_weights(
    client: InferenceEngineClient,
    names: list[str],
    training: dict[str, torch.Tensor],
    bias_names: list[str],
    serving_expert_index_by_name: dict[str, int],
) -> dict[str, int]:
    """Check that every serving rank holding a named weight matches the training snapshot.

    Router biases and router weights must arrive in fp32 and everything else in bf16.
    Split experts use their named training tensor; stacked experts use its slice.
    Returns the serving EP rank that owns each named expert.
    """
    found = {name: False for name in names}
    expert_owners = {name: set() for name in serving_expert_index_by_name}
    for engine in client.engines:
        per_rank = ray.get(engine.inference_engine_actor.read_engine_weights.remote(names, False))
        if isinstance(per_rank, dict):
            per_rank = [per_rank]
        for rank_values in per_rank:
            serving_ep_rank = int(rank_values["__ranks__"]["ep_rank"])
            # A pipeline stage does not hold the weights of other stages' layers. The check at the
            # end still requires every name to be found on some stage.
            staged = int(rank_values["__ranks__"]["pp_size"]) > 1
            for name in names:
                entry = rank_values[name]
                if entry.get("skip"):
                    continue
                if staged and not entry["found"] and "error" not in entry:
                    continue
                assert entry["found"], (name, entry)
                found[name] = True
                expected_dtype = "float32" if name in bias_names or name == ROUTER_NAME else "bfloat16"
                assert entry["dtype"] == expected_dtype, (name, entry["dtype"])
                expert_index = serving_expert_index_by_name.get(name)
                if expert_index is not None:
                    expert_owners[name].add(serving_ep_rank)
                expected = training[name] if name in training else training[STACKED_EXPERT_NAME][expert_index]
                actual = entry["tensor"]
                if name not in bias_names:
                    expected = expected.to(torch.bfloat16).to(actual.dtype)
                if name == LM_HEAD_NAME:
                    # vLLM aligns the vocabulary dimension (151665 -> 151680
                    # for this tokenizer). The HF weight occupies the prefix.
                    assert actual.shape[0] >= expected.shape[0], (actual.shape, expected.shape)
                    actual = actual[: expected.shape[0]]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(found.values()), found
    assert all(len(owners) == 1 for owners in expert_owners.values()), expert_owners
    return {name: next(iter(owners)) for name, owners in expert_owners.items()}
