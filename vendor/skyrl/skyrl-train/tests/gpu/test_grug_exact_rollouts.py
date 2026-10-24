import asyncio
import time

import numpy as np
import pytest
import ray
import torch
from transformers import AutoTokenizer

from skyrl_train.config.numerics import CheckFailure
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.entrypoints.main_base import create_ray_wrapped_inference_engines_from_config
from skyrl_train.exactness_check import ExactnessCheck
from skyrl_train.inference_engines.base import UNKNOWN_ENGINE_DP_RANK, InferenceEngineInput, InferenceEngineOutput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trajectory_runners.types import ENGINE_DP_RANKS_KEY, ROLLOUT_ENGINE_DP_RANKS_KEY
from skyrl_train.utils import initialize_ray
from skyrl_train.utils.utils import validate_cfg
from tests.gpu.grug_exact_checkpoint import exact_numerics_config, policy_logprobs, write_snowball_checkpoint
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.utils import init_worker_with_type


# One node: a decode-invariant engine of four data- and expert-parallel ranks, which add their expert partials in
# an order that depends on the serving rank, and a two-stage pipeline-parallel policy with two expert ranks.
ENGINE_RANKS = 4
POLICY_GPUS = 4
POLICY_EXPERT_PARALLEL = 2
PAD_TOKEN_ID = 0
# Prompts on both sides of the 2,048-token sliding window, served by every engine rank; one more row is a trajectory
# that failed before its model call (two placeholder tokens, no loss tokens, no serving rank).
PROMPT_LENGTHS = (1500, 1700, 1900, 2000, 2047, 2100, 2200, 2400, 2600, 2800, 3000, 3100, 1600, 2300, 2700)
RESPONSE_LENGTH = 384
# Each engine step prefills at most this many tokens, so prompts prefill in chunks over several steps while earlier
# requests decode.
MAX_BATCHED_TOKENS = 2048
SAMPLING_SEED = 11


def _config(model_path: str, router_replay: bool):
    rows = len(PROMPT_LENGTHS) + 1
    cfg = exact_numerics_config(
        model_path,
        policy_gpus=POLICY_GPUS,
        pipeline_stages=2,
        expert_parallel=POLICY_EXPERT_PARALLEL,
        engine_ranks=ENGINE_RANKS,
    )
    cfg.trainer.gradient_checkpointing = False
    cfg.trainer.max_prompt_length = max(PROMPT_LENGTHS)
    cfg.trainer.train_batch_size = rows
    cfg.trainer.policy_mini_batch_size = rows
    cfg.trainer.micro_train_batch_size_per_gpu = 4
    cfg.trainer.micro_forward_batch_size_per_gpu = 4
    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.numerics.check.every_weight_syncs = 1
    cfg.trainer.policy.megatron_config.moe_router_replay = router_replay
    # Large enough that one Adam step moves bf16 weights.
    cfg.trainer.policy.optimizer_config.lr = 2.0e-2
    cfg.trainer.policy.optimizer_config.max_grad_norm = 0.0
    cfg.generator.weight_sync_backend = "nccl"
    cfg.generator.max_num_batched_tokens = MAX_BATCHED_TOKENS
    cfg.generator.engine_init_kwargs = {
        "moe_backend": "triton",
        "enable_return_routed_experts": router_replay,
        "max_model_len": max(PROMPT_LENGTHS) + RESPONSE_LENGTH,
    }
    cfg.generator.sampling_params.max_generate_length = RESPONSE_LENGTH
    validate_cfg(cfg)
    return cfg


def _prompts(vocab_size: int) -> list[list[int]]:
    generator = torch.Generator().manual_seed(5)
    return [torch.randint(10, vocab_size, (length,), generator=generator).tolist() for length in PROMPT_LENGTHS]


def _generate(client: InferenceEngineClient, prompts: list[list[int]], sampling_params: dict) -> InferenceEngineOutput:
    started = time.monotonic()
    rollout = asyncio.run(
        client.generate(InferenceEngineInput(prompt_token_ids=prompts, sampling_params=sampling_params))
    )
    print(f"generated {len(prompts)} responses in {time.monotonic() - started:.1f}s: {rollout['stop_reasons']}")
    assert all(len(response) == RESPONSE_LENGTH for response in rollout["response_ids"]), rollout["stop_reasons"]
    assert set(rollout[ENGINE_DP_RANKS_KEY]) == set(range(ENGINE_RANKS)), rollout[ENGINE_DP_RANKS_KEY]
    return rollout


def _engine_logprobs(rollout: InferenceEngineOutput) -> np.ndarray:
    return np.asarray(rollout["response_logprobs"], dtype=np.float32)


def _batch(
    prompts: list[list[int]], rollout: InferenceEngineOutput, num_experts: int, route_shift: int = 0
) -> TrainingInputBatch:
    """The rollout's rows and the failed trajectory's placeholder row, left-padded, as the trainer receives them, with
    the rollout's expert routes when the engines returned them, each row's routes moved ``route_shift`` positions."""
    rows = [prompt + response for prompt, response in zip(prompts, rollout["response_ids"], strict=True)]
    rows.append([PAD_TOKEN_ID, PAD_TOKEN_ID])
    width = max(len(row) for row in rows)
    sequences = torch.full((len(rows), width), PAD_TOKEN_ID, dtype=torch.long)
    attention_mask = torch.zeros(len(rows), width, dtype=torch.long)
    for index, row in enumerate(rows):
        sequences[index, width - len(row) :] = torch.tensor(row)
        attention_mask[index, width - len(row) :] = 1
    loss_mask = attention_mask[:, -RESPONSE_LENGTH:].clone()
    loss_mask[-1] = 0
    engine_logprobs = torch.zeros(len(rows), RESPONSE_LENGTH)
    engine_logprobs[:-1] = torch.from_numpy(_engine_logprobs(rollout))
    advantages = torch.linspace(-1.0, 1.0, RESPONSE_LENGTH).unsqueeze(0).repeat(len(rows), 1) * loss_mask
    routes = rollout.get("routed_experts")
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "response_mask": attention_mask[:, -RESPONSE_LENGTH:].clone(),
            "rollout_logprobs": engine_logprobs,
            "base_action_log_probs": engine_logprobs.clone(),
            "values": torch.zeros_like(engine_logprobs),
            "returns": torch.zeros_like(engine_logprobs),
            "advantages": advantages,
            ROLLOUT_ENGINE_DP_RANKS_KEY: torch.tensor(
                [*rollout[ENGINE_DP_RANKS_KEY], UNKNOWN_ENGINE_DP_RANK], dtype=torch.long
            ),
        },
        routed_expert_rows=(
            RoutedExpertRows(
                (
                    *(np.roll(route, route_shift, axis=0) for route in routes),
                    np.zeros((1, *routes[0].shape[1:]), np.uint8),
                ),
                RESPONSE_LENGTH,
                num_experts,
            )
            if routes is not None
            else None
        ),
    )
    batch.metadata = {"response_length": RESPONSE_LENGTH, "global_step": 0}
    return batch


def _mismatches(name: str, trainer: torch.Tensor, rollout: InferenceEngineOutput) -> int:
    """How many of the rollout's log-probabilities the trainer's rows (the rollout's rows first) do not reproduce bit
    for bit."""
    engine = _engine_logprobs(rollout)
    scores = trainer[: engine.shape[0]].numpy()
    differing = scores.view(np.int32) != engine.view(np.int32)
    largest = np.abs(scores.astype(np.float64) - engine.astype(np.float64)).max()
    print(
        f"{name}: {int(differing.sum())} of {engine.size} log-probabilities differ "
        f"({int(differing.any(axis=1).sum())} of {engine.shape[0]} rows; largest difference {largest:.3g})"
    )
    return int(differing.sum())


def _run_check(check: ExactnessCheck, client: InferenceEngineClient) -> None:
    async def sample():
        check.start_generation(client)
        return await check.finish_generation()

    assert asyncio.run(sample()) == {}


@pytest.mark.parametrize("router_replay", [False, True], ids=["native_routing", "router_replay"])
def test_exact_trainer_reproduces_decode_invariant_engine_logprobs_across_a_weight_update(tmp_path, router_replay):
    """The exact Megatron trainer gives every response token of a decode-invariant engine the engine's
    log-probability bit for bit, with a failed trajectory's placeholder row in the batch, and the online exactness
    check passes. Routes moved by one position no longer match. The training forward reproduces the scoring
    forward's log-probabilities, and after one optimizer step and the production weight sync the new responses match again.
    """
    require_hoppers(POLICY_GPUS + ENGINE_RANKS)
    model_path = tmp_path / "model"
    model_path.mkdir()
    model_config = write_snowball_checkpoint(model_path)
    cfg = _config(str(model_path), router_replay)
    num_experts = model_config.num_local_experts
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = _prompts(model_config.vocab_size)
    sampling_params = {
        **get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params),
        "max_tokens": RESPONSE_LENGTH,
        "logprobs": 0,
        "seed": SAMPLING_SEED,
        "stop": None,
    }
    initialize_ray(cfg)
    try:
        engines = create_ray_wrapped_inference_engines_from_config(cfg, colocate_pg=None, tokenizer=tokenizer)
        client = InferenceEngineClient(engines, tokenizer, cfg)
        first = _generate(client, prompts, sampling_params)
        assert (first.get("routed_experts") is not None) == router_replay

        policy = init_worker_with_type(
            "policy", shared_pg=None, colocate_all=False, num_gpus_per_node=POLICY_GPUS, num_nodes=1, cfg=cfg
        )
        batch = _batch(prompts, first, num_experts)
        logprobs = policy_logprobs(policy, batch)
        assert _mismatches("initial weights", logprobs, first) == 0
        if router_replay:
            shifted = policy_logprobs(policy, _batch(prompts, first, num_experts, route_shift=1))
            assert _mismatches("routes moved by one position", shifted, first) > 0
        check = ExactnessCheck(
            cfg.trainer.numerics.check.every_weight_syncs,
            CheckFailure(cfg.trainer.numerics.check.on_failure),
            cfg.generator.sampling_params,
            tokenizer,
        )
        _run_check(check, client)
        check_batch = check.batch(policy.actor_infos[0].rank.dp_size, num_experts)
        metrics = check.compare(policy_logprobs(policy, check_batch), score_seconds=0.0)
        assert metrics["exactness_check/tokens"] > 0 and metrics["exactness_check/mismatched_tokens"] == 0

        batch["action_log_probs"] = logprobs * batch["loss_mask"]
        status = ray.get(policy.async_run_ray_method("mesh", "ppo_train", batch))[0].metadata["train_status"]
        print(f"policy step: loss {status['policy_loss']}, log-ratio max abs {status['log_ratio_abs_max']}")
        assert status["log_ratio_diagnostics_failed"] == 0.0 and status["log_ratio_abs_max"] == 0.0, status
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))

        updated = _generate(client, prompts, sampling_params)
        assert updated["response_ids"] != first["response_ids"]
        assert (
            _mismatches("updated weights", policy_logprobs(policy, _batch(prompts, updated, num_experts)), updated) == 0
        )
    finally:
        ray.shutdown()
