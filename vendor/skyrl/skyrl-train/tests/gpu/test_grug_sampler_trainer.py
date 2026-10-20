from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

import numpy as np
import pytest
import ray
import torch
from omegaconf import OmegaConf
from vllm.transformers_utils.configs.grugmoe import GrugMoeConfig as VllmGrugMoeConfig

from skyrl_train.config.behavior_logprobs import configure_behavior_logprob_sampling
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.mismatch_probe.modes import NATIVE_MODE, REPLAY_MODE
from skyrl_train.models.grug_moe import GrugMoeConfig, grug_long_layer_flags
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trajectory_runners.model_clients import DirectModelClient
from skyrl_train.utils import initialize_ray
from skyrl_train.workers.worker import PPORayActorGroup
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_serving import grug_engine_client, rollout_training_batch
from tests.gpu.grug_tiny_checkpoint import TOY_SHAPE, write_tiny_checkpoint
from tests.gpu.grug_training import ROLLOUT_WORLD_SIZE, _config, _init_policy, _train_step

NUM_EXPERTS = TOY_SHAPE["num_local_experts"]
NUM_LAYERS = TOY_SHAPE["num_hidden_layers"]

TOP_K = 2
POLICY_WORLD_SIZE = 1
NUM_PROMPTS = 16
PROMPT_DIGITS = 12
RESPONSE_LENGTH = 24
# Router weights large enough that adjacent tokens usually choose different experts.
ROUTER_WEIGHT_STD = 0.2
MIN_ADJACENT_ROUTE_CHANGE = 0.5
LOGPROB_MEAN_ABS_TOLERANCE = 0.0009
LOGPROB_P99_ABS_TOLERANCE = 0.004
LAYER0_ROUTE_AGREEMENT_MIN = 0.90
ROUTE_AGREEMENT_MIN = 0.90
STATED_NUMERICS = {
    "vocab_size": 1024,
    **TOY_SHAPE,
    "num_experts_per_tok": TOP_K,
    "max_position_embeddings": 128,
    "qk_mult": 1.37,
}
RESOLVED_NUMERICS = (
    "sliding_window",
    "qk_mult",
    "qk_mult_long_scale",
    "rope_theta",
    "rms_norm_eps",
    "max_position_embeddings",
    "num_experts_per_tok",
    "global_every",
    "disable_pko",
    "disable_long_rope",
    "rope_fused",
    "local_kv_heads",
    "global_kv_heads",
    "latent_dim",
    "num_shared_experts",
    "sconv",
    "sconv_kernel",
    "grugmoe_artifact_schema_version",
)


@dataclass(frozen=True)
class ProbeScores:
    logprobs: torch.Tensor  # [batch, response]
    experts: np.ndarray  # [layer, batch, response, top_k], sorted per row
    natives: np.ndarray  # the router's own choice, same layout
    replayed: np.ndarray  # [layer, batch, response] bool


@dataclass(frozen=True)
class GrugRollout:
    vllm_logprobs: torch.Tensor
    native: ProbeScores
    replay: ProbeScores
    batch: TrainingInputBatch
    policy: PPORayActorGroup


def _probe_scores(
    policy: PPORayActorGroup, batch: TrainingInputBatch, routes: RoutedExpertRows, mode: str
) -> ProbeScores:
    data = batch.select(["sequences", "attention_mask"], ["response_length"])
    data["probe_row_indices"] = torch.arange(data.batch_size, dtype=torch.long)
    data.routed_expert_rows = routes
    data.metadata.update(probe_mode=mode, probe_keep_fraction=None, probe_micro_batch_size=None)
    outputs = ray.get(policy.async_run_ray_method("mesh", "probe_forward", data=data))
    logprobs = concatenate_outputs_after_mesh_dispatch(policy.actor_infos, outputs)["output"].float()
    shape = (NUM_LAYERS, data.batch_size, data.metadata["response_length"], TOP_K)
    experts, natives = np.full(shape, -1), np.full(shape, -1)
    replayed = np.zeros(shape[:3], dtype=bool)
    for output in outputs:
        for observation in output.metadata["probe_routes"]:
            row = (observation["layer"], observation["sample"], observation["position"])
            experts[row] = sorted(observation["effective"])
            natives[row] = sorted(observation["native"])
            replayed[row] = observation["route_valid"]
    return ProbeScores(logprobs, experts, natives, replayed)


@pytest.fixture(scope="module")
def grug_rollout(tmp_path_factory):
    require_hoppers(POLICY_WORLD_SIZE + ROLLOUT_WORLD_SIZE)
    model_path = tmp_path_factory.mktemp("grug") / "model"
    model_path.mkdir()
    write_tiny_checkpoint(model_path, num_experts_per_tok=TOP_K, router_weight_std=ROUTER_WEIGHT_STD)
    cfg = _config(str(model_path), world_size=POLICY_WORLD_SIZE, pp=1, ep=1)
    cfg.trainer.train_batch_size = cfg.trainer.policy_mini_batch_size = NUM_PROMPTS
    cfg.trainer.policy.megatron_config.moe_router_replay = True
    cfg.trainer.policy.optimizer_config.lr = 0.0
    cfg.generator.sampling_params.max_generate_length = RESPONSE_LENGTH
    configure_behavior_logprob_sampling(cfg.generator)
    OmegaConf.update(cfg.generator.engine_init_kwargs, "enable_return_routed_experts", True, force_add=True)
    initialize_ray(cfg)
    try:
        client = grug_engine_client(cfg, str(model_path))
        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
        sampling_params.update({"ignore_eos": True, "logprobs": 0})
        digits = torch.randint(0, 10, (NUM_PROMPTS, PROMPT_DIGITS), generator=torch.Generator().manual_seed(29))
        prompts = [[{"role": "user", "content": " ".join(map(str, row))}] for row in digits.tolist()]
        rollout = asyncio.run(
            DirectModelClient(client).generate(
                {
                    "prompts": prompts,
                    "chat_completion_params": [{"seed": 31 + index} for index in range(NUM_PROMPTS)],
                    "sampling_params": sampling_params,
                }
            )
        )
        assert len({len(ids) for ids in rollout["prompt_ids"]}) == 1
        batch = rollout_training_batch(rollout["prompt_ids"], rollout)
        routes = RoutedExpertRows(tuple(rollout["routed_experts"]), RESPONSE_LENGTH, NUM_EXPERTS)
        batch.routed_expert_rows = routes
        policy = _init_policy(cfg, POLICY_WORLD_SIZE)
        unreplayed = replace(routes, rows=tuple(np.zeros_like(rows) for rows in routes.rows))
        yield GrugRollout(
            vllm_logprobs=batch["rollout_logprobs"],
            native=_probe_scores(policy, batch, unreplayed, NATIVE_MODE),
            replay=_probe_scores(policy, batch, routes, REPLAY_MODE),
            batch=batch,
            policy=policy,
        )
    finally:
        ray.shutdown()


@pytest.mark.vllm
@pytest.mark.parametrize(
    "theta",
    [
        {"rope_theta": 500000.0},
        {"rope_parameters": {"rope_type": "default", "rope_theta": 500000.0}},
        {"rope": {"theta": 500000.0}},
    ],
    ids=["rope_theta", "rope_parameters", "rope"],
)
@pytest.mark.parametrize(
    "model_fields",
    [
        pytest.param({}, id="snowball"),
        pytest.param(
            {
                "hidden_size": 128,
                "num_hidden_layers": 4,
                "num_local_experts": 16,
                "num_experts_per_tok": 8,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "head_dim": 16,
                "sliding_window": 8,
                "latent_dim": 64,
                "num_shared_experts": 2,
                "local_kv_heads": 4,
                "global_kv_heads": 2,
                "rope_fused": True,
                "sconv": True,
                "sconv_kernel": 4,
                "grugmoe_artifact_schema_version": 2,
                "qk_mult_long_scale": 1.1386,
            },
            id="hero",
        ),
    ],
)
def test_grug_config_classes_resolve_the_same_numerics(theta, model_fields):
    stated = {**STATED_NUMERICS, **model_fields, **theta}
    trainer, sampler = GrugMoeConfig(**stated), VllmGrugMoeConfig(**stated)
    differing = {
        name: (getattr(trainer, name), getattr(sampler, name))
        for name in RESOLVED_NUMERICS
        if getattr(trainer, name) != getattr(sampler, name)
    }
    assert differing == {}
    assert tuple(trainer.sconv_sites) == tuple(sampler.sconv_sites)
    assert list(grug_long_layer_flags(trainer.num_hidden_layers, trainer.global_every)) == [
        kind == "full_attention" for kind in sampler.layer_types
    ]


def test_vllm_and_megatron_score_grug_tokens_alike_with_the_same_experts(grug_rollout):
    gap = (grug_rollout.replay.logprobs - grug_rollout.vllm_logprobs).abs()
    native_gap = (grug_rollout.native.logprobs - grug_rollout.vllm_logprobs).abs()
    mean, p99 = gap.mean().item(), torch.quantile(gap.flatten(), 0.99).item()
    print(
        f"sampler-trainer log-prob gap over {gap.numel()} tokens: replayed experts mean abs {mean:.6f}, "
        f"p99 abs {p99:.6f}, max abs {gap.max().item():.6f}; native experts mean abs "
        f"{native_gap.mean().item():.6f}, p99 abs {torch.quantile(native_gap.flatten(), 0.99).item():.6f}"
    )
    assert mean < LOGPROB_MEAN_ABS_TOLERANCE, mean
    assert p99 < LOGPROB_P99_ABS_TOLERANCE, p99


def test_replayed_experts_match_the_trainer_native_choices(grug_rollout):
    replay = grug_rollout.replay
    layer0 = replay.natives[0]
    adjacent_change = (layer0[:, 1:] != layer0[:, :-1]).any(-1).mean()
    agree = (replay.experts == replay.natives).all(-1)
    layer0_agreement = agree[0][replay.replayed[0]].mean()
    agreement = agree[replay.replayed].mean()
    print(
        f"router agreement over {int(replay.replayed.sum())} replayed rows: layer 0 {layer0_agreement:.4f}, "
        f"all layers {agreement:.4f}; layer-0 adjacent change {adjacent_change:.4f}"
    )
    assert adjacent_change >= MIN_ADJACENT_ROUTE_CHANGE, adjacent_change
    assert replay.replayed.any()
    assert layer0_agreement >= LAYER0_ROUTE_AGREEMENT_MIN, layer0_agreement
    assert agreement >= ROUTE_AGREEMENT_MIN, agreement
    assert replay.replayed.sum(axis=(1, 2)).tolist() == [NUM_PROMPTS * RESPONSE_LENGTH] * NUM_LAYERS


# Correction weights reach the real worker using the policy shared by the sampler and replay checks.
def test_correction_weights_scale_the_megatron_policy_loss(grug_rollout):
    batch = grug_rollout.batch
    weights = (0.25 + 0.5 * (torch.arange(RESPONSE_LENGTH) % 5)).float().repeat(batch.batch_size, 1)
    statuses = []
    for scale in (1.0, 0.5):
        weighted = TrainingInputBatch(
            {**dict(batch), "correction_weights": weights * scale}, routed_expert_rows=batch.routed_expert_rows
        )
        weighted.metadata = dict(batch.metadata)
        statuses.append(_train_step(grug_rollout.policy, weighted))
    assert statuses[0]["raw_grad_norm"] > 0
    assert statuses[0]["policy_loss"] != 0
    print(
        f"correction-weight ratios: policy_loss {statuses[1]['policy_loss'] / statuses[0]['policy_loss']:.8f}, "
        f"raw_grad_norm {statuses[1]['raw_grad_norm'] / statuses[0]['raw_grad_norm']:.8f}"
    )
    for key in ("policy_loss", "raw_grad_norm"):
        torch.testing.assert_close(statuses[1][key], 0.5 * statuses[0][key], rtol=1e-4, atol=0)
