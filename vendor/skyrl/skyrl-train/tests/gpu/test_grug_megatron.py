"""Grug on the Megatron trainer, pipeline-parallel first.

The parity tests compare Megatron policy log-probabilities against the eager
HF reference at PP1, PP2, and PP2+EP2. The training test takes one PPO step
at PP2, checks that exported weights moved while the frozen router bias did
not, and checks that the exported HF checkpoint reproduces the post-update
Megatron log-probabilities. The four-H100 test adds disaggregated vLLM
serving: rollout, PP2 update, weight broadcast, serving readback, rollout.
The resume test reloads a checkpoint saved with Megatron's CPU optimizer offload.
"""

from __future__ import annotations

import asyncio
import copy
import math
import os
import uuid
from types import SimpleNamespace

import pytest
import ray
import torch
from megatron.bridge import AutoBridge
from megatron.core import parallel_state, tensor_parallel
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.transformer.module import Float16Module
from omegaconf import open_dict
from ray.util.placement_group import placement_group
from rigging.filesystem.cluster_config import marin_temp_bucket
from transformers import AutoTokenizer

import skyrl_train.models.grug_megatron_bridge  # noqa: F401  # registers the Grug bridge
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.distributed.megatron.grug_muonh import MegatronGrugMuonH, _offloaded_muon_direction_in_grad_
from skyrl_train.distributed.megatron.megatron_utils import load_megatron_optimizer, offload_megatron_optimizer
from skyrl_train.distributed.megatron.optimizer import HybridDeviceOptimizer
from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeForCausalLM
from skyrl_train.workers.megatron.bridge_provider_config import configure_bridge_provider
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import initialize_ray
from skyrl_train.utils.torch_utils import logprobs_from_logits
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_serving import (
    GRUG_BIAS_NAMES as BIAS_NAMES,
    GRUG_PARAMETER_NAMES as PARAMETER_NAMES,
    GRUG_SERVING_EXPERT_INDEX_BY_NAME as SERVING_EXPERT_INDEX_BY_NAME,
    LM_HEAD_NAME,
    ROUTER_NAME,
    assert_engine_weights,
    grug_engine_client,
    rank0_validation_snapshot,
    rollout_training_batch,
)
from tests.gpu.grug_tiny_checkpoint import TOY_SHAPE, write_tiny_checkpoint
from tests.gpu.utils import get_test_actor_config, init_worker_with_type
from skyrl_train.weight_sync.expert_block.driver import ExpertBlockSync

ROLLOUT_WORLD_SIZE = 2
RESPONSE_LENGTH = 8
PROMPT_LENGTH = 12


def _spot_check_name(suffix: str) -> str:
    """First broad-coverage parameter name with the given suffix."""
    return next(name for name in PARAMETER_NAMES if name.endswith(suffix))


ATTN_GATE_NAME = _spot_check_name(".self_attn.attn_gate.weight")
GATED_NORM_NAME = _spot_check_name(".attn_gated_norm.up_proj.weight")
LOGPROB_MAX_ABS_TOLERANCE = 2e-1
LOGPROB_MEAN_ABS_TOLERANCE = 3e-2
TRAIN_EVAL_LOGPROB_MAX_ABS_TOLERANCE = 1e-3


# Snowball's attention geometry, expert count, and window at a fraction of its width and depth.
SNOWBALL_LIKE_SHAPE = dict(
    hidden_size=2560,
    intermediate_size=128,
    shared_expert_intermediate_size=256,
    num_local_experts=256,
    num_hidden_layers=4,
    num_attention_heads=20,
    num_key_value_heads=5,
    head_dim=128,
    sliding_window=2048,
)
# Snowball's width with few experts, so each expert sees as many tokens per micro-batch
# as it does at full scale (about 700 for 2700-token rows across two EP ranks).
SNOWBALL_LIKE_DENSE_EXPERTS_SHAPE = {**SNOWBALL_LIKE_SHAPE, "num_local_experts": 32}


@pytest.fixture(params=[torch.float32, torch.bfloat16])
def embedding_preprocess_model(request, tmp_path):
    require_hoppers(1)
    torch.cuda.set_device(0)
    torch.distributed.init_process_group("nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        parallel_state.initialize_model_parallel()
        tensor_parallel.model_parallel_cuda_manual_seed(19)
        config = GrugMoeConfig(
            vocab_size=1024,
            num_experts_per_tok=2,
            max_position_embeddings=128,
            **{**TOY_SHAPE, "num_hidden_layers": 2},
        )
        provider = AutoBridge.from_hf_config(config).to_megatron_provider(load_weights=False)
        provider.params_dtype = request.param
        provider.bf16 = request.param == torch.bfloat16
        provider.fp16 = False
        configure_bridge_provider(
            provider,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
            bf16=provider.bf16,
            attention_backend="fused" if provider.bf16 else "unfused",
        )
        provider.recompute_granularity = "full"
        provider.recompute_method = "uniform"
        provider.recompute_num_layers = 1
        provider.finalize()
        chunk = provider.provide_distributed_model(wrap_with_ddp=False, bf16=provider.bf16)[0]
        model = chunk.module if isinstance(chunk, Float16Module) else chunk
        yield model
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("freeze_embeddings", [False, True])
def test_grug_embedding_recompute_preserves_gradients_and_reduces_saved_activations(
    embedding_preprocess_model, freeze_embeddings
):
    model = embedding_preprocess_model.train()
    model.embedding.word_embeddings.weight.requires_grad_(not freeze_embeddings)
    tokens = torch.arange(32, device="cuda").reshape(1, 32)
    parameters = [p for module in (model.embedding, model.embed_norm) for p in module.parameters() if p.requires_grad]
    parameter_storages = {p.untyped_storage().data_ptr() for p in model.parameters()}
    results = []
    saved_bytes = []
    for recompute in (None, "full"):
        model.config.recompute_granularity = recompute
        model.zero_grad(set_to_none=True)
        saved = {}

        def pack(value):
            storage = value.untyped_storage()
            if storage.data_ptr() not in parameter_storages:
                saved[storage.data_ptr()] = storage.nbytes()
            return value

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            output = model._preprocess(tokens, tokens)[0]
        saved_bytes.append(sum(saved.values()))
        output.float().square().mean().backward()
        gradients = [p.grad for p in parameters]
        assert all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients)
        results.append((output.detach(), [gradient.clone() for gradient in gradients]))
    torch.testing.assert_close(results[1], results[0], rtol=1e-5, atol=1e-5)
    if not freeze_embeddings:
        assert saved_bytes[1] < saved_bytes[0], saved_bytes


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


def _hf_response_logprobs_direct(model_path: str, batch: TrainingInputBatch) -> torch.Tensor:
    model = GrugMoeForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation="eager")
    model.eval().to("cuda")
    sequences = batch["sequences"].to("cuda")
    attention_mask = batch["attention_mask"].to("cuda")
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    num_actions = batch.metadata["response_length"]
    with torch.no_grad():
        logits = model(sequences, attention_mask=attention_mask, position_ids=position_ids).logits
        log_probs = logprobs_from_logits(logits, torch.roll(sequences, shifts=-1, dims=1))
    return log_probs[:, -num_actions - 1 : -1].float().cpu()


@ray.remote(num_gpus=1)
def _hf_response_logprobs(model_path: str, batch: TrainingInputBatch) -> torch.Tensor:
    return _hf_response_logprobs_direct(model_path, batch)


def _megatron_response_logprobs(policy, batch: TrainingInputBatch) -> torch.Tensor:
    outputs = ray.get(policy.async_run_ray_method("mesh", "forward", data=batch))
    return concatenate_outputs_after_mesh_dispatch(policy.actor_infos, outputs)["output"].float()


def _assert_logprobs_close(actual: torch.Tensor, expected: torch.Tensor, response_mask: torch.Tensor) -> None:
    valid = response_mask.bool()
    diff = (actual[valid] - expected[valid]).abs()
    print(
        f"logprob parity over {valid.sum().item()} tokens: max abs {diff.max().item():.4f}, mean abs {diff.mean().item():.4f}"
    )
    assert diff.max().item() < LOGPROB_MAX_ABS_TOLERANCE, diff.max()
    assert diff.mean().item() < LOGPROB_MEAN_ABS_TOLERANCE, diff.mean()


def _init_policy(cfg, world_size: int):
    return init_worker_with_type(
        "policy", shared_pg=None, colocate_all=False, num_gpus_per_node=world_size, num_nodes=1, cfg=cfg
    )


def _train_step(policy, batch: TrainingInputBatch) -> dict[str, float]:
    train_output = ray.get(policy.async_run_ray_method("mesh", "ppo_train", batch))[0]
    status = train_output.metadata["train_status"]
    assert math.isfinite(status["policy_loss"])
    return status


@pytest.mark.parametrize(
    ("world_size", "pp", "ep"),
    [(1, 1, 1), (2, 2, 1), (4, 2, 2)],
    ids=["pp1", "pp2", "pp2_ep2"],
)
def test_grug_megatron_forward_matches_hf(tmp_path, world_size: int, pp: int, ep: int):
    require_hoppers(world_size)
    model_path = tmp_path / "model"
    model_path.mkdir()
    write_tiny_checkpoint(model_path)
    cfg = _config(str(model_path), world_size=world_size, pp=pp, ep=ep)
    pad_token_id = AutoTokenizer.from_pretrained(model_path).pad_token_id
    batch = _padded_batch(pad_token_id)
    initialize_ray(cfg)
    try:
        expected = ray.get(_hf_response_logprobs.remote(str(model_path), batch))
        policy = _init_policy(cfg, world_size)
        actual = _megatron_response_logprobs(policy, batch)
        _assert_logprobs_close(actual, expected, batch["response_mask"])
    finally:
        ray.shutdown()


@pytest.mark.parametrize(
    ("world_size", "pp", "ep"),
    [(1, 1, 1), (2, 2, 1), (4, 2, 2), (4, 2, 1), (4, 1, 2)],
    ids=["pp1", "pp2", "pp2_ep2", "pp2_dp2", "ep2_dp2"],
)
@pytest.mark.parametrize(
    ("shape", "num_experts_per_tok", "prompt_length", "response_length"),
    [
        (TOY_SHAPE, 2, 24, 16),
        (TOY_SHAPE, 4, 1000, 200),
        (SNOWBALL_LIKE_SHAPE, 4, 2400, 300),
        (SNOWBALL_LIKE_SHAPE, 4, 1024, 8192),
        (SNOWBALL_LIKE_DENSE_EXPERTS_SHAPE, 4, 2400, 300),
    ],
    ids=[
        "toy_top2_short",
        "toy_top4_long",
        "snowball_like_top4_long",
        "snowball_like_top4_full_length",
        "snowball_like_dense_experts",
    ],
)
def test_grug_megatron_train_forward_matches_eval_forward(
    tmp_path,
    world_size: int,
    pp: int,
    ep: int,
    shape: dict,
    num_experts_per_tok: int,
    prompt_length: int,
    response_length: int,
):
    """The training forward must reproduce the eval-mode log-probs it is scored against.

    With one update per batch the PPO ratio is exp(train_logprob - eval_logprob), so any
    train/eval drift shows up as spurious clipping. Top-4
    routing exposed Megatron's unfused, atomic unpermute (the bridge now forces the fused
    kernels), and Snowball's width exposed cuBLAS kernel selection changing with the
    micro-batch shape (the two passes must use equal micro-batch sizes).
    """
    require_hoppers(world_size)
    model_path = tmp_path / "model"
    model_path.mkdir()
    write_tiny_checkpoint(
        model_path, max_position_embeddings=16384, num_experts_per_tok=num_experts_per_tok, shape=shape
    )
    cfg = _config(str(model_path), world_size=world_size, pp=pp, ep=ep)
    # The forward and training passes must see the same micro-batch shapes; see the trainer docs.
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    # The production DDP settings from cloud/iris/configs/snowball_megatron_full.yaml.
    cfg.trainer.policy.megatron_config.ddp_config.overlap_grad_reduce = True
    cfg.trainer.policy.megatron_config.ddp_config.overlap_param_gather = True
    cfg.trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32 = False
    pad_token_id = AutoTokenizer.from_pretrained(model_path).pad_token_id
    batch = _padded_batch(
        pad_token_id, prompt_length=prompt_length, response_length=response_length, variable_lengths=True
    )
    initialize_ray(cfg)
    try:
        policy = _init_policy(cfg, world_size)
        # The trainer offloads the optimizer for rollouts and reloads it before training.
        ray.get(
            policy.async_run_ray_method("pass_through", "offload_to_cpu", offload_optimizer=True, offload_model=False)
        )
        eval_logprobs = _megatron_response_logprobs(policy, batch)
        batch["action_log_probs"] = (eval_logprobs * batch["response_mask"]).float()
        ray.get(
            policy.async_run_ray_method(
                "pass_through", "backload_to_gpu", backload_optimizer=True, backload_model=False
            )
        )
        train_output = ray.get(policy.async_run_ray_method("mesh", "ppo_train", batch))[0]
        status = train_output.metadata["train_status"]
        print(
            f"train/eval log-ratio: mean abs {status['log_ratio_abs_mean']:.6f}, max abs {status['log_ratio_abs_max']:.6f}, "
            f"exact-unit fraction {status['ppo_ratio_exact_unit_fraction']:.4f}"
        )
        assert status["log_ratio_abs_max"] < TRAIN_EVAL_LOGPROB_MAX_ABS_TOLERANCE, status["log_ratio_abs_max"]
    finally:
        ray.shutdown()


@pytest.mark.parametrize(("world_size", "pp", "ep"), [(2, 1, 2), (4, 2, 2)], ids=["pp1_ep2", "pp2_ep2"])
@pytest.mark.parametrize(
    ("shape", "num_experts_per_tok", "prompt_length", "response_length"),
    [(TOY_SHAPE, 4, PROMPT_LENGTH, RESPONSE_LENGTH), (SNOWBALL_LIKE_SHAPE, 4, 2400, 300)],
    ids=["toy_top4", "snowball_like_top4_long"],
)
def test_grug_megatron_eval_forward_is_independent_of_peer_rank_batch(
    tmp_path,
    world_size: int,
    pp: int,
    ep: int,
    shape: dict,
    num_experts_per_tok: int,
    prompt_length: int,
    response_length: int,
):
    """A rank's log-probs must not change when only the peer expert-parallel rank's rows change.

    Expert parallelism co-batches tokens from every rank in the expert group, so a
    composition-dependent kernel would make old log-probs depend on which rows the
    other ranks happened to hold.
    """
    require_hoppers(world_size)
    model_path = tmp_path / "model"
    model_path.mkdir()
    write_tiny_checkpoint(
        model_path, max_position_embeddings=4096, num_experts_per_tok=num_experts_per_tok, shape=shape
    )
    cfg = _config(str(model_path), world_size=world_size, pp=pp, ep=ep)
    pad_token_id = AutoTokenizer.from_pretrained(model_path).pad_token_id
    rows = _padded_batch(pad_token_id, batch_size=6, prompt_length=prompt_length, response_length=response_length)
    first = rows.slice(0, 4)
    peer_swapped = TrainingInputBatch({key: torch.cat([rows[key][0:2], rows[key][4:6]]) for key in rows.keys()})
    peer_swapped.metadata = rows.metadata
    initialize_ray(cfg)
    try:
        policy = _init_policy(cfg, world_size)
        baseline = _megatron_response_logprobs(policy, first)
        repeated = _megatron_response_logprobs(policy, first)
        swapped = _megatron_response_logprobs(policy, peer_swapped)
        valid = first["response_mask"][:2].bool()
        repeat_diff = (repeated[:2][valid] - baseline[:2][valid]).abs().max().item()
        swap_diff = (swapped[:2][valid] - baseline[:2][valid]).abs().max().item()
        print(f"rank-0 rows: repeat max abs {repeat_diff:.6f}, peer-swapped max abs {swap_diff:.6f}")
        assert repeat_diff == 0.0, repeat_diff
        assert swap_diff == 0.0, swap_diff
    finally:
        ray.shutdown()


def test_grug_megatron_pp2_train_step_updates_weights_and_exports(tmp_path):
    world_size = 2
    require_hoppers(world_size)
    model_path = tmp_path / "model"
    model_path.mkdir()
    write_tiny_checkpoint(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    batch = _padded_batch(tokenizer.pad_token_id)
    names = [*PARAMETER_NAMES, *BIAS_NAMES]
    original = GrugMoeForCausalLM.from_pretrained(model_path, dtype=torch.float32).state_dict()
    results = []
    for checkpoint_cpu_offload in (False, True):
        cfg = _config(str(model_path), world_size=world_size, pp=2, ep=1)
        with open_dict(cfg.trainer.policy.megatron_config.transformer_config_kwargs):
            cfg.trainer.policy.megatron_config.transformer_config_kwargs.grug_checkpoint_cpu_offload = (
                checkpoint_cpu_offload
            )
        export_dir = tmp_path / ("checkpoint_cpu" if checkpoint_cpu_offload else "device")
        initialize_ray(cfg)
        try:
            policy = _init_policy(cfg, world_size)
            before = rank0_validation_snapshot(policy, names)
            for name in PARAMETER_NAMES:
                # Megatron holds bf16 parameters; the fp32 checkpoint rounds once on load.
                torch.testing.assert_close(before[name], original[name].to(torch.bfloat16).float(), rtol=0, atol=0)
            for name in BIAS_NAMES:
                torch.testing.assert_close(before[name], original[name].float(), rtol=0, atol=0)

            status = _train_step(policy, batch)
            after = rank0_validation_snapshot(policy, names)
            for name in PARAMETER_NAMES:
                assert not torch.equal(after[name], before[name]), f"{name} did not update"
            for name in BIAS_NAMES:
                torch.testing.assert_close(after[name], before[name], rtol=0, atol=0)

            post_update = _megatron_response_logprobs(policy, batch)
            ray.get(policy.async_run_ray_method("pass_through", "save_hf_model", str(export_dir), tokenizer))
            exported = GrugMoeForCausalLM.from_pretrained(export_dir, dtype=torch.float32).state_dict()
            for name in names:
                torch.testing.assert_close(exported[name].float(), after[name], rtol=0, atol=0)
            assert all(exported[name].dtype == torch.float32 for name in BIAS_NAMES)
            policy.kill_actors()
            reloaded = _hf_response_logprobs_direct(str(export_dir), batch)
            _assert_logprobs_close(post_update, reloaded, batch["response_mask"])
            results.append({"loss": status["policy_loss"], "weights": exported, "logprobs": post_update})
        finally:
            ray.shutdown()
    torch.testing.assert_close(results[1], results[0], rtol=1e-5, atol=1e-5)


@pytest.mark.vllm
@pytest.mark.parametrize(
    ("transport", "nccl_max_channels"),
    [("broadcast", None), ("broadcast", 32), ("expert_block", None)],
)
def test_grug_megatron_four_gpu_disaggregated_rollout_train_sync_rollout(tmp_path, nccl_max_channels, transport):
    """Rollout, Megatron update, checked weight transfer, serving readback, rollout."""
    policy_world_size = 2
    require_hoppers(policy_world_size + ROLLOUT_WORLD_SIZE)
    model_path = tmp_path / "model"
    model_path.mkdir()
    write_tiny_checkpoint(model_path)
    cfg = _config(
        str(model_path),
        world_size=policy_world_size,
        pp=2 if transport == "broadcast" else 1,
        ep=1 if transport == "broadcast" else 2,
    )
    cfg.generator.weight_sync_transport = transport
    cfg.generator.expert_block_sync.verify = transport == "expert_block"
    if nccl_max_channels is not None:
        cfg.trainer.algorithm.batch_invariant = True
        cfg.trainer.policy.megatron_config.nccl_max_channels = nccl_max_channels
    initialize_ray(cfg)
    client = grug_engine_client(
        cfg,
        str(model_path),
        require_verified_placements=transport == "expert_block",
        moe_backend="triton" if transport == "expert_block" else None,
    )
    try:
        prompt_pattern = [[1, 17, 29, 5, 11, 3], [1, 19, 31, 7, 13, 3]]
        prompts = [prompt_pattern[idx % 2] for idx in range(cfg.trainer.train_batch_size)]
        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
        sampling_params.update({"temperature": 0.0, "ignore_eos": True, "logprobs": 1})
        first_rollout = asyncio.run(
            client.generate(InferenceEngineInput(prompt_token_ids=prompts, sampling_params=sampling_params))
        )
        first_token = first_rollout["response_ids"][0][0]
        score_params = dict(sampling_params)
        score_params.update({"max_tokens": 1, "prompt_logprobs": 1})
        score_input = InferenceEngineInput(prompt_token_ids=[prompts[0] + [first_token]], sampling_params=score_params)
        first_logprob = asyncio.run(client.generate(score_input))["prompt_logprobs"][0][-1][first_token]

        policy = _init_policy(cfg, policy_world_size)
        names = [*PARAMETER_NAMES, *BIAS_NAMES]
        before = rank0_validation_snapshot(policy, names)
        _train_step(policy, rollout_training_batch(prompts, first_rollout))
        training = rank0_validation_snapshot(policy, names)
        for name in PARAMETER_NAMES:
            assert not torch.equal(training[name], before[name]), f"{name} did not update"

        if transport == "expert_block":
            sync = ExpertBlockSync(policy_model=policy, inference_engine_client=client, timeout_seconds=600)
            asyncio.run(sync.prepare())
            asyncio.run(client.pause_generation())
            asyncio.run(sync.sync(0))
            asyncio.run(sync.verify(0))
            asyncio.run(sync.close())
            asyncio.run(client.resume_generation(policy_step=1))
        else:
            ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
            ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
        sync_names = [
            *BIAS_NAMES,
            *SERVING_EXPERT_INDEX_BY_NAME,
            LM_HEAD_NAME,
            ROUTER_NAME,
            ATTN_GATE_NAME,
            GATED_NORM_NAME,
        ]
        assert_engine_weights(client, sync_names, training, BIAS_NAMES, SERVING_EXPERT_INDEX_BY_NAME)

        asyncio.run(client.reset_prefix_cache())
        second_logprob = asyncio.run(client.generate(score_input))["prompt_logprobs"][0][-1][first_token]
        assert abs(second_logprob - first_logprob) > 1e-7
    finally:
        ray.shutdown()


@pytest.mark.vllm
@pytest.mark.parametrize(
    "moe_backend",
    ["triton"] + (["flashinfer_cutlass"] if os.environ.get("GRUG_CUTLASS_WEEKLY") == "1" else []),
)
def test_grug_megatron_two_gpu_colocated_sleep_sync_preserves_grouped_experts(tmp_path, moe_backend):
    """A sleep-level-2 CUDA-IPC sync preserves grouped experts and serving output."""

    world_size = 2
    require_hoppers(world_size)
    model_path = tmp_path / "model"
    model_path.mkdir()
    write_tiny_checkpoint(model_path)
    cfg = _config(str(model_path), world_size=world_size, pp=1, ep=2)
    cfg.trainer.placement.colocate_all = True
    if moe_backend == "flashinfer_cutlass":
        cfg.generator.engine_init_timeout_seconds = 4200
    # Force each completed tensor into its own transport chunk. Before grouped-export-safe
    # chunking, this threshold split the conversion tasks and silently omitted the experts.
    cfg.generator.weight_transfer_threshold_cuda_ipc_GB = 1e-9
    initialize_ray(cfg)
    shared_pg = placement_group([{"GPU": 1, "CPU": 1}] * world_size, strategy="PACK")
    ray.get(shared_pg.ready(), timeout=120)
    client = grug_engine_client(
        cfg,
        str(model_path),
        shared_pg=shared_pg,
        inference_engine_enable_sleep=True,
        moe_backend=moe_backend,
    )
    try:
        asyncio.run(client.wake_up())
        prompts = [[1, 17, 29, 5, 11, 3], [1, 19, 31, 7, 13, 3]]
        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
        sampling_params.update({"temperature": 0.0, "max_tokens": 1, "prompt_logprobs": 1})
        target_tokens = [47, 53]
        score_prompts = [prompt + [target] for prompt, target in zip(prompts, target_tokens, strict=True)]
        score_input = InferenceEngineInput(prompt_token_ids=score_prompts, sampling_params=sampling_params)

        def target_logprobs(result) -> torch.Tensor:
            return torch.tensor(
                [row[-1][target] for row, target in zip(result["prompt_logprobs"], target_tokens, strict=True)]
            )

        before_logprobs = target_logprobs(asyncio.run(client.generate(score_input)))

        policy = init_worker_with_type(
            "policy",
            shared_pg=shared_pg,
            colocate_all=True,
            num_gpus_per_node=world_size,
            num_nodes=1,
            cfg=cfg,
        )
        # Expert storage is reordered by FlashInfer-CUTLASS finalization, so raw
        # readback is not an HF-layout invariant. Serving parity below exercises the experts in
        # their actual kernel layout; keep bytewise readback for layout-neutral weights.
        sync_names = [LM_HEAD_NAME, ROUTER_NAME]
        training = rank0_validation_snapshot(policy, sync_names)

        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        ray.get(
            policy.async_run_ray_method("pass_through", "offload_to_cpu", offload_optimizer=True, offload_model=False)
        )
        asyncio.run(client.wake_up(tags=["weights"]))
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))

        assert_engine_weights(client, sync_names, training, [], {})

        ray.get(
            policy.async_run_ray_method("pass_through", "offload_to_cpu", offload_optimizer=False, offload_model=True)
        )
        asyncio.run(client.wake_up(tags=["kv_cache"]))
        after_logprobs = target_logprobs(asyncio.run(client.generate(score_input)))
        torch.testing.assert_close(
            after_logprobs,
            before_logprobs,
            rtol=0,
            atol=LOGPROB_MAX_ABS_TOLERANCE,
        )
    finally:
        ray.util.remove_placement_group(shared_pg)
        ray.shutdown()


@pytest.mark.vllm
def test_grug_megatron_four_gpu_colocated_expert_block_matches_fresh_load_after_two_updates(tmp_path):
    """Each sleep/wake update must match a fresh vLLM load on every EP rank."""
    require_hoppers(4)
    model_path = tmp_path / "model"
    model_path.mkdir()
    write_tiny_checkpoint(model_path)
    cfg = _config(str(model_path), world_size=4, pp=1, ep=4)
    cfg.trainer.placement.colocate_all = True
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.generator.inference_engine_data_parallel_size = 4
    cfg.generator.inference_engine_expert_parallel_size = 4
    cfg.generator.weight_sync_transport = "expert_block"
    cfg.generator.expert_block_sync.verify = True
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = [[1, 17, 29, 5, 11, 3], [1, 19, 31, 7, 13, 3]] * 2
    rollout_sampling = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
    rollout_sampling.update({"temperature": 0.0, "ignore_eos": True, "max_tokens": 4, "logprobs": 1})
    score_sampling = {**rollout_sampling, "max_tokens": 1, "prompt_logprobs": 1}
    target = 47
    score_input = InferenceEngineInput(prompt_token_ids=[prompts[0] + [target]], sampling_params=score_sampling)
    expert_names = [
        f"model.layers.0.mlp.experts.{expert}.{part}_proj.weight"
        for expert in range(TOY_SHAPE["num_local_experts"])
        for part in ("gate", "down")
    ]
    names = [LM_HEAD_NAME, ROUTER_NAME, *expert_names]
    observations = []

    def read_weights(engine_client):
        rows = {}
        for engine in engine_client.engines:
            result = ray.get(engine.inference_engine_actor.read_engine_weights.remote(names, False))
            for row in result if isinstance(result, list) else [result]:
                rank = int(row["__ranks__"]["ep_rank"])
                assert rank not in rows
                rows[rank] = row
        assert set(rows) == set(range(4))
        return rows

    def logprob(engine_client):
        result = asyncio.run(engine_client.generate(score_input))
        return float(result["prompt_logprobs"][0][-1][target])

    initialize_ray(cfg)
    shared_pg = placement_group([{"GPU": 1, "CPU": 1}] * 4, strategy="PACK")
    ray.get(shared_pg.ready(), timeout=120)
    client = grug_engine_client(
        cfg,
        str(model_path),
        shared_pg=shared_pg,
        inference_engine_enable_sleep=True,
        require_verified_placements=True,
        moe_backend="triton",
    )
    try:
        policy = init_worker_with_type("policy", shared_pg=shared_pg, colocate_all=True, num_gpus_per_node=4, cfg=cfg)
        sync = ExpertBlockSync(policy_model=policy, inference_engine_client=client, timeout_seconds=600, colocated=True)
        for step in range(3):
            if step:
                rollout = asyncio.run(
                    client.generate(InferenceEngineInput(prompt_token_ids=prompts, sampling_params=rollout_sampling))
                )
                asyncio.run(client.pause_generation())
                asyncio.run(client.sleep())
                policy.backload_to_gpu()
                batch = rollout_training_batch(prompts, rollout)
                batch.metadata["global_step"] = step
                _train_step(policy, batch)
            policy.offload_to_cpu(offload_optimizer=True, offload_model=False)
            asyncio.run(client.wake_up(tags=["weights"]))
            if step == 0:
                asyncio.run(client.pause_generation())
            asyncio.run(sync.prepare())
            asyncio.run(sync.sync(step))
            asyncio.run(sync.verify(step))
            asyncio.run(sync.close())
            if step:
                export_dir = tmp_path / f"checkpoint-{step}"
                ray.get(policy.async_run_ray_method("pass_through", "save_hf_model", str(export_dir), tokenizer))
                observations.append((export_dir, read_weights(client)))
            policy.offload_to_cpu(offload_optimizer=False, offload_model=True)
            asyncio.run(client.wake_up(tags=["kv_cache"]))
            asyncio.run(client.resume_generation(policy_step=step + 1))
            if step:
                observations[-1] += (logprob(client),)
        first, second = observations
        assert any(
            not torch.equal(first[1][rank][name]["tensor"], second[1][rank][name]["tensor"])
            for rank in range(4)
            for name in expert_names
            if first[1][rank][name].get("found") and second[1][rank][name].get("found")
        ), "No sampled expert changed across updates"
    finally:
        ray.shutdown()

    for export_dir, synced_weights, synced_logprob in observations:
        initialize_ray(cfg)
        try:
            fresh = grug_engine_client(cfg, str(export_dir), moe_backend="triton")
            reference_weights = read_weights(fresh)
            for rank in range(4):
                for name in names:
                    actual, expected = synced_weights[rank][name], reference_weights[rank][name]
                    assert actual.get("found") == expected.get("found"), (rank, name)
                    if actual.get("found"):
                        torch.testing.assert_close(actual["tensor"], expected["tensor"], rtol=0, atol=0)
            # Both sides use the same vLLM backend and BF16 weights. This only allows
            # small kernel-order variation, not the trainer-vLLM numerical gap.
            assert abs(synced_logprob - logprob(fresh)) < 2e-3
        finally:
            ray.shutdown()


def _idle_pinned_host_bytes(_worker) -> int:
    """Pinned host memory that PyTorch's caching allocator holds but no tensor uses."""
    stats = torch.cuda.host_memory_stats()
    return stats["allocated_bytes.current"] - stats["active_bytes.current"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_offloaded_optimizer_load_does_not_stage_moments_on_cuda(dtype):
    require_hoppers(1)
    torch.manual_seed(19)
    parameters = [torch.nn.Parameter(torch.randn(1024, 1024, device="cuda", dtype=dtype)) for _ in range(4)]
    optimizer = HybridDeviceOptimizer(
        parameters,
        offload_fraction=1.0,
        cpu_optimizer_cls=torch.optim.AdamW,
        gpu_optimizer_cls=torch.optim.AdamW,
        param_update_in_fp32=True,
        overlap_cpu_optimizer_d2h_h2d=False,
        lr=1e-3,
    )
    for parameter in parameters:
        parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    torch.cuda.synchronize()
    saved_weights = [parameter.detach().cpu().clone() for parameter in parameters]
    saved_state = copy.deepcopy(optimizer.state_dict())
    for parameter in parameters:
        parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    torch.cuda.synchronize()
    expected = [parameter.detach().cpu().clone() for parameter in parameters]

    with torch.no_grad():
        for parameter, saved in zip(parameters, saved_weights, strict=True):
            parameter.copy_(saved)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    optimizer.load_state_dict(saved_state)
    torch.cuda.synchronize()
    extra_allocated = torch.cuda.max_memory_allocated() - allocated
    # Rebuilding CPU parameter copies can clone one shard on CUDA, but loading
    # all moments on CUDA would require at least eight FP32 shards.
    assert extra_allocated <= 2 * parameters[0].numel() * torch.float32.itemsize

    optimizer.step()
    torch.cuda.synchronize()
    for parameter, continued in zip(parameters, expected, strict=True):
        torch.testing.assert_close(parameter.cpu(), continued, rtol=0, atol=0)


def _adam_checkpoint_snapshot(worker):
    optimizer = worker.optimizer
    children = optimizer.chained_optimizers if isinstance(optimizer, ChainedOptimizer) else [optimizer]
    states = []
    for child in children:
        for group in child.optimizer.param_groups:
            for parameter in group["params"]:
                state = child.optimizer.state[parameter]
                states.append(
                    {
                        "param": parameter.detach().cpu().clone(),
                        **{name: state[name].detach().cpu().clone() for name in ("step", "exp_avg", "exp_avg_sq")},
                    }
                )
    assert states
    return states


def _assert_adam_checkpoint_matches(actual, expected):
    for actual_rank, expected_rank in zip(actual, expected, strict=True):
        for actual_parameter, expected_parameter in zip(actual_rank, expected_rank, strict=True):
            for name in ("param", "step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(actual_parameter[name], expected_parameter[name], rtol=0, atol=0)


@pytest.mark.parametrize("world_size", [1, 2])
def test_grug_megatron_cpu_offloaded_optimizer_resumes_checkpoint(tmp_path, world_size):
    """A checkpoint saved with CPU optimizer offload loads, restores the weights, and continues.

    Training from the checkpoint again must repeat the step taken right after saving it, which
    holds only if the CPU-offloaded masters and moments were restored too. The load must also
    return the pinned host copies it replaced instead of leaving them cached.
    """
    require_hoppers(world_size)
    model_path = tmp_path / "model"
    model_path.mkdir()
    write_tiny_checkpoint(model_path)
    cfg = _config(str(model_path), world_size=world_size, pp=1, ep=1)
    megatron = cfg.trainer.policy.megatron_config
    megatron.optimizer_checkpoint_sharding_type = "dp_reshardable"
    megatron.optimizer_config_kwargs.optimizer_cpu_offload = True
    megatron.optimizer_config_kwargs.optimizer_offload_fraction = 1.0
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    batch = _padded_batch(tokenizer.pad_token_id)
    # Megatron restore reads only CoreWeave S3 checkpoints.
    checkpoint_dir = marin_temp_bucket(1, f"skyrl-gpu-tests/offload-resume/{uuid.uuid4().hex}")
    initialize_ray(cfg)
    try:
        policy = _init_policy(cfg, world_size)
        for _ in range(5):
            assert _train_step(policy, batch)["skipped_steps"] == 0
        saved = copy.deepcopy(rank0_validation_snapshot(policy, PARAMETER_NAMES))
        saved_optimizer = copy.deepcopy(
            ray.get([actor.__ray_call__.remote(_adam_checkpoint_snapshot) for actor in policy._actor_handlers])
        )
        ray.get(policy.async_run_ray_method("pass_through", "save_checkpoint", checkpoint_dir, tokenizer))
        assert _train_step(policy, batch)["skipped_steps"] == 0
        after_saved_step = copy.deepcopy(rank0_validation_snapshot(policy, PARAMETER_NAMES))
        after_saved_optimizer = copy.deepcopy(
            ray.get([actor.__ray_call__.remote(_adam_checkpoint_snapshot) for actor in policy._actor_handlers])
        )

        ray.shutdown()
        initialize_ray(cfg)
        policy = _init_policy(cfg, world_size)
        ray.get(policy.async_run_ray_method("pass_through", "load_checkpoint", checkpoint_dir))
        idle_pinned = ray.get([actor.__ray_call__.remote(_idle_pinned_host_bytes) for actor in policy._actor_handlers])
        restored = rank0_validation_snapshot(policy, PARAMETER_NAMES)
        restored_optimizer = ray.get(
            [actor.__ray_call__.remote(_adam_checkpoint_snapshot) for actor in policy._actor_handlers]
        )
        _assert_adam_checkpoint_matches(restored_optimizer, saved_optimizer)
        assert _train_step(policy, batch)["skipped_steps"] == 0
        after_restored_step = rank0_validation_snapshot(policy, PARAMETER_NAMES)
        after_restored_optimizer = ray.get(
            [actor.__ray_call__.remote(_adam_checkpoint_snapshot) for actor in policy._actor_handlers]
        )
        _assert_adam_checkpoint_matches(after_restored_optimizer, after_saved_optimizer)

        assert idle_pinned == [0] * world_size

        for name in PARAMETER_NAMES:
            torch.testing.assert_close(restored[name], saved[name], rtol=0, atol=0)
            torch.testing.assert_close(after_restored_step[name], after_saved_step[name], rtol=0, atol=0)
    finally:
        ray.shutdown()


@pytest.mark.parametrize("nesterov", [False, True])
def test_megatron_muonh_offloaded_direction_matches_cuda_reference(nesterov):
    require_hoppers(1)
    torch.manual_seed(11)
    shape = (1025, 4096)  # Crosses the 16 MiB transfer-chunk boundary.
    gradient = torch.randn(shape, device="cuda")
    momentum = torch.randn(shape, device="cuda")
    expected_momentum = momentum.clone().mul_(0.95).add_(gradient)
    expected_direction = gradient.clone()
    if nesterov:
        expected_direction.add_(expected_momentum, alpha=0.95)
    else:
        expected_direction.copy_(expected_momentum)

    actual_direction = gradient.clone()
    actual_momentum = momentum.cpu().pin_memory()
    _offloaded_muon_direction_in_grad_(actual_direction, actual_momentum, beta=0.95, nesterov=nesterov)

    torch.testing.assert_close(actual_direction, expected_direction, rtol=0, atol=0)
    torch.testing.assert_close(actual_momentum, expected_momentum.cpu(), rtol=0, atol=0)


def test_megatron_muonh_momentum_stays_on_cpu_after_rollout_backload_and_resume():
    require_hoppers(1)
    torch.manual_seed(17)
    parameters = [torch.nn.Parameter(torch.randn(4, 8, device="cuda")) for _ in range(2)]

    def new_optimizer(weights):
        return MegatronGrugMuonH(
            [
                {"params": [weights[0]], "optimizer": "grug_muonh"},
                {"params": [weights[1]], "optimizer": "grug_adamh"},
            ],
            lr=1e-3,
            betas=(0.9, 0.95),
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            eps=1e-8,
            muon_eps=1e-8,
            qkv_split_shapes=(4, 2, 2),
            offload_momentum=True,
        )

    optimizer = new_optimizer(parameters)
    for parameter in parameters:
        parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    saved_parameters = [parameter.detach().clone() for parameter in parameters]
    saved_state = copy.deepcopy(optimizer.state_dict())

    wrapped_optimizer = SimpleNamespace(optimizer=optimizer)
    offload_megatron_optimizer(wrapped_optimizer)
    load_megatron_optimizer(wrapped_optimizer)
    assert optimizer.state[parameters[0]]["momentum_buffer"].device.type == "cpu"
    assert optimizer.state[parameters[0]]["momentum_buffer"].is_pinned()
    assert optimizer.state[parameters[1]]["exp_avg"].device.type == "cuda"
    assert optimizer.state[parameters[1]]["exp_avg_sq"].device.type == "cuda"

    restored_parameters = [torch.nn.Parameter(parameter.clone()) for parameter in saved_parameters]
    restored_optimizer = new_optimizer(restored_parameters)
    restored_optimizer.initialize_state()
    restored_optimizer.load_state_dict(saved_state)
    assert restored_optimizer.state[restored_parameters[0]]["momentum_buffer"].device.type == "cpu"

    for parameter, restored in zip(parameters, restored_parameters, strict=True):
        gradient = torch.randn_like(parameter)
        parameter.grad = gradient.clone()
        restored.grad = gradient.clone()
    optimizer.step()
    restored_optimizer.step()
    for parameter, restored in zip(parameters, restored_parameters, strict=True):
        torch.testing.assert_close(parameter, restored, rtol=0, atol=0)
