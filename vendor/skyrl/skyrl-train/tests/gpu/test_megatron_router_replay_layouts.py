"""Parallel-layout replay tests for Megatron MoE router replay (R3).

Each layout attacks one layout assumption: TP2 exercises the
sequence-parallel slice, PP2 the 1F1B recompute FIFO and the layer-number
mapping across pipeline stages, EP2 the alltoall dispatch, and packing on/off
the two target transforms. CP2 needs the separate dense-CP runtime and is
excluded from this mainline matrix. The oracle is behavioral and needs no in-actor hooks: a
completed training step proves token-exact replay on every rank (the
per-rank hit-fraction check and the FIFO drain assert turn a wrong layout
into a loud failure), an empty capture must reproduce native log-probs
exactly, a real capture must move them only on captured samples, and
perturbed-replay log-probs must agree across layouts at the bf16
tolerance the Grug HF-parity tests already accept.

Virtual pipelining is not exercised here (the Grug bridge cannot build
VPP models); per-chunk layer arming is covered on CPU.

Requires Hopper GPUs; run on an otherwise idle node (not part of the CPU
PR gate). Layouts whose world size exceeds the node are skipped.
"""

from __future__ import annotations

import math
import copy
from types import SimpleNamespace

import numpy as np
from finestore.rl import mismatch_probe as mismatch
from finestore.reader import ReadView
from omegaconf import open_dict
from safetensors.torch import load_file

import pytest
import ray
import torch
from transformers import AutoTokenizer

from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.mismatch_probe.collect import BatchLayout, ProbeCollector
from skyrl_train.mismatch_probe.archive import MismatchArchive
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import initialize_ray
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.router_replay_fixtures import random_unique_routes
from tests.gpu.grug_tiny_checkpoint import TOY_SHAPE, write_tiny_checkpoint
from tests.gpu.test_grug_megatron import (
    RESPONSE_LENGTH,
    _config,
    _padded_batch,
    _train_step,
)
from tests.gpu.utils import get_available_gpus, init_worker_with_type

NUM_EXPERTS = TOY_SHAPE["num_local_experts"]
NUM_LAYERS = TOY_SHAPE["num_hidden_layers"]
TOPK = 2
LOGPROB_MAX_ABS_TOLERANCE = 2e-1
LOGPROB_MEAN_ABS_TOLERANCE = 3e-2

# id, world_size, tp, pp, ep, cp, sample packing
LAYOUTS = [
    ("tp1", 1, 1, 1, 1, 1, False),
    ("tp1_packed", 1, 1, 1, 1, 1, True),
    ("tp2", 2, 2, 1, 1, 1, False),
    ("pp2", 2, 1, 2, 1, 1, False),
    ("ep2", 2, 1, 1, 2, 1, False),
    ("tp2_pp2", 4, 2, 2, 1, 1, False),
]


def _layout_config(tmp_path, layout) -> tuple:
    _, world_size, tp, pp, ep, cp, packing = layout
    model_path = tmp_path / "model"
    model_path.mkdir(parents=True)
    write_tiny_checkpoint(model_path, num_experts_per_tok=TOPK, vocab_size_multiple=2)
    cfg = _config(str(model_path), world_size=world_size, pp=pp, ep=ep)
    cfg.trainer.use_sample_packing = packing
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
    cfg.trainer.policy.megatron_config.context_parallel_size = cp
    cfg.trainer.policy.megatron_config.moe_router_replay = True
    return cfg, model_path


def _routed_batch(pad_token_id: int, *, captured: bool, variable_lengths: bool = False) -> TrainingInputBatch:
    batch = _padded_batch(
        pad_token_id,
        response_length=2 * RESPONSE_LENGTH if variable_lengths else RESPONSE_LENGTH,
        variable_lengths=variable_lengths,
    )
    generator = torch.Generator().manual_seed(31)
    shape = (batch["sequences"].shape[0], batch.metadata["response_length"], NUM_LAYERS, TOPK)
    if not captured:
        routes = torch.zeros(shape, dtype=torch.long)  # empty capture: everything routes natively
    else:
        routes = random_unique_routes(shape, NUM_EXPERTS, generator=generator)
        routes[1] = 0  # one sample with fully-lost capture routes natively
    batch["rollout_routed_experts"] = routes.to(torch.int32)
    return batch


def _response_logprobs(policy, batch: TrainingInputBatch) -> torch.Tensor:
    outputs = ray.get(policy.async_run_ray_method("mesh", "forward", data=batch))
    return concatenate_outputs_after_mesh_dispatch(policy.actor_infos, outputs)["output"].float()


@pytest.mark.parametrize("packing", [False, True], ids=["unpacked", "packed"])
def test_all_placeholder_replay_matches_flag_off_at_same_weights(tmp_path, packing):
    """The native probe control must be identical to the route-disabled model."""
    require_hoppers(1)
    cfg_on, model_path = _layout_config(tmp_path, ("placeholder", 1, 1, 1, 1, 1, packing))
    cfg_off = copy.deepcopy(cfg_on)
    cfg_off.trainer.policy.megatron_config.moe_router_replay = False
    pad_token_id = AutoTokenizer.from_pretrained(model_path).pad_token_id
    empty = _routed_batch(pad_token_id, captured=False)
    dense_routes = empty.pop("rollout_routed_experts")
    empty.routed_expert_rows = RoutedExpertRows(
        tuple(row[: int(mask.sum())].numpy() for row, mask in zip(dense_routes, empty["response_mask"], strict=True)),
        empty.metadata["response_length"],
        NUM_EXPERTS,
    )
    without_routes = empty.select(["sequences", "attention_mask"], ["response_length"])
    scores = []
    for cfg, batch in ((cfg_off, without_routes), (cfg_on, empty)):
        initialize_ray(cfg)
        try:
            policy = init_worker_with_type(
                "policy", shared_pg=None, colocate_all=False, num_gpus_per_node=1, num_nodes=1, cfg=cfg
            )
            scores.append(_response_logprobs(policy, batch))
        finally:
            ray.shutdown()
    assert torch.equal(scores[0], scores[1]), f"placeholder control differs with packing={packing}"


@pytest.mark.parametrize(
    "layout",
    [
        ("probe-pp2", 2, 1, 2, 1, 1, False),
        ("probe-tp2", 2, 2, 1, 1, 1, False),
        ("probe-dp2-ragged", 2, 1, 1, 1, 1, False),
    ],
    ids=["pp2", "tp2-sp", "dp2-ragged"],
)
def test_probe_forward_scores_all_modes_and_records_pipeline_routes(tmp_path, layout):
    require_hoppers(layout[1])
    cfg, model_path = _layout_config(tmp_path, layout)
    cfg.trainer.mismatch_probe.archive_uri = str(tmp_path / "probe")
    cfg.trainer.mismatch_probe.extra_trainer_modes = ["router_replay", "router_replay_filtered"]
    cfg.trainer.mismatch_probe.filtered_replay.keep_fraction = 1.0
    cfg.trainer.mismatch_probe.enabled = True
    cfg.trainer.max_steps = 2
    with open_dict(cfg.trainer.policy.optimizer_config):
        cfg.trainer.policy.optimizer_config.lr_warmup_steps_ratio = 0.1
    schedule = RayPPOTrainer.__new__(RayPPOTrainer)
    schedule.cfg = cfg
    schedule.train_dataset = range(100 * cfg.trainer.train_batch_size)
    schedule._configure_training_schedule()
    pad_token_id = AutoTokenizer.from_pretrained(model_path).pad_token_id
    batch = _routed_batch(pad_token_id, captured=True, variable_lengths=True)
    if layout[0] == "probe-dp2-ragged":
        original_metadata = batch.metadata
        batch = TrainingInputBatch({key: value[:3] for key, value in batch.items()})
        batch.metadata = original_metadata
    width = batch.metadata["response_length"]
    prompt_width = batch["sequences"].shape[1] - width
    probes = []
    for position in range(batch.batch_size):
        length = int(batch["response_mask"][position].sum())
        batch["rollout_routed_experts"][position, length:] = 0
        routes = batch["rollout_routed_experts"][position, :length].numpy()
        response = batch["sequences"][position, prompt_width : prompt_width + length].tolist()
        prompt = batch["sequences"][position, :prompt_width][
            batch["attention_mask"][position, :prompt_width].bool()
        ].tolist()
        probes.append(
            mismatch.ProbeRow(
                probe_hash="fixture",
                sample_id=f"sample-{position}",
                prompt_id=f"prompt-{position}",
                prompt_token_ids=prompt,
                trainer_prompt_ids=prompt,
                vllm_output_ids=response,
                trainer_input_ids=response,
                response_mask=[True] * length,
                loss_mask=[True] * length,
                request_seed=31 + position,
                batch_position=position,
                routed_experts=routes.tobytes(),
                routed_experts_shape=list(routes.shape),
                routed_experts_dtype=str(routes.dtype),
                route_valid_mask=(routes != 0).any(-1).tolist(),
            )
        )
    dense_routes = batch.pop("rollout_routed_experts")
    batch.routed_expert_rows = RoutedExpertRows(
        tuple(row[: int(mask.sum())].numpy() for row, mask in zip(dense_routes, batch["response_mask"], strict=True)),
        width,
        NUM_EXPERTS,
    )
    batch.metadata["uids"] = [row.prompt_id for row in probes]
    initialize_ray(cfg)
    try:
        policy = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=layout[1],
            num_nodes=1,
            cfg=cfg,
            num_training_steps=schedule.total_training_steps,
        )
        trainer = SimpleNamespace(policy_model=policy, critic_model=None, ref_model=None, global_step=0)
        padded = RayPPOTrainer.pad_batch(trainer, batch)
        collector = ProbeCollector(cfg)
        collector.probes = probes
        collector.probe_hash = "fixture"
        collector.training_input = padded
        collector.batch_layout = BatchLayout(
            sample_ids=[row.sample_id for row in probes],
            native_order=list(range(len(probes))),
            repeat_order=list(reversed(range(len(probes)))),
            padded_rows=padded.metadata["pad_size"],
            native_micro_batch_size=cfg.trainer.micro_forward_batch_size_per_gpu,
            repeat_micro_batch_size=4,
            collated_bytes=sum(value.numel() * value.element_size() for value in padded.values()),
            collated_route_bytes=padded.routed_experts.nbytes,
            nonzero_advantage_samples=int((padded["advantages"].abs().sum(dim=1) > 0).sum()),
        )
        scores = collector._trainer_scores(trainer, 0)
        archive = MismatchArchive(collector.archive_uri, writer_id="gpu-owner")
        try:
            archive.write(probes=probes, scores=scores)
        finally:
            archive.close()
        view = ReadView(collector.archive_uri)
        stored = [mismatch.ScoreRow.model_validate(row) for row in view.scan(mismatch.SCORES_TABLE).to_pylist()]
        for score in stored:
            probe = probes[int(score.sample_id.split("-")[1])]
            assert len(score.logprobs) == len(probe.vllm_output_ids)
            assert all(math.isfinite(value) for value in score.logprobs)
            choices = np.frombuffer(score.expert_choices, dtype=score.expert_choices_dtype).reshape(
                score.expert_choices_shape
            )
            valid = np.asarray(probe.route_valid_mask, dtype=bool)
            assert np.all(choices[valid] >= 0)
            if score.mode == "router_replay":
                captured = np.frombuffer(probe.routed_experts, dtype=probe.routed_experts_dtype).reshape(
                    probe.routed_experts_shape
                )
                np.testing.assert_array_equal(choices[valid], captured[valid])
        by_mode = {
            mode: [row for row in stored if row.mode == mode]
            for mode in ("native", "repeat", "router_replay", "router_replay_filtered")
        }
        assert any(a.logprobs != b.logprobs for a, b in zip(by_mode["native"], by_mode["router_replay"], strict=True))
        assert any(np.frombuffer(row.replacement_mask, dtype=bool).any() for row in by_mode["router_replay_filtered"])
        before_update, after_update = tmp_path / "before-update", tmp_path / "after-update"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        ray.get(policy.async_run_ray_method("pass_through", "save_hf_model", str(before_update), tokenizer))
        _train_step(policy, padded)
        ray.get(policy.async_run_ray_method("pass_through", "save_hf_model", str(after_update), tokenizer))
        before_parameters = {
            name: tensor
            for shard in before_update.glob("*.safetensors")
            for name, tensor in load_file(shard).items()
            if name.endswith(".weight")
        }
        after_parameters = {
            name: tensor
            for shard in after_update.glob("*.safetensors")
            for name, tensor in load_file(shard).items()
            if name.endswith(".weight")
        }
        assert before_parameters and before_parameters.keys() == after_parameters.keys()
        assert any(not torch.equal(tensor, after_parameters[name]) for name, tensor in before_parameters.items())
    finally:
        ray.shutdown()


def test_router_replay_is_token_exact_across_layouts(tmp_path):
    max_gpus = len(get_available_gpus())
    if max_gpus < 1:
        pytest.skip("no GPUs available")
    layouts = [layout for layout in LAYOUTS if layout[1] <= max_gpus]
    require_hoppers(1)
    # The cross-layout parity anchor must run.
    assert layouts[0][0] == "tp1"

    replayed_by_layout: dict[str, torch.Tensor] = {}
    for layout in layouts:
        layout_id = layout[0]
        cfg, model_path = _layout_config(tmp_path / layout_id, layout)
        pad_token_id = AutoTokenizer.from_pretrained(model_path).pad_token_id
        initialize_ray(cfg)
        try:
            policy = init_worker_with_type(
                "policy", shared_pg=None, colocate_all=False, num_gpus_per_node=layout[1], num_nodes=1, cfg=cfg
            )

            empty = _routed_batch(pad_token_id, captured=False)
            native = _response_logprobs(policy, empty)
            repeated = _response_logprobs(policy, empty)
            captured = _routed_batch(pad_token_id, captured=True)
            replayed = _response_logprobs(policy, captured)

            # An empty capture reproduces native routing exactly and repeats.
            assert torch.equal(native, repeated), layout_id
            sample_diff = (replayed - native).abs().amax(dim=1)
            assert (sample_diff[torch.tensor([0, 2, 3])] > 1e-4).all(), (layout_id, sample_diff)
            assert sample_diff[1].item() == 0.0, (layout_id, sample_diff)

            # The training step replays on every rank and recomputes in FIFO
            # order; any layout bug fails inside the step before the assert.
            status = _train_step(policy, captured)
            assert status["router_replay/hit_fraction"] == 1.0, (layout_id, status)
            assert math.isfinite(status["policy_loss"]), (layout_id, status)
        finally:
            ray.shutdown()
        replayed_by_layout[layout_id] = replayed

    baseline = replayed_by_layout["tp1"]
    for layout_id, replayed in replayed_by_layout.items():
        diff = (replayed - baseline).abs()
        print(
            f"{layout_id} vs tp1 replayed log-probs: max abs {diff.max().item():.4f}, mean abs {diff.mean().item():.4f}"
        )
        assert diff.max().item() < LOGPROB_MAX_ABS_TOLERANCE, (layout_id, diff.max())
        assert diff.mean().item() < LOGPROB_MEAN_ABS_TOLERANCE, (layout_id, diff.mean())
