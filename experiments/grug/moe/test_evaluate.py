# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import json
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.checkpoint import save_checkpoint
from levanter.data.text import GrugLmExample
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.distributed import DistributedConfig
from levanter.tracker import NoopConfig

from experiments.datasets.mrcr import MrcrPromptVariant
from experiments.grug.moe.evaluate import (
    GrugCheckpointEvalConfig,
    GrugCheckpointEvalRuntimeConfig,
    MrcrConditionLoss,
    MrcrExampleLoss,
    _canonical_67b_model,
    _gather_flattened,
    _loss_sums,
    derive_mrcr_metrics,
    evaluate_grug_checkpoint,
    load_grug_checkpoint_params,
    pair_mrcr_condition_losses,
    persist_grug_checkpoint_eval,
    summarize_per_example_losses,
    validate_grug_checkpoint_eval_config,
)
from experiments.grug.moe.model import Transformer


class _TrackerInitialized(Exception):
    pass


class _ShapeExemplarObserved(Exception):
    pass


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _LossModel:
    weight: jax.Array

    def next_token_loss(self, tokens, loss_weight, **_kwargs):
        return self.weight * jnp.ones_like(loss_weight)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _TrainingCheckpoint:
    step: jax.Array
    params: jax.Array
    opt_state: jax.Array
    ema_params: jax.Array
    pending_qb_betas: jax.Array


def _example(
    source_id: str,
    *,
    scored_tokens: int,
    full_loss_sum: float,
    query_loss_sum: float,
    n_needles: int = 2,
    distance_band: str = "distance_le_32768",
) -> MrcrExampleLoss:
    return MrcrExampleLoss(
        source_id=source_id,
        prompt_variant="two_shot",
        context_cap=65_536,
        n_needles=n_needles,
        distance_band=distance_band,
        evidence_distance_tokens=100,
        scored_tokens=scored_tokens,
        full_context_loss_sum=full_loss_sum,
        query_only_loss_sum=query_loss_sum,
        full_context_scored_bytes=scored_tokens * 2,
        query_only_scored_bytes=scored_tokens * 2,
    )


def _config(tmp_path, *, component_names: tuple[str, ...] | None = None) -> GrugCheckpointEvalConfig:
    model = _canonical_67b_model()
    cell = "two_shot/cap_65536/2needle/distance_le_32768"
    if component_names is None:
        component_names = (f"{cell}/full_context", f"{cell}/query_only")
    data = LmDataConfig(
        tokenizer="passthrough",
        vocab_size=model.vocab_size,
        components={name: DatasetComponent(cache_dir=str(tmp_path / name)) for name in component_names},
    )
    return GrugCheckpointEvalConfig(
        run_id="test-eval",
        checkpoint_path=str(tmp_path / "checkpoint"),
        context_cap=65_536,
        prompt_variant=MrcrPromptVariant.TWO_SHOT,
        qk_mult=1.57,
        model=model,
        data=data,
        dataset_stats_path=str(tmp_path / "stats.json"),
        dataset_manifest_paths={cell: str(tmp_path / "manifest.jsonl")},
        runtime=GrugCheckpointEvalRuntimeConfig(
            mp="f32",
            tracker=NoopConfig(),
            eval_batch_size=1,
            data_axis_size=1,
            context_axis_size=1,
        ),
        output_path=str(tmp_path / "output"),
        bootstrap_samples=100,
    )


def test_pair_mrcr_condition_losses_requires_complete_equal_pairs():
    common = dict(
        source_id="a",
        prompt_variant="two_shot",
        context_cap=65_536,
        n_needles=2,
        distance_band="distance_le_32768",
        evidence_distance_tokens=100,
        scored_tokens=3,
        scored_bytes=6,
    )
    full = MrcrConditionLoss(condition="full_context", loss_sum=3.0, **common)
    query = MrcrConditionLoss(condition="query_only", loss_sum=6.0, **common)
    paired = pair_mrcr_condition_losses((query, full))
    assert paired[0].source_id == "a"
    assert paired[0].context_gain_nll == 1.0

    with pytest.raises(ValueError, match="missing pair"):
        pair_mrcr_condition_losses((full,))
    with pytest.raises(ValueError, match="unequal scored-token"):
        pair_mrcr_condition_losses((full, dataclasses.replace(query, scored_tokens=2)))
    with pytest.raises(ValueError, match="duplicate condition"):
        pair_mrcr_condition_losses((full, full, query))


def test_derive_mrcr_metrics_computes_micro_macro_and_reproducible_paired_intervals():
    rows = (
        _example("a", scored_tokens=1, full_loss_sum=1.0, query_loss_sum=3.0),
        _example("b", scored_tokens=3, full_loss_sum=3.0, query_loss_sum=3.0),
        _example("c", scored_tokens=2, full_loss_sum=4.0, query_loss_sum=2.0),
    )
    first = derive_mrcr_metrics(rows, bootstrap_samples=2_000, bootstrap_seed=17)
    second = derive_mrcr_metrics(rows, bootstrap_samples=2_000, bootstrap_seed=17)
    prefix = "eval/mrcr/two_shot/cap_65536/2needle/distance_le_32768"

    assert first == second
    assert first[f"{prefix}/micro_context_gain_nll"] == pytest.approx(0.0)
    assert first[f"{prefix}/macro_context_gain_nll"] == pytest.approx(1.0 / 3.0)
    assert first[f"{prefix}/micro_context_ppl_ratio"] == pytest.approx(1.0)
    assert first[f"{prefix}/micro_context_gain_nll_ci95_low"] <= 0
    assert first[f"{prefix}/micro_context_gain_nll_ci95_high"] >= 0
    assert first[f"{prefix}/macro_context_gain_nll_ci95_low"] <= 1.0 / 3.0
    assert first[f"{prefix}/macro_context_gain_nll_ci95_high"] >= 1.0 / 3.0


def test_summarize_per_example_losses_scores_only_response_body_tokens():
    # Prompt, supplied nonce prefix, and EOS occupy positions 0, 1, and 4.
    per_position_loss = jnp.asarray([[100.0, 100.0, 2.0, 3.0, 100.0]])
    loss_weight = jnp.asarray([[0.0, 0.0, 1.0, 1.0, 0.0]])
    bytes_per_position = jnp.asarray([[8, 8, 4, 6, 8]])

    loss_sum, token_count, byte_count = summarize_per_example_losses(per_position_loss, loss_weight, bytes_per_position)

    assert loss_sum.tolist() == [5.0]
    assert token_count.tolist() == [2.0]
    assert byte_count.tolist() == [10.0]


def test_loss_sums_treats_model_parameters_as_dynamic_arguments():
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape((1, 1, 1)),
        ("replica_dcn", "data", "expert"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
    example = GrugLmExample.causal(jnp.asarray([0, 1, 2], dtype=jnp.int32))
    batch = dataclasses.replace(example, tokens=example.tokens[None, :], loss_weight=example.loss_weight[None, :])
    byte_lengths = jnp.asarray([1, 2, 3], dtype=jnp.int32)
    compiled_loss_sums = jax.jit(_loss_sums, static_argnames="sharding")

    def total_loss(weight):
        model = cast(Transformer, _LossModel(weight))
        loss_sum, _, _ = compiled_loss_sums(model, batch, byte_lengths, sharding=sharding)
        return jnp.sum(loss_sum)

    assert jax.grad(total_loss)(jnp.asarray(2.0)) == pytest.approx(2.0)


def test_gather_flattened_uses_tiled_global_array_semantics(monkeypatch):
    def process_allgather(_values, *, tiled):
        assert tiled is True
        return jnp.asarray([[3, 1], [4, 2]], dtype=jnp.int32)

    monkeypatch.setattr("experiments.grug.moe.evaluate.multihost_utils.process_allgather", process_allgather)

    gathered = _gather_flattened(jnp.asarray([0], dtype=jnp.int32))

    assert gathered.tolist() == [3, 1, 4, 2]


def test_validate_grug_checkpoint_eval_config_rejects_static_and_dataset_mismatches(tmp_path):
    config = _config(tmp_path)
    validate_grug_checkpoint_eval_config(config)

    with pytest.raises(ValueError, match="context_cap"):
        validate_grug_checkpoint_eval_config(dataclasses.replace(config, context_cap=8192))
    with pytest.raises(ValueError, match="qk_mult"):
        validate_grug_checkpoint_eval_config(dataclasses.replace(config, qk_mult=1.75))
    changed_model = dataclasses.replace(config.model, sliding_window=4096)
    with pytest.raises(ValueError, match="sliding_window"):
        validate_grug_checkpoint_eval_config(dataclasses.replace(config, model=changed_model))

    wrong_variant = _config(
        tmp_path,
        component_names=(
            "one_shot/cap_65536/2needle/distance_le_32768/full_context",
            "one_shot/cap_65536/2needle/distance_le_32768/query_only",
        ),
    )
    with pytest.raises(ValueError, match="different cap or prompt variant"):
        validate_grug_checkpoint_eval_config(wrong_variant)

    incomplete = _config(
        tmp_path,
        component_names=("two_shot/cap_65536/2needle/distance_le_32768/full_context",),
    )
    with pytest.raises(ValueError, match="incomplete"):
        validate_grug_checkpoint_eval_config(incomplete)


def test_load_grug_checkpoint_params_restores_only_step_and_params(tmp_path):
    checkpoint_path = tmp_path / "checkpoint"
    save_checkpoint(
        _TrainingCheckpoint(
            step=jnp.asarray(157_000, dtype=jnp.int32),
            params=jnp.asarray([1.0, 2.0]),
            opt_state=jnp.asarray([99.0]),
            ema_params=jnp.asarray([88.0]),
            pending_qb_betas=jnp.asarray([77.0]),
        ),
        step=157_000,
        checkpoint_path=checkpoint_path,
        is_temporary=False,
    )
    step, params = load_grug_checkpoint_params(str(checkpoint_path), initialized_params=jnp.zeros(2), mesh=None)

    assert step == 157_000
    assert params.tolist() == [1.0, 2.0]


def test_evaluate_grug_checkpoint_initializes_distributed_before_tracker(monkeypatch, tmp_path):
    events = []

    def initialize_distributed(_self):
        events.append("distributed")

    def initialize_tracker(_self, _run_id):
        events.append("tracker")
        raise _TrackerInitialized

    monkeypatch.setattr(DistributedConfig, "initialize", initialize_distributed)
    monkeypatch.setattr(NoopConfig, "init", initialize_tracker)

    with pytest.raises(_TrackerInitialized):
        evaluate_grug_checkpoint(_config(tmp_path))

    assert events == ["distributed", "tracker"]


def test_evaluate_grug_checkpoint_restores_with_shape_only_param_exemplar(monkeypatch, tmp_path):
    def initialize_distributed(_self):
        pass

    def initialize_model(_config, *, key):
        del key
        return {"weight": jnp.ones((2, 3), dtype=jnp.float32)}

    def load_checkpoint_with_shape_exemplar(initialized, _checkpoint_path, **_kwargs):
        param_leaves = jax.tree.leaves(initialized["params"])
        assert param_leaves
        assert all(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in param_leaves)
        assert not any(isinstance(leaf, jax.Array) for leaf in param_leaves)
        raise _ShapeExemplarObserved

    monkeypatch.setattr(DistributedConfig, "initialize", initialize_distributed)
    monkeypatch.setattr(Transformer, "init", initialize_model)
    monkeypatch.setattr("experiments.grug.moe.evaluate.load_checkpoint", load_checkpoint_with_shape_exemplar)

    with pytest.raises(_ShapeExemplarObserved):
        evaluate_grug_checkpoint(_config(tmp_path))


def test_persist_grug_checkpoint_eval_is_idempotent_and_rejects_conflicts(monkeypatch, tmp_path):
    monkeypatch.setenv("GIT_COMMIT", "abc123")
    config = _config(tmp_path)
    row = _example("a", scored_tokens=2, full_loss_sum=4.0, query_loss_sum=6.0)
    metrics = derive_mrcr_metrics((row,), bootstrap_samples=10, bootstrap_seed=0)

    persist_grug_checkpoint_eval(config, checkpoint_step=157_000, paired_rows=(row,), metrics=metrics)
    first_metrics = (tmp_path / "output" / "eval_metrics.jsonl").read_bytes()
    first_examples = (tmp_path / "output" / "mrcr_example_losses.jsonl").read_bytes()
    persist_grug_checkpoint_eval(config, checkpoint_step=157_000, paired_rows=(row,), metrics=metrics)

    assert (tmp_path / "output" / "eval_metrics.jsonl").read_bytes() == first_metrics
    assert (tmp_path / "output" / "mrcr_example_losses.jsonl").read_bytes() == first_examples
    record = json.loads(first_metrics)
    assert record["checkpoint_step"] == 157_000
    assert record["evaluator_commit"] == "abc123"
    assert record["metrics"] == metrics

    with pytest.raises(ValueError, match="conflicting output"):
        persist_grug_checkpoint_eval(
            config,
            checkpoint_step=157_000,
            paired_rows=(dataclasses.replace(row, query_only_loss_sum=7.0),),
            metrics=metrics,
        )


def test_persist_grug_checkpoint_eval_requires_propagated_commit(monkeypatch, tmp_path):
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    config = _config(tmp_path)
    row = _example("a", scored_tokens=2, full_loss_sum=4.0, query_loss_sum=6.0)

    with pytest.raises(KeyError, match="GIT_COMMIT"):
        persist_grug_checkpoint_eval(config, checkpoint_step=157_000, paired_rows=(row,), metrics={})

    assert not (tmp_path / "output").exists()
