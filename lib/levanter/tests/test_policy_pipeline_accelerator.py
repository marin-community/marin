# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit GPU acceptance for compiled policy updates and checkpoint continuation.

Run with the root pipeline extra on four GPUs, one pytest worker, and the
manual marker enabled. The default unit-test selection excludes this module.
"""

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
from iris.jax.init import initialize_jax

from levanter.checkpoint import discover_latest_checkpoint, save_checkpoint
from levanter.grpo import grpo_objective_weights
from levanter.mpmd_checkpoint import checkpoint_arrays, restore_checkpoint
from levanter.policy import ClippedPolicyConfig
from levanter.policy_model import PolicyBatch
from levanter.policy_pipeline import (
    BATCH_AXES,
    ROUTING_METRIC_NAMES,
    PipelinePolicyConfig,
    PipelineState,
    initialize_pipeline_state,
    make_pipeline_mesh,
    make_pipeline_policy_learner,
    make_pipeline_train_step,
    pipeline_batch,
    prepare_pipeline_train,
)
from levanter.testing.policy_reference import full_softmax_logprobs

logger = logging.getLogger(__name__)
pytestmark = [pytest.mark.manual, pytest.mark.timeout(600)]

_VOCAB = 32
_HIDDEN = 16
_MAX_GRAD_NORM = 0.01
_OBJECTIVE = ClippedPolicyConfig(0.2, 0.2, 0.1)


@pytest.fixture(scope="session", autouse=True)
def initialized_accelerator_cluster():
    initialize_jax()


class _Stage(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    embedding: jax.Array | None
    head: jax.Array | None

    def embed(self, tokens):
        assert self.embedding is not None
        embedding = jax.sharding.reshard(self.embedding, P())
        return embedding.at[tokens].get(out_sharding=P(BATCH_AXES, None, None))

    def run_blocks(self, hidden, segment_ids, position_ids):
        weight = jax.sharding.reshard(self.weight, P())
        bias = jax.sharding.reshard(self.bias, P())
        return jnp.tanh(hidden @ weight + bias + position_ids[..., None] * 0.01)

    def run_blocks_with_stats(self, hidden, segment_ids, position_ids):
        hidden = self.run_blocks(hidden, segment_ids, position_ids)
        counters = {
            name: jnp.asarray(-1 if name == "routing_drop_layer" else 0, dtype=jnp.int32)
            for name in ROUTING_METRIC_NAMES
        }
        return hidden, counters

    def finish(self, hidden):
        return hidden

    def get_lm_head(self):
        assert self.head is not None
        return jax.sharding.reshard(self.head, P())

    def trainable_filter(self):
        return eqx.is_inexact_array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _LearnerCheckpoint:
    state: PipelineState
    batch: PolicyBatch
    completed_steps: jax.Array


def _stages(mesh):
    rng = np.random.default_rng(31)

    def matrix(shape):
        return jax.device_put(rng.normal(0, 0.125, shape).astype(np.float32), NamedSharding(mesh, P("data", None)))

    return tuple(
        _Stage(
            matrix((_HIDDEN, _HIDDEN)),
            jax.device_put(rng.normal(0, 0.02, _HIDDEN).astype(np.float32), NamedSharding(mesh, P("data"))),
            matrix((_VOCAB, _HIDDEN)) if index == 0 else None,
            matrix((_HIDDEN, _VOCAB)) if index == 1 else None,
        )
        for index in range(2)
    )


def _batch(mesh):
    rows = 8 * jax.device_count()
    Batch, Position, Response = hax.Axis("batch", rows), hax.Axis("position", 5), hax.Axis("response", 2)
    tokens = np.random.default_rng(19).integers(0, _VOCAB, (rows, Position.size), dtype=np.int32)
    mask = np.resize(np.array([[1, 1], [1, 0], [0, 0], [1, 1]], dtype=np.float32), (rows, 2))
    # Four microbatches include an entirely empty final contribution.
    mask[-rows // 4 :] = 0

    def named(value, axes):
        return hax.named(jax.device_put(value, NamedSharding(mesh, P(BATCH_AXES, None))), axes)

    response_mask = named(mask, (Batch, Response))
    policy_weights, kl_weights = grpo_objective_weights(response_mask, Batch=Batch, Position=Response)
    return PolicyBatch(
        tokens=named(tokens, (Batch, Position)),
        attention_mask=named(np.ones_like(tokens), (Batch, Position)),
        position_ids=named(np.broadcast_to(np.arange(5, dtype=np.int32), tokens.shape), (Batch, Position)),
        response_mask=response_mask,
        advantages=named(
            mask * np.resize(np.array([1.3, -0.7, 0.2], dtype=np.float32), rows)[:, None], (Batch, Response)
        ),
        policy_weights=policy_weights,
        kl_weights=kl_weights,
        old_logprobs=named(np.full_like(mask, -9), (Batch, Response)),
        reference_logprobs=named(np.full_like(mask, -3.5), (Batch, Response)),
    )


def _host_leaves(tree):
    return [np.asarray(value).copy() for value in jax.tree.leaves(checkpoint_arrays(tree))]


def _assert_close(actual, expected):
    errors = []
    absolute_sum, count = 0.0, 0
    actual_arrays, expected_arrays = _host_leaves(actual), _host_leaves(expected)
    assert len(actual_arrays) == len(expected_arrays)
    for observed, wanted in zip(actual_arrays, expected_arrays, strict=True):
        if np.issubdtype(wanted.dtype, np.integer):
            np.testing.assert_array_equal(observed, wanted)
        else:
            np.testing.assert_allclose(observed, wanted, atol=1e-7, rtol=1e-5)
            difference = np.abs(observed.astype(np.float64) - wanted.astype(np.float64))
            errors.append(float(np.max(difference)))
            absolute_sum += float(difference.sum())
            count += difference.size
    return max(errors, default=0.0), absolute_sum / max(count, 1)


def _reference_scores(params, frozen, batch, policy):
    stages = eqx.combine(policy.cast_to_compute(params), frozen)
    hidden = stages[0].embed(batch.tokens.array)
    for stage in stages:
        hidden = stage.run_blocks(hidden, batch.attention_mask.array, batch.position_ids.array)
    logits = hidden[:, -3:-1].astype(jnp.float32) @ stages[-1].get_lm_head().astype(jnp.float32)
    return jnp.take_along_axis(full_softmax_logprobs(logits), batch.tokens.array[:, -2:, None], axis=-1)[..., 0]


class _ReferenceUpdate(NamedTuple):
    state: PipelineState
    loss: jax.Array
    grad_norm: jax.Array


def _reference_step(optimizer, policy, microbatches):
    @jax.jit
    def update(state, batch):
        def loss(params, part):
            scores = _reference_scores(params, state.frozen, part, policy)
            advantages, pw, kw = (getattr(part, name).array for name in ("advantages", "policy_weights", "kl_weights"))
            ratio = jnp.exp(jnp.clip(scores - part.old_logprobs.array, -20, 20))
            surrogate = -jnp.minimum(ratio * advantages, jnp.clip(ratio, 0.8, 1.2) * advantages)
            assert part.reference_logprobs is not None
            delta = jnp.clip(part.reference_logprobs.array - scores, -20, 20)
            kl = jnp.clip(jnp.exp(delta) - delta - 1, -10, 10)
            return jnp.sum(surrogate * pw) + _OBJECTIVE.kl_loss_coef * jnp.sum(kl * kw)

        rows = batch.tokens.axis_size("batch") // microbatches
        gradients = jax.tree.map(jnp.zeros_like, state.trainable)
        objective = jnp.asarray(0.0)
        for index in range(microbatches):

            def slice_rows(value):
                axes = tuple(hax.Axis("batch", rows) if axis.name == "batch" else axis for axis in value.axes)
                array = value.array.at[index * rows : (index + 1) * rows].get(out_sharding=P(BATCH_AXES, None))
                return hax.named(array, axes)

            part = jax.tree.map(slice_rows, batch, is_leaf=lambda value: isinstance(value, hax.NamedArray))
            value, contribution = jax.value_and_grad(loss)(state.trainable, part)
            objective += value
            gradients = jax.tree.map(jnp.add, gradients, contribution)
        norm = optax.global_norm(gradients)
        scale = jnp.minimum(1.0, _MAX_GRAD_NORM / jnp.maximum(norm, _MAX_GRAD_NORM))
        next_params, next_opt = [], []
        for params, opt_state, gradient in zip(state.trainable, state.opt_state, gradients, strict=True):
            updates, opt_state = optimizer.update(
                jax.tree.map(lambda value: value * scale, gradient), opt_state, params
            )
            next_params.append(eqx.apply_updates(params, updates))
            next_opt.append(opt_state)
        return _ReferenceUpdate(PipelineState(tuple(next_params), state.frozen, tuple(next_opt)), objective, norm)

    return update


@pytest.mark.parametrize("compute_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("microbatches", [2, 4])
def test_compiled_pipeline_updates_and_checkpoint_match_spmd(
    compute_dtype, microbatches, tmp_path: Path, record_property
):
    assert jax.default_backend() == "gpu", "Run this acceptance test with the root pipeline extra on GPUs"
    assert (
        jax.process_count() == 1 and jax.device_count() == 4
    ), "This bounded acceptance requires one four-GPU process"
    config = PipelinePolicyConfig(stages=2, microbatches=microbatches, max_grad_norm=_MAX_GRAD_NORM)
    mesh, mpmd_mesh = make_pipeline_mesh(config)
    reference_mesh = Mesh(
        np.asarray(jax.devices()).reshape(1, 4, 1, 1),
        (*BATCH_AXES, "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    policy = jmp.get_policy(f"params=float32,compute={compute_dtype},output=float32")
    optimizer = optax.adamw(3e-3, b1=0.9, b2=0.95, weight_decay=0.05)
    with jax.default_matmul_precision("float32"):
        with jax.set_mesh(reference_mesh):
            reference_stages = _stages(reference_mesh)
            reference_state = PipelineState(
                reference_stages,
                jax.tree.map(lambda _: None, reference_stages),
                tuple(optimizer.init(stage) for stage in reference_stages),
            )
            reference_batch = _batch(reference_mesh)
            scores = jax.jit(_reference_scores, static_argnums=(3,))(
                reference_state.trainable, reference_state.frozen, reference_batch, policy
            )
        with jax.set_mesh(mesh):
            state = initialize_pipeline_state(_stages(mesh), optimizer, mpmd_mesh)
            batch = _batch(mesh)
            learner = make_pipeline_policy_learner(
                optimizer, policy, state, batch, config=config, objective=_OBJECTIVE, mpmd_mesh=mpmd_mesh
            )
            scored = learner.score(state, batch)
            frozen_scores = np.asarray(scored.old_logprobs.array).copy()
            np.testing.assert_allclose(frozen_scores, scores, atol=1e-7, rtol=1e-5)
        with jax.set_mesh(reference_mesh):
            reference_batch = dataclasses.replace(
                reference_batch,
                old_logprobs=hax.named(
                    jax.device_put(frozen_scores, NamedSharding(reference_mesh, P(BATCH_AXES, None))),
                    reference_batch.old_logprobs.axes,
                ),
            )
            reference_update = _reference_step(optimizer, policy, microbatches)
            expected_first, expected_loss, expected_norm = reference_update(reference_state, reference_batch)
            if compute_dtype == "float32":
                full_first, full_loss, full_norm = _reference_step(optimizer, policy, 1)(
                    reference_state, reference_batch
                )
                _assert_close(expected_first, full_first)
                np.testing.assert_allclose(expected_loss, full_loss, atol=1e-7, rtol=1e-5)
                np.testing.assert_allclose(expected_norm, full_norm, atol=1e-7, rtol=1e-5)
        # The compiled primitive exposes the pre-clip norm in addition to state.
        with jax.set_mesh(mesh):
            probe_state = initialize_pipeline_state(_stages(mesh), optimizer, mpmd_mesh)
            probe_batch = _batch(mesh)
            probe_batch = dataclasses.replace(
                probe_batch,
                old_logprobs=hax.named(
                    jax.device_put(frozen_scores, NamedSharding(mesh, P(BATCH_AXES, None))),
                    probe_batch.old_logprobs.axes,
                ),
            )
            batches = pipeline_batch(probe_batch, microbatches=microbatches)
            train_step = make_pipeline_train_step(
                optimizer, policy, probe_state, batches, config=config, objective=_OBJECTIVE, mpmd_mesh=mpmd_mesh
            )
            prepared = prepare_pipeline_train(train_step, probe_state, batches, mpmd_mesh)
            probe_state, metrics = prepared.update(prepared.state, prepared.batches)
        actual_norm = float(checkpoint_arrays(metrics["grad_norm"]))
        np.testing.assert_allclose(actual_norm, expected_norm, atol=1e-7, rtol=1e-5)
        assert float(expected_norm) > _MAX_GRAD_NORM
        _assert_close(probe_state, expected_first)
        # The public learner must retain its caller's snapshot across updates.
        with jax.set_mesh(mesh):
            first = learner.update(state, scored)
        np.testing.assert_allclose(first.loss, expected_loss, atol=1e-7, rtol=1e-5)
        _assert_close(first.state, expected_first)
        np.testing.assert_array_equal(scored.old_logprobs.array, frozen_scores)
        payload = _LearnerCheckpoint(
            first.state, scored, jax.device_put(np.asarray(1, np.int32), NamedSharding(mesh, P()))
        )
        path = str(tmp_path / "step-1")
        save_checkpoint(checkpoint_arrays(payload), 1, path, metadata={"completed_steps": 1})
        candidate = discover_latest_checkpoint(str(tmp_path))
        assert candidate is not None and candidate.step == 1
        shardings = jax.tree.map(lambda value: value.sharding, payload)
        before = _host_leaves(payload)
        with jax.set_mesh(mesh):
            exemplar = _LearnerCheckpoint(
                initialize_pipeline_state(_stages(mesh), optimizer, mpmd_mesh),
                _batch(mesh),
                jax.device_put(np.asarray(0, np.int32), NamedSharding(mesh, P())),
            )
            restored = restore_checkpoint(exemplar, path, shardings)
        for observed, wanted in zip(_host_leaves(restored), before, strict=True):
            np.testing.assert_array_equal(observed, wanted)
        with jax.set_mesh(reference_mesh):
            current_scores = jax.jit(_reference_scores, static_argnums=(3,))(
                expected_first.trainable, expected_first.frozen, reference_batch, policy
            )
            active = np.asarray(reference_batch.policy_weights.array) > 0
            assert np.max(np.abs(np.exp(np.asarray(current_scores)[active] - frozen_scores[active]) - 1)) > 1e-4
            expected_second, expected_loss, _ = reference_update(expected_first, reference_batch)
        with jax.set_mesh(mesh):
            second = learner.update(first.state, scored)
            resumed = learner.update(restored.state, restored.batch)
        np.testing.assert_allclose(second.loss, expected_loss, atol=1e-7, rtol=1e-5)
        np.testing.assert_array_equal(resumed.loss, second.loss)
        maximum_error, mean_error = _assert_close(second.state, expected_second)
        for observed, wanted in zip(_host_leaves(resumed.state), _host_leaves(second.state), strict=True):
            np.testing.assert_array_equal(observed, wanted)
        np.testing.assert_array_equal(scored.old_logprobs.array, frozen_scores)
        np.testing.assert_array_equal(restored.batch.old_logprobs.array, frozen_scores)
        assert any(
            np.any(observed != initial)
            for observed, initial in zip(
                _host_leaves(second.state.trainable), _host_leaves(reference_stages), strict=True
            )
        )
        record_property("compute_dtype", compute_dtype)
        record_property("microbatches", microbatches)
        record_property("device_count", jax.device_count())
        record_property("process_count", jax.process_count())
        record_property("preclip_norm", actual_norm)
        record_property("reference_preclip_norm", float(expected_norm))
        record_property("maximum_state_absolute_error", maximum_error)
        record_property("mean_state_absolute_error", mean_error)
        logger.info(
            "policy acceptance dtype=%s microbatches=%d preclip_norm=%s maximum_state_absolute_error=%s mean_state_absolute_error=%s checkpoint=%s",
            compute_dtype,
            microbatches,
            float(expected_norm),
            maximum_error,
            mean_error,
            path,
        )
