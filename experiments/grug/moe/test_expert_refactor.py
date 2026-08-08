# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from levanter.grug.grug_moe import MoEExpertMlp
from levanter.grug.sharding import compact_grug_mesh
from levanter.utils.activation import ActivationFunctionEnum

from experiments.grug.moe.expert_merge import MoeLayerTrace
from experiments.grug.moe.expert_refactor import (
    ExpertRefactorBatch,
    ExpertRefactorParameters,
    RefactorSplit,
    expert_refactor_forward,
    expert_refactor_loss,
    expert_refactor_predictions,
    expert_refactor_step,
    initial_expert_refactor_state,
    make_expert_refactor_dataset,
    sample_expert_refactor_batch,
)
from experiments.grug.moe.model import GrugModelConfig, MoEMLP

_ROUTING_RENORM_SUM = 2.5


def _config() -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=32,
        hidden_dim=4,
        intermediate_dim=6,
        shared_expert_intermediate_dim=8,
        num_experts=5,
        num_experts_per_token=4,
        num_layers=2,
        expert_bank_for_layer=(0, 0),
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=8,
        sliding_window=4,
        moe_implementation="scatter",
    )


def _parameters(seed: int) -> ExpertRefactorParameters:
    config = _config()
    keys = jax.random.split(jax.random.key(seed), 9)
    bank = MoEExpertMlp(
        w_gate=jax.random.normal(keys[0], (5, 4, 6)) * 0.25,
        w_up=jax.random.normal(keys[1], (5, 4, 6)) * 0.25,
        w_down=jax.random.normal(keys[2], (5, 6, 4)) * 0.25,
        implementation="scatter",
        activation=ActivationFunctionEnum.silu,
        capacity_factor=1.0,
    )
    routers = tuple(
        MoEMLP(
            router=jax.random.normal(keys[3 + layer], (4, 5)) * 0.3,
            router_bias=jax.random.normal(keys[5 + layer], (5,)) * 0.05,
            cfg=config,
        )
        for layer in range(2)
    )
    return ExpertRefactorParameters(bank=bank, routers=routers)


def _batch(seed: int = 0) -> ExpertRefactorBatch:
    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(2, 8, 4)).astype(np.float32)
    targets = rng.normal(size=(2, 8, 4)).astype(np.float32)
    target_power = np.asarray(
        [np.mean(np.sum(np.square(targets[layer]), axis=-1)) for layer in range(2)],
        dtype=np.float32,
    )
    return ExpertRefactorBatch(
        inputs=jnp.asarray(inputs),
        targets=jnp.asarray(targets),
        layer_indices=jnp.asarray([2, 3], dtype=jnp.int32),
        target_power_by_layer=jnp.asarray(target_power),
    )


def _reference_predictions(parameters: ExpertRefactorParameters, batch: ExpertRefactorBatch) -> jax.Array:
    predictions = []
    for layer, router in enumerate(parameters.routers):
        inputs = batch.inputs[layer]
        logits = inputs @ router.router
        biased_logits = logits + jax.lax.stop_gradient(router.router_bias)
        _, selected = jax.lax.top_k(biased_logits, router.cfg.num_experts_per_token + 1)
        selected = selected[:, :-1]
        selected_logits = jnp.take_along_axis(logits, selected, axis=-1)
        combine = jax.nn.sigmoid(selected_logits)
        combine *= _ROUTING_RENORM_SUM / (jnp.sum(combine, axis=-1, keepdims=True) + 1e-9)

        gate = jnp.einsum("nd,nkdi->nki", inputs, parameters.bank.w_gate[selected])
        up = jnp.einsum("nd,nkdi->nki", inputs, parameters.bank.w_up[selected])
        hidden = jax.nn.silu(gate) * up
        expert_output = jnp.einsum("nki,nkid->nkd", hidden, parameters.bank.w_down[selected])
        predictions.append(jnp.sum(expert_output * combine[:, :, None], axis=1))
    return jnp.stack(predictions, axis=0)


def _reference_loss(parameters: ExpertRefactorParameters, batch: ExpertRefactorBatch) -> jax.Array:
    predictions = _reference_predictions(parameters, batch)
    squared_error = jnp.sum(jnp.square(predictions - batch.targets), axis=-1)
    normalized_mse = jnp.mean(squared_error, axis=1) / (batch.target_power_by_layer + 1e-8)
    return jnp.mean(normalized_mse)


def _reference_routing_health(
    parameters: ExpertRefactorParameters,
    batch: ExpertRefactorBatch,
) -> tuple[jax.Array, jax.Array]:
    counts_by_layer = []
    entropy_by_layer = []
    for layer, router in enumerate(parameters.routers):
        logits = batch.inputs[layer] @ router.router
        _, selected = jax.lax.top_k(
            logits + jax.lax.stop_gradient(router.router_bias),
            router.cfg.num_experts_per_token + 1,
        )
        selected = selected[:, :-1]
        counts = jnp.sum(jax.nn.one_hot(selected, router.cfg.num_experts, dtype=jnp.float32), axis=(0, 1))
        fractions = counts / jnp.sum(counts)
        counts_by_layer.append(counts)
        entropy_by_layer.append(-jnp.sum(fractions * jnp.log(fractions + 1e-6)))
    return jnp.stack(entropy_by_layer), jnp.stack(counts_by_layer)


def _array_leaves(tree) -> list[jax.Array]:
    return jax.tree.leaves(tree)


def _permuted(parameters: ExpertRefactorParameters, permutation: jax.Array) -> ExpertRefactorParameters:
    bank = dataclasses.replace(
        parameters.bank,
        w_gate=parameters.bank.w_gate[permutation],
        w_up=parameters.bank.w_up[permutation],
        w_down=parameters.bank.w_down[permutation],
    )
    routers = tuple(
        dataclasses.replace(
            router,
            router=router.router[:, permutation],
            router_bias=router.router_bias[permutation],
        )
        for router in parameters.routers
    )
    return ExpertRefactorParameters(bank=bank, routers=routers)


def test_refactor_loss_and_gradients_match_independent_top4_reference():
    parameters = _parameters(0)
    batch = _batch(1)
    mesh = compact_grug_mesh(expert_axis_size=1)

    with jax.set_mesh(mesh):
        forward = expert_refactor_forward(parameters, batch)
        actual_loss, actual_nrmse = expert_refactor_loss(parameters, batch)
        expected_loss = _reference_loss(parameters, batch)
        expected_entropy, expected_counts = _reference_routing_health(parameters, batch)
        actual_grads = jax.grad(lambda current: expert_refactor_loss(current, batch)[0])(parameters)
        expected_grads = jax.grad(lambda current: _reference_loss(current, batch))(parameters)

    np.testing.assert_allclose(forward.predictions, _reference_predictions(parameters, batch), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(forward.routing_entropy_by_layer, expected_entropy, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(forward.routing_counts_by_layer, expected_counts)
    np.testing.assert_array_equal(forward.capacity_overflow_by_layer, jnp.zeros((2,), dtype=jnp.float32))
    np.testing.assert_allclose(actual_loss, expected_loss, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(actual_loss, jnp.mean(jnp.square(actual_nrmse)), rtol=1e-6, atol=1e-7)
    for actual, expected in zip(_array_leaves(actual_grads), _array_leaves(expected_grads), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    assert all(float(jnp.linalg.norm(leaf)) > 0 for leaf in _array_leaves(actual_grads.bank))
    assert all(float(jnp.linalg.norm(router_grad.router)) > 0 for router_grad in actual_grads.routers)
    assert all(float(jnp.linalg.norm(router_grad.router_bias)) == 0 for router_grad in actual_grads.routers)


def test_teacher_routing_fields_cannot_affect_refactor_batches_loss_or_gradients():
    rng = np.random.default_rng(2)
    inputs = rng.normal(size=(16, 4)).astype(np.float32)
    targets = rng.normal(size=(16, 4)).astype(np.float32)
    first_trace = MoeLayerTrace(
        mlp_input=inputs,
        selected_experts=np.zeros((16, 1), dtype=np.int32),
        combine_weights=np.full((16, 1), np.nan, dtype=np.float32),
        routed_output=targets,
    )
    second_trace = MoeLayerTrace(
        mlp_input=inputs,
        selected_experts=np.full((16, 7), 10_000, dtype=np.int32),
        combine_weights=rng.normal(size=(16, 7)).astype(np.float32),
        routed_output=targets,
    )
    first_datasets = tuple(
        make_expert_refactor_dataset(first_trace, source_layer=layer, heldout_fraction=0.25, seed=10 + layer)
        for layer in (2, 3)
    )
    second_datasets = tuple(
        make_expert_refactor_dataset(second_trace, source_layer=layer, heldout_fraction=0.25, seed=10 + layer)
        for layer in (2, 3)
    )
    first_batch = sample_expert_refactor_batch(
        first_datasets,
        examples_per_layer=5,
        split=RefactorSplit.TRAIN,
        rng=np.random.default_rng(4),
    )
    second_batch = sample_expert_refactor_batch(
        second_datasets,
        examples_per_layer=5,
        split=RefactorSplit.TRAIN,
        rng=np.random.default_rng(4),
    )
    parameters = _parameters(3)
    mesh = compact_grug_mesh(expert_axis_size=1)

    for first, second in zip(first_datasets, second_datasets, strict=True):
        np.testing.assert_array_equal(first.train_inputs, second.train_inputs)
        np.testing.assert_array_equal(first.train_targets, second.train_targets)
        np.testing.assert_array_equal(first.heldout_inputs, second.heldout_inputs)
        np.testing.assert_array_equal(first.heldout_targets, second.heldout_targets)
    for first, second in zip(_array_leaves(first_batch), _array_leaves(second_batch), strict=True):
        np.testing.assert_array_equal(first, second)
    assert first_batch.inputs.shape == first_batch.targets.shape == (2, 5, 4)
    np.testing.assert_array_equal(first_batch.layer_indices, [2, 3])
    np.testing.assert_allclose(
        first_batch.target_power_by_layer,
        [dataset.train_target_power for dataset in first_datasets],
        rtol=1e-6,
        atol=1e-7,
    )
    with jax.set_mesh(mesh):
        first_loss, first_grads = jax.value_and_grad(lambda current: expert_refactor_loss(current, first_batch)[0])(
            parameters
        )
        second_loss, second_grads = jax.value_and_grad(lambda current: expert_refactor_loss(current, second_batch)[0])(
            parameters
        )
    np.testing.assert_array_equal(first_loss, second_loss)
    for first, second in zip(_array_leaves(first_grads), _array_leaves(second_grads), strict=True):
        np.testing.assert_array_equal(first, second)


def test_joint_expert_slot_permutation_preserves_predictions_loss_and_gradients():
    parameters = _parameters(4)
    permutation = jnp.asarray([2, 4, 1, 0, 3], dtype=jnp.int32)
    permuted_parameters = _permuted(parameters, permutation)
    batch = _batch(5)
    mesh = compact_grug_mesh(expert_axis_size=1)

    with jax.set_mesh(mesh):
        original_predictions = expert_refactor_predictions(parameters, batch)
        permuted_predictions = expert_refactor_predictions(permuted_parameters, batch)
        original_loss, original_grads = jax.value_and_grad(lambda current: expert_refactor_loss(current, batch)[0])(
            parameters
        )
        permuted_loss, permuted_grads = jax.value_and_grad(lambda current: expert_refactor_loss(current, batch)[0])(
            permuted_parameters
        )

    np.testing.assert_allclose(permuted_predictions, original_predictions, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(permuted_loss, original_loss, rtol=1e-6, atol=1e-7)
    expected_permuted_grads = _permuted(original_grads, permutation)
    for actual, expected in zip(_array_leaves(permuted_grads), _array_leaves(expected_permuted_grads), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_predictions_are_sensitive_only_to_the_router_for_the_corresponding_layer():
    parameters = _parameters(6)
    batch = _batch(7)
    first_router = parameters.routers[0]
    changed_bias = jnp.asarray([100.0, -100.0, 0.0, 0.0, 0.0], dtype=first_router.router_bias.dtype)
    changed = dataclasses.replace(
        parameters,
        routers=(dataclasses.replace(first_router, router_bias=changed_bias), parameters.routers[1]),
    )
    mesh = compact_grug_mesh(expert_axis_size=1)

    with jax.set_mesh(mesh):
        before = expert_refactor_predictions(parameters, batch)
        after = expert_refactor_predictions(changed, batch)

    before_array = np.asarray(before)
    after_array = np.asarray(after)
    assert not np.allclose(after_array[0], before_array[0])
    np.testing.assert_array_equal(after_array[1], before_array[1])


def test_jitted_update_changes_bank_and_both_router_projections_but_no_frozen_teacher_data():
    parameters = _parameters(8)
    batch = _batch(9)
    frozen_inputs = np.asarray(batch.inputs).copy()
    frozen_targets = np.asarray(batch.targets).copy()
    optimizer = optax.adam(1e-2)
    state = initial_expert_refactor_state(parameters.bank, parameters.routers, optimizer)
    mesh = compact_grug_mesh(expert_axis_size=1)

    with jax.set_mesh(mesh):
        updated, loss = jax.jit(expert_refactor_step, static_argnums=(2, 3))(state, batch, optimizer, 1e-8)

    assert jnp.isfinite(loss)
    assert int(updated.step) == 1
    assert all(
        not np.array_equal(before, after)
        for before, after in zip(
            _array_leaves(state.parameters.bank), _array_leaves(updated.parameters.bank), strict=True
        )
    )
    for before, after in zip(state.parameters.routers, updated.parameters.routers, strict=True):
        assert not np.array_equal(before.router, after.router)
        np.testing.assert_array_equal(before.router_bias, after.router_bias)
        assert before.cfg == after.cfg
    np.testing.assert_array_equal(batch.inputs, frozen_inputs)
    np.testing.assert_array_equal(batch.targets, frozen_targets)


def test_jitted_update_supports_a_data_sharded_mesh():
    script = """
import jax
import jax.numpy as jnp
import optax
from levanter.grug.sharding import compact_grug_mesh
from experiments.grug.moe.expert_refactor import expert_refactor_step, initial_expert_refactor_state
from experiments.grug.moe.test_expert_refactor import _batch, _parameters

parameters = _parameters(20)
batch = _batch(21)
optimizer = optax.adam(1e-3)
state = initial_expert_refactor_state(parameters.bank, parameters.routers, optimizer)
mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
with jax.set_mesh(mesh):
    updated, loss = jax.jit(expert_refactor_step, static_argnums=(2, 3))(state, batch, optimizer, 1e-8)

assert jax.device_count() == 4
assert bool(jnp.isfinite(loss))
assert int(updated.step) == 1
assert updated.parameters.bank.w_gate.sharding.mesh.shape["data"] == 4
"""
    environment = os.environ.copy()
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[3],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_joint_refactor_tiny_problem_overfits_complete_routed_outputs():
    teacher = _parameters(10)
    student = _parameters(11)
    rng = np.random.default_rng(12)
    inputs = rng.normal(size=(40, 4)).astype(np.float32)
    teacher_inputs = jnp.stack([jnp.asarray(inputs), jnp.asarray(inputs)], axis=0)
    placeholder = ExpertRefactorBatch(
        inputs=teacher_inputs,
        targets=jnp.zeros_like(teacher_inputs),
        layer_indices=jnp.asarray([2, 3], dtype=jnp.int32),
        target_power_by_layer=jnp.ones((2,), dtype=jnp.float32),
    )
    mesh = compact_grug_mesh(expert_axis_size=1)
    with jax.set_mesh(mesh):
        teacher_outputs = expert_refactor_predictions(teacher, placeholder)
    traces = tuple(
        MoeLayerTrace(
            mlp_input=inputs,
            selected_experts=np.full((40, 3), -1, dtype=np.int32),
            combine_weights=np.full((40, 3), np.nan, dtype=np.float32),
            routed_output=np.asarray(teacher_outputs[layer]),
        )
        for layer in range(2)
    )
    datasets = tuple(
        make_expert_refactor_dataset(trace, source_layer=layer + 2, heldout_fraction=0.2, seed=20 + layer)
        for layer, trace in enumerate(traces)
    )
    train_batch = sample_expert_refactor_batch(
        datasets,
        examples_per_layer=32,
        split=RefactorSplit.TRAIN,
        rng=np.random.default_rng(13),
    )
    optimizer = optax.adam(3e-3)
    state = initial_expert_refactor_state(student.bank, student.routers, optimizer)

    with jax.set_mesh(mesh):
        initial_loss = expert_refactor_loss(state.parameters, train_batch)[0]
        step = jax.jit(expert_refactor_step, static_argnums=(2, 3))
        for _ in range(60):
            state, _ = step(state, train_batch, optimizer, 1e-8)
        final_loss = expert_refactor_loss(state.parameters, train_batch)[0]

    assert float(final_loss) < 0.5 * float(initial_loss)
