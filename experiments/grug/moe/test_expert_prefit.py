# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax
import numpy as np
from levanter.grug.grug_moe import MoEExpertMlp
from levanter.utils.activation import ActivationFunctionEnum

from experiments.grug.moe.expert_merge import MoeLayerTrace
from experiments.grug.moe.expert_prefit import (
    PrefitConfig,
    PrefitDataset,
    PrefitSplit,
    aggregate_routed_moe_nrmse,
    make_prefit_dataset,
    prefit_loss,
    prefit_shared_bank,
    sample_prefit_batch,
)


def _bank(seed: int) -> MoEExpertMlp:
    key_gate, key_up, key_down = jax.random.split(jax.random.key(seed), 3)
    return MoEExpertMlp(
        w_gate=jax.random.normal(key_gate, (2, 3, 4)) * 0.2,
        w_up=jax.random.normal(key_up, (2, 3, 4)) * 0.2,
        w_down=jax.random.normal(key_down, (2, 4, 3)) * 0.2,
        implementation="scatter",
        activation=ActivationFunctionEnum.silu,
        capacity_factor=1.0,
    )


def _datasets() -> tuple[MoEExpertMlp, tuple[PrefitDataset, ...]]:
    representative = _bank(0)
    second_source = dataclasses.replace(
        representative,
        w_gate=representative.w_gate + 0.03,
        w_up=representative.w_up - 0.02,
        w_down=representative.w_down + 0.01,
    )
    rng = np.random.default_rng(1)
    datasets = []
    for source_layer, source_bank in ((2, representative), (3, second_source)):
        for expert in range(2):
            datasets.append(
                make_prefit_dataset(
                    source_bank,
                    source_layer=source_layer,
                    source_expert=expert,
                    shared_expert=expert,
                    train_inputs=rng.normal(size=(24, 3)).astype(np.float32),
                    heldout_inputs=rng.normal(size=(12, 3)).astype(np.float32),
                )
            )
    return representative, tuple(datasets)


def test_prefit_batch_balances_source_experts_and_layers():
    _, datasets = _datasets()
    batch = sample_prefit_batch(
        datasets,
        examples_per_source=3,
        split=PrefitSplit.TRAIN,
        rng=np.random.default_rng(2),
    )

    np.testing.assert_array_equal(np.bincount(np.asarray(batch.source_indices)), [3, 3, 3, 3])
    np.testing.assert_array_equal(np.bincount(np.asarray(batch.shared_experts)), [6, 6])
    assert batch.inputs.shape == batch.targets.shape == (12, 3)


def test_prefit_reduces_balanced_heldout_functional_error():
    initial_bank, datasets = _datasets()
    heldout = sample_prefit_batch(
        datasets,
        examples_per_source=12,
        split=PrefitSplit.HELDOUT,
        rng=np.random.default_rng(3),
    )
    initial_loss, _ = prefit_loss(initial_bank, heldout)

    result = prefit_shared_bank(
        initial_bank,
        datasets,
        config=PrefitConfig(
            learning_rate=3e-3,
            steps=80,
            examples_per_source=4,
            heldout_examples_per_source=12,
            eval_every=20,
            early_stopping_patience=5,
        ),
        seed=3,
    )
    final_loss, _ = prefit_loss(result.bank, heldout)

    assert float(final_loss) < 0.99 * float(initial_loss)
    assert result.evaluations[0].step == 0
    assert result.evaluations[-1].step <= 80


def test_aggregate_routed_moe_nrmse_matches_independent_topk_reference():
    teacher = _bank(4)
    shared = _bank(5)
    inputs = np.random.default_rng(6).normal(size=(5, 3)).astype(np.float32)
    selected = np.asarray([[0, 1], [1, 0], [0, 1], [1, 0], [1, 0]], dtype=np.int32)
    combine = np.asarray([[0.7, 0.3], [0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.55, 0.45]], dtype=np.float32)
    assignment = (1, 0)

    def explicit_expert(bank: MoEExpertMlp, expert: int, x: np.ndarray) -> np.ndarray:
        gate = x @ np.asarray(bank.w_gate[expert])
        up = x @ np.asarray(bank.w_up[expert])
        hidden = np.asarray(jax.nn.silu(gate)) * up
        return hidden @ np.asarray(bank.w_down[expert])

    teacher_output = np.zeros_like(inputs)
    shared_output = np.zeros_like(inputs)
    for token in range(inputs.shape[0]):
        for route in range(selected.shape[1]):
            source_expert = int(selected[token, route])
            weight = combine[token, route]
            teacher_output[token] += weight * explicit_expert(teacher, source_expert, inputs[token])
            shared_output[token] += weight * explicit_expert(shared, assignment[source_expert], inputs[token])
    expected = np.sqrt(np.sum(np.square(shared_output - teacher_output)) / np.sum(np.square(teacher_output)))
    trace = MoeLayerTrace(
        mlp_input=inputs,
        selected_experts=selected,
        combine_weights=combine,
        routed_output=teacher_output,
    )

    actual = aggregate_routed_moe_nrmse(shared, trace, assignment)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert not np.isclose(float(actual), float(aggregate_routed_moe_nrmse(shared, trace, (0, 1))))


def test_prefit_rejects_unbalanced_shared_clusters():
    initial_bank, datasets = _datasets()
    unbalanced = datasets[:-1]

    with np.testing.assert_raises(ValueError):
        prefit_shared_bank(initial_bank, unbalanced, config=PrefitConfig(steps=1))

    missing_expert = tuple(dataset for dataset in datasets if dataset.shared_expert == 0)
    with np.testing.assert_raises(ValueError):
        prefit_shared_bank(initial_bank, missing_expert, config=PrefitConfig(steps=1))
