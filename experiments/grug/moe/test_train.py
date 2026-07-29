# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp

from experiments.grug.moe.train import _drop_metrics


def test_drop_metrics_sum_layer_counts_without_int32_overflow():
    batch_size = 2048
    sequence_length = 4096
    top_k = 8
    num_layers = 48
    dropped_assignments = 2_576_980_368
    total_assignments = batch_size * sequence_length * top_k * num_layers
    per_layer, remainder = divmod(dropped_assignments, num_layers)
    dropped_assignments_per_layer = jnp.asarray(
        [per_layer + 1] * remainder + [per_layer] * (num_layers - remainder),
        dtype=jnp.int32,
    )

    metrics = _drop_metrics(
        dropped_assignments_per_layer,
        batch_size=batch_size,
        sequence_length=sequence_length,
        top_k=top_k,
        num_layers=num_layers,
    )

    assert metrics["moe/dropped_assignments"] == dropped_assignments
    assert metrics["moe/drop_fraction"] == dropped_assignments / total_assignments
