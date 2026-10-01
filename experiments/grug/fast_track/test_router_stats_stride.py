# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``router_stats_stride``: the logging-only router sums on a token subsample, scaled back up."""

import jax
import numpy as np
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

import experiments.grug.fast_track.test_kda_local as t
from experiments.grug.fast_track.router_metrics import local_routing_stats

_TOKENS, _EXPERTS, _K = 64, 8, 2


def _stats(stride: int):
    mesh = t._mesh()
    keys = jax.random.split(jax.random.PRNGKey(0))
    selected = jax.random.randint(keys[0], (_TOKENS, _K), 0, _EXPERTS)
    logits = jax.random.normal(keys[1], (_TOKENS, _EXPERTS))
    axes = ("replica_dcn", "data", "expert")
    with jax.set_mesh(mesh):
        selected, logits = (reshard(x, P(axes, None)) for x in (selected, logits))
        out = local_routing_stats(
            selected,
            logits,
            mesh,
            num_experts=_EXPERTS,
            batch_axes=("replica_dcn", "data", "expert"),
            token_stride=stride,
        )
    return {k: np.asarray(v).sum(axis=0) for k, v in out.items()}


def test_strided_sums_keep_their_totals():
    full, strided = _stats(1), _stats(4)
    # Every token makes K assignments and its probabilities sum to 1, sampled or not.
    for stats in (full, strided):
        assert stats["routing_counts_local"].sum() == _TOKENS * _K
        np.testing.assert_allclose(stats["router_prob_sum_local"].sum(), _TOKENS, rtol=1e-5)
    # The sampled per-expert probability mass tracks the full one.
    np.testing.assert_allclose(strided["router_prob_sum_local"], full["router_prob_sum_local"], rtol=0.6)
