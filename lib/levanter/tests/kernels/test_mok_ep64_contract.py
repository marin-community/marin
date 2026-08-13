# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest

from levanter.kernels.mixture_of_kittens.api import _failure_agreement_axes
from levanter.kernels.mixture_of_kittens.config import (
    MokLikeBackwardPeerStorage,
    MokLikeConfig,
    MokLikeForwardXStorage,
    MokLikeTopology,
)


def test_ep64_config_requires_one_staged_workspace_slot():
    config = MokLikeConfig(topology=MokLikeTopology.NVLINK_EP64, workspace_slots=1)

    assert config.topology.expert_axis_size == 64

    with pytest.raises(ValueError, match="exactly one workspace slot"):
        MokLikeConfig(topology=MokLikeTopology.NVLINK_EP64, workspace_slots=2)
    with pytest.raises(ValueError, match="runtime-staged forward"):
        MokLikeConfig(
            topology=MokLikeTopology.NVLINK_EP64,
            workspace_slots=1,
            forward_x_storage=MokLikeForwardXStorage.XLA_PEER_EXPERIMENTAL,
        )
    with pytest.raises(ValueError, match="runtime-staged backward"):
        MokLikeConfig(
            topology=MokLikeTopology.NVLINK_EP64,
            workspace_slots=1,
            backward_peer_storage=MokLikeBackwardPeerStorage.XLA_PEER_EXPERIMENTAL,
        )


def test_ep64_failure_status_agrees_over_the_full_mesh():
    mesh = jax.sharding.AbstractMesh(
        (1, 1, 64, 1),
        ("replica_dcn", "data", "expert", "model"),
    )

    agreement_axes = _failure_agreement_axes(mesh, MokLikeTopology.NVLINK_EP64)
    jaxpr = jax.make_jaxpr(
        lambda status: jax.lax.pmax(status, agreement_axes),
        axis_env=tuple(zip(mesh.axis_names, mesh.axis_sizes, strict=True)),
    )(jnp.asarray(0, dtype=jnp.int32))

    assert agreement_axes == mesh.axis_names
    pmax_equation = next(equation for equation in jaxpr.jaxpr.eqns if equation.primitive.name == "pmax")
    assert pmax_equation.params["axes"] == mesh.axis_names


def test_ep4_failure_status_still_excludes_process_local_expert_axis():
    mesh = jax.sharding.AbstractMesh(
        (8, 4, 4, 2),
        ("replica_dcn", "data", "expert", "model"),
    )

    assert _failure_agreement_axes(mesh, MokLikeTopology.LOCAL_EP4) == ("replica_dcn", "data", "model")
