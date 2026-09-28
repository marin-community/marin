# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track: the ragged all-to-all MoE transport on the d512 expert recipe (LatentMoE, ungated ReLU^2 experts on
the fused-activation path) matches the pooled-wave transport, and selecting it applies its XLA transport flags."""

import os
import subprocess
import sys
import textwrap

import pytest

from experiments.grug.fast_track.model import GrugModelConfig
from experiments.grug.fast_track.train import (
    RAGGED_TRANSPORT_XLA_FLAGS,
    RUNTIME_ENV,
    RaggedTransport,
    _apply_runtime_defaults,
)

# XLA:CPU has no ragged-all-to-all thunk; this all_gather emulation of the collective stands in for it.
_SCRIPT = """
import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from experiments.grug.fast_track.model import GrugModelConfig, MoEMLP


def emulated_ragged_all_to_all(
    operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, *, axis_name, axis_index_groups=None
):
    del recv_sizes, axis_index_groups
    me = jax.lax.axis_index(axis_name)
    operands = jax.lax.all_gather(operand, axis_name)
    input_offsets = jax.lax.all_gather(input_offsets, axis_name)
    send_sizes = jax.lax.all_gather(send_sizes, axis_name)
    output_offsets = jax.lax.all_gather(output_offsets, axis_name)
    num_senders, updates = input_offsets.shape
    destination = jnp.arange(updates) // (updates // num_senders)
    rows = jnp.arange(output.shape[0])[:, None, None]
    hit = (destination == me) & (rows >= output_offsets) & (rows < output_offsets + send_sizes)
    source_rows = (input_offsets + rows - output_offsets).reshape(output.shape[0], -1)
    update = jnp.argmax(hit.reshape(output.shape[0], -1), axis=1)
    source_row = jnp.take_along_axis(source_rows, update[:, None], axis=1)[:, 0]
    values = operands[update // updates, jnp.clip(source_row, 0, operand.shape[0] - 1)]
    return jnp.where(hit.any(axis=(1, 2))[:, None], values.astype(output.dtype), output)


jax.lax.ragged_all_to_all = emulated_ragged_all_to_all

mesh = Mesh(
    np.asarray(jax.devices()).reshape(1, 2, 4, 1),
    ("replica_dcn", "data", "expert", "model"),
    axis_types=(AxisType.Explicit,) * 4,
)
base = GrugModelConfig(
    hidden_dim=32,
    intermediate_dim=12,
    shared_expert_intermediate_dim=8,
    num_experts=16,
    num_experts_per_token=4,
    latent_dim=16,
    capacity_factor=8.0,
    pooled_transport_capacity_factor=8.0,
    moe_ungated_relu2=True,
    moe_ungated_kernel=True,
    moe_fused_relu2=True,
)
x = jax.random.normal(jax.random.key(1), (8, 16, base.hidden_dim), jnp.float32)
cotangent = jax.random.normal(jax.random.key(2), x.shape, jnp.float32)


def loss_and_grads(implementation):
    cfg = dataclasses.replace(base, moe_implementation=implementation)
    with jax.set_mesh(mesh):
        mlp = MoEMLP.init(cfg, key=jax.random.key(0))
        assert mlp.expert_mlp.implementation == implementation
        assert mlp.expert_mlp.w_gate is None
        batch = jax.device_put(x, NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None, None)))
        ct = jax.device_put(cotangent, NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None, None)))

        def loss(mlp, x):
            out, stats = mlp(x)
            return jnp.sum(out * ct), stats

        (value, stats), grads = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))(mlp, batch)
    assert float(stats["capacity_overflow"]) == 0.0
    return value, eqx.filter(grads, eqx.is_array)


ragged_loss, ragged_grads = loss_and_grads("ragged_all_to_all")
pooled_loss, pooled_grads = loss_and_grads("fixed_pooled_wave_all_to_all")
np.testing.assert_allclose(ragged_loss, pooled_loss, rtol=1e-5)
flat = jax.tree_util.tree_flatten_with_path(ragged_grads)[0]
reference = jax.tree_util.tree_leaves(pooled_grads)
assert len(flat) == len(reference)
for (path, grad), ref in zip(flat, reference, strict=True):
    scale = float(jnp.max(jnp.abs(ref))) + 1e-12
    assert float(jnp.max(jnp.abs(grad - ref))) / scale < 1e-5, jax.tree_util.keystr(path)
"""


def test_ragged_moe_block_matches_pooled_wave_on_ungated_relu2_latent_experts():
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_SCRIPT)], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_unknown_moe_implementation_is_rejected():
    with pytest.raises(ValueError, match="moe_implementation"):
        GrugModelConfig(moe_implementation="ragged")


@pytest.fixture
def runtime_env(monkeypatch):
    """Pre-set the runtime env so `_apply_runtime_defaults` leaves nothing behind after the test."""
    for name, value in RUNTIME_ENV.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


@pytest.mark.parametrize("transport", list(RaggedTransport))
def test_ragged_transport_flags_replace_conflicting_kernel_flags(transport: RaggedTransport, runtime_env):
    monkeypatch = runtime_env
    stray = "--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=maybe"
    monkeypatch.setenv("XLA_FLAGS", f"--xla_dump_to=/tmp/hlo {stray}")
    _apply_runtime_defaults(inline_watch_enabled=False, ragged_transport=transport)
    flags = os.environ["XLA_FLAGS"].split()
    assert stray not in flags
    assert "--xla_dump_to=/tmp/hlo" in flags
    assert set(RAGGED_TRANSPORT_XLA_FLAGS[transport]) <= set(flags)
    assert "--xla_gpu_experimental_parallel_collective_overlap_limit=1" in flags


def test_pooled_wave_leaves_ragged_kernel_flags_alone(runtime_env):
    runtime_env.setenv("XLA_FLAGS", "")
    _apply_runtime_defaults(inline_watch_enabled=False, ragged_transport=None)
    assert "ragged_all_to_all" not in os.environ["XLA_FLAGS"]
