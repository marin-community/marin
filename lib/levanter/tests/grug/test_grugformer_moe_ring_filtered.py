# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Ring-filtered expert-parallel MoE against ring and a dense reference, with sender and receiver drops."""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax.nn.ragged_dot import ragged_dot
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
from levanter.grug._moe import ep_ring, ep_ring_filtered
from levanter.grug._moe.ep_ring_filtered import destination_hit_fraction
from levanter.grug.grug_moe import moe_mlp
from levanter.testing.devices import use_test_devices


@pytest.mark.timeout(240)
@pytest.mark.parametrize(
    ("transport_capacity_factor", "capacity_factor", "padding"),
    [
        (8.0, 8.0, "none"),
        (0.5, 8.0, "none"),
        (8.0, 0.75, "none"),
        (0.5, 0.3, "uneven"),
        (0.5, 0.3, "all"),
    ],
    ids=["no_drops", "sender_drops", "receiver_drops", "both_drops_uneven_padding", "all_padding"],
)
@pytest.mark.multidevice
def test_ring_filtered_matches_ring_and_dense_with_drops(
    transport_capacity_factor: float, capacity_factor: float, padding: str
):
    """On 4 expert shards, ring_filtered matches a dense reference that applies its sender capacity and then ring's
    receiver capacity, and matches ring whenever no sender drops. Gradients include the combine weights, which
    carry the router gradient. The uneven padding leaves shard 0 with no valid tokens."""
    with use_test_devices(4) as devices, jax.default_matmul_precision("highest"):
        shards, tokens_per_shard, hidden, inter, experts, topk = 4, 48, 8, 6, 32, 4

        tokens = shards * tokens_per_shard
        local_experts = experts // shards
        mesh = Mesh(
            np.asarray(devices).reshape(1, shards, 1),
            axis_names=("data", "expert", "model"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )
        keys = jax.random.split(jax.random.key(0), 7)
        x = jax.random.normal(keys[0], (tokens, hidden))
        selected = jnp.argsort(jax.random.uniform(keys[1], (tokens, experts)), axis=1)[:, :topk].astype(jnp.int32)
        combine_weights = jax.nn.softmax(jax.random.normal(keys[2], (tokens, topk)), axis=-1)
        w_up_gate = jax.random.normal(keys[3], (experts, hidden, 2 * inter)) / math.sqrt(hidden)
        w_down = jax.random.normal(keys[4], (experts, inter, hidden)) / math.sqrt(inter)
        cotangent = jax.random.normal(keys[5], (tokens, hidden))
        if padding == "uneven":
            # Shard s keeps about s / 3 of its tokens.
            keep_fraction = np.repeat(np.arange(shards) / (shards - 1), tokens_per_shard)
            token_valid = np.asarray(jax.random.uniform(keys[6], (tokens,))) < keep_fraction
        else:
            token_valid = np.full(tokens, padding == "none")

        # Sender capacity: per (source, destination) shard pair, the first C valid tokens that select an expert
        # on the destination keep their assignments there, with C scaled by the source's valid tokens.
        sel_np = np.asarray(selected)
        keep = np.repeat(token_valid[:, None], topk, axis=1)
        hit_fraction = destination_hit_fraction(num_experts=experts, local_experts=local_experts, topk=topk)
        pair_capacity = min(tokens_per_shard, math.ceil(transport_capacity_factor * hit_fraction * tokens_per_shard))
        for source in range(shards):
            first = source * tokens_per_shard
            source_valid = int(token_valid[first : first + tokens_per_shard].sum())
            capacity = min(pair_capacity, max(1, math.ceil(transport_capacity_factor * hit_fraction * source_valid)))
            for dest in range(shards):
                seen = 0
                for t in range(first, first + tokens_per_shard):
                    on_dest = sel_np[t] // local_experts == dest
                    if token_valid[t] and on_dest.any():
                        if seen >= capacity:
                            keep[t, on_dest] = False
                        seen += 1
        valid_assignments = int(token_valid.sum()) * topk
        expected_sender_dropped = valid_assignments - int(keep.sum())

        # Ring's receiver capacity: each destination runs at most L of the assignments it receives, in (expert,
        # token) order, with L set by the valid demand before sender drops.
        receiver_capacity = min(
            max(local_experts, math.ceil(capacity_factor * valid_assignments / shards)),
            max(local_experts, math.ceil(capacity_factor * tokens_per_shard * topk)),
        )
        expected_receiver_dropped = 0
        for dest in range(shards):
            accepted = 0
            for expert in range(dest * local_experts, (dest + 1) * local_experts):
                for t, k in zip(*np.nonzero(keep & (sel_np == expert))):
                    if accepted < receiver_capacity:
                        accepted += 1
                    else:
                        keep[t, k] = False
                        expected_receiver_dropped += 1
        keep = jnp.asarray(keep, jnp.float32)

        def dense(x, cw, w_up_gate, w_down):
            hidden_act = jnp.einsum("th,tkhi->tki", x, w_up_gate[selected])
            gate, up = jnp.split(hidden_act, [inter], axis=-1)
            out = jnp.einsum("tki,tkih->tkh", jax.nn.silu(gate) * up, w_down[selected])
            return jnp.einsum("tkh,tk->th", out, cw * keep)

        batch = NamedSharding(mesh, P(("data", "expert"), None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        args = (
            jax.device_put(x, batch),
            jax.device_put(combine_weights, batch),
            jax.device_put(w_up_gate, expert_sharding),
            jax.device_put(w_down, expert_sharding),
        )
        selected_sharded = jax.device_put(selected, batch)
        token_valid_sharded = jax.device_put(jnp.asarray(token_valid), NamedSharding(mesh, P(("data", "expert"))))
        cotangent_sharded = jax.device_put(cotangent, batch)

        def backend(implementation):
            def loss(x, cw, w_up_gate, w_down):
                out, counts = moe_mlp(
                    x,
                    selected_sharded,
                    cw,
                    w_up_gate,
                    w_down,
                    token_valid=token_valid_sharded,
                    activation=jax.nn.silu,
                    implementation=implementation,
                    mesh=mesh,
                    capacity_factor=capacity_factor,
                    pooled_transport_capacity_factor=transport_capacity_factor,
                    report_capacity_overflow=True,
                )
                return jnp.sum(out * cotangent_sharded), (out, counts)

            with jax.set_mesh(mesh):
                grad_fn = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True))
                (_, (out, counts)), grads = grad_fn(*args)
            return out, counts, grads

        expected = dense(x, combine_weights, w_up_gate, w_down)
        expected_grads = jax.grad(lambda *a: jnp.sum(dense(*a) * cotangent), argnums=(0, 1, 2, 3))(
            x, combine_weights, w_up_gate, w_down
        )
        actual, counts, actual_grads = backend("ring_filtered")
        references = [(expected, expected_grads)]
        if expected_sender_dropped == 0:
            ring_out, ring_counts, ring_grads = backend("ring")
            assert int(ring_counts.dropped) == expected_receiver_dropped
            references.append((ring_out, ring_grads))

        for ref_out, ref_grads in references:
            np.testing.assert_allclose(np.asarray(actual), np.asarray(ref_out), rtol=1e-5, atol=1e-5)
            for a, e in zip(actual_grads, ref_grads, strict=True):
                np.testing.assert_allclose(np.asarray(a), np.asarray(e), rtol=1e-5, atol=1e-5)
        assert int(counts.sender_dropped) == expected_sender_dropped
        assert int(counts.receiver_dropped) == expected_receiver_dropped
        # Each case drops what its parameters ask for; with no valid tokens nothing can drop.
        assert (expected_sender_dropped > 0) == (padding != "all" and transport_capacity_factor < 1.0)
        assert (expected_receiver_dropped > 0) == (padding != "all" and capacity_factor < 1.0)


def _fill_rows_past_groups(out: jax.Array, group_sizes: jax.Array, fill: float) -> jax.Array:
    return jnp.where(jnp.arange(out.shape[0])[:, None] < jnp.sum(group_sizes), out, fill)


def _ragged_dot_with_unwritten_rows(fill: float) -> Callable[..., jax.Array]:
    """``ragged_dot`` whose output and lhs gradient rows past the groups hold ``fill``, as a kernel that leaves them
    unwritten may."""

    @jax.custom_vjp
    def filled(lhs, rhs, group_sizes):
        return _fill_rows_past_groups(ragged_dot(lhs, rhs, group_sizes), group_sizes, fill)

    def filled_fwd(lhs, rhs, group_sizes):
        return filled(lhs, rhs, group_sizes), (lhs, rhs, group_sizes)

    def filled_bwd(residual, cotangent):
        lhs, rhs, group_sizes = residual
        _, vjp = jax.vjp(lambda a, b: ragged_dot(a, b, group_sizes), lhs, rhs)
        dlhs, drhs = vjp(cotangent)
        return _fill_rows_past_groups(dlhs, group_sizes, fill), drhs, None

    filled.defvjp(filled_fwd, filled_bwd)
    return filled


@pytest.mark.timeout(240)
@pytest.mark.parametrize("implementation", ["ring", "ring_gather_combine", "ring_filtered"])
@pytest.mark.multidevice
def test_ring_paths_never_read_the_padding_rows_past_the_groups(implementation: str, monkeypatch):
    """The ring paths leave unused dispatch capacity outside every ragged_dot group, where the kernels' output is
    unspecified. NaN there instead of zero must leave every output and gradient finite and unchanged up to the
    rounding of a different compiled program."""
    with use_test_devices(4) as devices, jax.default_matmul_precision("highest"):
        shards, tokens_per_shard, hidden, inter, experts, topk = 4, 16, 8, 6, 16, 2
        tokens = shards * tokens_per_shard
        mesh = Mesh(
            np.asarray(devices).reshape(1, shards, 1),
            axis_names=("data", "expert", "model"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )
        keys = jax.random.split(jax.random.key(1), 6)
        batch = NamedSharding(mesh, P(("data", "expert"), None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        args = (
            jax.device_put(jax.random.normal(keys[0], (tokens, hidden)), batch),
            jax.device_put(jax.nn.softmax(jax.random.normal(keys[1], (tokens, topk)), axis=-1), batch),
            jax.device_put(
                jax.random.normal(keys[2], (experts, hidden, 2 * inter)) / math.sqrt(hidden), expert_sharding
            ),
            jax.device_put(jax.random.normal(keys[3], (experts, inter, hidden)) / math.sqrt(inter), expert_sharding),
        )
        selected = jnp.argsort(jax.random.uniform(keys[4], (tokens, experts)), axis=1)[:, :topk].astype(jnp.int32)
        selected = jax.device_put(selected, batch)
        # A quarter of the tokens are padding, so every shard has unused capacity.
        token_valid = jax.device_put(jnp.arange(tokens) % 4 != 1, NamedSharding(mesh, P(("data", "expert"))))
        cotangent = jax.device_put(jax.random.normal(keys[5], (tokens, hidden)), batch)

        def run(fill):
            for module in (ep_ring, ep_ring_filtered):
                monkeypatch.setattr(module, "ragged_dot", _ragged_dot_with_unwritten_rows(fill))

            def loss(x, cw, w_up_gate, w_down):
                out, _ = moe_mlp(
                    x,
                    selected,
                    cw,
                    w_up_gate,
                    w_down,
                    token_valid=token_valid,
                    activation=jax.nn.silu,
                    implementation=implementation,
                    mesh=mesh,
                    capacity_factor=2.0,
                    pooled_transport_capacity_factor=2.0,
                    report_capacity_overflow=True,
                )
                return jnp.sum(out * cotangent), out

            with jax.set_mesh(mesh):
                (_, out), grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True))(*args)
            return out, grads

        zero_filled, nan_filled = run(0.0), run(jnp.nan)
        for zero, nan in zip(jax.tree.leaves(zero_filled), jax.tree.leaves(nan_filled), strict=True):
            assert np.isfinite(np.asarray(nan)).all()
            np.testing.assert_allclose(np.asarray(nan), np.asarray(zero), rtol=1e-6, atol=1e-6)


def _bits(x: jax.Array) -> np.ndarray:
    x = np.asarray(x)
    return x.view(np.uint16 if x.dtype.itemsize == 2 else np.uint32)


def _expert_mesh(devices, shards=4):
    return Mesh(
        np.asarray(devices).reshape(1, shards, 1),
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def _ring_filtered_loss(*, hidden, dtype, experts, mesh, shards=4, tokens_per_shard=16, inter=8, topk=2):
    """A ring_filtered value-and-grad problem with sender and receiver drops and uneven padding: the loss of
    (x, combine weights, w_up_gate, w_down) with the drop counts as aux, and those four arguments.

    The inputs come from NumPy so that building them compiles nothing.
    """
    tokens = shards * tokens_per_shard
    rng = np.random.default_rng(0)
    batch = NamedSharding(mesh, P(("data", "expert"), None))
    expert_sharding = NamedSharding(mesh, P("expert", None, None))
    logits = rng.standard_normal((tokens, topk), dtype=np.float32)
    args = (
        jax.device_put(rng.standard_normal((tokens, hidden), dtype=np.float32).astype(dtype), batch),
        jax.device_put(np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True), batch),
        jax.device_put(
            (rng.standard_normal((experts, hidden, 2 * inter), dtype=np.float32) / math.sqrt(hidden)).astype(dtype),
            expert_sharding,
        ),
        jax.device_put(
            (rng.standard_normal((experts, inter, hidden), dtype=np.float32) / math.sqrt(inter)).astype(dtype),
            expert_sharding,
        ),
    )
    selected = jax.device_put(rng.random((tokens, experts)).argsort(axis=1)[:, :topk].astype(np.int32), batch)
    cotangent = jax.device_put(rng.standard_normal((tokens, hidden), dtype=np.float32).astype(dtype), batch)
    # Shard s keeps between 30% and all of its tokens.
    keep_fraction = np.repeat(0.3 + 0.7 * np.arange(shards) / (shards - 1), tokens_per_shard)
    token_valid = jax.device_put(rng.random(tokens) < keep_fraction, NamedSharding(mesh, P(("data", "expert"))))

    def loss(x, cw, w_up_gate, w_down):
        out, counts = moe_mlp(
            x,
            selected,
            cw,
            w_up_gate,
            w_down,
            token_valid=token_valid,
            activation=jax.nn.silu,
            implementation="ring_filtered",
            mesh=mesh,
            capacity_factor=0.3,
            pooled_transport_capacity_factor=0.5,
            report_capacity_overflow=True,
        )
        return jnp.sum((out * cotangent).astype(jnp.float32)), (out, counts)

    return loss, args


_GATHER_KERNELS = {"grug_gather_rows", "grug_gather_sum"}


@pytest.mark.skipif(jax.default_backend() == "tpu", reason="Pallas-Triton kernels need a GPU or CPU")
@pytest.mark.parametrize(
    ("hidden", "dtype", "kernels_on_instinct"),
    [(16, jnp.float32, _GATHER_KERNELS), (16, jnp.bfloat16, _GATHER_KERNELS), (24, jnp.float32, set())],
    ids=["tileable_f32", "tileable_bf16", "untileable_f32"],
)
@pytest.mark.multidevice
def test_ring_filtered_on_amd_instinct_matches_xla_bitwise(hidden, dtype, kernels_on_instinct, monkeypatch):
    """On AMD Instinct, ring_filtered moves rows with the Pallas-Triton gathers when they can tile the hidden width
    and with XLA otherwise. Either way it gives the outputs, gradients and drop counts of the XLA gathers used on
    other devices, bit for bit, with sender and receiver drops and uneven padding."""
    with use_test_devices(4) as devices:
        mesh = _expert_mesh(devices)
        loss, args = _ring_filtered_loss(hidden=hidden, dtype=dtype, experts=16, mesh=mesh)

        def run(amd_instinct):
            monkeypatch.setattr(ep_ring_filtered, "is_amd_instinct", lambda: amd_instinct)
            with jax.set_mesh(mesh):
                traced = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True)).trace(*args)
                # Which kernels run: without this, the bitwise comparison passes trivially.
                jaxpr = str(traced.jaxpr)
                kernels = {name for name in _GATHER_KERNELS if name in jaxpr}
                (_, (out, counts)), grads = traced.lower().compile()(*args)
            return kernels, out, counts, grads

        xla_kernels, xla_out, xla_counts, xla_grads = run(amd_instinct=False)
        kernels, out, counts, grads = run(amd_instinct=True)
        assert xla_kernels == set()
        # The sender's send gather and combine backward run grug_gather_rows; the expert side's dispatch backward
        # runs grug_gather_sum.
        assert kernels == kernels_on_instinct
        assert int(xla_counts.sender_dropped) > 0 and int(xla_counts.receiver_dropped) > 0
        assert int(counts.sender_dropped) == int(xla_counts.sender_dropped)
        assert int(counts.receiver_dropped) == int(xla_counts.receiver_dropped)
        np.testing.assert_array_equal(_bits(out), _bits(xla_out))
        for a, e in zip(grads, xla_grads, strict=True):
            assert np.any(np.asarray(e).astype(np.float32) != 0)
            np.testing.assert_array_equal(_bits(a), _bits(e))
