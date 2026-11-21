# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Shadow experts in the ragged expert-parallel backend: which assignments survive, and their function."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from levanter.grug._moe import ep_ragged_all_to_all
from levanter.grug._moe.ep_ragged_all_to_all import _shadow_routing, capacity_block_experts
from levanter.grug.grug_moe import ShadowExperts, moe_mlp

# The cases below sit on drop boundaries of this exact layout, so it must not follow the device count.
_SHARDS = 4
_EXPERTS_PER_SHARD = 4
_NUM_EXPERTS = _SHARDS * _EXPERTS_PER_SHARD
_TOKENS_PER_SHARD = 32
_TOPK = 4
_ROUTING_SEED = 4
_FP32_RELATIVE_TOLERANCE = 1e-5
_BF16_RELATIVE_TOLERANCE = 2e-2
# Shadowing on and off sum the same bf16 terms in another order.
_BF16_REORDER_TOLERANCE = 1e-2


@pytest.fixture
def expert_mesh(monkeypatch: pytest.MonkeyPatch) -> Mesh:
    devices = jax.devices()
    if devices[0].platform == "cpu" or len(devices) < _SHARDS:
        pytest.skip(f"requires {_SHARDS} accelerators; ragged_all_to_all is not implemented on XLA:CPU")
    if devices[0].platform == "tpu":
        monkeypatch.setenv("RAGGED_DOT_IMPL", "megablox")
    return Mesh(
        np.array(devices[:_SHARDS]).reshape(1, _SHARDS, 1),
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 3,
    )


def _dims() -> tuple[int, int]:
    """Hidden and intermediate sizes; the TPU GMM wants a wide hidden size and a rectangular VJP."""
    return (16, 24) if jax.devices()[0].platform == "gpu" else (128, 16)


def _activation_dtype() -> jnp.dtype:
    # QuACK's SM100 expert kernels take 16-bit activations; on TPU, follow the neighbouring EP tests.
    if jax.devices()[0].platform == "tpu" or ep_ragged_all_to_all._quack_grouped_gemm_available():
        return jnp.bfloat16
    return jnp.float32


def _skewed_routing() -> np.ndarray:
    """Distinct top-k experts per token, with two hot experts on different shards."""
    rng = np.random.default_rng(_ROUTING_SEED)
    probability = np.ones(_NUM_EXPERTS)
    probability[[1, 6]] = [12.0, 8.0]
    probability /= probability.sum()
    tokens = _SHARDS * _TOKENS_PER_SHARD
    return np.stack([rng.choice(_NUM_EXPERTS, _TOPK, replace=False, p=probability) for _ in range(tokens)]).astype(
        np.int32
    )


def _token_valid(padded: bool) -> np.ndarray:
    valid = np.ones((_SHARDS, _TOKENS_PER_SHARD), dtype=bool)
    if padded:
        valid[:, -5:] = False
    return valid.reshape(-1)


def _shadowed_counts(counts: np.ndarray, shadow: ShadowExperts | None) -> np.ndarray:
    """Each shard's shadowed rows per expert: the most-loaded experts first, up to its buffer."""
    shadowed = np.zeros_like(counts)
    if shadow is None:
        return shadowed
    hot = np.argsort(-counts.sum(0), kind="stable")[: shadow.count]
    rows = math.ceil(min(shadow.rows_per_token, shadow.count, _TOPK) * _TOKENS_PER_SHARD)
    for s in range(_SHARDS):
        left = rows
        for e in hot:
            shadowed[s, e] = min(counts[s, e], left)
            left -= shadowed[s, e]
    return shadowed


def _surviving_assignments(
    selected: np.ndarray, valid: np.ndarray, *, capacity_factor: float, shadow: ShadowExperts | None
) -> np.ndarray:
    """A numpy model of the clip with shadow experts; returns the ``[T, K]`` mask of kept assignments.

    Each shard sorts its valid assignments by expert, stably. The group's ``count`` most-loaded experts keep
    the last rows of each shard's group, up to the shard's shadow rows. Receivers clip the rest per capacity
    block, expert by expert and sender by sender, taking each group's first rows.
    """
    block = capacity_block_experts(_EXPERTS_PER_SHARD)
    blocks = _EXPERTS_PER_SHARD // block
    # Invalid assignments sort after every expert.
    flat = np.where(valid[:, None], selected, _NUM_EXPERTS).reshape(_SHARDS, _TOKENS_PER_SHARD * _TOPK)
    counts = np.stack([np.bincount(f, minlength=_NUM_EXPERTS + 1)[:_NUM_EXPERTS] for f in flat])  # [S, E]
    shadowed = _shadowed_counts(counts, shadow)
    transport = counts - shadowed

    assignments = _TOKENS_PER_SHARD * _TOPK
    physical = max(_EXPERTS_PER_SHARD, math.ceil(capacity_factor * assignments))
    logical = min(max(math.ceil(counts.sum() * capacity_factor / _SHARDS), _EXPERTS_PER_SHARD), physical)
    block_capacity = max(math.ceil(logical / blocks), block)
    accepted = np.zeros_like(counts)
    for receiver in range(_SHARDS):
        for b in range(blocks):
            left = block_capacity
            first = receiver * _EXPERTS_PER_SHARD + b * block
            for e in range(first, first + block):
                for s in range(_SHARDS):
                    accepted[s, e] = min(transport[s, e], left)
                    left -= accepted[s, e]

    keep = np.zeros(flat.shape, dtype=bool)
    for s in range(_SHARDS):
        order = np.argsort(flat[s], kind="stable")
        start = 0
        for e in range(_NUM_EXPERTS):
            group = order[start : start + counts[s, e]]
            keep[s, group[: accepted[s, e]]] = True
            keep[s, group[counts[s, e] - shadowed[s, e] :]] = True
            start += counts[s, e]
    return keep.reshape(selected.shape)


def _dense(x, selected, weights, w13, w2):
    hidden = jnp.einsum("th,tkhi->tki", x, w13[selected])
    gate, up = jnp.split(hidden, [w2.shape[1]], axis=-1)
    return jnp.einsum("tki,tkih,tk->th", jax.nn.silu(gate) * up, w2[selected], weights)


def _relative_error(actual, expected) -> float:
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float(np.max(np.abs(actual - expected)) / np.max(np.abs(expected)))


@pytest.mark.tpu
@pytest.mark.parametrize(
    ("capacity_factor", "shadow", "padded", "expected_drops"),
    [
        (1.0, None, False, 126),
        (16.0, ShadowExperts(count=2, rows_per_token=0.25), False, 0),
        (1.0, ShadowExperts(count=2, rows_per_token=2.0), False, 0),
        # Clips both the spilled hot rows (56) and the other experts' rows (21).
        (1.0, ShadowExperts(count=2, rows_per_token=0.25), True, 77),
    ],
    ids=["transport_only_drops", "split_group_no_drops", "shadow_keeps_hot_tails", "spill_is_clipped"],
)
def test_ragged_shadow_experts_keep_the_modelled_assignments_with_their_function(
    expert_mesh: Mesh,
    capacity_factor: float,
    shadow: ShadowExperts | None,
    padded: bool,
    expected_drops: int,
):
    selected_np = _skewed_routing()
    valid_np = _token_valid(padded)
    keep = _surviving_assignments(selected_np, valid_np, capacity_factor=capacity_factor, shadow=shadow)
    hidden, intermediate = _dims()
    keys = jax.random.split(jax.random.key(0), 5)
    tokens = selected_np.shape[0]
    dtype = _activation_dtype()
    tolerance = _FP32_RELATIVE_TOLERANCE if dtype == jnp.float32 else _BF16_RELATIVE_TOLERANCE
    # The reference runs in float32 on the same rounded inputs.
    x = jax.random.normal(keys[0], (tokens, hidden)).astype(dtype).astype(jnp.float32)
    weights = jax.nn.softmax(jax.random.normal(keys[1], (tokens, _TOPK)), axis=-1).astype(dtype).astype(jnp.float32)
    w13 = (jax.random.normal(keys[2], (_NUM_EXPERTS, hidden, 2 * intermediate)) / math.sqrt(hidden)).astype(dtype)
    w13 = w13.astype(jnp.float32)
    w2 = (jax.random.normal(keys[3], (_NUM_EXPERTS, intermediate, hidden)) / math.sqrt(intermediate)).astype(dtype)
    w2 = w2.astype(jnp.float32)
    cotangent = jax.random.normal(keys[4], (tokens, hidden)).astype(dtype).astype(jnp.float32)
    selected = jnp.asarray(selected_np)
    keep_mask = jnp.asarray(keep)

    with jax.default_matmul_precision("highest"):
        expected = _dense(x, selected, weights * keep_mask, w13, w2)
        expected_grads = jax.grad(
            lambda x, w, w13, w2: jnp.sum(_dense(x, selected, w * keep_mask, w13, w2) * cotangent),
            argnums=(0, 1, 2, 3),
        )(x, weights, w13, w2)

    tokens_spec = NamedSharding(expert_mesh, P(("data", "expert"), None))
    valid_spec = NamedSharding(expert_mesh, P(("data", "expert")))
    experts_spec = NamedSharding(expert_mesh, P("expert", None, None))
    args = [jax.device_put(a, tokens_spec) for a in (x.astype(dtype), selected, weights.astype(dtype))]
    args += [jax.device_put(a.astype(dtype), experts_spec) for a in (w13, w2)]
    token_valid = jax.device_put(jnp.asarray(valid_np), valid_spec)
    cotangent_sharded = jax.device_put(cotangent.astype(dtype), tokens_spec)

    def backend(x, selected, weights, w13, w2):
        return moe_mlp(
            x,
            selected,
            weights,
            w13,
            w2,
            token_valid=token_valid,
            implementation="ragged_all_to_all",
            mesh=expert_mesh,
            capacity_factor=capacity_factor,
            report_capacity_overflow=True,
            shadow_experts=shadow,
        )

    # TPU megablox rejects "highest" on bf16 operands.
    backend_precision = "highest" if dtype == jnp.float32 else "default"
    with jax.set_mesh(expert_mesh), jax.default_matmul_precision(backend_precision):
        actual, counts = jax.jit(backend)(*args)
        actual_grads = jax.jit(
            jax.grad(
                lambda x, w, w13, w2: jnp.sum(
                    (backend(x, args[1], w, w13, w2)[0] * cotangent_sharded).astype(jnp.float32)
                ),
                argnums=(0, 1, 2, 3),
            )
        )(args[0], args[2], args[3], args[4])

    assert int(counts.dropped) == expected_drops
    assert _relative_error(actual, expected) < tolerance
    for name, got, want in zip(("x", "weights", "w13", "w2"), actual_grads, expected_grads, strict=True):
        assert _relative_error(got, want) < tolerance, name


def test_shadow_routing_takes_the_tail_of_each_shadowed_group_and_inverts_it():
    num_experts, topk, rows = 6, 2, 5
    all_group_sizes = np.array(
        [
            [4, 0, 1, 3, 0, 2],
            [1, 2, 0, 1, 0, 6],
            [2, 1, 0, 2, 1, 4],
        ],
        dtype=np.int32,
    )  # totals [7, 3, 1, 6, 1, 12]: experts 5, 0, 3 are the three most loaded
    rng = np.random.default_rng(0)
    routing = jax.jit(lambda sizes, order, shard: _shadow_routing(sizes, order, shard, count=3, rows=rows, topk=topk))

    for shard, shard_sizes in enumerate(all_group_sizes):
        selected = rng.permutation(np.repeat(np.arange(num_experts), shard_sizes)).astype(np.int32)
        sorted_indices = np.argsort(selected, kind="stable").astype(np.int32)
        shadow, shadowed_by_expert = routing(
            jnp.asarray(all_group_sizes), jnp.asarray(sorted_indices), jnp.int32(shard)
        )
        np.testing.assert_array_equal(shadow.experts, [5, 0, 3])

        expected = []
        left = rows
        for e in (5, 0, 3):
            group = sorted_indices[selected[sorted_indices] == e]
            taken = min(len(group), left)
            left -= taken
            assert np.asarray(shadowed_by_expert)[shard, e] == taken
            expected.extend(group[len(group) - taken :])
        active = len(expected)
        # Inactive slots hold distinct out-of-range assignments.
        np.testing.assert_array_equal(shadow.assignments, [*expected, *(len(selected) + np.arange(active, rows))])
        want_positions = np.full(len(selected), rows)
        want_positions[expected] = np.arange(active)
        np.testing.assert_array_equal(np.asarray(shadow.positions).reshape(-1), want_positions)


@pytest.mark.tpu
def test_ragged_shadow_experts_change_nothing_when_nothing_is_dropped(expert_mesh: Mesh):
    """With room for every assignment, shadowing only moves rows between GEMM calls.

    Each row's expert output comes from the same per-row products; the outputs, input gradients and
    weight gradients sum the same terms in another order, so they match to bf16 rounding.
    """
    dtype = jnp.bfloat16
    hidden, intermediate = _dims()
    selected = jnp.asarray(_skewed_routing())
    tokens = selected.shape[0]
    keys = jax.random.split(jax.random.key(1), 5)
    x = jax.random.normal(keys[0], (tokens, hidden)).astype(dtype)
    weights = jax.nn.softmax(jax.random.normal(keys[1], (tokens, _TOPK)), axis=-1).astype(dtype)
    w13 = (jax.random.normal(keys[2], (_NUM_EXPERTS, hidden, 2 * intermediate)) / math.sqrt(hidden)).astype(dtype)
    w2 = (jax.random.normal(keys[3], (_NUM_EXPERTS, intermediate, hidden)) / math.sqrt(intermediate)).astype(dtype)
    cotangent = jax.random.normal(keys[4], (tokens, hidden)).astype(dtype)
    tokens_spec = NamedSharding(expert_mesh, P(("data", "expert"), None))
    experts_spec = NamedSharding(expert_mesh, P("expert", None, None))
    x, selected, weights, cotangent = (jax.device_put(a, tokens_spec) for a in (x, selected, weights, cotangent))
    w13, w2 = (jax.device_put(a, experts_spec) for a in (w13, w2))

    def run(shadow):
        def backend(x, weights, w13, w2):
            return moe_mlp(
                x,
                selected,
                weights,
                w13,
                w2,
                implementation="ragged_all_to_all",
                mesh=expert_mesh,
                capacity_factor=16.0,
                report_capacity_overflow=True,
                shadow_experts=shadow,
            )

        with jax.set_mesh(expert_mesh):
            out, counts = jax.jit(backend)(x, weights, w13, w2)
            grads = jax.jit(
                jax.grad(lambda *a: jnp.sum((backend(*a)[0] * cotangent).astype(jnp.float32)), argnums=(0, 1, 2, 3))
            )(x, weights, w13, w2)
        assert int(counts.dropped) == 0
        return out, grads

    out_off, grads_off = run(None)
    out_on, grads_on = run(ShadowExperts(count=3, rows_per_token=3.0))
    for name, on, off in zip(("out", "x", "weights", "w13", "w2"), (out_on, *grads_on), (out_off, *grads_off)):
        assert _relative_error(on, off) < _BF16_REORDER_TOLERANCE, name
