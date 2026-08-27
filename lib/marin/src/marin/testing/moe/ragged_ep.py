# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""4-GPU correctness check for the ragged all-to-all expert-parallel MoE transport.

A hero arm costs a production rack, so a change to the transport's offset arithmetic or expert
kernels earns one only after this passes. It runs as a remote job, so it lives here rather than in
a test module: the callable is pickled by reference and the worker imports whatever defines it.
`tests/cluster/grug/test_ragged_ep_check.py` submits it.

Each seed is checked twice against an exact fp32 dense MoE. Without drops, at a capacity that
cannot clip, the transport must equal dense outright. Under drops, at a capacity that clips, it
must equal dense restricted to the rows it kept -- and must keep exactly the rows a NumPy model of
the capacity gate says it should, since matching values while dropping a different set would mean
the two disagree about which assignments exist.

The second case is the one worth having. The hero trains with assignments being clipped, so a
check that only ever runs dropless leaves the surviving-set arithmetic untested.
"""

import dataclasses
import json
import logging
import math
import os
from enum import StrEnum

import jax
import jax.numpy as jnp
import numpy as np
from fray.types import ANY_REGION, ResourceConfig
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug._moe.ep_common import _clip_receiver_group_sizes
from levanter.grug._moe.ep_ragged_all_to_all import (
    _EXPERT_CHUNKS,
    RAGGED_REQUIRED_XLA_FLAGS,
    _quack_grouped_gemm_available,
    _select_expert_mlp,
)
from levanter.grug.grug_moe import moe_mlp
from pydantic import BaseModel
from rigging.filesystem.storage_path import StoragePath

from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.namespacing import user_namespaced_name

logger = logging.getLogger(__name__)

TOKENS_PER_DEVICE = 64
HIDDEN_DIM = 64
INTERMEDIATE_DIM = 96
NUM_EXPERTS = 8
TOPK = 2
SEEDS = (0, 1, 2, 3)
# The expert axis `_make_ep_mesh` builds, whatever the device count.
EP_SIZE = 2


def _no_drop_capacity() -> float:
    """Return the capacity factor at which the transport cannot clip an assignment.

    A receiver's worst case is every assignment on the expert axis arriving at one of its chunks:
    ``EP_SIZE * assignments_per_shard`` rows against a chunk buffer holding
    ``capacity_factor * assignments_per_shard / chunks``. Size the buffer for that and nothing
    drops, so the factor is ``EP_SIZE * chunks``. It must be computed: a factor that ignores the
    chunk count leaves each chunk holding a fraction of what can arrive, and droplessness becomes
    a property of how a seed happens to route rather than of the capacity.
    """
    local_experts = NUM_EXPERTS // EP_SIZE
    chunks = _EXPERT_CHUNKS if local_experts % _EXPERT_CHUNKS == 0 and _EXPERT_CHUNKS > 1 else 1
    return float(EP_SIZE * chunks)


NO_DROP_CAPACITY = _no_drop_capacity()
# `ring` keeps the pre-chunking value: it does not chunk, and larger factors make its top_k
# selection ask for more rows than exist and fail. It is a diagnostic, so it does not need the
# structural bound the graded ragged run does.
RING_DIAGNOSTIC_CAPACITY = 2.0
# At or below the mean assignment count per expert, so the skewed router in `_inputs` drives
# real clipping and the surviving set is worth checking.
SKEWED_CAPACITY = 1.0

# The maximum allowed relative deviation between the ragged and dense kernels.
#
# Gradients are gated on a median rather than the max. Both paths compute the same mathematical
# gradient, so every difference is bf16 reduction-order noise: the two orders leave most entries
# bit-identical, while a few cancellation-heavy entries lose their leading digits and push the
# relative max to around 0.5 on a correct run. Gate on the max and a correct transport fails.
#
# Take the median per slice of the leading axis and gate the worst slice. A tensor-wide median
# tolerates corruption confined to under half the entries, which is the shape of the faults worth
# catching: one expert's weight gradient is an eighth of `w13`, one shard's token block a quarter
# of the `x` gradient. Neither moves a tensor-wide median; both move their own slice's.
TOLERANCE = 5e-2


class TransportKernel(StrEnum):
    """Which ragged all-to-all kernel the run exercises.

    ``DEVICE`` matches the hero and is the default. It needs a jaxlib that defines the two flags;
    older ones abort at import on an unknown ``XLA_FLAGS`` entry. ``STOCK`` clears them and takes
    the runtime default, the host-launched kernel, which still covers the expert-MLP kernels and
    the offset arithmetic.
    """

    DEVICE = "device-kernel"
    STOCK = "stock"


def _transport_flags(kernel: TransportKernel) -> tuple[str, ...]:
    return RAGGED_REQUIRED_XLA_FLAGS if kernel is TransportKernel.DEVICE else ()


def _benchmark_resources(target_cluster: str) -> ResourceConfig:
    """One 4-GPU GB200 node on ``target_cluster``, which the caller names."""
    return ResourceConfig.with_gpu(
        "GB200",
        count=4,
        cpu=16,
        ram="256g",
        disk="128g",
        regions=[ANY_REGION],
        target_cluster=target_cluster,
    )


class GradDiff(BaseModel):
    """Relative gradient deviation, summarized three ways. ``worst_slice_median`` is the gate."""

    worst_entry: float
    tensor_median: float
    worst_slice_median: float


class SeedRow(BaseModel):
    seed: int
    ragged_vs_dense: float
    ring_vs_dense: float
    grad_vs_dense: GradDiff
    dropped_no_drop_case: int
    dropped_no_drop_case_ring: int
    ragged_vs_dense_dropped: float
    grad_vs_dense_dropped: GradDiff
    dropped_skewed: int
    dropped_skewed_expected: int


class RuntimeRow(BaseModel):
    """What the run actually exercised, so a green verdict names the code it covers.

    The expert-MLP kernels are chosen at trace time from the device's compute capability and the
    installed packages, and the transport kernel from XLA flags the runtime may not recognize.
    Both are environment-dependent, so recording them is the difference between "the ragged EP
    path is correct" and "some ragged EP path was correct somewhere".
    """

    jax_version: str
    device_kind: str
    quack_grouped_gemm_available: bool
    expert_mlp_silu: str
    expert_mlp_gelu: str
    xla_flags: str


def _runtime_row() -> RuntimeRow:
    return RuntimeRow(
        jax_version=jax.__version__,
        device_kind=jax.devices()[0].device_kind,
        quack_grouped_gemm_available=_quack_grouped_gemm_available(),
        expert_mlp_silu=_select_expert_mlp(jax.nn.silu).__name__,
        expert_mlp_gelu=_select_expert_mlp(jax.nn.gelu).__name__,
        xla_flags=os.environ.get("XLA_FLAGS", ""),
    )


class RaggedEpResult(Artifact):
    """Per-seed deviations for the ragged-EP ground-truth check."""


@dataclasses.dataclass(frozen=True)
class RaggedEpConfig:
    output_path: str


def _make_ep_mesh() -> Mesh:
    devices = jax.devices()
    if len(devices) < 2 or len(devices) % 2 != 0:
        raise RuntimeError(f"need an even device count >= 2, got {len(devices)}")
    mesh_devices = np.array(devices).reshape(len(devices) // 2, 2, 1)
    return Mesh(
        mesh_devices,
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def _inputs(key, tokens, *, skew):
    k_x, k_sel, k_cw, k_w13, k_w2 = jax.random.split(key, 5)
    x = jax.random.normal(k_x, (tokens, HIDDEN_DIM), dtype=jnp.bfloat16)
    bias = jnp.array([3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) if skew else jnp.zeros((NUM_EXPERTS,))
    logits = jax.random.normal(k_sel, (tokens, NUM_EXPERTS)) + bias
    selected = jax.lax.top_k(logits, TOPK)[1].astype(jnp.int32)
    combine_weights = jax.nn.softmax(jax.random.normal(k_cw, (tokens, TOPK)), axis=-1).astype(jnp.bfloat16)
    w13 = jax.random.normal(k_w13, (NUM_EXPERTS, HIDDEN_DIM, 2 * INTERMEDIATE_DIM), dtype=jnp.bfloat16)
    w2 = jax.random.normal(k_w2, (NUM_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM), dtype=jnp.bfloat16)
    return x, selected, combine_weights, w13, w2


def _accepted_counts(group_sizes: np.ndarray, capacity_factor: float, assignments_per_shard: int) -> np.ndarray:
    """Rows each (sender, global expert) pair gets to keep, summed over the backend's chunks.

    Mirrors the backend's per-chunk gate: local experts are split into ``_EXPERT_CHUNKS`` groups
    processed in sequence, each with its own share of the receiver buffer, so an expert competes
    for capacity only with the other experts in its chunk.
    """
    local_experts = NUM_EXPERTS // EP_SIZE
    chunks = _EXPERT_CHUNKS if local_experts % _EXPERT_CHUNKS == 0 and _EXPERT_CHUNKS > 1 else 1
    chunk_experts = local_experts // chunks
    local_capacity = max(local_experts, math.ceil(capacity_factor * assignments_per_shard))
    chunk_capacity = max(chunk_experts, math.ceil(local_capacity / chunks))
    chunk_of_expert = (np.arange(NUM_EXPERTS) % local_experts) // chunk_experts

    accepted = np.zeros_like(group_sizes)
    for chunk in range(chunks):
        masked = np.where(chunk_of_expert[None, :] == chunk, group_sizes, 0)
        accepted += np.asarray(
            _clip_receiver_group_sizes(
                jnp.asarray(masked), local_expert_size=local_experts, receiver_capacity=chunk_capacity
            )
        )
    return accepted


def _keep_mask(selected: np.ndarray, tokens_per_shard: int, capacity_factor: float) -> np.ndarray:
    """Which (token, route) assignments survive capacity clipping, as a [tokens, TOPK] 0/1 mask.

    The batch is sharded over ``("data", "expert")``, so shard ``j`` owns a contiguous token block
    and the all-to-all runs between the ``EP_SIZE`` shards that share a ``data`` index. Within a
    shard the dispatch buffer is expert-sorted with a stable sort, so the accepted prefix of each
    expert group is its lowest-indexed assignments.
    """
    tokens = selected.shape[0]
    num_shards = tokens // tokens_per_shard
    keep = np.zeros(selected.shape, dtype=np.float32)
    for group_start in range(0, num_shards, EP_SIZE):
        shards = range(group_start, group_start + EP_SIZE)
        flat = [selected[s * tokens_per_shard : (s + 1) * tokens_per_shard].reshape(-1) for s in shards]
        group_sizes = np.stack([np.bincount(f, minlength=NUM_EXPERTS) for f in flat]).astype(np.int32)
        accepted = _accepted_counts(group_sizes, capacity_factor, tokens_per_shard * TOPK)
        for sender, shard in enumerate(shards):
            for expert in range(NUM_EXPERTS):
                # Stable expert-sorted order == ascending flat assignment index within the group.
                positions = np.flatnonzero(flat[sender] == expert)[: accepted[sender, expert]]
                rows, routes = np.divmod(positions, TOPK)
                keep[shard * tokens_per_shard + rows, routes] = 1.0
    return keep


def _dense_reference(x, selected, combine_weights, w13, w2):
    """Exact fp32 dense MoE with the same scalar loss: forward output and gradients.

    Uses replicated fp32 jnp ops (no capacity, no transport), so it is exact up to fp32
    rounding and serves as the arbiter for every EP implementation.
    """
    sel = jnp.asarray(selected)
    cw = jnp.asarray(combine_weights).astype(jnp.float32)

    def loss(xf, w13f, w2f):
        hidden = jnp.einsum("th,ehi->tei", xf, w13f)
        gate, up = hidden[..., :INTERMEDIATE_DIM], hidden[..., INTERMEDIATE_DIM:]
        per_expert = jnp.einsum("tei,eih->teh", jax.nn.silu(gate) * up, w2f)
        per_route = jnp.take_along_axis(per_expert, sel[..., None], axis=1)
        out = jnp.sum(per_route * cw[..., None], axis=1)
        return jnp.sum(out * out), out

    (_l, out), grads = jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True)(
        jnp.asarray(x, jnp.float32), jnp.asarray(w13, jnp.float32), jnp.asarray(w2, jnp.float32)
    )
    return np.asarray(out), grads


def _maxdiff_vs_dense(out, dense) -> float:
    denom = float(np.max(np.abs(dense))) + 1e-6
    return float(np.max(np.abs(np.asarray(out, np.float32) - dense))) / denom


def _graddiff(a, b) -> GradDiff:
    """Compare a gradient tuple against the reference, worst tensor wins each statistic.

    Slices run along the leading axis: expert for the two weight gradients, token for the
    activation gradient. See the ``TOLERANCE`` comment for why the gate reads a per-slice median.
    """
    maxes, medians, slice_medians = [], [], []
    for u, v in zip(a, b, strict=True):
        u = np.asarray(u, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        denom = float(np.max(np.abs(u)) + np.max(np.abs(v))) + 1e-6
        absdiff = np.abs(u - v)
        maxes.append(float(np.max(absdiff)) / denom)
        medians.append(float(np.median(absdiff)) / denom)
        per_slice = np.median(absdiff.reshape(absdiff.shape[0], -1), axis=1)
        slice_medians.append(float(np.max(per_slice)) / denom)
    return GradDiff(worst_entry=max(maxes), tensor_median=max(medians), worst_slice_median=max(slice_medians))


def _run() -> list[SeedRow]:
    mesh = _make_ep_mesh()
    tokens = len(jax.devices()) * TOKENS_PER_DEVICE
    batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
    expert_sharding = NamedSharding(mesh, P("expert", None, None))

    def loss_and_grad(impl, x, sel, cw, w13, w2, *, capacity_factor):
        def loss(x, w13, w2):
            out, dropped = moe_mlp(
                x,
                sel,
                cw,
                w13,
                w2,
                implementation=impl,
                mesh=None,
                report_capacity_overflow=True,
                capacity_factor=capacity_factor,
            )
            return (out * out).sum(), (out, dropped)

        (_v, (out, dropped)), grads = jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True)(x, w13, w2)
        return out, grads, int(dropped.total)

    def reshard(x, sel, cw, w13, w2):
        return (
            jax.sharding.reshard(x, batch_sharding),
            jax.sharding.reshard(sel, batch_sharding),
            jax.sharding.reshard(cw, batch_sharding),
            jax.sharding.reshard(w13, expert_sharding),
            jax.sharding.reshard(w2, expert_sharding),
        )

    def compare_without_drops(seed: int):
        """Balanced routing at a capacity that cannot clip, so the transport must equal dense.

        `ring` runs on the same inputs and is recorded as a diagnostic, never as a reference.
        """
        raw = _inputs(jax.random.key(1000 + seed), tokens, skew=False)
        dense, g_dense = _dense_reference(*raw)
        xb, selb, cwb, w13b, w2b = reshard(*raw)
        o_ragged, g_ragged, dropped = loss_and_grad(
            "ragged_all_to_all", xb, selb, cwb, w13b, w2b, capacity_factor=NO_DROP_CAPACITY
        )
        o_ring, _g_ring, dropped_ring = loss_and_grad(
            "ring", xb, selb, cwb, w13b, w2b, capacity_factor=RING_DIAGNOSTIC_CAPACITY
        )
        return dense, g_dense, o_ragged, g_ragged, dropped, o_ring, dropped_ring

    def compare_under_drops(seed: int):
        """Skewed routing at a capacity that clips, against dense restricted to surviving rows.

        The NumPy oracle decides which assignments survive; masking the combine weights with it
        gives a dense reference that computes only those.
        """
        raw_skew = _inputs(jax.random.key(seed), tokens, skew=True)
        xs_raw, sel_raw, cw_raw, w13_raw, w2_raw = raw_skew
        keep = _keep_mask(np.asarray(sel_raw), TOKENS_PER_DEVICE, SKEWED_CAPACITY)
        dense_dropped, g_dense_dropped = _dense_reference(
            xs_raw, sel_raw, jnp.asarray(cw_raw, jnp.float32) * keep, w13_raw, w2_raw
        )
        xs, sels, cws, w13s, w2s = reshard(*raw_skew)
        o_drop, g_drop, dropped_skewed = loss_and_grad(
            "ragged_all_to_all", xs, sels, cws, w13s, w2s, capacity_factor=SKEWED_CAPACITY
        )
        return keep, dense_dropped, g_dense_dropped, o_drop, g_drop, dropped_skewed

    rows: list[SeedRow] = []
    for seed in SEEDS:
        with jax.set_mesh(mesh):
            dense, g_dense, o_ragged, g_ragged, dropped, o_ring, dropped_ring = compare_without_drops(seed)
            keep, dense_dropped, g_dense_dropped, o_drop, g_drop, dropped_skewed = compare_under_drops(seed)

        grad = _graddiff(g_ragged, g_dense)
        grad_dropped = _graddiff(g_drop, g_dense_dropped)
        rows.append(
            SeedRow(
                seed=seed,
                ragged_vs_dense=_maxdiff_vs_dense(o_ragged, dense),
                ring_vs_dense=_maxdiff_vs_dense(o_ring, dense),
                grad_vs_dense=grad,
                dropped_no_drop_case=dropped,
                dropped_no_drop_case_ring=dropped_ring,
                ragged_vs_dense_dropped=_maxdiff_vs_dense(o_drop, dense_dropped),
                grad_vs_dense_dropped=grad_dropped,
                dropped_skewed=dropped_skewed,
                dropped_skewed_expected=round(float((1.0 - keep).sum())),
            )
        )
        logger.info("ragged_ep_seed %s", rows[-1].model_dump_json())
    return rows


def run_benchmark(config: RaggedEpConfig) -> None:
    rows = _run()
    payload = [r.model_dump(mode="json") for r in rows]
    ground_truth_ok = all(
        r.ragged_vs_dense <= TOLERANCE
        and r.grad_vs_dense.worst_slice_median <= TOLERANCE
        and r.dropped_no_drop_case == 0
        for r in rows
    )
    # A dropping transport is correct when it equals dense over the rows it kept AND keeps the
    # rows the gate says it should: matching values while dropping a different set would mean the
    # oracle and the backend disagree about which assignments exist.
    drops_ok = all(
        r.ragged_vs_dense_dropped <= TOLERANCE
        and r.grad_vs_dense_dropped.worst_slice_median <= TOLERANCE
        and r.dropped_skewed == r.dropped_skewed_expected
        for r in rows
    )
    verdict = {
        "ragged_correct": bool(ground_truth_ok and drops_ok),
        "matches_dense_no_drop": ground_truth_ok,
        "matches_dense_under_drops": drops_ok,
    }
    runtime = _runtime_row()
    logger.info(
        "ragged_ep_result %s verdict %s runtime %s",
        json.dumps(payload),
        json.dumps(verdict),
        runtime.model_dump_json(),
    )
    output_dir = StoragePath(config.output_path)
    output_dir.mkdirs(exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps({"rows": payload, "verdict": verdict, "runtime": runtime.model_dump(mode="json")}, indent=2)
    )
    if not verdict["ragged_correct"]:
        raise RuntimeError(f"ragged EP NOT validated: {verdict}")


def build_benchmark(
    *,
    target_cluster: str,
    version: str | None = None,
    transport_kernel: TransportKernel = TransportKernel.DEVICE,
) -> ArtifactStep[RaggedEpResult]:
    name = "grug/ragged-ep-check"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> RaggedEpConfig:
        return RaggedEpConfig(output_path=ctx.output_path)

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=RaggedEpResult,
        run=remote(
            run_benchmark,
            name="ragged-ep-check-gb200",
            resources=_benchmark_resources(target_cluster),
            env_vars={"JAX_ENABLE_PGLE": "false", "XLA_FLAGS": " ".join(_transport_flags(transport_kernel))},
        ),
        build_config=build_config,
    )
