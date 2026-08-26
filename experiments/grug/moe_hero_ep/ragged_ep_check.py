# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the ragged all-to-all EP path on a 4-GPU node.

The guard for the EP64 tuning loop: a hero arm costs a production rack, so any change to the
transport's offset arithmetic or expert kernels earns one only after this passes. By default the
run carries the hero's ragged XLA flags, so it covers the device-initiated kernel the hero uses;
``--transport-kernel`` selects otherwise and the verdict records which one ran.

Nothing invokes this on a schedule. All-to-all is GPU-only so it cannot run in CPU CI, and the
repository has no marker, fixture, or workflow for a test that needs particular accelerators
(#8704) -- ``tests/cluster/`` covers submitting a job to a standing cluster but names only the
TPU and H100 peers. So this is launched by hand, and it lives beside the other remote harnesses
in this directory rather than under ``tests/``. It belongs in ``tests/cluster/`` once #8704 adds
a GB200 lane and a cost tier.

A. Ground truth. At a capacity factor high enough that nothing is dropped, every token reaches
   every expert it selected, so the transport computes exactly the dense MoE. The gate compares
   forward and gradients against an exact fp32 dense reference. The EP ``ring`` implementation is
   recorded as a diagnostic only: measured 2026-08-21, it deviates from dense by 0.8-4.5x relative
   at this shape (its gradients contain NaN), so it cannot serve as a reference.

B. Drop regime. The hero trains with assignments being clipped, so the no-drop case above leaves
   the interesting half of the transport unchecked: which rows survive, and whether the survivors
   land where the combine expects them. Accepted rows are the prefix of each expert group under a
   greedy first-sender-wins gate, so the surviving set is a pure function of the routing and the
   capacity. This section computes it in NumPy, feeds it back as a mask on the combine weights,
   and compares against the same exact fp32 dense reference -- a dropping transport is correct
   exactly when it equals dense restricted to the rows it kept.
"""

import dataclasses
import json
import logging
import math
import os
from enum import StrEnum

import click
import jax
import jax.numpy as jnp
import numpy as np
from fray.types import ANY_REGION, ResourceConfig
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug._moe.ep_common import _clip_receiver_group_sizes
from levanter.grug._moe.ep_ragged_all_to_all import _EXPERT_CHUNKS, _quack_grouped_gemm_available, _select_expert_mlp
from levanter.grug.grug_moe import moe_mlp
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from pydantic import BaseModel
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.moe_hero_ep.train import RAGGED_REQUIRED_XLA_FLAGS

logger = logging.getLogger(__name__)

TOKENS_PER_DEVICE = 64
HIDDEN_DIM = 64
INTERMEDIATE_DIM = 96
NUM_EXPERTS = 8
TOPK = 2
SEEDS = (0, 1, 2, 3)
# The expert axis this harness builds; `_make_ep_mesh` fixes it at 2 whatever the device count.
EP_SIZE = 2


def _no_drop_capacity() -> float:
    """Capacity factor at which the ragged transport provably cannot clip an assignment.

    A receiver's worst case is every assignment on the expert axis landing on a single one of its
    chunks: ``EP_SIZE * assignments_per_shard`` rows arriving against a chunk buffer holding
    ``capacity_factor * assignments_per_shard / chunks``. The structural bound is therefore
    ``EP_SIZE * chunks``. Unchunked it is just ``EP_SIZE`` -- which is why the pre-chunking value
    of 2.0 no longer guarantees anything here: at four local experts the backend runs two chunks,
    so 2.0 leaves each chunk holding only half the rows that can arrive, and droplessness becomes
    a property of how evenly the seed happens to route rather than of the capacity.
    """
    local_experts = NUM_EXPERTS // EP_SIZE
    chunks = _EXPERT_CHUNKS if local_experts % _EXPERT_CHUNKS == 0 and _EXPERT_CHUNKS > 1 else 1
    return float(EP_SIZE * chunks)


NO_DROP_CAPACITY = _no_drop_capacity()
# `ring` keeps the pre-chunking value: it does not chunk, and larger factors make its top_k
# selection ask for more rows than exist and fail. It is a diagnostic (see the module docstring),
# so it does not need the structural bound the graded ragged run does.
RING_DIAGNOSTIC_CAPACITY = 2.0
# Section B's capacity: at or below the mean assignment count per expert, so the skewed router in
# `_inputs` drives real clipping and the surviving set is worth checking.
SKEWED_CAPACITY = 1.0

# Gradients are gated on a median, not the max. Both paths compute the same mathematical gradient,
# so every observed difference is bf16 reduction-order noise: weight gradients sum rows per expert
# and the two paths order that sum differently. Reordering leaves the bulk of the entries
# bit-identical while a handful of cancellation-heavy entries lose their leading digits and show a
# large relative max, around 0.5 on a correct run. Gating on that rejects a correct transport.
#
# The median is taken per slice of the leading axis and the worst slice gates, because a
# tensor-wide median tolerates any corruption confined to under half the entries -- which is the
# shape of the localized faults worth catching here. One expert's grouped weight gradient is an
# eighth of `w13`; one shard's token block is a quarter of the `x` gradient. Neither moves a
# tensor-wide median. Both move the median of the slice they live in. The tensor-wide median and
# the max are recorded as diagnostics.
TOLERANCE = 5e-2


class TransportKernel(StrEnum):
    """Which ragged all-to-all kernel the run exercises.

    ``DEVICE`` matches the hero and is the default, but jaxlib vintages that predate the two flags
    abort at import on unknown ``XLA_FLAGS`` entries, so it needs the pinned jax/XLA build.
    ``STOCK`` drops the flags and takes whatever the runtime defaults to -- the host-launched
    kernel -- which lets the expert-MLP kernels and the offset arithmetic be validated on a stock
    image while that build is pending. The choice is recorded in the results either way, so a
    ``STOCK`` run is never mistaken for coverage of the transport the hero uses.
    """

    DEVICE = "device-kernel"
    STOCK = "stock"


def _transport_flags(kernel: TransportKernel) -> tuple[str, ...]:
    return RAGGED_REQUIRED_XLA_FLAGS if kernel is TransportKernel.DEVICE else ()


# TODO(https://github.com/marin-community/marin/issues/8704): move this to `tests/cluster/` once a
# GB200 fixture and a cost tier exist. Today nothing selects it, so the gate below runs only when
# someone launches it by hand.
BENCHMARK_RESOURCES = ResourceConfig.with_gpu("GB200", count=4, cpu=16, ram="256g", disk="128g", regions=[ANY_REGION])


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

    rows: list[SeedRow] = []
    for seed in SEEDS:
        with jax.set_mesh(mesh):
            # A. ground truth: balanced routing, no drops -- ragged must match the exact fp32
            # dense MoE; ring runs alongside as a diagnostic only (see module docstring).
            raw = _inputs(jax.random.key(1000 + seed), tokens, skew=False)
            dense, g_dense = _dense_reference(*raw)
            xb, selb, cwb, w13b, w2b = reshard(*raw)
            o_ragged, g_ragged, dropped = loss_and_grad(
                "ragged_all_to_all", xb, selb, cwb, w13b, w2b, capacity_factor=NO_DROP_CAPACITY
            )
            o_ring, _g_ring, dropped_ring = loss_and_grad(
                "ring", xb, selb, cwb, w13b, w2b, capacity_factor=RING_DIAGNOSTIC_CAPACITY
            )

            # B. drop regime: skewed routing at a capacity that clips, checked against the dense
            # reference restricted to the assignments the NumPy oracle says survive.
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
    *, version: str | None = None, transport_kernel: TransportKernel = TransportKernel.DEVICE
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
            resources=BENCHMARK_RESOURCES,
            env_vars={"JAX_ENABLE_PGLE": "false", "XLA_FLAGS": " ".join(_transport_flags(transport_kernel))},
        ),
        build_config=build_config,
    )


@click.command()
@click.option(
    "--transport-kernel",
    type=click.Choice([kernel.value for kernel in TransportKernel]),
    default=TransportKernel.DEVICE.value,
    show_default=True,
    help=(
        "Ragged all-to-all kernel to validate. The default matches the hero and needs the pinned "
        "jax/XLA build; 'stock' drops the flags so the run works on an image without it."
    ),
)
@build_options
def main(transport_kernel: str) -> ArtifactStep[RaggedEpResult]:
    return build_benchmark(transport_kernel=TransportKernel(transport_kernel))


if __name__ == "__main__":
    main()
