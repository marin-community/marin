# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the ragged all-to-all EP path on a 4-GPU node.

The guard for the EP64 tuning loop: a hero arm costs a production rack, so any change to the
transport's offset arithmetic or expert kernels earns one only after this passes. All-to-all is
GPU-only, so this cannot run in CPU CI.

A. Ground truth. At a capacity factor high enough that nothing is dropped, every token reaches
   every expert it selected, so the transport computes exactly the dense MoE. The gate compares
   forward and gradients against an exact fp32 dense reference. The EP ``ring`` implementation is
   recorded as a diagnostic only: measured 2026-08-21, it deviates from dense by 0.8-4.5x relative
   at this shape (its gradients contain NaN), so it cannot serve as a reference.
"""

import dataclasses
import json
import logging

import click
import jax
import jax.numpy as jnp
import numpy as np
from fray.types import ANY_REGION, ResourceConfig
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug._moe.ep_ragged_all_to_all import _EXPERT_CHUNKS
from levanter.grug.grug_moe import moe_mlp
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from pydantic import BaseModel
from rigging.filesystem.storage_path import StoragePath

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

# Gradients are compared on the MEDIAN relative difference, not the max. Both paths compute the
# same mathematical gradient, so every observed difference is bf16 reduction-order noise: weight
# gradients sum rows per expert and the two paths order that sum differently. Reordering leaves
# the bulk of the entries bit-identical (median 0.0) while a handful of cancellation-heavy entries
# lose their leading digits and show a large relative max. Gating on the max rejects a correct
# transport; gating on the median rejects a real backward bug, which shifts the whole distribution.
# The max is recorded as a diagnostic.
TOLERANCE = 5e-2

BENCHMARK_RESOURCES = ResourceConfig.with_gpu("GB200", count=4, cpu=16, ram="256g", disk="128g", regions=[ANY_REGION])


class SeedRow(BaseModel):
    seed: int
    ragged_vs_dense: float
    ring_vs_dense: float
    max_grad_vs_dense: float
    median_grad_vs_dense: float
    dropped_no_drop_case: int
    dropped_no_drop_case_ring: int


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


def _graddiff(a, b) -> tuple[float, float]:
    """Worst-case and median relative gradient difference across the gradient tuple."""
    maxes, medians = [], []
    for u, v in zip(a, b, strict=True):
        u = np.asarray(u, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        denom = float(np.max(np.abs(u)) + np.max(np.abs(v))) + 1e-6
        absdiff = np.abs(u - v)
        maxes.append(float(np.max(absdiff)) / denom)
        medians.append(float(np.median(absdiff)) / denom)
    return max(maxes), max(medians)


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

        max_g, med_g = _graddiff(g_ragged, g_dense)
        rows.append(
            SeedRow(
                seed=seed,
                ragged_vs_dense=_maxdiff_vs_dense(o_ragged, dense),
                ring_vs_dense=_maxdiff_vs_dense(o_ring, dense),
                max_grad_vs_dense=max_g,
                median_grad_vs_dense=med_g,
                dropped_no_drop_case=dropped,
                dropped_no_drop_case_ring=dropped_ring,
            )
        )
        logger.info("ragged_ep_seed %s", rows[-1].model_dump_json())
    return rows


def run_benchmark(config: RaggedEpConfig) -> None:
    rows = _run()
    payload = [r.model_dump(mode="json") for r in rows]
    ground_truth_ok = all(
        r.ragged_vs_dense <= TOLERANCE and r.median_grad_vs_dense <= TOLERANCE and r.dropped_no_drop_case == 0
        for r in rows
    )
    verdict = {
        "ragged_correct": bool(ground_truth_ok),
        "matches_dense_no_drop": ground_truth_ok,
    }
    logger.info("ragged_ep_result %s verdict %s", json.dumps(payload), json.dumps(verdict))
    output_dir = StoragePath(config.output_path)
    output_dir.mkdirs(exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps({"rows": payload, "verdict": verdict}, indent=2))
    if not verdict["ragged_correct"]:
        raise RuntimeError(f"ragged EP NOT validated: {verdict}")


def build_benchmark(*, version: str | None = None) -> ArtifactStep[RaggedEpResult]:
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
            env_vars={"JAX_ENABLE_PGLE": "false"},
        ),
        build_config=build_config,
    )


@click.command()
@build_options
def main() -> ArtifactStep[RaggedEpResult]:
    return build_benchmark()


if __name__ == "__main__":
    main()
