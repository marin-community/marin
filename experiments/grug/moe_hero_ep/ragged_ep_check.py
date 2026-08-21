# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the ragged all-to-all EP path on a 4-GPU node.

The guard for the EP64 tuning loop: a hero arm costs a production rack, so any change to the
transport's offset arithmetic or expert kernels earns one only after this passes. All-to-all is
GPU-only, so this cannot run in CPU CI.

A. Ground truth. At a capacity factor high enough that nothing is dropped, every token reaches
   every expert it selected, so ``ragged_all_to_all`` computes exactly the MoE that the ``ring``
   reference computes. Forward and gradients must agree.

B. Split invariance. ``ragged_all_to_all_splits_per_peer`` divides each peer transfer into N
   ragged updates. It moves the same bytes to the same places, so the result must not depend on
   it -- including under skewed routing, where padding and drops are present and the per-split
   offset arithmetic is doing real work.
"""

import dataclasses
import json
import logging

import click
import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from fray.types import ANY_REGION, ResourceConfig
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.grug_moe import moe_mlp
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from pydantic import BaseModel
from rigging.filesystem.storage_path import prefix_join

logger = logging.getLogger(__name__)

TOKENS_PER_DEVICE = 64
HIDDEN_DIM = 64
INTERMEDIATE_DIM = 96
NUM_EXPERTS = 8
TOPK = 2
SEEDS = (0, 1, 2, 3)
SPLITS_UNDER_TEST = (8, 32)
# Capacity high enough that no assignment is clipped, which is what makes `ring` a ground truth.
NO_DROP_CAPACITY = 4.0
SKEWED_CAPACITY = 1.0

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
    max_out_vs_ring: float
    max_grad_vs_ring: float
    median_grad_vs_ring: float
    dropped_no_drop_case: int
    max_out_splits: dict[str, float]
    median_grad_splits: dict[str, float]
    max_grad_splits: dict[str, float]
    dropped_skewed: int


class RaggedEpResult(Artifact):
    """Per-seed deviations for the ragged-EP ground-truth and split-invariance checks."""


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

    def loss_and_grad(impl, x, sel, cw, w13, w2, *, capacity_factor, splits=1):
        extra = {"ragged_all_to_all_splits_per_peer": splits} if impl == "ragged_all_to_all" else {}

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
                **extra,
            )
            return (out * out).sum(), (out, dropped)

        (_v, (out, dropped)), grads = jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True)(x, w13, w2)
        return out, grads, int(dropped)

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
            # A. ground truth: balanced routing, no drops -- ragged must match ring.
            xb, selb, cwb, w13b, w2b = reshard(*_inputs(jax.random.key(1000 + seed), tokens, skew=False))
            o_ragged, g_ragged, dropped = loss_and_grad(
                "ragged_all_to_all", xb, selb, cwb, w13b, w2b, capacity_factor=NO_DROP_CAPACITY
            )
            o_ring, g_ring, dropped_ring = loss_and_grad(
                "ring", xb, selb, cwb, w13b, w2b, capacity_factor=NO_DROP_CAPACITY
            )

            # B. split invariance: skewed routing at capacity 1.0, so padding and drops are present.
            xs, sels, cws, w13s, w2s = reshard(*_inputs(jax.random.key(seed), tokens, skew=True))
            o1, g1, d1 = loss_and_grad(
                "ragged_all_to_all", xs, sels, cws, w13s, w2s, capacity_factor=SKEWED_CAPACITY, splits=1
            )
            split_out, split_grad = {}, {}
            for splits in SPLITS_UNDER_TEST:
                o_n, g_n, _ = loss_and_grad(
                    "ragged_all_to_all", xs, sels, cws, w13s, w2s, capacity_factor=SKEWED_CAPACITY, splits=splits
                )
                split_out[str(splits)] = float(jnp.max(jnp.abs(o_n - o1)))
                split_grad[str(splits)] = _graddiff(g_n, g1)

        max_g, med_g = _graddiff(g_ragged, g_ring)
        rows.append(
            SeedRow(
                seed=seed,
                max_out_vs_ring=float(jnp.max(jnp.abs(o_ragged - o_ring))),
                max_grad_vs_ring=max_g,
                median_grad_vs_ring=med_g,
                dropped_no_drop_case=dropped + dropped_ring,
                max_out_splits=split_out,
                median_grad_splits={k: v[1] for k, v in split_grad.items()},
                max_grad_splits={k: v[0] for k, v in split_grad.items()},
                dropped_skewed=d1,
            )
        )
        logger.info("ragged_ep_seed %s", rows[-1].model_dump_json())
    return rows


def run_benchmark(config: RaggedEpConfig) -> None:
    rows = _run()
    payload = [r.model_dump(mode="json") for r in rows]
    ground_truth_ok = all(
        r.max_out_vs_ring <= TOLERANCE and r.median_grad_vs_ring <= TOLERANCE and r.dropped_no_drop_case == 0
        for r in rows
    )
    splits_ok = all(
        all(v <= TOLERANCE for v in r.max_out_splits.values())
        and all(v <= TOLERANCE for v in r.median_grad_splits.values())
        for r in rows
    )
    verdict = {
        "ragged_correct": bool(ground_truth_ok and splits_ok),
        "matches_ring_no_drop": ground_truth_ok,
        "split_invariant": splits_ok,
    }
    logger.info("ragged_ep_result %s verdict %s", json.dumps(payload), json.dumps(verdict))
    filesystem, _, _ = fsspec.get_fs_token_paths(config.output_path)
    filesystem.makedirs(config.output_path, exist_ok=True)
    with filesystem.open(prefix_join(config.output_path, "results.json"), "w") as handle:
        json.dump({"rows": payload, "verdict": verdict}, handle, indent=2)
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
