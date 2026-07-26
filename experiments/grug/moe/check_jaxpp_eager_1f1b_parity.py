# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare direct and automatic JaxPP eager-1F1B Grug MoE gradients.

The authoritative mode uses four ranks with one device per rank. It preserves
FP32 master parameters and casts to BF16 inside the differentiated loss,
matching ``train.py``. Loss and every gradient leaf must have relative-L2 at
most 0.002. On one Iris task with ``--gpu=H100x4``, the script self-spawns one
JAX process per GPU:

    uv run python experiments/grug/moe/check_jaxpp_eager_1f1b_parity.py \
      --platform gpu --precision production-mixed

The externally supervised ``iris.hooks.multigpu_main`` launch pattern remains
supported when the caller already owns process supervision.

The FP32 mode is a single-process schedule-algebra check. Expose four CPU
devices before Python imports JAX:

    XLA_FLAGS=--xla_force_host_platform_device_count=4 \
      uv run python experiments/grug/moe/check_jaxpp_eager_1f1b_parity.py \
      --platform cpu --precision fp32

JaxPP must be installed with ``jaxpp_setup_scripts()`` from ``train.py``.
"""

import argparse
import dataclasses
import functools
import json
import math
import multiprocessing as mp
import os
import time
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

import jax
import jax.numpy as jnp
import jmp
import numpy as np
from iris.cluster.client.job_info import get_job_info
from iris.hooks.multigpu import IRIS_MULTIGPU_PROCESS_COUNT_ENV
from iris.runtime.jax_init import initialize_jax
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.text.examples import GrugLmExample

from experiments.grug.moe import train as grug_train
from experiments.grug.moe.model import GrugModelConfig, Transformer

DEFAULT_TOLERANCE = 0.002
PIPELINE_STAGES = 4
MICROBATCHES = 4
RESHARD_THRESHOLD_BYTES = 1 << 40
WORKER_TIMEOUT = 900.0
WORKER_SHUTDOWN_TIMEOUT = 10.0
_GRUG_MESH_AXES = ("replica_dcn", "data", "expert", "model")
_MIXED_PRECISION = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")


class PrecisionMode(StrEnum):
    PRODUCTION_MIXED = "production-mixed"
    FP32 = "fp32"


@dataclasses.dataclass(frozen=True)
class ValueParity:
    reference_l2: float
    actual_l2: float
    absolute_l2: float
    max_absolute_error: float
    relative_l2: float
    norm_ratio: float
    cosine_similarity: float
    finite: bool
    passed: bool


@dataclasses.dataclass(frozen=True)
class GradientParity:
    path: str
    reference_l2: float
    actual_l2: float
    absolute_l2: float
    max_absolute_error: float
    relative_l2: float
    norm_ratio: float
    cosine_similarity: float
    finite: bool
    passed: bool


@dataclasses.dataclass(frozen=True)
class ParityReport:
    tolerance: float
    loss: ValueParity
    gradients: tuple[GradientParity, ...]
    passed: bool

    @property
    def max_gradient_relative_l2(self) -> float:
        return max((gradient.relative_l2 for gradient in self.gradients), default=0.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tolerance": self.tolerance,
            "loss": dataclasses.asdict(self.loss),
            "gradients": [dataclasses.asdict(gradient) for gradient in self.gradients],
            "max_gradient_relative_l2": self.max_gradient_relative_l2,
            "passed": self.passed,
        }


def relative_l2(actual: jax.Array, reference: jax.Array) -> tuple[float, float, float]:
    metrics = parity_metrics(actual, reference)
    return metrics.reference_l2, metrics.absolute_l2, metrics.relative_l2


@dataclasses.dataclass(frozen=True)
class TensorParityMetrics:
    reference_l2: float
    actual_l2: float
    absolute_l2: float
    max_absolute_error: float
    relative_l2: float
    norm_ratio: float
    cosine_similarity: float
    finite: bool


def parity_metrics(actual: jax.Array, reference: jax.Array) -> TensorParityMetrics:
    """Return error, scale, direction, and finite-value diagnostics."""
    actual_f32 = jnp.asarray(actual, dtype=jnp.float32)
    reference_f32 = jnp.asarray(reference, dtype=jnp.float32)
    difference = actual_f32 - reference_f32
    absolute = float(jnp.linalg.norm(difference))
    max_absolute_error = float(jnp.max(jnp.abs(difference), initial=0.0))
    reference_norm = float(jnp.linalg.norm(reference_f32))
    actual_norm = float(jnp.linalg.norm(actual_f32))
    relative = absolute / max(reference_norm, 1e-12)
    if reference_norm == 0.0:
        norm_ratio = 1.0 if actual_norm == 0.0 else math.inf
    else:
        norm_ratio = actual_norm / reference_norm
    if actual_norm == 0.0 or reference_norm == 0.0:
        cosine_similarity = 1.0 if actual_norm == reference_norm else 0.0
    else:
        inner_product = float(jnp.sum(actual_f32 * reference_f32))
        cosine_similarity = max(-1.0, min(1.0, inner_product / (actual_norm * reference_norm)))
    finite = bool(
        jnp.all(jnp.isfinite(actual_f32))
        & jnp.all(jnp.isfinite(reference_f32))
        & jnp.isfinite(absolute)
        & jnp.isfinite(max_absolute_error)
        & jnp.isfinite(reference_norm)
        & jnp.isfinite(actual_norm)
        & jnp.isfinite(relative)
    )
    if not finite:
        relative = math.inf
    return TensorParityMetrics(
        reference_l2=reference_norm,
        actual_l2=actual_norm,
        absolute_l2=absolute,
        max_absolute_error=max_absolute_error,
        relative_l2=relative,
        norm_ratio=norm_ratio,
        cosine_similarity=cosine_similarity,
        finite=finite,
    )


def build_value_parity(
    actual: jax.Array,
    reference: jax.Array,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ValueParity:
    """Build one finite, relative-L2-gated value result."""
    metrics = parity_metrics(actual, reference)
    return ValueParity(
        reference_l2=metrics.reference_l2,
        actual_l2=metrics.actual_l2,
        absolute_l2=metrics.absolute_l2,
        max_absolute_error=metrics.max_absolute_error,
        relative_l2=metrics.relative_l2,
        norm_ratio=metrics.norm_ratio,
        cosine_similarity=metrics.cosine_similarity,
        finite=metrics.finite,
        passed=metrics.finite and metrics.relative_l2 <= tolerance,
    )


def build_parity_report(
    *,
    automatic_loss: jax.Array,
    direct_loss: jax.Array,
    automatic_gradients,
    direct_gradients,
    tolerance: float = DEFAULT_TOLERANCE,
    gradient_root: str = "params",
) -> ParityReport:
    """Build the per-leaf parity result and enforce one relative-L2 ceiling."""
    if tolerance <= 0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")

    loss = build_value_parity(automatic_loss, direct_loss, tolerance=tolerance)

    automatic_with_paths, automatic_tree = jax.tree.flatten_with_path(automatic_gradients)
    direct_with_paths, direct_tree = jax.tree.flatten_with_path(direct_gradients)
    if automatic_tree != direct_tree:
        raise ValueError("automatic and direct gradient trees differ")

    gradients = []
    for (automatic_path, automatic_leaf), (direct_path, direct_leaf) in zip(
        automatic_with_paths,
        direct_with_paths,
        strict=True,
    ):
        if automatic_path != direct_path:
            raise ValueError(f"automatic and direct gradient paths differ: {automatic_path} != {direct_path}")
        metrics = parity_metrics(
            automatic_leaf,
            direct_leaf,
        )
        gradients.append(
            GradientParity(
                path=f"{gradient_root}{jax.tree_util.keystr(automatic_path)}",
                reference_l2=metrics.reference_l2,
                actual_l2=metrics.actual_l2,
                absolute_l2=metrics.absolute_l2,
                max_absolute_error=metrics.max_absolute_error,
                relative_l2=metrics.relative_l2,
                norm_ratio=metrics.norm_ratio,
                cosine_similarity=metrics.cosine_similarity,
                finite=metrics.finite,
                passed=metrics.finite and metrics.relative_l2 <= tolerance,
            )
        )

    gradient_results = tuple(gradients)
    return ParityReport(
        tolerance=tolerance,
        loss=loss,
        gradients=gradient_results,
        passed=loss.passed and all(gradient.passed for gradient in gradient_results),
    )


def _tiny_model_config() -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=8,
        shared_expert_intermediate_dim=0,
        num_experts=4,
        num_experts_per_token=1,
        num_layers=PIPELINE_STAGES,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=8,
        sliding_window=8,
        attention_implementation="reference",
        moe_implementation="scatter",
        remat_mode="save_moe",
    )


def _pipeline_mesh(platform: str) -> Mesh:
    devices = jax.devices(platform)
    if len(devices) < PIPELINE_STAGES:
        raise ValueError(
            f"eager-1F1B parity requires {PIPELINE_STAGES} {platform} devices, found {len(devices)}. "
            "For CPU, set XLA_FLAGS=--xla_force_host_platform_device_count=4 before starting Python."
        )
    mesh_devices = np.asarray(devices[:PIPELINE_STAGES], dtype=object).reshape((PIPELINE_STAGES, 1, 1, 1, 1))
    return Mesh(
        mesh_devices,
        ("pipeline", *_GRUG_MESH_AXES),
        axis_types=(AxisType.Explicit,) * 5,
    )


def _tiny_problem(mesh: Mesh) -> tuple[Transformer, GrugLmExample]:
    config = _tiny_model_config()
    with jax.set_mesh(mesh):
        model = Transformer.init(config, key=jax.random.PRNGKey(0))
        tokens = jnp.arange(MICROBATCHES * config.max_seq_len, dtype=jnp.int32).reshape(MICROBATCHES, 1, -1) * 7 + 3
        tokens %= config.vocab_size
        loss_weight = jnp.ones_like(tokens, dtype=jnp.float32).at[:, :, -1].set(0)
        batch = GrugLmExample(tokens=tokens, loss_weight=loss_weight)
    return model, batch


def _loss(
    params: Transformer,
    fixed_router_biases: tuple[jax.Array, ...],
    microbatch: GrugLmExample,
    *,
    pipeline_stages: int | None,
    precision: PrecisionMode,
    mixed_precision_policy: Any = _MIXED_PRECISION,
) -> jax.Array:
    model = grug_train._replace_router_biases(params, fixed_router_biases)
    if precision is PrecisionMode.PRODUCTION_MIXED:
        model = grug_train._cast_preserving_overwrites(model, mixed_precision_policy.cast_to_compute)
    return model.next_token_loss(
        microbatch.tokens,
        microbatch.loss_weight,
        mask=microbatch.attn_mask,
        reduction="mean",
        logsumexp_weight=None,
        return_router_metrics=False,
        pipeline_stages=pipeline_stages,
    )


def _direct_microbatch_mean(
    params: Transformer,
    batch: GrugLmExample,
    *,
    precision: PrecisionMode,
    mixed_precision_policy: Any = _MIXED_PRECISION,
) -> tuple[jax.Array, Transformer]:
    pipeline_params, fixed_router_biases = grug_train._detach_router_biases(params)
    losses = []
    gradients = []
    for microbatch_index in range(MICROBATCHES):
        microbatch = jax.tree.map(lambda value, index=microbatch_index: value[index], batch)
        microbatch_loss = functools.partial(
            _loss,
            fixed_router_biases=fixed_router_biases,
            microbatch=microbatch,
            pipeline_stages=None,
            precision=precision,
            mixed_precision_policy=mixed_precision_policy,
        )
        loss, gradient = jax.value_and_grad(microbatch_loss)(pipeline_params)
        losses.append(loss)
        gradients.append(gradient)

    scale = jnp.asarray(1.0 / MICROBATCHES, dtype=jnp.float32)
    loss_mean = sum(losses) * scale
    gradient_mean = jax.tree.map(lambda *leaves: sum(leaves) * scale, *gradients)
    return loss_mean, grug_train._restore_zero_router_bias_gradients(gradient_mean, fixed_router_biases)


def _automatic_treduce(
    params: Transformer,
    batch: GrugLmExample,
    *,
    mpmd_mesh,
    precision: PrecisionMode,
) -> tuple[jax.Array, Transformer]:
    pp = grug_train._require_jaxpp()

    def automatic_step(candidate: Transformer, microbatches: GrugLmExample):
        pipeline_params, fixed_router_biases = grug_train._detach_router_biases(candidate)
        microbatch_grad = functools.partial(
            jax.value_and_grad(
                lambda model, microbatch: _loss(
                    model,
                    fixed_router_biases,
                    microbatch,
                    pipeline_stages=PIPELINE_STAGES,
                    precision=precision,
                )
            ),
            pipeline_params,
        )
        loss_sum, gradient_sum = pp.treduce(
            microbatch_grad,
            microbatches,
            schedule=pp.Eager1F1B(num_stages=PIPELINE_STAGES),
            operation=(pp.Add, pp.Add),
        )
        scale = jnp.asarray(1.0 / MICROBATCHES, dtype=loss_sum.dtype)
        gradient_mean = jax.tree.map(lambda gradient: gradient * scale, gradient_sum)
        return (
            loss_sum * scale,
            grug_train._restore_zero_router_bias_gradients(gradient_mean, fixed_router_biases),
        )

    mesh = mpmd_mesh.jax_mesh
    input_shardings = (
        grug_train._tree_named_shardings_on_mesh(mesh, params),
        grug_train._tree_named_shardings_on_mesh(mesh, batch),
    )
    output_shardings = (
        NamedSharding(mesh, P()),
        grug_train._tree_named_shardings_on_mesh(mesh, params),
    )
    automatic = pp.mpmd_jit_with_loop(
        automatic_step,
        mpmd_mesh=mpmd_mesh,
        in_shardings=input_shardings,
        out_shardings=output_shardings,
    )
    compiled = grug_train._compile_automatic_jaxpp_with_phase_validation(automatic, params, batch)
    compiled = grug_train._localize_automatic_jaxpp_shardings(compiled, mpmd_mesh)
    grug_train._validate_automatic_jaxpp_task_jaxprs(compiled)

    argument_shardings, keyword_shardings = compiled.in_shardings
    if keyword_shardings:
        raise ValueError(f"unexpected JaxPP keyword shardings: {keyword_shardings}")
    mpmd_params, mpmd_batch = pp.spmd_to_mpmd_reshard(
        mpmd_mesh,
        (params, batch),
        argument_shardings,
        threshold=RESHARD_THRESHOLD_BYTES,
    )
    automatic_loss, automatic_gradients = compiled(mpmd_params, mpmd_batch)
    return pp.mpmd_to_spmd_reshard(
        mpmd_mesh,
        (automatic_loss, automatic_gradients),
        output_shardings,
        threshold=RESHARD_THRESHOLD_BYTES,
    )


def _validate_authoritative_run(precision: PrecisionMode, tolerance: float) -> None:
    if precision is PrecisionMode.FP32:
        return
    if tolerance != DEFAULT_TOLERANCE:
        raise ValueError(f"production-mixed parity requires tolerance={DEFAULT_TOLERANCE}, got {tolerance}")
    if jax.process_count() != PIPELINE_STAGES or jax.local_device_count() != 1:
        raise ValueError(
            "production-mixed parity requires four JAX processes with one device each; "
            f"found process_count={jax.process_count()}, local_device_count={jax.local_device_count()}. "
            "Use the H100x4 self-spawned command in this module's docstring."
        )


def run_parity(*, platform: str, precision: PrecisionMode, tolerance: float) -> ParityReport:
    _validate_authoritative_run(precision, tolerance)
    mesh = _pipeline_mesh(platform)
    pp = grug_train._require_jaxpp()
    mpmd_mesh = pp.MpmdMesh(mesh, "pipeline")
    grug_train._install_jaxpp_bind_meshes_patch()
    grug_train._install_jaxpp_const_sharding_patch()

    params, batch = _tiny_problem(mesh)
    with jax.set_mesh(mesh):
        direct_step = jax.jit(functools.partial(_direct_microbatch_mean, precision=precision))
        direct_loss, direct_gradients = direct_step(
            params,
            batch,
        )
        jax.block_until_ready((direct_loss, direct_gradients))
    automatic_loss, automatic_gradients = _automatic_treduce(
        params,
        batch,
        mpmd_mesh=mpmd_mesh,
        precision=precision,
    )

    return build_parity_report(
        automatic_loss=automatic_loss,
        direct_loss=direct_loss,
        automatic_gradients=automatic_gradients,
        direct_gradients=direct_gradients,
        tolerance=tolerance,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=("cpu", "gpu"),
        default="cpu",
        help="JAX platform providing at least four devices.",
    )
    parser.add_argument(
        "--precision",
        type=PrecisionMode,
        choices=tuple(PrecisionMode),
        default=PrecisionMode.PRODUCTION_MIXED,
        help="production-mixed is authoritative; fp32 is a local schedule-algebra check.",
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    return parser.parse_args(argv)


def _run_current_process(*, platform: str, precision: PrecisionMode, tolerance: float) -> int:
    report = run_parity(platform=platform, precision=precision, tolerance=tolerance)
    if jax.process_index() == 0:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


def _run_distributed_rank(
    process_id: int,
    coordinator_address: str,
    platform: str,
    precision: PrecisionMode,
    tolerance: float,
) -> None:
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=PIPELINE_STAGES,
        process_id=process_id,
        local_device_ids=[process_id],
        cluster_detection_method="deactivate",
    )
    try:
        if jax.process_index() != process_id:
            raise ValueError(f"initialized JAX process {jax.process_index()}, expected {process_id}")
        raise SystemExit(_run_current_process(platform=platform, precision=precision, tolerance=tolerance))
    finally:
        jax.distributed.shutdown()


def _monitor_workers(processes: list[mp.Process]) -> int:
    deadline = time.monotonic() + WORKER_TIMEOUT
    try:
        while any(process.is_alive() for process in processes):
            failed = next(
                (process.exitcode for process in processes if process.exitcode not in (None, 0)),
                None,
            )
            if failed is not None:
                return failed
            if time.monotonic() >= deadline:
                raise TimeoutError(f"distributed parity exceeded {WORKER_TIMEOUT:g} seconds")
            time.sleep(0.25)
        return next((process.exitcode for process in processes if process.exitcode), 0)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=WORKER_SHUTDOWN_TIMEOUT)
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()


def _run_self_spawned(*, platform: str, precision: PrecisionMode, tolerance: float) -> int:
    context = mp.get_context("spawn")
    coordinator_address = "127.0.0.1:5789"
    processes = [
        context.Process(
            target=_run_distributed_rank,
            args=(process_id, coordinator_address, platform, precision, tolerance),
            name=f"jaxpp-parity-rank-{process_id}",
        )
        for process_id in range(PIPELINE_STAGES)
    ]
    for process in processes:
        process.start()
    return _monitor_workers(processes)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    job_info = get_job_info()
    externally_distributed = IRIS_MULTIGPU_PROCESS_COUNT_ENV in os.environ or (
        job_info is not None and job_info.num_tasks > 1
    )
    if externally_distributed:
        initialize_jax()
        return _run_current_process(platform=args.platform, precision=args.precision, tolerance=args.tolerance)
    if args.precision is PrecisionMode.PRODUCTION_MIXED:
        return _run_self_spawned(platform=args.platform, precision=args.precision, tolerance=args.tolerance)
    return _run_current_process(platform=args.platform, precision=args.precision, tolerance=args.tolerance)


if __name__ == "__main__":
    raise SystemExit(main())
