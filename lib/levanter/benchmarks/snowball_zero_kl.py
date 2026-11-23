# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded Snowball teacher, prefill, and cached-decode comparison.

TPU runs compare native Splash against all supported fixed-arithmetic opt-ins
on the four-slot, 256-token NoPE fixture. CPU runs exercise the same benchmark
helpers with a small real reference model; TPU qualification remains unexecuted.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from haliax import Axis
from jax.sharding import Mesh

from levanter.grug.attention.fixed_schedule import FixedScheduleSpec
from levanter.grug.sharding import compact_grug_mesh
from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.kernels.pallas.splash_attention import splash_attention_block_sizes
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel

BASE_OID = "135a9f284668e9db4a807947c472ca84c16fa712"
MODEL_SEED = 38127
TOKEN_SEED = 90211
FIXTURE = "four-slot256token-oneNoPElayer-twoexperts-vocab23"
PAGE_SIZE = 128
NUM_SLOTS = 4
SEQUENCE_LENGTH = 256
VOCAB_SIZE = 23
SELECTOR_FIELDS = (
    "attention_implementation",
    "inference_attention_implementation",
    "moe_implementation",
    "projection_implementation",
    "gated_norm_implementation",
    "rms_norm_implementation",
    "qk_rms_norm_implementation",
    "shared_attention_implementation",
)


@dataclass(frozen=True)
class BenchmarkArgs:
    warmup: int = 2
    measured: int = 8
    backend: Literal["auto", "cpu", "tpu"] = "auto"
    output: Path | None = None
    source_sha: str | None = None


@dataclass(frozen=True)
class Fixture:
    tokens: hax.NamedArray
    example: LmExample
    page_maps: tuple[np.ndarray, ...]
    page_size: int


@dataclass(frozen=True)
class PagedRequest:
    tokens: hax.NamedArray
    info: PageBatchInfo
    positions: hax.NamedArray


def qualified_config(*, implementation: Literal["native", "fixed_schedule"], backend: str) -> SnowballConfig:
    """Configure every arithmetic selector explicitly for the bounded comparison."""
    fixed = implementation == "fixed_schedule"
    return SnowballConfig(
        vocab_size=VOCAB_SIZE,
        hidden_dim=128,
        intermediate_dim=32,
        shared_expert_intermediate_dim=32,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=128,
        max_seq_len=SEQUENCE_LENGTH,
        attention_implementation="tpu_splash" if backend == "tpu" else "reference",
        inference_attention_implementation="tpu" if backend == "tpu" else "reference",
        moe_implementation="ring",
        projection_implementation=implementation,
        gated_norm_implementation=implementation,
        rms_norm_implementation=("pallas_tpu" if backend == "tpu" else "reference") if fixed else "native",
        qk_rms_norm_implementation=("pallas_tpu" if backend == "tpu" else "reference") if fixed else "native",
        shared_attention_implementation=implementation,
    )


def make_fixture(*, sequence_length: int = SEQUENCE_LENGTH, page_size: int = PAGE_SIZE) -> Fixture:
    """Build seeded tokens, nonuniform causal loss weights, and two physical maps."""
    if sequence_length != 2 * page_size:
        raise ValueError("each benchmark sequence must occupy exactly two pages")
    tokens = hax.named(
        jax.random.randint(
            jax.random.key(TOKEN_SEED),
            (NUM_SLOTS, sequence_length),
            0,
            VOCAB_SIZE,
            dtype=jnp.int32,
        ),
        ("batch", "position"),
    )
    weights = hax.named(jnp.broadcast_to(jnp.linspace(0.5, 1.5, sequence_length), tokens.array.shape), tokens.axes)
    return Fixture(
        tokens=tokens,
        example=LmExample(
            tokens=tokens,
            loss_weight=weights * LmExample.causal_loss_mask(tokens.resolve_axis("position")),
            attn_mask=AttentionMask.causal(),
        ),
        page_maps=(
            np.asarray([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int32),
            np.asarray([[6, 1], [7, 0], [4, 3], [5, 2]], dtype=np.int32),
        ),
        page_size=page_size,
    )


def page_batch(fixture: Fixture, *, page_map: np.ndarray, prefix_length: int) -> PagedRequest:
    """Pack a prefix into explicit physical cache destinations outside tracing."""
    batch_size, sequence_length = fixture.tokens.array.shape
    if page_map.shape != (batch_size, 2) or prefix_length not in range(1, sequence_length + 1):
        raise ValueError("invalid page map or prefix for the bounded fixture")
    positions = np.tile(np.arange(prefix_length, dtype=np.int32), batch_size)
    rows = np.repeat(np.arange(batch_size), prefix_length)
    destinations = page_map[rows, positions // fixture.page_size] * fixture.page_size + positions % fixture.page_size
    info = PageBatchInfo(
        slot_ids=hax.named(jnp.arange(batch_size, dtype=jnp.int32), "seq"),
        page_indices=hax.named(jnp.asarray(page_map), ("seq", "page")),
        seq_lens=hax.named(jnp.full((batch_size,), prefix_length, dtype=jnp.int32), "seq"),
        cu_q_lens=hax.named(jnp.arange(batch_size + 1, dtype=jnp.int32) * prefix_length, "seq"),
        num_seqs=jnp.asarray(batch_size, dtype=jnp.int32),
        new_token_dests=hax.named(jnp.asarray(destinations), "position"),
        page_size=fixture.page_size,
    )
    return PagedRequest(
        hax.named(jnp.asarray(np.asarray(fixture.tokens.array)[:, :prefix_length].reshape(-1)), "position"),
        info,
        hax.named(jnp.asarray(positions), "position"),
    )


def decode_batch(fixture: Fixture, *, page_map: np.ndarray, position: int) -> PagedRequest:
    """Pack the one-token step after a prepared prefix."""
    prefix = page_batch(fixture, page_map=page_map, prefix_length=position)
    batch_size = fixture.tokens.array.shape[0]
    positions = np.full((batch_size,), position, dtype=np.int32)
    destinations = page_map[np.arange(batch_size), positions // fixture.page_size] * fixture.page_size
    destinations += positions % fixture.page_size
    return PagedRequest(
        hax.named(fixture.tokens.array[:, position], "position"),
        dataclasses.replace(
            prefix.info,
            seq_lens=hax.named(jnp.full((batch_size,), position + 1, dtype=jnp.int32), "seq"),
            cu_q_lens=hax.named(jnp.arange(batch_size + 1, dtype=jnp.int32), "seq"),
            new_token_dests=hax.named(jnp.asarray(destinations), "position"),
        ),
        hax.named(jnp.asarray(positions), "position"),
    )


def _block(value: Any) -> Any:
    return jax.block_until_ready(value)


def _timed(fn: Callable[[], Any], *, warmup: int, measured: int) -> tuple[float, float, Any]:
    """Measure a synchronized first call including compilation, then steady calls."""
    if warmup < 0 or measured <= 0:
        raise ValueError("warmup must be non-negative and measured must be positive")
    started = time.perf_counter()
    result = _block(fn())
    first_call_time = time.perf_counter() - started
    for _ in range(warmup):
        _block(fn())
    samples = []
    for _ in range(measured):
        started = time.perf_counter()
        result = _block(fn())
        samples.append(time.perf_counter() - started)
    return first_call_time, float(np.mean(samples)), result


def teacher_call(model: SnowballLMHeadModel, tokens: hax.NamedArray) -> hax.NamedArray:
    return model(tokens)


def loss_and_grad(model: SnowballLMHeadModel, example: LmExample) -> Any:
    return eqx.filter_value_and_grad(lambda current: current.compute_next_token_loss(example).array)(model)


def paged_call(
    model: SnowballLMHeadModel, cache: Any, tokens: hax.NamedArray, info: PageBatchInfo, positions: hax.NamedArray
) -> Any:
    # The canonical public decode protocol returns (logits, cache).
    return model.decode(tokens, cache, info, positions)


def prepare_cache(
    model: SnowballLMHeadModel,
    fixture: Fixture,
    page_map: np.ndarray,
    *,
    prefix_length: int,
    decode_fn: Callable[..., Any],
) -> Any:
    """Allocate and synchronize the input prefix cache before decode timing."""
    cache = model.initial_cache(PageTableSpec(8, fixture.page_size, NUM_SLOTS), dtype=jnp.bfloat16)
    prefix = page_batch(fixture, page_map=page_map, prefix_length=prefix_length)
    _, cache = decode_fn(model, cache, prefix.tokens, prefix.info, prefix.positions)
    return _block(cache)


def _git_sha() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _fingerprint(value: Any) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(eqx.filter(value, eqx.is_array)):
        array = np.asarray(leaf)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _array_metrics(left: Any, right: Any) -> dict[str, Any]:
    left_array, right_array = np.asarray(left), np.asarray(right)
    if left_array.shape != right_array.shape or left_array.dtype != np.float32 or right_array.dtype != np.float32:
        raise ValueError("accuracy requires equal-shaped FP32 arrays")
    delta = np.abs(left_array - right_array)
    bit_mismatches = int(np.count_nonzero(left_array.view(np.uint32) != right_array.view(np.uint32)))
    finite = bool(np.isfinite(left_array).all() and np.isfinite(right_array).all())
    return {
        "status": "exact" if bit_mismatches == 0 else "different",
        "shape": list(left_array.shape),
        "elements": int(delta.size),
        "bitwise_fp32": bit_mismatches == 0,
        "bit_mismatch_count": bit_mismatches,
        "numerical_mismatch_count": int(np.count_nonzero(left_array != right_array)),
        "max_abs_error": float(np.max(delta)),
        "mean_abs_error": float(np.mean(delta)),
        "finite": finite,
    }


def _logit_metrics(left: Any, right: Any) -> dict[str, Any]:
    """Compare every vocabulary element and teacher-weighted distributions."""
    metrics = _array_metrics(left, right)
    left_array, right_array = np.asarray(left), np.asarray(right)
    # Use float64 host diagnostics to avoid adding metric work to measured kernels.
    left64, right64 = left_array.astype(np.float64), right_array.astype(np.float64)
    left64 -= np.max(left64, axis=-1, keepdims=True)
    right64 -= np.max(right64, axis=-1, keepdims=True)
    log_left = left64 - np.log(np.exp(left64).sum(axis=-1, keepdims=True))
    log_right = right64 - np.log(np.exp(right64).sum(axis=-1, keepdims=True))
    probability = np.exp(log_left)
    metrics["teacher_weighted_cross_entropy"] = float(np.mean(-np.sum(probability * log_right, axis=-1)))
    metrics["teacher_kl"] = float(np.mean(np.sum(probability * (log_left - log_right), axis=-1)))
    return metrics


def _gradient_metrics(left: Any, right: Any) -> dict[str, Any]:
    a = jax.tree_util.tree_leaves(eqx.filter(left, eqx.is_array))
    b = jax.tree_util.tree_leaves(eqx.filter(right, eqx.is_array))
    leaves = [_array_metrics(np.asarray(x).reshape(-1), np.asarray(y).reshape(-1)) for x, y in zip(a, b, strict=True)]
    return {
        "finite": all(leaf["finite"] for leaf in leaves),
        "bit_mismatch_count": sum(leaf["bit_mismatch_count"] for leaf in leaves),
        "max_abs_error": max(leaf["max_abs_error"] for leaf in leaves),
        "mean_abs_error": sum(leaf["mean_abs_error"] * leaf["elements"] for leaf in leaves)
        / sum(leaf["elements"] for leaf in leaves),
        "leaf_count": len(leaves),
    }


def _metadata(args: BenchmarkArgs, *, backend: str, mesh: Mesh) -> dict[str, Any]:
    versions = {}
    for package in ("jax", "jaxlib", "libtpu", "equinox", "haliax"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "schema": "snowball_zero_kl_performance.v2",
        "source_sha": args.source_sha if args.source_sha is not None else _git_sha(),
        "base_oid": BASE_OID,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture": FIXTURE if backend == "tpu" else "CPU-reference-four-slot8token-H128-D128",
        "model_seed": MODEL_SEED,
        "token_seed": TOKEN_SEED,
        "warmup": args.warmup,
        "measured": args.measured,
        "backend": backend,
        "python": platform.python_version(),
        "runtime_versions": versions,
        "host": platform.platform(),
        "devices": [
            {"id": device.id, "kind": device.device_kind, "platform": device.platform} for device in jax.devices()
        ],
        "mesh": dict(mesh.shape),
        "device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "process_count": jax.process_count(),
        "default_matmul_precision": str(jax.config.jax_default_matmul_precision),
        "jax_enable_x64": jax.config.jax_enable_x64,
        "jax_disable_jit": jax.config.jax_disable_jit,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "backend_env": {
            key: os.environ[key] for key in sorted(os.environ) if key.startswith("JAX_") or key.startswith("LIBTPU_")
        },
        "precision": {"parameter_dtype": "float32", "activation_dtype": "float32", "cache_dtype": "bfloat16"},
        "timing": "first call includes compilation when not already cached; synchronized steady mean; no donation",
        "accuracy_note": "full-vocabulary diagnostics outside timing; exact means FP32 bit equality",
    }


def _selector_metadata(config: SnowballConfig) -> dict[str, Any]:
    selectors = {field: getattr(config, field) for field in SELECTOR_FIELDS}
    fixed = config.shared_attention_implementation == "fixed_schedule"
    selectors["executed_teacher_attention"] = "fixed_schedule" if fixed else config.attention_implementation
    # The TPU paged dispatcher explicitly routes BF16 cache inputs to its reference math.
    selectors["executed_paged_attention"] = "fixed_schedule" if fixed else "reference (BF16 cache)"
    selectors["teacher_kv_dtype"] = "bfloat16" if fixed else "float32"
    selectors["paged_query_dtype"] = "float32"
    selectors["paged_kv_dtype"] = "bfloat16"
    selectors["tiles"] = {"compiler_selected": "unobserved"}
    if config.projection_implementation == "fixed_schedule":
        selectors["tiles"]["projection"] = {
            "m": 128,
            "k": 128,
            "n": 128,
            "source": "levanter/kernels/pallas/fixed_schedule_linear.py:_BLOCK_SIZE",
        }
    if config.rms_norm_implementation == "pallas_tpu":
        selectors["tiles"]["rms_norm"] = {
            "rows": 128,
            "hidden": config.hidden_dim,
            "source": "levanter/kernels/pallas/rms_norm/pallas_tpu.py:_ROW_BLOCK_SIZE",
        }
    if config.qk_rms_norm_implementation == "pallas_tpu":
        selectors["tiles"]["qk_rms_norm"] = {"rows": 128, "hidden": config.inferred_head_dim}
    if fixed:
        spec = FixedScheduleSpec()
        selectors["tiles"]["fixed_attention"] = dataclasses.asdict(spec)
    elif config.attention_implementation == "tpu_splash":
        selectors["tiles"]["teacher_splash"] = dataclasses.asdict(
            splash_attention_block_sizes(
                q_seq_len=SEQUENCE_LENGTH,
                kv_seq_len=SEQUENCE_LENGTH,
                q_seq_shards=1,
                kv_seq_shards=1,
            )
        )
    return selectors


def _measure(
    report: dict[str, Any],
    args: BenchmarkArgs,
    fn: Callable[[], Any],
    *,
    implementation: str,
    phase: str,
    token_count: int,
    page_map_index: int | None = None,
    position: int | None = None,
) -> Any:
    identity = {
        "implementation": implementation,
        "phase": phase,
        "page_map_index": page_map_index,
        "position": position,
        "backend": report["metadata"]["backend"],
        "devices": report["metadata"]["devices"],
        "mesh": report["metadata"]["mesh"],
        "selectors": report["configurations"][implementation]["selectors"],
        "input_tokens_shape": (
            [token_count] if phase in ("prefill", "cached_decode") else [NUM_SLOTS, token_count // NUM_SLOTS]
        ),
        "token_dtype": "int32",
        "logit_dtype": "float32",
        "cache_dtype": "bfloat16" if phase in ("prefill", "cached_decode") else None,
    }
    row = {
        **identity,
        "row_key": json.dumps(identity, sort_keys=True),
        "status": "running",
        "token_count": token_count,
    }
    report["rows"].append(row)
    try:
        first, steady, result = _timed(fn, warmup=args.warmup, measured=args.measured)
    except Exception as error:
        row.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            failure_kind="unsupported" if isinstance(error, NotImplementedError) else "execution_failure",
        )
        raise
    row.update(
        status="success",
        first_call_including_compile=first,
        steady_state_time=steady,
        throughput_tokens_per_second=token_count / steady,
    )
    return result


def _run_benchmark(
    args: BenchmarkArgs,
    *,
    backend: str,
    fixture: Fixture,
    mesh: Mesh,
    models: tuple[tuple[str, SnowballLMHeadModel], ...],
) -> dict[str, Any]:
    report = {
        "metadata": _metadata(args, backend=backend, mesh=mesh),
        "rows": [],
        "accuracy": {},
        "configurations": {},
        "status": "running",
    }
    try:
        _execute(report, args, fixture=fixture, mesh=mesh, models=models)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), accuracy_status="incomplete")
        if args.output is not None:
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        raise
    report["status"] = "success"
    report["accuracy_status"] = "observed"
    return report


def _execute(
    report: dict[str, Any],
    args: BenchmarkArgs,
    *,
    fixture: Fixture,
    mesh: Mesh,
    models: tuple[tuple[str, SnowballLMHeadModel], ...],
) -> None:
    parameter_hashes = {}
    for name, model in models:
        parameter_hashes[name] = _fingerprint(model)
        report["configurations"][name] = {
            "config": dataclasses.asdict(model.config),
            "selectors": _selector_metadata(model.config),
            "parameter_sha256": parameter_hashes[name],
        }
    if len(set(parameter_hashes.values())) != 1:
        raise AssertionError("benchmark arms must have identical parameter bytes")
    report["parameter_identity"] = {"all_equal_bytes": True, "sha256": parameter_hashes}
    report["token_sha256"] = _fingerprint(fixture.tokens)
    report["loss_weight_sha256"] = _fingerprint(fixture.example.loss_weight)
    report["page_maps"] = [page_map.tolist() for page_map in fixture.page_maps]
    teacher_fn = eqx.filter_jit(teacher_call)
    loss_fn = eqx.filter_jit(loss_and_grad)
    # Explicit dynamic inputs prevent closed-over token/cache arrays becoming constants.
    paged_fn = eqx.filter_jit(paged_call, donate="none")
    teachers, losses, gradients, prefills, decodes = {}, {}, {}, {}, {}
    batch_size, sequence_length = fixture.tokens.array.shape
    positions = (fixture.page_size - 1, fixture.page_size, sequence_length - 1)
    with (
        jax.set_mesh(mesh),
        hax.axis_mapping({"batch": ("replica_dcn", "data", "expert"), "position": None, "vocab": "model"}),
    ):
        fixture = dataclasses.replace(
            fixture, tokens=hax.auto_sharded(fixture.tokens), example=hax.auto_sharded(fixture.example)
        )
        _block(tuple(model for _, model in models))
        _block(fixture.tokens)
        _block(fixture.example)
        for name, model in models:
            teachers[name] = _measure(
                report,
                args,
                lambda: teacher_fn(model, fixture.tokens),
                implementation=name,
                phase="teacher_forward",
                token_count=batch_size * sequence_length,
            )
            loss, gradient = _measure(
                report,
                args,
                lambda: loss_fn(model, fixture.example),
                implementation=name,
                phase="teacher_loss_grad",
                token_count=batch_size * sequence_length,
            )
            losses[name], gradients[name] = float(np.asarray(loss)), gradient
            for map_index, page_map in enumerate(fixture.page_maps):
                full = page_batch(fixture, page_map=page_map, prefix_length=sequence_length)
                empty_cache = _block(
                    model.initial_cache(
                        PageTableSpec(8, fixture.page_size, NUM_SLOTS),
                        dtype=jnp.bfloat16,
                    )
                )
                _block((full.tokens, full.info, full.positions))
                prefill_logits, _ = _measure(
                    report,
                    args,
                    lambda: paged_fn(model, empty_cache, full.tokens, full.info, full.positions),
                    implementation=name,
                    phase="prefill",
                    token_count=batch_size * sequence_length,
                    page_map_index=map_index,
                )
                prefills[name, map_index] = prefill_logits.array.reshape(batch_size, sequence_length, VOCAB_SIZE)
                for position in positions:
                    # Prefix construction and prefill are outside the hot one-token region.
                    cache = prepare_cache(model, fixture, page_map, prefix_length=position, decode_fn=paged_fn)
                    input_cache_sha256 = _fingerprint(cache)
                    request = decode_batch(fixture, page_map=page_map, position=position)
                    _block((request.tokens, request.info, request.positions))
                    logits, _ = _measure(
                        report,
                        args,
                        lambda: paged_fn(model, cache, request.tokens, request.info, request.positions),
                        implementation=name,
                        phase="cached_decode",
                        token_count=batch_size,
                        page_map_index=map_index,
                        position=position,
                    )
                    input_cache_unchanged = _fingerprint(cache) == input_cache_sha256
                    report["rows"][-1].update(
                        input_cache_sha256=input_cache_sha256, input_cache_unchanged=input_cache_unchanged
                    )
                    if not input_cache_unchanged:
                        raise AssertionError("cached-decode timing mutated its input prefix cache")
                    decodes[name, map_index, position] = logits.array

    # Diagnostic slicing must run on host arrays, outside Explicit-sharded device indexing.
    teacher_logits = {name: np.asarray(value.array) for name, value in teachers.items()}
    prefills = {key: np.asarray(value) for key, value in prefills.items()}
    decodes = {key: np.asarray(value) for key, value in decodes.items()}
    accuracy = report["accuracy"]
    for name, _ in models:
        teacher = teacher_logits[name]
        accuracy[name] = {
            "teacher_weighted_next_token_loss": losses[name],
            "gradient_finite": all(
                bool(np.isfinite(np.asarray(x)).all())
                for x in jax.tree_util.tree_leaves(eqx.filter(gradients[name], eqx.is_array))
            ),
            "serving": [],
        }
        for map_index in range(len(fixture.page_maps)):
            accuracy[name]["serving"].append(
                {
                    "page_map_index": map_index,
                    "teacher_vs_full_prefill": _logit_metrics(teacher, prefills[name, map_index]),
                    "decode": [
                        {
                            "position": position,
                            "teacher_vs_cached_decode": _logit_metrics(
                                teacher[:, position],
                                decodes[name, map_index, position],
                            ),
                            "prefill_vs_cached_decode": _logit_metrics(
                                prefills[name, map_index][:, position],
                                decodes[name, map_index, position],
                            ),
                        }
                        for position in positions
                    ],
                }
            )
    left, right = (name for name, _ in models)
    accuracy["cross_arm"] = {
        "teacher": _logit_metrics(teacher_logits[left], teacher_logits[right]),
        "loss_abs_error": abs(losses[left] - losses[right]),
        "gradient": _gradient_metrics(gradients[left], gradients[right]),
        "serving": [
            {
                "page_map_index": index,
                "prefill": _logit_metrics(prefills[left, index], prefills[right, index]),
                "decode": [
                    {
                        "position": position,
                        **_logit_metrics(decodes[left, index, position], decodes[right, index, position]),
                    }
                    for position in positions
                ],
            }
            for index in range(len(fixture.page_maps))
        ],
    }
    if not all(accuracy[name]["gradient_finite"] and np.isfinite(losses[name]) for name, _ in models):
        raise FloatingPointError("non-finite teacher loss or gradient")
    if not all(
        np.isfinite(np.asarray(array)).all()
        for array in (
            *teacher_logits.values(),
            *prefills.values(),
            *decodes.values(),
        )
    ):
        raise FloatingPointError("non-finite teacher or serving logits")


def run_tpu_benchmark(args: BenchmarkArgs) -> dict[str, Any]:
    """Run the qualified four-device TPU fixture without fallback substitution."""
    if (
        jax.default_backend() != "tpu"
        or jax.device_count() != NUM_SLOTS
        or jax.local_device_count() != NUM_SLOTS
        or jax.process_count() != 1
        or any("tpu v4" not in device.device_kind.lower() for device in jax.devices())
    ):
        raise RuntimeError("the qualified comparison requires exactly four local TPUv4 devices in one process")
    mesh = compact_grug_mesh(expert_axis_size=1)
    with jax.set_mesh(mesh):
        models = tuple(
            (
                name,
                SnowballLMHeadModel.init(
                    Axis("vocab", VOCAB_SIZE),
                    qualified_config(implementation=implementation, backend="tpu"),
                    key=jax.random.key(MODEL_SEED),
                ),
            )
            for name, implementation in (("native_splash", "native"), ("fixed_schedule", "fixed_schedule"))
        )
    return _run_benchmark(args, backend="tpu", fixture=make_fixture(), mesh=mesh, models=models)


def run_cpu_reference(args: BenchmarkArgs) -> dict[str, Any]:
    """Exercise real teacher/prefill/decode helpers on small CPU reference arms."""
    if jax.default_backend() != "cpu":
        raise RuntimeError("CPU reference controls require a CPU runtime")
    mesh = compact_grug_mesh(expert_axis_size=1)
    with jax.set_mesh(mesh):
        models = []
        for name, norm in (("cpu_native_reference", "native"), ("cpu_batch_invariant_norm_reference", "reference")):
            config = dataclasses.replace(
                qualified_config(implementation="native", backend="cpu"),
                max_seq_len=8,
                rms_norm_implementation=norm,
                qk_rms_norm_implementation=norm,
            )
            models.append(
                (
                    name,
                    SnowballLMHeadModel.init(
                        Axis("vocab", VOCAB_SIZE),
                        config,
                        key=jax.random.key(MODEL_SEED),
                    ),
                )
            )
    report = _run_benchmark(
        args, backend="cpu", fixture=make_fixture(sequence_length=8, page_size=4), mesh=mesh, models=tuple(models)
    )
    report["tpu_qualification"] = {
        "status": "unexecuted",
        "reason": "CPU reference controls do not execute TPU kernels",
    }
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "cpu", "tpu"), default="auto")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measured", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-sha", help="Exact source commit when executing a snapshot without Git metadata")
    parsed = parser.parse_args(argv)
    args = BenchmarkArgs(**vars(parsed))
    backend = jax.default_backend() if args.backend == "auto" else args.backend
    if backend not in ("cpu", "tpu"):
        raise ValueError("this benchmark supports CPU reference controls or qualified TPU execution")
    report = run_cpu_reference(args) if backend == "cpu" else run_tpu_benchmark(args)
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.write_text(output)


if __name__ == "__main__":
    main()
