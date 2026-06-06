# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Qwen3 8B decode-heavy TPU inference against vLLM TPU.

This is the phase-1 parity harness from
`.agents/projects/levanter_tpu_inference_parity/design.md`. It intentionally
uses the OpenAI completions endpoint for both engines so the first signal is
decode throughput rather than chat-template rendering.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import gzip
import importlib.metadata
import json
import logging
import os
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jmp
import requests

import haliax as hax

from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.inference.engine import InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams, SequenceTable
from levanter.inference.page_table import PageTable
from levanter.inference.tpu_kernels import TpuPagedAttentionBackend, TpuPagedAttentionConfig
from levanter.layers.attention import AttentionMask
from levanter.layers.sampler import SamplerTopKMode
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from levanter.tokenizers import load_tokenizer
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import sharded_tree_size
from levanter.utils.mesh import MeshConfig

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_MATRIX = "rl_decode_qwen3_8b_v1"
DENSE_EXPANSION_MATRIX = "dense_qwen3_8b_v2"
DEFAULT_WARMUP_ROUNDS = 2
LEVANTER_COMPILED_SHAPE_BUCKETS = 2
PARITY_DECODE_RATIO_TARGET = 0.85
HLO_SUMMARY_MAX_EXAMPLES_PER_FILE = 80
HLO_SUMMARY_PATTERNS = {
    "collective": ("all-gather", "all_gather", "all-reduce", "all_reduce", "all-to-all", "all_to_all"),
    "rng_sampling": ("rng", "random", "threefry", "logistic", "exponential", "categorical"),
    "sort_or_topk": ("stablehlo.sort", "argsort", "topk", "top-k", "top_k"),
    "logprob": ("log-softmax", "log_softmax", "logsumexp", "reduce-logsumexp"),
    "custom_call": ("custom-call", "custom_call", "ragged_paged_attention", "pallas_call"),
}
PROFILED_BACKEND_PREFIX = "levanter:"
REFERENCE_LOGIT_ARTIFACT_JSON = "levanter_reference_logits.json"
REFERENCE_LOGIT_ARTIFACT_MD = "levanter_reference_logits.md"
REFERENCE_LOGIT_TOP_KS = (1, 10, 100, 1000, 4096)
REFERENCE_LOGIT_HISTOGRAM_BOUNDS = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
REFERENCE_LOGIT_CACHE_DTYPE_POLICIES = ("auto", "default", "bfloat16", "float32")
VLLM_STARTUP_ENV_KEYS = (
    "JAX_COMPILATION_CACHE_DIR",
    "JAX_ENABLE_COMPILATION_CACHE",
    "JAX_PLATFORMS",
    "LIBTPU_INIT_ARGS",
    "MODEL_IMPL_TYPE",
    "TPU_CHIPS_PER_HOST_BOUNDS",
    "TPU_HOST_BOUNDS",
    "TPU_MIN_LOG_LEVEL",
    "TPU_STDERR_LOG_LEVEL",
    "TPU_VISIBLE_DEVICES",
    "VLLM_XLA_CACHE_PATH",
    "XLA_FLAGS",
)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    active_sequences: int
    input_tokens: int
    output_tokens: int
    n: int = 1

    @property
    def request_count(self) -> int:
        if self.active_sequences % self.n != 0:
            raise ValueError(f"Case {self.name} active_sequences must be divisible by n")
        return self.active_sequences // self.n


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_name: str
    backend: str
    request_count: int
    active_sequences: int
    n: int
    input_tokens_target: int
    output_tokens_target: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    compile_including_seconds: float | None
    steady_state_seconds: float
    request_latency_ms_p50: float
    request_latency_ms_p90: float
    ttft_ms_p50: float | None
    decode_tokens_per_second: float
    total_tokens_per_second: float
    hbm_used_bytes: int | None
    compiled_shape_count: int | None
    prefill_admissions: int | None = None
    prefill_prompt_tokens_per_admission: list[int] | None = None
    prefill_seconds_per_admission: list[float] | None = None
    decode_seconds_per_iteration: list[float] | None = None
    decode_device_seconds_per_iteration: list[float] | None = None
    decode_host_seconds_per_iteration: list[float] | None = None
    decode_submit_seconds_per_iteration: list[float] | None = None
    decode_extract_seconds_per_iteration: list[float] | None = None
    decode_tokens_per_iteration: list[int] | None = None
    prefill_drain_seconds_per_iteration: list[float] | None = None
    prefill_drain_tokens_per_iteration: list[int] | None = None
    generation_seconds_per_iteration: list[float] | None = None
    generation_host_seconds_per_iteration: list[float] | None = None
    generation_tokens_per_iteration: list[int] | None = None


@dataclass(frozen=True, slots=True)
class StressResult:
    case_name: str
    backend: str
    concurrent_requests: int
    active_sequences: int
    n: int
    input_tokens_target: int
    output_tokens_target: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    timeout_requests: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    load_seconds: float
    wall_clock_seconds: float
    steady_decode_tokens_per_second: float
    wall_clock_decode_tokens_per_second: float
    wall_clock_total_tokens_per_second: float
    request_latency_ms_p50: float
    request_latency_ms_p90: float
    request_latency_ms_p99: float
    request_latency_ms_max: float
    hbm_used_bytes: int | None
    compiled_shape_count: int | None
    max_request_queue_depth: int | None
    max_batch_queue_depth: int | None
    page_size: int | None
    max_pages: int | None
    retry_errors: int
    error_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class ServerHandle:
    name: str
    base_url: str
    model_id: str
    close: Callable[[], None]
    command: list[str] | None = None
    hbm_used_bytes: int | None = None
    compiled_shape_count: int | None = None
    supports_seed: bool = True
    metrics_snapshot: Callable[[], dict[str, Any]] | None = None
    diagnose_without_lm_head: (
        Callable[
            [BenchmarkCase, str, int, bool, int, float, float, float],
            CaseResult,
        ]
        | None
    ) = None
    diagnose_with_lm_head_no_sampling: (
        Callable[
            [BenchmarkCase, str, int, bool, int, float, float, float],
            CaseResult,
        ]
        | None
    ) = None


@dataclass(frozen=True, slots=True)
class ReferenceLogitCheckResult:
    prompt_index: int
    case_names: list[str]
    prompt_token_ids: list[int]
    decode_backend: str
    kv_cache_dtype: str
    kv_cache_dtype_policy: str
    reference_hidden_dtype: str
    decode_hidden_dtype: str
    reference_logits_dtype: str
    decode_logits_dtype: str
    tpu_inference_out_dtype: str | None
    preserve_attention_output_dtype: bool
    positions: int
    vocab_size: int
    hidden_max_abs_error: float
    hidden_mean_abs_error: float
    hidden_rms_abs_error: float
    max_abs_error: float
    max_abs_error_if_reference_rounded_to_decode_dtype: float
    residual_max_abs_error_after_reference_rounding: float
    max_rel_error: float
    mean_abs_error: float
    rms_abs_error: float
    abs_error_p50: float
    abs_error_p90: float
    abs_error_p99: float
    abs_error_p999: float
    top1_agreement: float
    top_k_diagnostics: list[dict[str, float | int]]
    abs_error_histogram: list[dict[str, float | int | None]]
    passed: bool
    single_token_direct_hidden_dtype: str | None = None
    single_token_direct_vs_reference_hidden_max_abs_error: float | None = None
    single_token_direct_vs_decode_hidden_max_abs_error: float | None = None
    single_token_direct_vs_reference_logits_max_abs_error: float | None = None
    single_token_direct_vs_decode_logits_max_abs_error: float | None = None


def matrix_cases(name: str) -> list[BenchmarkCase]:
    if name == DEFAULT_MATRIX:
        return [
            BenchmarkCase("decode_b8_i1_o128_n1", active_sequences=8, input_tokens=1, output_tokens=128, n=1),
            BenchmarkCase("decode_b8_i1_o512_n1", active_sequences=8, input_tokens=1, output_tokens=512, n=1),
            BenchmarkCase("decode_b32_i1_o128_n1", active_sequences=32, input_tokens=1, output_tokens=128, n=1),
            BenchmarkCase("decode_b32_i1_o512_n1", active_sequences=32, input_tokens=1, output_tokens=512, n=1),
            BenchmarkCase("decode_b128_i1_o128_n1", active_sequences=128, input_tokens=1, output_tokens=128, n=1),
            BenchmarkCase("decode_b128_i1_o512_n1", active_sequences=128, input_tokens=1, output_tokens=512, n=1),
            BenchmarkCase("decode_b32_i1_o128_n4", active_sequences=32, input_tokens=1, output_tokens=128, n=4),
            BenchmarkCase("decode_b32_i1_o512_n4", active_sequences=32, input_tokens=1, output_tokens=512, n=4),
        ]
    if name == DENSE_EXPANSION_MATRIX:
        return [
            BenchmarkCase("decode_b32_i1_o128_n1", active_sequences=32, input_tokens=1, output_tokens=128, n=1),
            BenchmarkCase("decode_b32_i1_o512_n1", active_sequences=32, input_tokens=1, output_tokens=512, n=1),
            BenchmarkCase("decode_b32_i1_o2048_n1", active_sequences=32, input_tokens=1, output_tokens=2048, n=1),
            BenchmarkCase("decode_b32_i1_o128_n4", active_sequences=32, input_tokens=1, output_tokens=128, n=4),
            BenchmarkCase("decode_b32_i1_o512_n4", active_sequences=32, input_tokens=1, output_tokens=512, n=4),
            BenchmarkCase("decode_b32_i1_o2048_n4", active_sequences=32, input_tokens=1, output_tokens=2048, n=4),
            BenchmarkCase("mixed_b32_i128_o512_n1", active_sequences=32, input_tokens=128, output_tokens=512, n=1),
            BenchmarkCase("mixed_b32_i512_o128_n1", active_sequences=32, input_tokens=512, output_tokens=128, n=1),
            BenchmarkCase("mixed_b32_i512_o512_n1", active_sequences=32, input_tokens=512, output_tokens=512, n=1),
            BenchmarkCase("mixed_b32_i512_o512_n4", active_sequences=32, input_tokens=512, output_tokens=512, n=4),
            BenchmarkCase("mixed_b8_i512_o512_n1", active_sequences=8, input_tokens=512, output_tokens=512, n=1),
            BenchmarkCase("prefill_b8_i2048_o128_n1", active_sequences=8, input_tokens=2048, output_tokens=128, n=1),
            BenchmarkCase("prefill_b8_i2048_o128_n4", active_sequences=8, input_tokens=2048, output_tokens=128, n=4),
            BenchmarkCase("pressure_b128_i1_o128_n1", active_sequences=128, input_tokens=1, output_tokens=128, n=1),
            BenchmarkCase("pressure_b128_i1_o128_n4", active_sequences=128, input_tokens=1, output_tokens=128, n=4),
            BenchmarkCase("churn_b64_i128_o512_n1", active_sequences=64, input_tokens=128, output_tokens=512, n=1),
            BenchmarkCase("churn_b64_i128_o512_n4", active_sequences=64, input_tokens=128, output_tokens=512, n=4),
        ]
    raise ValueError(f"Unknown matrix {name!r}. Supported: {DEFAULT_MATRIX}, {DENSE_EXPANSION_MATRIX}")


def find_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _poll_json(url: str, *, timeout: float) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"{url} did not become ready within {timeout}s; last_error={last_error}")


def _poll_json_while_process_alive(url: str, *, timeout: float, process: subprocess.Popen) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"process exited with code {return_code} before {url} became ready; last_error={last_error}"
            )
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"{url} did not become ready within {timeout}s; last_error={last_error}")


def _tail_text_file(path: Path, *, max_bytes: int = 12000) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes), os.SEEK_SET)
        return f.read().decode("utf-8", errors="replace").strip()


def _vllm_startup_error_message(
    *,
    log_dir: Path,
    cmd: list[str],
    process_return_code: int | None,
    cause: Exception,
    startup_snapshot: dict[str, Any] | None = None,
) -> str:
    process_state = (
        f"vLLM process exited with code {process_return_code}"
        if process_return_code is not None
        else "vLLM process was still running when startup timed out"
    )
    parts = [
        f"vLLM failed to start. Logs: {log_dir}. Command: {' '.join(cmd)}",
        f"{process_state}. Startup error: {cause}",
    ]
    if startup_snapshot is not None:
        parts.append(f"startup snapshot:\n{json.dumps(startup_snapshot, indent=2, sort_keys=True)}")
    stderr_tail = _tail_text_file(log_dir / "stderr.log")
    if stderr_tail:
        parts.append(f"stderr tail:\n{stderr_tail}")
    stdout_tail = _tail_text_file(log_dir / "stdout.log")
    if stdout_tail:
        parts.append(f"stdout tail:\n{stdout_tail}")
    return "\n\n".join(parts)


def _vllm_startup_snapshot(env: dict[str, str]) -> dict[str, Any]:
    return {
        "runtime": _runtime_env_snapshot(include_jax_devices=False),
        "env": {key: env.get(key) for key in VLLM_STARTUP_ENV_KEYS},
    }


def _prompt_for_token_count(tokenizer, target_tokens: int) -> tuple[str, int]:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    seed_texts = ["Hello", "x", "The", "Q"]
    for text in seed_texts:
        if len(tokenizer.encode(text, add_special_tokens=False)) == target_tokens:
            return text, target_tokens

    base = "The quick brown fox jumps over the lazy dog. "
    text = base
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text += base

    # Greedy token-level trim, then decode back to text. Re-encoding can differ
    # for some tokenizers, so record the actual prompt length separately.
    token_ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
    actual = len(tokenizer.encode(prompt, add_special_tokens=False))
    return prompt, actual


def write_prompt_corpus(
    output_dir: Path,
    cases: list[BenchmarkCase],
    prompts: dict[str, tuple[str, int]],
    tokenizer: Any,
) -> dict[str, Any]:
    """Write the exact prompt corpus used by the benchmark."""
    records = []
    for case in cases:
        prompt, prompt_token_count = prompts[case.name]
        token_ids = [int(token_id) for token_id in tokenizer.encode(prompt, add_special_tokens=False)]
        if len(token_ids) != prompt_token_count:
            raise ValueError(
                f"Prompt token count drift for {case.name}: recorded {prompt_token_count}, encoded {len(token_ids)}"
            )
        records.append(
            {
                "case_name": case.name,
                "active_sequences": case.active_sequences,
                "input_tokens_target": case.input_tokens,
                "output_tokens_target": case.output_tokens,
                "n": case.n,
                "request_count": case.request_count,
                "prompt": prompt,
                "prompt_token_ids": token_ids,
                "prompt_tokens": prompt_token_count,
            }
        )
    corpus = {"prompts": records}
    with open(output_dir / "prompt_corpus.json", "w") as f:
        json.dump(corpus, f, indent=2, sort_keys=True)
    return corpus


def _deduplicated_prompt_token_groups(
    cases: list[BenchmarkCase],
    prompts: dict[str, tuple[str, int]],
    tokenizer: Any,
    *,
    max_prompts: int | None,
) -> list[tuple[list[str], list[int]]]:
    groups: dict[tuple[int, ...], list[str]] = {}
    for case in cases:
        prompt, prompt_token_count = prompts[case.name]
        token_ids = tuple(int(token_id) for token_id in tokenizer.encode(prompt, add_special_tokens=False))
        if len(token_ids) != prompt_token_count:
            raise ValueError(
                f"Prompt token count drift for {case.name}: recorded {prompt_token_count}, encoded {len(token_ids)}"
            )
        groups.setdefault(token_ids, []).append(case.name)

    items = [(case_names, list(token_ids)) for token_ids, case_names in groups.items()]
    if max_prompts is not None:
        return items[:max_prompts]
    return items


def _abs_error_histogram(flat_abs_error: list[float]) -> list[dict[str, float | int | None]]:
    buckets: list[dict[str, float | int | None]] = []
    total = len(flat_abs_error)
    lower_bounds = REFERENCE_LOGIT_HISTOGRAM_BOUNDS
    upper_bounds: tuple[float | None, ...] = (*REFERENCE_LOGIT_HISTOGRAM_BOUNDS[1:], None)

    for lower, upper in zip(lower_bounds, upper_bounds, strict=True):
        if upper is None:
            count = sum(1 for value in flat_abs_error if value >= lower)
        else:
            count = sum(1 for value in flat_abs_error if lower <= value < upper)
        buckets.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "fraction": count / total if total else 0.0,
            }
        )

    return buckets


def _reference_logit_error_metrics(reference: jax.Array, candidate: jax.Array) -> dict[str, Any]:
    reference_dtype = jnp.dtype(reference.dtype)
    candidate_dtype = jnp.dtype(candidate.dtype)
    reference_f32 = jnp.asarray(reference, dtype=jnp.float32)
    candidate_f32 = jnp.asarray(candidate, dtype=jnp.float32)
    abs_error = jnp.abs(reference_f32 - candidate_f32)
    rel_error = abs_error / jnp.maximum(jnp.abs(reference_f32), jnp.asarray(1e-6, dtype=jnp.float32))
    flat_abs_error = [float(value) for value in jax.device_get(jnp.ravel(abs_error))]
    vocab_size = int(reference_f32.shape[-1])
    positions = max(1, int(abs_error.size // vocab_size))

    reference_logprobs = jax.nn.log_softmax(reference_f32, axis=-1)
    candidate_logprobs = jax.nn.log_softmax(candidate_f32, axis=-1)
    top_k_diagnostics: list[dict[str, float | int]] = []
    top1_agreement = 0.0

    for requested_k in REFERENCE_LOGIT_TOP_KS:
        k = min(requested_k, vocab_size)
        if top_k_diagnostics and top_k_diagnostics[-1]["k"] == k:
            continue

        _, reference_top_indices = jax.lax.top_k(reference_f32, k)
        _, candidate_top_indices = jax.lax.top_k(candidate_f32, k)
        reference_top_candidate_logits = jnp.take_along_axis(candidate_f32, reference_top_indices, axis=-1)
        reference_top_logits = jnp.take_along_axis(reference_f32, reference_top_indices, axis=-1)
        reference_top_abs_error = jnp.abs(reference_top_logits - reference_top_candidate_logits)

        reference_top_candidate_logprobs = jnp.take_along_axis(candidate_logprobs, reference_top_indices, axis=-1)
        reference_top_logprobs = jnp.take_along_axis(reference_logprobs, reference_top_indices, axis=-1)
        reference_top_logprob_abs_error = jnp.abs(reference_top_logprobs - reference_top_candidate_logprobs)

        overlap = jnp.any(reference_top_indices[..., :, None] == candidate_top_indices[..., None, :], axis=-1)
        if k == 1:
            top1_agreement = float(jax.device_get(jnp.mean(overlap.astype(jnp.float32))))

        top_k_diagnostics.append(
            {
                "k": k,
                "max_abs_error": float(jax.device_get(jnp.max(reference_top_abs_error))),
                "mean_abs_error": float(jax.device_get(jnp.mean(reference_top_abs_error))),
                "max_logprob_abs_error": float(jax.device_get(jnp.max(reference_top_logprob_abs_error))),
                "mean_logprob_abs_error": float(jax.device_get(jnp.mean(reference_top_logprob_abs_error))),
                "overlap_fraction": float(jax.device_get(jnp.sum(overlap) / (positions * k))),
            }
        )

    return {
        "reference_logits_dtype": str(reference_dtype),
        "decode_logits_dtype": str(candidate_dtype),
        "max_abs_error": float(jax.device_get(jnp.max(abs_error))),
        "max_abs_error_if_reference_rounded_to_decode_dtype": float(
            jax.device_get(jnp.max(jnp.abs(reference_f32 - reference_f32.astype(candidate_dtype).astype(jnp.float32))))
        ),
        "residual_max_abs_error_after_reference_rounding": float(
            jax.device_get(jnp.max(jnp.abs(candidate_f32 - reference_f32.astype(candidate_dtype).astype(jnp.float32))))
        ),
        "max_rel_error": float(jax.device_get(jnp.max(rel_error))),
        "mean_abs_error": float(jax.device_get(jnp.mean(abs_error))),
        "rms_abs_error": float(jax.device_get(jnp.sqrt(jnp.mean(jnp.square(abs_error))))),
        "abs_error_p50": _percentile(flat_abs_error, 0.50),
        "abs_error_p90": _percentile(flat_abs_error, 0.90),
        "abs_error_p99": _percentile(flat_abs_error, 0.99),
        "abs_error_p999": _percentile(flat_abs_error, 0.999),
        "top1_agreement": top1_agreement,
        "top_k_diagnostics": top_k_diagnostics,
        "abs_error_histogram": _abs_error_histogram(flat_abs_error),
    }


def _array_error_metrics(reference: jax.Array, candidate: jax.Array) -> dict[str, Any]:
    reference_dtype = jnp.dtype(reference.dtype)
    candidate_dtype = jnp.dtype(candidate.dtype)
    reference_f32 = jnp.asarray(reference, dtype=jnp.float32)
    candidate_f32 = jnp.asarray(candidate, dtype=jnp.float32)
    abs_error = jnp.abs(reference_f32 - candidate_f32)
    return {
        "reference_dtype": str(reference_dtype),
        "candidate_dtype": str(candidate_dtype),
        "max_abs_error": float(jax.device_get(jnp.max(abs_error))),
        "mean_abs_error": float(jax.device_get(jnp.mean(abs_error))),
        "rms_abs_error": float(jax.device_get(jnp.sqrt(jnp.mean(jnp.square(abs_error))))),
    }


def _unrolled_full_attention_hidden(
    model: Qwen3LMHeadModel,
    input_ids: hax.NamedArray,
    attn_mask: AttentionMask | hax.NamedArray | None,
    pos_ids: hax.NamedArray,
) -> hax.NamedArray:
    """Run full-sequence transformer layers in the same explicit layer order as paged decode."""
    x = model.embeddings.embed(input_ids)
    for layer in model.transformer.layers.unstacked():
        x = layer(x, mask=attn_mask, key=None, pos_ids=pos_ids)
    return model.transformer.norm(x)


def _single_token_direct_attention_hidden(
    model: Qwen3LMHeadModel,
    input_ids: hax.NamedArray,
    pos_ids: hax.NamedArray,
) -> hax.NamedArray | None:
    """Run a one-token transformer pass where self-attention is replaced by the exact V path."""
    if input_ids.axis_size("position") != 1:
        return None

    x = model.embeddings.embed(input_ids)
    for layer in model.transformer.layers.unstacked():
        residual = x
        x_norm = layer.input_layernorm(x)
        _, _, value = layer.self_attn._compute_qkv(x_norm, key=None, pos_ids=pos_ids)
        attn_tokens = value.broadcast_axis(layer.self_attn.config.QHeadsPerGroup)

        gate_proj = getattr(layer.self_attn, "gate_proj", None)
        if gate_proj is not None:
            gate = hax.nn.sigmoid(gate_proj(x_norm))
            gate = gate.rename({"gate_size": "head_size"})
            attn_tokens = attn_tokens * gate

        attn_output = attn_tokens.flatten_axes(("kv_head", "q_heads_per_group"), "heads")
        attn_output = attn_output.astype(x_norm.dtype)
        attn_output = layer.self_attn.o_proj(attn_output, key=None)

        post_attn_layernorm = getattr(layer, "post_attn_layernorm", None)
        if post_attn_layernorm is not None:
            attn_output = post_attn_layernorm(attn_output)
        x = residual + attn_output

        residual = x
        x_norm = layer.post_attention_layernorm(x)
        mlp_output = layer.mlp(x_norm, key=None)
        post_mlp_layernorm = getattr(layer, "post_mlp_layernorm", None)
        if post_mlp_layernorm is not None:
            mlp_output = post_mlp_layernorm(mlp_output)
        x = residual + mlp_output

    return model.transformer.norm(x)


def _reference_logit_cache_dtype(tpu_paged_attention: TpuPagedAttentionConfig, default_dtype: jnp.dtype) -> jnp.dtype:
    backend = tpu_paged_attention.backend
    if isinstance(backend, TpuPagedAttentionBackend | str):
        backends = (TpuPagedAttentionBackend(backend),)
    else:
        backends = tuple(TpuPagedAttentionBackend(item) for item in backend)
    if TpuPagedAttentionBackend.AUTO in backends or TpuPagedAttentionBackend.TPU_INFERENCE in backends:
        return jnp.bfloat16
    return default_dtype


def _reference_logit_cache_dtype_for_policy(
    tpu_paged_attention: TpuPagedAttentionConfig,
    default_dtype: jnp.dtype,
    policy: str,
) -> jnp.dtype:
    if policy == "auto":
        return _reference_logit_cache_dtype(tpu_paged_attention, default_dtype)
    if policy == "default":
        return default_dtype
    if policy == "bfloat16":
        return jnp.bfloat16
    if policy == "float32":
        return jnp.float32
    raise ValueError(f"Unknown reference-logit cache dtype policy: {policy}")


def _reference_logit_backend_config(
    backend: TpuPagedAttentionBackend | str,
    base_config: TpuPagedAttentionConfig,
) -> TpuPagedAttentionConfig:
    backend = TpuPagedAttentionBackend(backend)
    return TpuPagedAttentionConfig(
        backend=backend,
        fail_on_reference_fallback=backend != TpuPagedAttentionBackend.REFERENCE,
        tpu_inference_out_dtype=base_config.tpu_inference_out_dtype,
        preserve_attention_output_dtype=base_config.preserve_attention_output_dtype,
    )


def check_levanter_reference_logits(
    *,
    trainer: TrainerConfig,
    model: Qwen3LMHeadModel,
    tokenizer: Any,
    cases: list[BenchmarkCase],
    prompts: dict[str, tuple[str, int]],
    tpu_paged_attention: TpuPagedAttentionConfig,
    decode_backends: list[TpuPagedAttentionBackend] | None,
    cache_dtype_policies: list[str],
    default_cache_dtype: jnp.dtype,
    max_model_len: int,
    atol: float,
    rtol: float,
    max_prompts: int | None,
) -> list[ReferenceLogitCheckResult]:
    """Compare paged decode logits against full causal Levanter logits for the fixed prompt corpus."""

    prompt_groups = _deduplicated_prompt_token_groups(cases, prompts, tokenizer, max_prompts=max_prompts)
    results: list[ReferenceLogitCheckResult] = []
    max_pages_per_seq = (max_model_len + 127) // 128
    if decode_backends is None:
        backend_configs = [("configured", tpu_paged_attention)]
    else:
        backend_configs = [
            (backend.value, _reference_logit_backend_config(backend, tpu_paged_attention))
            for backend in decode_backends
        ]

    with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):
        for prompt_index, (case_names, prompt_token_ids) in enumerate(prompt_groups):
            if not prompt_token_ids:
                raise ValueError(f"Prompt group {prompt_index} is empty")

            Pos = hax.Axis("position", len(prompt_token_ids))
            input_ids = hax.named(jnp.asarray(prompt_token_ids, dtype=jnp.int32), Pos)
            pos_ids = hax.arange(Pos, dtype=jnp.int32)

            reference_hidden = _unrolled_full_attention_hidden(
                model,
                input_ids,
                AttentionMask.causal(),
                pos_ids,
            )
            reference_logits = model.lm_head_logits(reference_hidden)
            direct_hidden = _single_token_direct_attention_hidden(model, input_ids, pos_ids)
            direct_logits = model.lm_head_logits(direct_hidden) if direct_hidden is not None else None

            for decode_backend_name, backend_config in backend_configs:
                seen_dtypes: set[str] = set()
                for cache_dtype_policy in cache_dtype_policies:
                    kv_cache_dtype = _reference_logit_cache_dtype_for_policy(
                        backend_config,
                        default_cache_dtype,
                        cache_dtype_policy,
                    )
                    dtype_key = str(jnp.dtype(kv_cache_dtype))
                    if dtype_key in seen_dtypes:
                        continue
                    seen_dtypes.add(dtype_key)

                    page_table = PageTable.init(
                        max_pages=max_pages_per_seq,
                        max_seqs=1,
                        page_size=128,
                        max_pages_per_seq=max_pages_per_seq,
                    )
                    sequences = SequenceTable.init(page_table.max_seqs, page_table.pages_per_seq, page_table.page_size)
                    sequences, seq_id_arr = sequences.reserve_slot(0)
                    seq_id = int(seq_id_arr)
                    slot_ids = hax.named(jnp.full((len(prompt_token_ids),), seq_id, dtype=jnp.int32), Pos)
                    sequences, page_table, batch_info = sequences.allocate_for_seq(page_table, slot_ids, pos_ids)
                    del sequences, page_table

                    kv_cache = model.initial_cache(
                        PageTable.init(
                            max_pages=max_pages_per_seq,
                            max_seqs=1,
                            page_size=128,
                            max_pages_per_seq=max_pages_per_seq,
                        ).spec(),
                        dtype=kv_cache_dtype,
                    )
                    decode_hidden, _ = model.decode_hidden(
                        input_ids,
                        kv_cache,
                        batch_info,
                        pos_ids,
                        tpu_paged_attention=backend_config,
                    )
                    decode_logits = model.lm_head_logits(decode_hidden)

                    hidden_metrics = _array_error_metrics(reference_hidden.array, decode_hidden.array)
                    error_metrics = _reference_logit_error_metrics(reference_logits.array, decode_logits.array)
                    direct_vs_reference_hidden = (
                        _array_error_metrics(reference_hidden.array, direct_hidden.array)
                        if direct_hidden is not None
                        else None
                    )
                    direct_vs_decode_hidden = (
                        _array_error_metrics(decode_hidden.array, direct_hidden.array)
                        if direct_hidden is not None
                        else None
                    )
                    direct_vs_reference_logits = (
                        _array_error_metrics(reference_logits.array, direct_logits.array)
                        if direct_logits is not None
                        else None
                    )
                    direct_vs_decode_logits = (
                        _array_error_metrics(decode_logits.array, direct_logits.array)
                        if direct_logits is not None
                        else None
                    )
                    results.append(
                        ReferenceLogitCheckResult(
                            prompt_index=prompt_index,
                            case_names=case_names,
                            prompt_token_ids=prompt_token_ids,
                            decode_backend=decode_backend_name,
                            kv_cache_dtype=str(jnp.dtype(kv_cache_dtype)),
                            kv_cache_dtype_policy=cache_dtype_policy,
                            reference_hidden_dtype=hidden_metrics["reference_dtype"],
                            decode_hidden_dtype=hidden_metrics["candidate_dtype"],
                            reference_logits_dtype=error_metrics["reference_logits_dtype"],
                            decode_logits_dtype=error_metrics["decode_logits_dtype"],
                            tpu_inference_out_dtype=backend_config.tpu_inference_out_dtype,
                            preserve_attention_output_dtype=backend_config.preserve_attention_output_dtype,
                            positions=len(prompt_token_ids),
                            vocab_size=reference_logits.axis_size("vocab"),
                            hidden_max_abs_error=hidden_metrics["max_abs_error"],
                            hidden_mean_abs_error=hidden_metrics["mean_abs_error"],
                            hidden_rms_abs_error=hidden_metrics["rms_abs_error"],
                            max_abs_error=error_metrics["max_abs_error"],
                            max_abs_error_if_reference_rounded_to_decode_dtype=error_metrics[
                                "max_abs_error_if_reference_rounded_to_decode_dtype"
                            ],
                            residual_max_abs_error_after_reference_rounding=error_metrics[
                                "residual_max_abs_error_after_reference_rounding"
                            ],
                            max_rel_error=error_metrics["max_rel_error"],
                            mean_abs_error=error_metrics["mean_abs_error"],
                            rms_abs_error=error_metrics["rms_abs_error"],
                            abs_error_p50=error_metrics["abs_error_p50"],
                            abs_error_p90=error_metrics["abs_error_p90"],
                            abs_error_p99=error_metrics["abs_error_p99"],
                            abs_error_p999=error_metrics["abs_error_p999"],
                            top1_agreement=error_metrics["top1_agreement"],
                            top_k_diagnostics=error_metrics["top_k_diagnostics"],
                            abs_error_histogram=error_metrics["abs_error_histogram"],
                            passed=error_metrics["max_abs_error"] <= atol or error_metrics["max_rel_error"] <= rtol,
                            single_token_direct_hidden_dtype=(
                                direct_vs_reference_hidden["candidate_dtype"]
                                if direct_vs_reference_hidden is not None
                                else None
                            ),
                            single_token_direct_vs_reference_hidden_max_abs_error=(
                                direct_vs_reference_hidden["max_abs_error"]
                                if direct_vs_reference_hidden is not None
                                else None
                            ),
                            single_token_direct_vs_decode_hidden_max_abs_error=(
                                direct_vs_decode_hidden["max_abs_error"]
                                if direct_vs_decode_hidden is not None
                                else None
                            ),
                            single_token_direct_vs_reference_logits_max_abs_error=(
                                direct_vs_reference_logits["max_abs_error"]
                                if direct_vs_reference_logits is not None
                                else None
                            ),
                            single_token_direct_vs_decode_logits_max_abs_error=(
                                direct_vs_decode_logits["max_abs_error"]
                                if direct_vs_decode_logits is not None
                                else None
                            ),
                        )
                    )

    return results


def write_reference_logit_check_outputs(
    output_dir: Path,
    results: list[ReferenceLogitCheckResult],
    *,
    atol: float,
    rtol: float,
    raise_on_failure: bool = True,
) -> None:
    payload = {
        "atol": atol,
        "rtol": rtol,
        "passed": all(result.passed for result in results),
        "results": [dataclasses.asdict(result) for result in results],
    }
    with open(output_dir / REFERENCE_LOGIT_ARTIFACT_JSON, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    with open(output_dir / REFERENCE_LOGIT_ARTIFACT_MD, "w") as f:

        def optional_metric(value: float | str | None) -> str:
            if value is None:
                return ""
            if isinstance(value, float):
                return f"{value:.6g}"
            return value

        f.write(
            "| prompt | backend | cache dtype | cache policy | rpa out dtype | preserve attn dtype | cases | "
            "positions | vocab | ref hidden dtype | decode hidden dtype | hidden max abs | hidden mean abs | hidden rms abs | "
            "direct hidden dtype | direct-ref hidden max abs | direct-decode hidden max abs | "
            "direct-ref logits max abs | direct-decode logits max abs | "
            "ref dtype | decode dtype | max abs error | ref-round max abs | round residual max abs | "
            "mean abs error | rms abs error | p99 abs error | p99.9 abs error | "
            "max rel error | top1 agreement | passed |\n"
        )
        f.write(
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        )
        for result in results:
            f.write(
                "| "
                + " | ".join(
                    [
                        str(result.prompt_index),
                        result.decode_backend,
                        result.kv_cache_dtype,
                        result.kv_cache_dtype_policy,
                        str(result.tpu_inference_out_dtype),
                        str(result.preserve_attention_output_dtype).lower(),
                        ",".join(result.case_names),
                        str(result.positions),
                        str(result.vocab_size),
                        result.reference_hidden_dtype,
                        result.decode_hidden_dtype,
                        f"{result.hidden_max_abs_error:.6g}",
                        f"{result.hidden_mean_abs_error:.6g}",
                        f"{result.hidden_rms_abs_error:.6g}",
                        optional_metric(result.single_token_direct_hidden_dtype),
                        optional_metric(result.single_token_direct_vs_reference_hidden_max_abs_error),
                        optional_metric(result.single_token_direct_vs_decode_hidden_max_abs_error),
                        optional_metric(result.single_token_direct_vs_reference_logits_max_abs_error),
                        optional_metric(result.single_token_direct_vs_decode_logits_max_abs_error),
                        result.reference_logits_dtype,
                        result.decode_logits_dtype,
                        f"{result.max_abs_error:.6g}",
                        f"{result.max_abs_error_if_reference_rounded_to_decode_dtype:.6g}",
                        f"{result.residual_max_abs_error_after_reference_rounding:.6g}",
                        f"{result.mean_abs_error:.6g}",
                        f"{result.rms_abs_error:.6g}",
                        f"{result.abs_error_p99:.6g}",
                        f"{result.abs_error_p999:.6g}",
                        f"{result.max_rel_error:.6g}",
                        f"{result.top1_agreement:.6g}",
                        str(result.passed).lower(),
                    ]
                )
                + " |\n"
            )

        for result in results:
            prompt_label = (
                f"prompt {result.prompt_index} / {result.decode_backend} / "
                f"{result.kv_cache_dtype_policy}:{result.kv_cache_dtype} / "
                f"out={result.tpu_inference_out_dtype} / preserve={result.preserve_attention_output_dtype}"
            )
            f.write(f"\n### {prompt_label} reference-top-k diagnostics\n\n")
            f.write(
                "| k | max logit abs error | mean logit abs error | max logprob abs error | "
                "mean logprob abs error | overlap fraction |\n"
            )
            f.write("| --- | --- | --- | --- | --- | --- |\n")
            for top_k in result.top_k_diagnostics:
                f.write(
                    "| "
                    + " | ".join(
                        [
                            str(top_k["k"]),
                            f"{top_k['max_abs_error']:.6g}",
                            f"{top_k['mean_abs_error']:.6g}",
                            f"{top_k['max_logprob_abs_error']:.6g}",
                            f"{top_k['mean_logprob_abs_error']:.6g}",
                            f"{top_k['overlap_fraction']:.6g}",
                        ]
                    )
                    + " |\n"
                )

            f.write(f"\n### {prompt_label} absolute-error histogram\n\n")
            f.write("| lower | upper | count | fraction |\n")
            f.write("| --- | --- | --- | --- |\n")
            for bucket in result.abs_error_histogram:
                upper = bucket["upper"]
                f.write(
                    "| "
                    + " | ".join(
                        [
                            f"{bucket['lower']:.6g}",
                            "inf" if upper is None else f"{upper:.6g}",
                            str(bucket["count"]),
                            f"{bucket['fraction']:.6g}",
                        ]
                    )
                    + " |\n"
                )

    if raise_on_failure and not payload["passed"]:
        raise AssertionError(
            "Levanter reference-logit check failed; see " f"{output_dir / REFERENCE_LOGIT_ARTIFACT_JSON} for details"
        )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    frac = position - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _completion_tokens(payload: dict[str, Any]) -> int:
    usage = payload.get("usage") or {}
    if "completion_tokens" in usage:
        return int(usage["completion_tokens"])
    total = 0
    for choice in payload.get("choices", []):
        total += len(str(choice.get("text", "")).split())
    return total


def _prompt_tokens(payload: dict[str, Any], fallback: int) -> int:
    usage = payload.get("usage") or {}
    return int(usage.get("prompt_tokens", fallback))


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _jax_device_snapshot() -> dict[str, Any]:
    try:
        devices = jax.devices()
        return {
            "devices": [str(device) for device in devices],
            "device_kind": getattr(devices[0], "device_kind", None) if devices else None,
        }
    except Exception as exc:
        return {
            "devices": None,
            "device_kind": None,
            "devices_error": repr(exc),
        }


def _runtime_env_snapshot(*, include_jax_devices: bool) -> dict[str, Any]:
    snapshot = {
        "jax_version": _distribution_version("jax"),
        "jaxlib_version": _distribution_version("jaxlib"),
        "libtpu_version": _distribution_version("libtpu"),
        "vllm_tpu_version": _distribution_version("vllm-tpu"),
        "tpu_inference_version": _distribution_version("tpu-inference"),
    }
    if include_jax_devices:
        snapshot.update(_jax_device_snapshot())
    else:
        snapshot.update(
            {
                "devices": None,
                "device_kind": None,
                "devices_skipped": "Skipped because another process, such as vLLM, may own libtpu.",
            }
        )
    return snapshot


def _send_completion(
    *,
    base_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    n: int,
    temperature: float,
    top_p: float,
    seed: int | None,
    timeout: float,
    return_logprobs: bool,
    top_k: int | None = None,
) -> tuple[dict[str, Any], float]:
    request = {
        "model": model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
    }
    if top_k is not None:
        request["top_k"] = top_k
    if seed is not None:
        request["seed"] = seed
    if return_logprobs:
        request["logprobs"] = 1
    start = time.perf_counter()
    response = requests.post(f"{base_url}/completions", json=request, timeout=timeout)
    elapsed = time.perf_counter() - start
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text.strip()
        if body:
            raise requests.HTTPError(f"{exc}; response body: {body}", response=response) from exc
        raise
    return response.json(), elapsed


def _send_completion_after_start(*, start_event: threading.Event, **kwargs) -> tuple[dict[str, Any], float]:
    start_event.wait()
    return _send_completion(**kwargs)


def _safe_artifact_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)


@contextlib.contextmanager
def _maybe_jax_profile(*, enabled: bool, profile_dir: Path | None, step_name: str):
    if not enabled:
        yield
        return
    if profile_dir is None:
        raise ValueError("profile_dir must be set when profiling is enabled")

    profile_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting JAX profile trace %s", profile_dir)
    jax.profiler.start_trace(str(profile_dir), create_perfetto_trace=True)
    try:
        with jax.profiler.StepTraceAnnotation(step_name):
            yield
    finally:
        jax.profiler.stop_trace()
        logger.info("Stopped JAX profile trace %s", profile_dir)


def run_case(
    *,
    handle: ServerHandle,
    case: BenchmarkCase,
    prompt: str,
    prompt_tokens: int,
    warmup: bool,
    seed: int,
    temperature: float,
    top_p: float,
    request_timeout: float,
    return_logprobs: bool,
    top_k: int | None = None,
) -> CaseResult:
    latencies: list[float] = []
    prompt_total = 0
    completion_total = 0

    start_event = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=case.request_count) as executor:
        futures = [
            executor.submit(
                _send_completion_after_start,
                start_event=start_event,
                base_url=handle.base_url,
                model_id=handle.model_id,
                prompt=prompt,
                max_tokens=case.output_tokens,
                n=case.n,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed + i if handle.supports_seed else None,
                timeout=request_timeout,
                return_logprobs=return_logprobs,
            )
            for i in range(case.request_count)
        ]
        start = time.perf_counter()
        start_event.set()
        for future in concurrent.futures.as_completed(futures):
            payload, elapsed = future.result()
            latencies.append(elapsed * 1000.0)
            prompt_total += _prompt_tokens(payload, prompt_tokens)
            completion_total += _completion_tokens(payload)

    elapsed = time.perf_counter() - start
    total_tokens = prompt_total + completion_total
    runtime_metrics = _case_runtime_metrics(handle.metrics_snapshot)
    return CaseResult(
        case_name=case.name,
        backend=handle.name,
        request_count=case.request_count,
        active_sequences=case.active_sequences,
        n=case.n,
        input_tokens_target=case.input_tokens,
        output_tokens_target=case.output_tokens,
        prompt_tokens=prompt_total,
        completion_tokens=completion_total,
        total_tokens=total_tokens,
        compile_including_seconds=elapsed if warmup else None,
        steady_state_seconds=elapsed,
        request_latency_ms_p50=statistics.median(latencies) if latencies else 0.0,
        request_latency_ms_p90=_percentile(latencies, 0.90),
        ttft_ms_p50=None,
        decode_tokens_per_second=completion_total / elapsed if elapsed > 0 else 0.0,
        total_tokens_per_second=total_tokens / elapsed if elapsed > 0 else 0.0,
        hbm_used_bytes=handle.hbm_used_bytes,
        compiled_shape_count=handle.compiled_shape_count,
        prefill_admissions=runtime_metrics.prefill_admissions,
        prefill_prompt_tokens_per_admission=runtime_metrics.prefill_prompt_tokens_per_admission,
        prefill_seconds_per_admission=runtime_metrics.prefill_seconds_per_admission,
        decode_seconds_per_iteration=runtime_metrics.decode_seconds_per_iteration,
        decode_device_seconds_per_iteration=runtime_metrics.decode_device_seconds_per_iteration,
        decode_host_seconds_per_iteration=runtime_metrics.decode_host_seconds_per_iteration,
        decode_submit_seconds_per_iteration=runtime_metrics.decode_submit_seconds_per_iteration,
        decode_extract_seconds_per_iteration=runtime_metrics.decode_extract_seconds_per_iteration,
        decode_tokens_per_iteration=runtime_metrics.decode_tokens_per_iteration,
        prefill_drain_seconds_per_iteration=runtime_metrics.prefill_drain_seconds_per_iteration,
        prefill_drain_tokens_per_iteration=runtime_metrics.prefill_drain_tokens_per_iteration,
        generation_seconds_per_iteration=runtime_metrics.generation_seconds_per_iteration,
        generation_host_seconds_per_iteration=runtime_metrics.generation_host_seconds_per_iteration,
        generation_tokens_per_iteration=runtime_metrics.generation_tokens_per_iteration,
    )


def _completion_error_key(exc: BaseException) -> str:
    if isinstance(exc, requests.exceptions.Timeout):
        return "timeout"
    return type(exc).__name__


def _update_stress_metric_maxima(
    maxima: dict[str, int],
    metrics_snapshot: Callable[[], dict[str, Any]] | None,
) -> None:
    if metrics_snapshot is None:
        return
    metrics = metrics_snapshot()
    for key in ("request_queue_depth", "batch_queue_depth"):
        value = metrics.get(key)
        if value is not None:
            maxima[key] = max(maxima.get(key, 0), int(value))


def _stress_service_static_metric(
    metrics_snapshot: Callable[[], dict[str, Any]] | None,
    key: str,
) -> int | None:
    if metrics_snapshot is None:
        return None
    value = metrics_snapshot().get(key)
    return None if value is None else int(value)


@dataclass(frozen=True, slots=True)
class CaseRuntimeMetrics:
    prefill_admissions: int | None
    prefill_prompt_tokens_per_admission: list[int] | None
    prefill_seconds_per_admission: list[float] | None
    decode_seconds_per_iteration: list[float] | None
    decode_device_seconds_per_iteration: list[float] | None
    decode_host_seconds_per_iteration: list[float] | None
    decode_submit_seconds_per_iteration: list[float] | None
    decode_extract_seconds_per_iteration: list[float] | None
    decode_tokens_per_iteration: list[int] | None
    prefill_drain_seconds_per_iteration: list[float] | None
    prefill_drain_tokens_per_iteration: list[int] | None
    generation_seconds_per_iteration: list[float] | None
    generation_host_seconds_per_iteration: list[float] | None
    generation_tokens_per_iteration: list[int] | None


def _optional_metric_float_list(metrics: dict[str, Any], key: str) -> list[float] | None:
    value = metrics.get(key)
    return None if value is None else [float(item) for item in value]


def _optional_metric_int_list(metrics: dict[str, Any], key: str) -> list[int] | None:
    value = metrics.get(key)
    return None if value is None else [int(item) for item in value]


def _case_runtime_metrics(metrics_snapshot: Callable[[], dict[str, Any]] | None) -> CaseRuntimeMetrics:
    if metrics_snapshot is None:
        return CaseRuntimeMetrics(
            prefill_admissions=None,
            prefill_prompt_tokens_per_admission=None,
            prefill_seconds_per_admission=None,
            decode_seconds_per_iteration=None,
            decode_device_seconds_per_iteration=None,
            decode_host_seconds_per_iteration=None,
            decode_submit_seconds_per_iteration=None,
            decode_extract_seconds_per_iteration=None,
            decode_tokens_per_iteration=None,
            prefill_drain_seconds_per_iteration=None,
            prefill_drain_tokens_per_iteration=None,
            generation_seconds_per_iteration=None,
            generation_host_seconds_per_iteration=None,
            generation_tokens_per_iteration=None,
        )
    metrics = metrics_snapshot()
    admissions = metrics.get("prefill_admissions")
    return CaseRuntimeMetrics(
        prefill_admissions=None if admissions is None else int(admissions),
        prefill_prompt_tokens_per_admission=_optional_metric_int_list(metrics, "prefill_prompt_tokens_per_admission"),
        prefill_seconds_per_admission=_optional_metric_float_list(metrics, "prefill_seconds_per_admission"),
        decode_seconds_per_iteration=_optional_metric_float_list(metrics, "decode_seconds_per_iteration"),
        decode_device_seconds_per_iteration=_optional_metric_float_list(
            metrics, "decode_device_seconds_per_iteration"
        ),
        decode_host_seconds_per_iteration=_optional_metric_float_list(metrics, "decode_host_seconds_per_iteration"),
        decode_submit_seconds_per_iteration=_optional_metric_float_list(
            metrics, "decode_submit_seconds_per_iteration"
        ),
        decode_extract_seconds_per_iteration=_optional_metric_float_list(
            metrics, "decode_extract_seconds_per_iteration"
        ),
        decode_tokens_per_iteration=_optional_metric_int_list(metrics, "decode_tokens_per_iteration"),
        prefill_drain_seconds_per_iteration=_optional_metric_float_list(
            metrics, "prefill_drain_seconds_per_iteration"
        ),
        prefill_drain_tokens_per_iteration=_optional_metric_int_list(metrics, "prefill_drain_tokens_per_iteration"),
        generation_seconds_per_iteration=_optional_metric_float_list(metrics, "generation_seconds_per_iteration"),
        generation_host_seconds_per_iteration=_optional_metric_float_list(
            metrics, "generation_host_seconds_per_iteration"
        ),
        generation_tokens_per_iteration=_optional_metric_int_list(metrics, "generation_tokens_per_iteration"),
    )


def run_stress_case(
    *,
    handle: ServerHandle,
    case: BenchmarkCase,
    prompt: str,
    prompt_tokens: int,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    request_timeout: float,
    return_logprobs: bool,
    duration_seconds: float,
    concurrent_requests: int,
    metrics_interval_seconds: float,
    max_requests: int | None,
) -> StressResult:
    if duration_seconds <= 0.0 and max_requests is None:
        raise ValueError("run_stress_case requires a positive duration_seconds or max_requests")
    if concurrent_requests < 1:
        raise ValueError("concurrent_requests must be positive")

    start_event = threading.Event()
    request_counter = 0
    counter_lock = threading.Lock()
    result_lock = threading.Lock()
    latencies: list[float] = []
    error_counts: dict[str, int] = {}
    prompt_total = 0
    completion_total = 0
    success_count = 0
    timeout_count = 0
    failure_count = 0
    metric_maxima: dict[str, int] = {}

    start_time = time.perf_counter()
    deadline = start_time + duration_seconds if duration_seconds > 0.0 else None

    def claim_request() -> int | None:
        nonlocal request_counter
        now = time.perf_counter()
        if deadline is not None and now >= deadline:
            return None
        with counter_lock:
            if max_requests is not None and request_counter >= max_requests:
                return None
            request_index = request_counter
            request_counter += 1
            return request_index

    def worker() -> None:
        nonlocal prompt_total, completion_total, success_count, timeout_count, failure_count
        start_event.wait()
        while True:
            request_index = claim_request()
            if request_index is None:
                return
            request_seed = seed + request_index if handle.supports_seed else None
            try:
                payload, elapsed = _send_completion(
                    base_url=handle.base_url,
                    model_id=handle.model_id,
                    prompt=prompt,
                    max_tokens=case.output_tokens,
                    n=case.n,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=request_seed,
                    timeout=request_timeout,
                    return_logprobs=return_logprobs,
                )
                with result_lock:
                    latencies.append(elapsed * 1000.0)
                    prompt_total += _prompt_tokens(payload, prompt_tokens)
                    completion_total += _completion_tokens(payload)
                    success_count += 1
                    _update_stress_metric_maxima(metric_maxima, handle.metrics_snapshot)
            except Exception as exc:
                key = _completion_error_key(exc)
                with result_lock:
                    failure_count += 1
                    timeout_count += int(key == "timeout")
                    error_counts[key] = error_counts.get(key, 0) + 1
                    _update_stress_metric_maxima(metric_maxima, handle.metrics_snapshot)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = [executor.submit(worker) for _ in range(concurrent_requests)]
        start_time = time.perf_counter()
        deadline = start_time + duration_seconds if duration_seconds > 0.0 else None
        _update_stress_metric_maxima(metric_maxima, handle.metrics_snapshot)
        start_event.set()
        while any(not future.done() for future in futures):
            _update_stress_metric_maxima(metric_maxima, handle.metrics_snapshot)
            time.sleep(metrics_interval_seconds)
        for future in futures:
            future.result()
        _update_stress_metric_maxima(metric_maxima, handle.metrics_snapshot)

    wall_clock_seconds = time.perf_counter() - start_time
    load_seconds = min(duration_seconds, wall_clock_seconds) if duration_seconds > 0.0 else wall_clock_seconds
    total_requests = success_count + failure_count
    total_tokens = prompt_total + completion_total
    return StressResult(
        case_name=case.name,
        backend=handle.name,
        concurrent_requests=concurrent_requests,
        active_sequences=concurrent_requests * case.n,
        n=case.n,
        input_tokens_target=case.input_tokens,
        output_tokens_target=case.output_tokens,
        total_requests=total_requests,
        successful_requests=success_count,
        failed_requests=failure_count,
        timeout_requests=timeout_count,
        prompt_tokens=prompt_total,
        completion_tokens=completion_total,
        total_tokens=total_tokens,
        load_seconds=load_seconds,
        wall_clock_seconds=wall_clock_seconds,
        steady_decode_tokens_per_second=completion_total / load_seconds if load_seconds > 0 else 0.0,
        wall_clock_decode_tokens_per_second=completion_total / wall_clock_seconds if wall_clock_seconds > 0 else 0.0,
        wall_clock_total_tokens_per_second=total_tokens / wall_clock_seconds if wall_clock_seconds > 0 else 0.0,
        request_latency_ms_p50=statistics.median(latencies) if latencies else 0.0,
        request_latency_ms_p90=_percentile(latencies, 0.90),
        request_latency_ms_p99=_percentile(latencies, 0.99),
        request_latency_ms_max=max(latencies) if latencies else 0.0,
        hbm_used_bytes=handle.hbm_used_bytes,
        compiled_shape_count=handle.compiled_shape_count,
        max_request_queue_depth=metric_maxima.get("request_queue_depth"),
        max_batch_queue_depth=metric_maxima.get("batch_queue_depth"),
        page_size=_stress_service_static_metric(handle.metrics_snapshot, "page_size"),
        max_pages=_stress_service_static_metric(handle.metrics_snapshot, "max_pages"),
        retry_errors=0,
        error_counts=error_counts,
    )


def start_vllm_server(
    *,
    model: str,
    port: int,
    timeout: float,
    max_model_len: int,
    tensor_parallel_size: int,
    load_format: str | None,
    cache_dir: str | None,
    extra_args: list[str],
    log_dir: Path | None,
) -> ServerHandle:
    vllm_bin = shutil.which("vllm") or "vllm"
    cmd = [
        vllm_bin,
        "serve",
        model,
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        *extra_args,
    ]
    if load_format is not None:
        cmd.extend(["--load-format", load_format])

    env = dict(os.environ)
    if cache_dir is not None:
        env.setdefault("JAX_COMPILATION_CACHE_DIR", cache_dir)
        env.setdefault("VLLM_XLA_CACHE_PATH", cache_dir)
    env.setdefault("MODEL_IMPL_TYPE", "vllm")
    env.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
    env.setdefault("TPU_MIN_LOG_LEVEL", "3")
    env.setdefault("TPU_STDERR_LOG_LEVEL", "3")
    startup_snapshot = _vllm_startup_snapshot(env)

    if log_dir is None:
        log_dir = Path(tempfile.mkdtemp(prefix="qwen3_vllm_bench_"))
    else:
        log_dir.mkdir(parents=True, exist_ok=True)
    stdout = open(log_dir / "stdout.log", "w")  # noqa: SIM115
    stderr = open(log_dir / "stderr.log", "w")  # noqa: SIM115
    process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, text=True, env=env)
    base_url = f"http://127.0.0.1:{port}/v1"

    def close() -> None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
        stdout.close()
        stderr.close()

    try:
        payload = _poll_json_while_process_alive(f"{base_url}/models", timeout=timeout, process=process)
        model_id = str(payload["data"][0]["id"])
    except Exception as exc:
        process_return_code = process.poll()
        close()
        raise RuntimeError(
            _vllm_startup_error_message(
                log_dir=log_dir,
                cmd=cmd,
                process_return_code=process_return_code,
                cause=exc,
                startup_snapshot=startup_snapshot,
            )
        ) from exc

    logger.info("Started vLLM server at %s with logs in %s", base_url, log_dir)
    return ServerHandle("vllm-tpu", base_url, model_id, close, cmd, supports_seed=False)


def levanter_trainer_config(tensor_parallel_size: int, mp_policy: str = "f32") -> TrainerConfig:
    return TrainerConfig(
        mp=jmp.get_policy(mp_policy),
        mesh=MeshConfig(
            axes={"data": 1, "replica": 1, "model": tensor_parallel_size},
            shared_mapping={"kv_head": "model", "vocab": "model"},
        ),
    )


def _with_rpa_overrides(
    model_config: Qwen3Config,
    *,
    rpa_num_kv_pages_per_block: int | None,
    rpa_num_queries_per_block: int | None,
    rpa_vmem_limit_bytes: int | None,
) -> Qwen3Config:
    rpa_overrides = {
        key: value
        for key, value in {
            "rpa_num_kv_pages_per_block": rpa_num_kv_pages_per_block,
            "rpa_num_queries_per_block": rpa_num_queries_per_block,
            "rpa_vmem_limit_bytes": rpa_vmem_limit_bytes,
        }.items()
        if value is not None
    }
    if not rpa_overrides:
        return model_config
    return dataclasses.replace(model_config, **rpa_overrides)


def run_levanter_without_lm_head_case(
    *,
    trainer: TrainerConfig,
    server: Any,
    tokenizer: Any,
    backend_name: str,
    hbm_used_bytes: int | None,
    compiled_shape_count: int | None,
    case: BenchmarkCase,
    prompt: str,
    prompt_tokens: int,
    warmup: bool,
    seed: int,
    temperature: float,
    top_p: float,
    engine_method: str = "generate_without_lm_head",
    top_k: int | None = None,
) -> CaseResult:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    requests = []
    for i in range(case.request_count):
        seq_params = SeqDecodingParams(
            max_num_tokens=jnp.asarray(len(prompt_ids) + case.output_tokens, dtype=jnp.int32),
            stop_tokens=None,
            temperature=jnp.asarray(temperature, dtype=jnp.float32),
            top_p=jnp.asarray(top_p, dtype=jnp.float32),
            top_k=jnp.asarray(0 if top_k is None else top_k, dtype=jnp.int32),
            key=jax.random.PRNGKey(seed + i),
        )
        requests.append(
            Request(
                prompt_tokens=prompt_ids,
                request_id=i,
                decode_params=seq_params,
                n_generations=case.n,
                return_logprobs=False,
            )
        )

    start = time.perf_counter()
    with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):
        result = getattr(server.inference_context.engine, engine_method)(requests)
    elapsed = time.perf_counter() - start

    completion_total = result.total_generated
    prompt_total = prompt_tokens * case.active_sequences
    total_tokens = prompt_total + completion_total
    elapsed_ms = elapsed * 1000.0
    return CaseResult(
        case_name=case.name,
        backend=backend_name,
        request_count=case.request_count,
        active_sequences=case.active_sequences,
        n=case.n,
        input_tokens_target=case.input_tokens,
        output_tokens_target=case.output_tokens,
        prompt_tokens=prompt_total,
        completion_tokens=completion_total,
        total_tokens=total_tokens,
        compile_including_seconds=elapsed if warmup else None,
        steady_state_seconds=elapsed,
        request_latency_ms_p50=elapsed_ms,
        request_latency_ms_p90=elapsed_ms,
        ttft_ms_p50=None,
        decode_tokens_per_second=completion_total / elapsed if elapsed > 0 else 0.0,
        total_tokens_per_second=total_tokens / elapsed if elapsed > 0 else 0.0,
        hbm_used_bytes=hbm_used_bytes,
        compiled_shape_count=compiled_shape_count,
        prefill_admissions=result.prefill_admissions,
        prefill_prompt_tokens_per_admission=result.prefill_prompt_tokens_per_admission,
        prefill_seconds_per_admission=result.prefill_seconds_per_admission,
        decode_seconds_per_iteration=result.decode_seconds_per_iteration,
        decode_device_seconds_per_iteration=result.decode_device_seconds_per_iteration,
        decode_host_seconds_per_iteration=result.decode_host_seconds_per_iteration,
        decode_submit_seconds_per_iteration=result.decode_submit_seconds_per_iteration,
        decode_extract_seconds_per_iteration=result.decode_extract_seconds_per_iteration,
        decode_tokens_per_iteration=result.decode_tokens_per_iteration,
        prefill_drain_seconds_per_iteration=result.prefill_drain_seconds_per_iteration,
        prefill_drain_tokens_per_iteration=result.prefill_drain_tokens_per_iteration,
        generation_seconds_per_iteration=result.generation_seconds_per_iteration,
        generation_host_seconds_per_iteration=result.generation_host_seconds_per_iteration,
        generation_tokens_per_iteration=result.generation_tokens_per_iteration,
    )


def run_levanter_with_lm_head_no_sampling_case(**kwargs: Any) -> CaseResult:
    return run_levanter_without_lm_head_case(
        engine_method="generate_with_lm_head_no_sampling",
        **kwargs,
    )


def start_levanter_server(
    *,
    model: str,
    checkpoint: str,
    tokenizer_name: str,
    port: int,
    timeout: float,
    max_model_len: int,
    max_seqs: int,
    max_pages: int | None,
    max_prefill_size: int | None,
    max_rounds: int,
    max_tokens_per_round: int | None,
    max_top_k: int | None,
    tensor_parallel_size: int,
    hbm_utilization: float,
    rpa_num_kv_pages_per_block: int | None,
    rpa_num_queries_per_block: int | None,
    rpa_vmem_limit_bytes: int | None,
    tpu_paged_attention_backend: TpuPagedAttentionBackend,
    allow_reference_fallback: bool,
    compute_dtype: str,
    trainer_mp_policy: str,
    tpu_inference_out_dtype: str | None,
    preserve_attention_output_dtype: bool,
    sampler_top_k_mode: SamplerTopKMode,
    use_streaming_greedy_lm_head: bool,
    batch_timeout: float,
    kernel_artifacts_dir: Path | None,
    return_logprobs: bool,
    reference_logit_check_dir: Path | None,
    reference_logit_check_cases: list[BenchmarkCase] | None,
    reference_logit_check_prompts: dict[str, tuple[str, int]] | None,
    reference_logit_atol: float,
    reference_logit_rtol: float,
    reference_logit_max_prompts: int | None,
    reference_logit_decode_backends: list[TpuPagedAttentionBackend] | None,
    reference_logit_cache_dtype_policies: list[str],
    reference_logit_only: bool,
) -> ServerHandle:
    # Keep this benchmark importable in Levanter's non-serve test environment.
    # The OpenAI server types require the optional serve extra.
    from levanter.inference.openai import InferenceServer, InferenceServerConfig

    trainer = levanter_trainer_config(tensor_parallel_size, trainer_mp_policy)
    tokenizer = load_tokenizer(tokenizer_name)
    service_compute_dtype = jnp.dtype(compute_dtype)
    service_config = InferenceEngineConfig(
        max_seq_len=max_model_len,
        max_seqs=max_seqs,
        max_pages=max_pages,
        page_size=128,
        compute_dtype=service_compute_dtype,
        hbm_utilization=hbm_utilization,
        max_queued_tokens=max(2 * max_seqs, 512),
        max_rounds=max_rounds,
        max_tokens_per_round=max_tokens_per_round,
        max_top_k=max_top_k,
        sampler_top_k_mode=sampler_top_k_mode,
        max_seqs_in_prefill=max_seqs,
        max_prefill_size=max_prefill_size,
        tpu_paged_attention=TpuPagedAttentionConfig(
            backend=tpu_paged_attention_backend,
            fail_on_reference_fallback=not allow_reference_fallback,
            tpu_inference_out_dtype=tpu_inference_out_dtype,
            preserve_attention_output_dtype=preserve_attention_output_dtype,
        ),
        use_streaming_greedy_lm_head=use_streaming_greedy_lm_head,
    )
    server_config = InferenceServerConfig(
        host="127.0.0.1",
        port=port,
        tokenizer=tokenizer_name,
        trainer=trainer,
        service=service_config,
        temperature=0.0,
        batch_timeout=batch_timeout,
    )

    server = None
    hbm_used_bytes = None
    with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):
        converter = HFCheckpointConverter(
            Qwen3Config,
            reference_checkpoint=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
        )
        model_config = converter.config_from_hf_checkpoint(ref=checkpoint)
        model_config = _with_rpa_overrides(
            model_config,
            rpa_num_kv_pages_per_block=rpa_num_kv_pages_per_block,
            rpa_num_queries_per_block=rpa_num_queries_per_block,
            rpa_vmem_limit_bytes=rpa_vmem_limit_bytes,
        )
        model_obj = converter.load_pretrained(
            Qwen3LMHeadModel,
            ref=checkpoint,
            config=model_config,
            dtype=trainer.mp.compute_dtype,
            axis_mapping=trainer.parameter_axis_mapping,
            resize_vocab_to_match_tokenizer=False,
        )
        if not reference_logit_only:
            server = InferenceServer.create(server_config, model_obj, tokenizer)
            hbm_used_bytes = sharded_tree_size(server.inference_context.engine.gen_state.cache)

    if reference_logit_check_dir is not None:
        if reference_logit_check_cases is None or reference_logit_check_prompts is None:
            raise ValueError("reference_logit_check_cases and prompts must be provided when checking logits")
        reference_results = check_levanter_reference_logits(
            trainer=trainer,
            model=model_obj,
            tokenizer=tokenizer,
            cases=reference_logit_check_cases,
            prompts=reference_logit_check_prompts,
            tpu_paged_attention=service_config.tpu_paged_attention,
            decode_backends=reference_logit_decode_backends,
            cache_dtype_policies=reference_logit_cache_dtype_policies,
            default_cache_dtype=service_compute_dtype,
            max_model_len=max_model_len,
            atol=reference_logit_atol,
            rtol=reference_logit_rtol,
            max_prompts=reference_logit_max_prompts,
        )
        write_reference_logit_check_outputs(
            reference_logit_check_dir,
            reference_results,
            atol=reference_logit_atol,
            rtol=reference_logit_rtol,
            raise_on_failure=not reference_logit_only,
        )
        if reference_logit_only:
            logger.info("Benchmark reference-logit-only artifacts:")
            log_output_artifacts(reference_logit_check_dir)
            return ServerHandle(
                name=f"levanter:{tpu_paged_attention_backend.value}:reference_logit_only",
                base_url=f"http://127.0.0.1:{port}/v1",
                model_id=model,
                close=lambda: None,
            )

    assert server is not None
    assert hbm_used_bytes is not None

    if kernel_artifacts_dir is not None:
        write_levanter_kernel_artifacts(
            trainer,
            server,
            kernel_artifacts_dir,
            return_logprobs=return_logprobs,
            use_streaming_greedy_lm_head=use_streaming_greedy_lm_head,
        )

    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}/v1"

    def close() -> None:
        server.shutdown()

    try:
        _poll_json(f"http://127.0.0.1:{port}/health", timeout=timeout)
    except Exception:
        close()
        raise

    diagnostic_backend_name = f"levanter:{tpu_paged_attention_backend.value}:no_lm_head"
    lm_head_no_sampling_backend_name = f"levanter:{tpu_paged_attention_backend.value}:lm_head_no_sampling"

    def diagnose_without_lm_head(
        case: BenchmarkCase,
        prompt: str,
        prompt_tokens: int,
        warmup: bool,
        seed: int,
        temperature: float,
        top_p: float,
        request_timeout: float,
    ) -> CaseResult:
        del request_timeout
        return run_levanter_without_lm_head_case(
            trainer=trainer,
            server=server,
            tokenizer=tokenizer,
            backend_name=diagnostic_backend_name,
            hbm_used_bytes=hbm_used_bytes,
            compiled_shape_count=LEVANTER_COMPILED_SHAPE_BUCKETS,
            case=case,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            warmup=warmup,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            top_k=None,
        )

    def diagnose_with_lm_head_no_sampling(
        case: BenchmarkCase,
        prompt: str,
        prompt_tokens: int,
        warmup: bool,
        seed: int,
        temperature: float,
        top_p: float,
        request_timeout: float,
    ) -> CaseResult:
        del request_timeout
        return run_levanter_with_lm_head_no_sampling_case(
            trainer=trainer,
            server=server,
            tokenizer=tokenizer,
            backend_name=lm_head_no_sampling_backend_name,
            hbm_used_bytes=hbm_used_bytes,
            compiled_shape_count=LEVANTER_COMPILED_SHAPE_BUCKETS,
            case=case,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            warmup=warmup,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            top_k=None,
        )

    def metrics_snapshot() -> dict[str, Any]:
        engine = server.inference_context.engine
        return {
            "request_queue_depth": server.inference_context.request_queue.qsize(),
            "batch_queue_depth": server.inference_context.batch_queue.qsize(),
            "page_size": engine.config.page_size if engine is not None else None,
            "max_pages": engine.config.max_pages if engine is not None else None,
            "prefill_admissions": server.inference_context.last_prefill_admissions,
            "prefill_prompt_tokens_per_admission": (
                list(server.inference_context.last_prefill_prompt_tokens_per_admission)
            ),
            "prefill_seconds_per_admission": list(server.inference_context.last_prefill_seconds_per_admission),
            "decode_seconds_per_iteration": list(server.inference_context.last_decode_seconds_per_iteration),
            "decode_device_seconds_per_iteration": list(
                server.inference_context.last_decode_device_seconds_per_iteration
            ),
            "decode_host_seconds_per_iteration": list(server.inference_context.last_decode_host_seconds_per_iteration),
            "decode_submit_seconds_per_iteration": list(
                server.inference_context.last_decode_submit_seconds_per_iteration
            ),
            "decode_extract_seconds_per_iteration": list(
                server.inference_context.last_decode_extract_seconds_per_iteration
            ),
            "decode_tokens_per_iteration": list(server.inference_context.last_decode_tokens_per_iteration),
            "prefill_drain_seconds_per_iteration": list(
                server.inference_context.last_prefill_drain_seconds_per_iteration
            ),
            "prefill_drain_tokens_per_iteration": list(
                server.inference_context.last_prefill_drain_tokens_per_iteration
            ),
            "generation_seconds_per_iteration": list(server.inference_context.last_generation_seconds_per_iteration),
            "generation_host_seconds_per_iteration": list(
                server.inference_context.last_generation_host_seconds_per_iteration
            ),
            "generation_tokens_per_iteration": list(server.inference_context.last_generation_tokens_per_iteration),
        }

    logger.info("Started Levanter server at %s", base_url)
    return ServerHandle(
        f"levanter:{tpu_paged_attention_backend.value}",
        base_url,
        "levanter",
        close,
        None,
        hbm_used_bytes=hbm_used_bytes,
        compiled_shape_count=LEVANTER_COMPILED_SHAPE_BUCKETS,
        metrics_snapshot=metrics_snapshot,
        diagnose_without_lm_head=diagnose_without_lm_head,
        diagnose_with_lm_head_no_sampling=diagnose_with_lm_head_no_sampling,
    )


def write_levanter_kernel_artifacts(
    trainer: TrainerConfig,
    server: Any,
    output_dir: Path,
    *,
    return_logprobs: bool,
    use_streaming_greedy_lm_head: bool,
) -> None:
    with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):
        server.inference_context.engine.write_kernel_jaxprs(
            str(output_dir),
            return_logprobs=return_logprobs,
            use_streaming_greedy_lm_head=use_streaming_greedy_lm_head,
            log_artifacts=False,
        )


def _artifact_entry(path: Path, *, root: Path) -> dict[str, Any]:
    relative_path = path.relative_to(root)
    if path.is_dir():
        files = [child for child in path.rglob("*") if child.is_file()]
        return {
            "path": str(relative_path),
            "kind": "directory",
            "file_count": len(files),
            "bytes": sum(child.stat().st_size for child in files),
        }
    return {
        "path": str(relative_path),
        "kind": "file",
        "bytes": path.stat().st_size,
    }


def _read_hlo_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return f.read()
    return path.read_text()


def _hlo_summary_matches(line: str) -> list[str]:
    lowered_line = line.lower()
    matches: list[str] = []
    for name, needles in HLO_SUMMARY_PATTERNS.items():
        if name == "sort_or_topk":
            is_sort_or_topk = (
                "stablehlo.sort" in lowered_line
                or "func.call @argsort" in lowered_line
                or "call @argsort" in lowered_line
                or "topk" in lowered_line
                or "top-k" in lowered_line
                or "top_k" in lowered_line
            )
            if is_sort_or_topk:
                matches.append(name)
            continue
        if any(needle in lowered_line for needle in needles):
            matches.append(name)
    return matches


def summarize_levanter_hlo(output_dir: Path) -> dict[str, Any] | None:
    hlo_dir = output_dir / "levanter_hlo"
    if not hlo_dir.exists():
        return None

    files: list[dict[str, Any]] = []
    for path in sorted(hlo_dir.rglob("*")):
        if not path.is_file():
            continue
        lowered_name = path.name.lower()
        if ".hlo." not in lowered_name and not lowered_name.endswith(".hlo"):
            continue

        text = _read_hlo_text(path)
        lines = text.splitlines()
        pattern_counts = {name: 0 for name in HLO_SUMMARY_PATTERNS}
        examples: list[dict[str, Any]] = []
        for line_number, line in enumerate(lines, start=1):
            matches = _hlo_summary_matches(line)
            if not matches:
                continue
            for name in matches:
                pattern_counts[name] += 1
            if len(examples) < HLO_SUMMARY_MAX_EXAMPLES_PER_FILE:
                examples.append({"line": line_number, "matches": matches, "text": line[:240]})

        files.append(
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "line_count": len(lines),
                "pattern_counts": pattern_counts,
                "examples": examples,
            }
        )

    if not files:
        return None

    summary = {"files": files}
    with open(output_dir / "hlo_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    with open(output_dir / "hlo_summary.md", "w") as f:
        f.write("| file | bytes | lines | collective | rng_sampling | sort_or_topk | logprob | custom_call |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for file_summary in files:
            counts = file_summary["pattern_counts"]
            f.write(
                "| "
                + " | ".join(
                    [
                        file_summary["path"],
                        str(file_summary["bytes"]),
                        str(file_summary["line_count"]),
                        str(counts["collective"]),
                        str(counts["rng_sampling"]),
                        str(counts["sort_or_topk"]),
                        str(counts["logprob"]),
                        str(counts["custom_call"]),
                    ]
                )
                + " |\n"
            )
        for file_summary in files:
            if not file_summary["examples"]:
                continue
            f.write(f"\n### {file_summary['path']}\n\n")
            for example in file_summary["examples"]:
                matches = ",".join(example["matches"])
                f.write(f"- L{example['line']} `{matches}`: `{example['text']}`\n")
    return summary


def write_artifact_manifest(output_dir: Path) -> dict[str, Any]:
    artifact_paths = [
        output_dir / "summary.json",
        output_dir / "summary.md",
        output_dir / "stress_summary.md",
        output_dir / "env.json",
        output_dir / "prompt_corpus.json",
        output_dir / REFERENCE_LOGIT_ARTIFACT_JSON,
        output_dir / REFERENCE_LOGIT_ARTIFACT_MD,
        output_dir / "hlo_summary.json",
        output_dir / "hlo_summary.md",
        output_dir / "levanter_hlo",
        output_dir / "levanter_profiles",
        output_dir / "vllm_profiles",
    ]
    artifacts = [_artifact_entry(path, root=output_dir) for path in artifact_paths if path.exists()]
    manifest = {"artifacts": artifacts}
    with open(output_dir / "artifacts.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def _ratio(numerator: float, denominator: float | None) -> str:
    if denominator is None or denominator == 0.0:
        return ""
    return f"{numerator / denominator:.3f}"


def _optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def _optional_int(value: int | None) -> str:
    if value is None:
        return ""
    return str(value)


def _optional_int_list(value: list[int] | None) -> str:
    if value is None:
        return ""
    return ",".join(str(item) for item in value)


def _optional_float_list(value: list[float] | None) -> str:
    if value is None:
        return ""
    return ",".join(f"{item:.3f}" for item in value)


def _optional_tokens_per_second(tokens: list[int] | None, seconds: list[float] | None) -> float | None:
    if not tokens or not seconds:
        return None
    elapsed = sum(seconds)
    if elapsed <= 0.0:
        return None
    return sum(tokens) / elapsed


def _parity_target_status(result: CaseResult, baseline: CaseResult | None) -> str:
    if baseline is None or result.backend == baseline.backend or baseline.decode_tokens_per_second == 0.0:
        return ""
    if result.decode_tokens_per_second / baseline.decode_tokens_per_second >= PARITY_DECODE_RATIO_TARGET:
        return "pass"
    return "fail"


def parity_comparisons(results: list[CaseResult]) -> list[dict[str, Any]]:
    vllm_by_case = {result.case_name: result for result in results if result.backend == "vllm-tpu"}
    comparisons = []
    for result in results:
        baseline = vllm_by_case.get(result.case_name)
        if baseline is None or result.backend == baseline.backend:
            continue
        decode_ratio = (
            result.decode_tokens_per_second / baseline.decode_tokens_per_second
            if baseline.decode_tokens_per_second
            else None
        )
        total_ratio = (
            result.total_tokens_per_second / baseline.total_tokens_per_second
            if baseline.total_tokens_per_second
            else None
        )
        comparisons.append(
            {
                "case_name": result.case_name,
                "backend": result.backend,
                "baseline_backend": baseline.backend,
                "decode_ratio": decode_ratio,
                "total_ratio": total_ratio,
                "meets_decode_ratio_target": (decode_ratio is not None and decode_ratio >= PARITY_DECODE_RATIO_TARGET),
            }
        )
    return comparisons


def write_outputs(
    output_dir: Path,
    results: list[CaseResult],
    env: dict[str, Any],
    stress_results: list[StressResult] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dicts = []
    for result in results:
        result_dict = dataclasses.asdict(result)
        result_dict["decode_iteration_tokens_per_second"] = _optional_tokens_per_second(
            result.decode_tokens_per_iteration,
            result.decode_seconds_per_iteration,
        )
        result_dict["decode_device_tokens_per_second"] = _optional_tokens_per_second(
            result.decode_tokens_per_iteration,
            result.decode_device_seconds_per_iteration,
        )
        result_dict["generation_tokens_per_second"] = _optional_tokens_per_second(
            result.generation_tokens_per_iteration,
            result.generation_seconds_per_iteration,
        )
        result_dicts.append(result_dict)
    stress_result_dicts = [dataclasses.asdict(result) for result in stress_results or []]
    comparisons = parity_comparisons(results)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(
            {
                "results": result_dicts,
                "stress_results": stress_result_dicts,
                "comparisons": comparisons,
                "parity_decode_ratio_target": PARITY_DECODE_RATIO_TARGET,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    with open(output_dir / "env.json", "w") as f:
        json.dump(env, f, indent=2, sort_keys=True)

    headers = [
        "case",
        "backend",
        "active seqs",
        "n",
        "decode tok/s",
        "total tok/s",
        "steady s",
        "compile incl s",
        "ttft p50 ms",
        "p50 ms",
        "p90 ms",
        "hbm bytes",
        "shape buckets",
        "prefill admissions",
        "prefill chunks",
        "prefill s",
        "decode iter s",
        "decode device s",
        "decode host s",
        "decode submit s",
        "decode extract s",
        "decode iter toks",
        "prefill drain s",
        "prefill drain toks",
        "generation s",
        "generation host s",
        "generation toks",
        "decode iter tok/s",
        "decode device tok/s",
        "generation tok/s",
        "decode/vllm",
        "total/vllm",
        "target",
    ]
    vllm_by_case = {result.case_name: result for result in results if result.backend == "vllm-tpu"}
    rows = [
        [
            result.case_name,
            result.backend,
            str(result.active_sequences),
            str(result.n),
            f"{result.decode_tokens_per_second:.2f}",
            f"{result.total_tokens_per_second:.2f}",
            f"{result.steady_state_seconds:.3f}",
            _optional_float(result.compile_including_seconds),
            _optional_float(result.ttft_ms_p50),
            f"{result.request_latency_ms_p50:.1f}",
            f"{result.request_latency_ms_p90:.1f}",
            _optional_int(result.hbm_used_bytes),
            _optional_int(result.compiled_shape_count),
            _optional_int(result.prefill_admissions),
            _optional_int_list(result.prefill_prompt_tokens_per_admission),
            _optional_float_list(result.prefill_seconds_per_admission),
            _optional_float_list(result.decode_seconds_per_iteration),
            _optional_float_list(result.decode_device_seconds_per_iteration),
            _optional_float_list(result.decode_host_seconds_per_iteration),
            _optional_float_list(result.decode_submit_seconds_per_iteration),
            _optional_float_list(result.decode_extract_seconds_per_iteration),
            _optional_int_list(result.decode_tokens_per_iteration),
            _optional_float_list(result.prefill_drain_seconds_per_iteration),
            _optional_int_list(result.prefill_drain_tokens_per_iteration),
            _optional_float_list(result.generation_seconds_per_iteration),
            _optional_float_list(result.generation_host_seconds_per_iteration),
            _optional_int_list(result.generation_tokens_per_iteration),
            _optional_float(
                _optional_tokens_per_second(result.decode_tokens_per_iteration, result.decode_seconds_per_iteration)
            ),
            _optional_float(
                _optional_tokens_per_second(
                    result.decode_tokens_per_iteration, result.decode_device_seconds_per_iteration
                )
            ),
            _optional_float(
                _optional_tokens_per_second(
                    result.generation_tokens_per_iteration, result.generation_seconds_per_iteration
                )
            ),
            (
                _ratio(result.decode_tokens_per_second, vllm_by_case.get(result.case_name).decode_tokens_per_second)
                if result.case_name in vllm_by_case
                else ""
            ),
            (
                _ratio(result.total_tokens_per_second, vllm_by_case.get(result.case_name).total_tokens_per_second)
                if result.case_name in vllm_by_case
                else ""
            ),
            _parity_target_status(result, vllm_by_case.get(result.case_name)),
        ]
        for result in results
    ]
    with open(output_dir / "summary.md", "w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join("---" for _ in headers) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row) + " |\n")
    if stress_results:
        write_stress_summary(output_dir, stress_results)
    summarize_levanter_hlo(output_dir)
    write_artifact_manifest(output_dir)


def write_stress_summary(output_dir: Path, stress_results: list[StressResult]) -> None:
    headers = [
        "case",
        "backend",
        "concurrent reqs",
        "active seqs",
        "n",
        "requests",
        "success",
        "failed",
        "timeouts",
        "completion toks",
        "steady decode tok/s",
        "wall decode tok/s",
        "wall total tok/s",
        "load s",
        "wall s",
        "p50 ms",
        "p90 ms",
        "p99 ms",
        "max ms",
        "max request q",
        "max batch q",
        "max pages",
        "hbm bytes",
        "errors",
    ]
    rows = [
        [
            result.case_name,
            result.backend,
            str(result.concurrent_requests),
            str(result.active_sequences),
            str(result.n),
            str(result.total_requests),
            str(result.successful_requests),
            str(result.failed_requests),
            str(result.timeout_requests),
            str(result.completion_tokens),
            f"{result.steady_decode_tokens_per_second:.2f}",
            f"{result.wall_clock_decode_tokens_per_second:.2f}",
            f"{result.wall_clock_total_tokens_per_second:.2f}",
            f"{result.load_seconds:.3f}",
            f"{result.wall_clock_seconds:.3f}",
            f"{result.request_latency_ms_p50:.1f}",
            f"{result.request_latency_ms_p90:.1f}",
            f"{result.request_latency_ms_p99:.1f}",
            f"{result.request_latency_ms_max:.1f}",
            _optional_int(result.max_request_queue_depth),
            _optional_int(result.max_batch_queue_depth),
            _optional_int(result.max_pages),
            _optional_int(result.hbm_used_bytes),
            json.dumps(result.error_counts, sort_keys=True),
        ]
        for result in stress_results
    ]
    with open(output_dir / "stress_summary.md", "w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join("---" for _ in headers) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row) + " |\n")


def log_output_artifacts(output_dir: Path) -> None:
    summary = output_dir / "summary.md"
    env = output_dir / "env.json"
    artifacts = output_dir / "artifacts.json"
    if summary.exists():
        logger.info("Benchmark summary.md:\n%s", summary.read_text())
    stress_summary = output_dir / "stress_summary.md"
    if stress_summary.exists():
        logger.info("Benchmark stress_summary.md:\n%s", stress_summary.read_text())
    if env.exists():
        logger.info("Benchmark env.json:\n%s", env.read_text())
    hlo_summary = output_dir / "hlo_summary.md"
    if hlo_summary.exists():
        logger.info("Benchmark hlo_summary.md:\n%s", hlo_summary.read_text())
    reference_logits = output_dir / REFERENCE_LOGIT_ARTIFACT_MD
    if reference_logits.exists():
        logger.info("Benchmark levanter_reference_logits.md:\n%s", reference_logits.read_text())
    if artifacts.exists():
        logger.info("Benchmark artifacts.json:\n%s", artifacts.read_text())


def log_case_result(result: CaseResult) -> None:
    logger.info("Benchmark case result:\n%s", json.dumps(dataclasses.asdict(result), indent=2, sort_keys=True))


def selected_backends(backend: str) -> list[str]:
    if backend == "both":
        return ["vllm", "levanter"]
    return [backend]


def start_backend(
    args: argparse.Namespace,
    backend: str,
    *,
    output_dir: Path,
    checkpoint: str,
    tokenizer_name: str,
    cases: list[BenchmarkCase],
    prompts: dict[str, tuple[str, int]],
):
    if backend == "vllm":
        return start_vllm_server(
            model=args.model,
            port=args.vllm_port or find_open_port(),
            timeout=args.startup_timeout,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            load_format=args.load_format,
            cache_dir=args.cache_dir,
            extra_args=args.vllm_extra_arg,
            log_dir=output_dir / "vllm_profiles",
        )
    if backend == "levanter":
        return start_levanter_server(
            model=args.model,
            checkpoint=checkpoint,
            tokenizer_name=tokenizer_name,
            port=args.levanter_port or find_open_port(),
            timeout=args.startup_timeout,
            max_model_len=args.max_model_len,
            max_seqs=args.max_seqs,
            max_pages=args.max_pages,
            max_prefill_size=args.max_prefill_size,
            max_rounds=args.max_rounds,
            max_tokens_per_round=args.max_tokens_per_round,
            max_top_k=args.top_k,
            tensor_parallel_size=args.tensor_parallel_size,
            hbm_utilization=args.hbm_utilization,
            rpa_num_kv_pages_per_block=args.rpa_num_kv_pages_per_block,
            rpa_num_queries_per_block=args.rpa_num_queries_per_block,
            rpa_vmem_limit_bytes=args.rpa_vmem_limit_bytes,
            tpu_paged_attention_backend=TpuPagedAttentionBackend(args.levanter_tpu_paged_attention_backend),
            allow_reference_fallback=args.levanter_allow_reference_fallback,
            compute_dtype=args.levanter_compute_dtype,
            trainer_mp_policy=args.levanter_trainer_mp,
            tpu_inference_out_dtype=args.levanter_tpu_inference_out_dtype,
            preserve_attention_output_dtype=args.levanter_preserve_attention_output_dtype,
            sampler_top_k_mode=SamplerTopKMode(args.levanter_sampler_top_k_mode),
            use_streaming_greedy_lm_head=args.levanter_streaming_greedy_lm_head,
            batch_timeout=args.batch_timeout,
            kernel_artifacts_dir=output_dir / "levanter_hlo" if args.dump_levanter_kernels else None,
            return_logprobs=args.return_logprobs,
            reference_logit_check_dir=output_dir if args.check_levanter_reference_logits else None,
            reference_logit_check_cases=cases,
            reference_logit_check_prompts=prompts,
            reference_logit_atol=args.reference_logit_atol,
            reference_logit_rtol=args.reference_logit_rtol,
            reference_logit_max_prompts=args.reference_logit_max_prompts,
            reference_logit_decode_backends=(
                [TpuPagedAttentionBackend(backend) for backend in args.reference_logit_decode_backend]
                if args.reference_logit_decode_backend
                else None
            ),
            reference_logit_cache_dtype_policies=args.reference_logit_cache_dtype,
            reference_logit_only=args.reference_logit_only,
        )
    raise ValueError(f"Unknown backend {backend!r}")


def run_cases_for_backend(
    *,
    args: argparse.Namespace,
    handle: ServerHandle,
    cases: list[BenchmarkCase],
    prompts: dict[str, tuple[str, int]],
    output_dir: Path | None = None,
) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in cases:
        prompt, prompt_tokens = prompts[case.name]
        compile_including_seconds = 0.0 if args.warmup_rounds > 0 else None
        for i in range(args.warmup_rounds):
            logger.info("Warmup %s %s round %d", handle.name, case.name, i)
            warmup_result = run_case(
                handle=handle,
                case=case,
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                warmup=True,
                seed=args.seed,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                request_timeout=args.request_timeout,
                return_logprobs=args.return_logprobs,
            )
            assert compile_including_seconds is not None
            compile_including_seconds += warmup_result.steady_state_seconds
        for i in range(args.measure_rounds):
            logger.info("Measure %s %s round %d", handle.name, case.name, i)
            profile_enabled = args.profile_levanter and handle.name.startswith(PROFILED_BACKEND_PREFIX)
            profile_dir = None
            if output_dir is not None and profile_enabled:
                profile_dir = (
                    output_dir
                    / "levanter_profiles"
                    / _safe_artifact_name(handle.name)
                    / _safe_artifact_name(case.name)
                    / f"measure_{i}"
                )
            with _maybe_jax_profile(
                enabled=profile_enabled,
                profile_dir=profile_dir,
                step_name=f"{handle.name}:{case.name}:measure_{i}",
            ):
                result = run_case(
                    handle=handle,
                    case=case,
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    warmup=False,
                    seed=args.seed + i,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    request_timeout=args.request_timeout,
                    return_logprobs=args.return_logprobs,
                )
            result = dataclasses.replace(result, compile_including_seconds=compile_including_seconds)
            log_case_result(result)
            results.append(result)
        if args.levanter_diagnose_without_lm_head and handle.diagnose_without_lm_head is not None:
            diagnostic_compile_seconds = 0.0 if args.warmup_rounds > 0 else None
            for i in range(args.warmup_rounds):
                logger.info("Warmup %s:no_lm_head %s round %d", handle.name, case.name, i)
                warmup_result = handle.diagnose_without_lm_head(
                    case,
                    prompt,
                    prompt_tokens,
                    True,
                    args.seed,
                    args.temperature,
                    args.top_p,
                    args.request_timeout,
                )
                assert diagnostic_compile_seconds is not None
                diagnostic_compile_seconds += warmup_result.steady_state_seconds
            for i in range(args.measure_rounds):
                logger.info("Measure %s:no_lm_head %s round %d", handle.name, case.name, i)
                result = handle.diagnose_without_lm_head(
                    case,
                    prompt,
                    prompt_tokens,
                    False,
                    args.seed + i,
                    args.temperature,
                    args.top_p,
                    args.request_timeout,
                )
                result = dataclasses.replace(result, compile_including_seconds=diagnostic_compile_seconds)
                log_case_result(result)
                results.append(result)
        if args.levanter_diagnose_lm_head_no_sampling and handle.diagnose_with_lm_head_no_sampling is not None:
            diagnostic_compile_seconds = 0.0 if args.warmup_rounds > 0 else None
            for i in range(args.warmup_rounds):
                logger.info("Warmup %s:lm_head_no_sampling %s round %d", handle.name, case.name, i)
                warmup_result = handle.diagnose_with_lm_head_no_sampling(
                    case,
                    prompt,
                    prompt_tokens,
                    True,
                    args.seed,
                    args.temperature,
                    args.top_p,
                    args.request_timeout,
                )
                assert diagnostic_compile_seconds is not None
                diagnostic_compile_seconds += warmup_result.steady_state_seconds
            for i in range(args.measure_rounds):
                logger.info("Measure %s:lm_head_no_sampling %s round %d", handle.name, case.name, i)
                result = handle.diagnose_with_lm_head_no_sampling(
                    case,
                    prompt,
                    prompt_tokens,
                    False,
                    args.seed + i,
                    args.temperature,
                    args.top_p,
                    args.request_timeout,
                )
                result = dataclasses.replace(result, compile_including_seconds=diagnostic_compile_seconds)
                log_case_result(result)
                results.append(result)
    return results


def run_stress_cases_for_backend(
    *,
    args: argparse.Namespace,
    handle: ServerHandle,
    cases: list[BenchmarkCase],
    prompts: dict[str, tuple[str, int]],
) -> list[StressResult]:
    stress_results: list[StressResult] = []
    for case in cases:
        prompt, prompt_tokens = prompts[case.name]
        for i in range(args.warmup_rounds):
            logger.info("Stress warmup %s %s round %d", handle.name, case.name, i)
            run_case(
                handle=handle,
                case=case,
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                warmup=True,
                seed=args.seed,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                request_timeout=args.request_timeout,
                return_logprobs=args.return_logprobs,
            )
        concurrent_requests = args.stress_concurrent_requests or case.request_count
        logger.info(
            "Stress %s %s for %.1fs with %d concurrent HTTP requests (%d active sequences)",
            handle.name,
            case.name,
            args.stress_duration_seconds,
            concurrent_requests,
            concurrent_requests * case.n,
        )
        result = run_stress_case(
            handle=handle,
            case=case,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            request_timeout=args.request_timeout,
            return_logprobs=args.return_logprobs,
            duration_seconds=args.stress_duration_seconds,
            concurrent_requests=concurrent_requests,
            metrics_interval_seconds=args.stress_metrics_interval_seconds,
            max_requests=args.stress_max_requests,
        )
        logger.info("Benchmark stress result:\n%s", json.dumps(dataclasses.asdict(result), indent=2, sort_keys=True))
        stress_results.append(result)
    return stress_results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id used by vLLM and tokenizer.")
    parser.add_argument("--levanter-checkpoint", default=None, help="HF id or object-store checkpoint for Levanter.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer id/path. Defaults to --model.")
    parser.add_argument("--backend", choices=["levanter", "vllm", "both"], default="both")
    parser.add_argument("--matrix", default=DEFAULT_MATRIX)
    parser.add_argument("--case", action="append", default=None, help="Run only the named case. Repeatable.")
    parser.add_argument("--list-cases", action="store_true", help="Print benchmark cases and exit.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=DEFAULT_WARMUP_ROUNDS,
        help=(
            "Warmup request rounds per case before measuring steady state. "
            "Defaults to two because Levanter compiles the target prefill shape on its first two requests."
        ),
    )
    parser.add_argument("--measure-rounds", type=int, default=3)
    parser.add_argument(
        "--stress-duration-seconds",
        type=float,
        default=0.0,
        help="Run a sustained request-loop stress pass per selected case after warmup. Disabled by default.",
    )
    parser.add_argument(
        "--stress-concurrent-requests",
        type=int,
        default=None,
        help="Concurrent HTTP requests for stress mode. Defaults to each case's request_count.",
    )
    parser.add_argument(
        "--stress-metrics-interval-seconds",
        type=float,
        default=5.0,
        help="Queue-depth sampling interval during stress mode.",
    )
    parser.add_argument(
        "--stress-max-requests",
        type=int,
        default=None,
        help="Optional total request cap for bounded smoke tests. Production stress runs should leave this unset.",
    )
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--startup-timeout", type=float, default=3600.0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--load-format", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-seqs", type=int, default=128)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-prefill-size", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=32)
    parser.add_argument("--max-tokens-per-round", type=int, default=None)
    parser.add_argument("--hbm-utilization", type=float, default=0.85)
    parser.add_argument("--rpa-num-kv-pages-per-block", type=int, default=None)
    parser.add_argument("--rpa-num-queries-per-block", type=int, default=None)
    parser.add_argument("--rpa-vmem-limit-bytes", type=int, default=None)
    parser.add_argument(
        "--levanter-tpu-paged-attention-backend",
        choices=[backend.value for backend in TpuPagedAttentionBackend],
        default=TpuPagedAttentionBackend.AUTO.value,
        help="Paged attention backend used by Levanter. AUTO requires tpu-inference on TPU.",
    )
    parser.add_argument(
        "--levanter-allow-reference-fallback",
        action="store_true",
        help="Allow explicit reference fallback in Levanter on TPU for diagnostics.",
    )
    parser.add_argument(
        "--levanter-compute-dtype",
        choices=["bfloat16", "float32"],
        default="bfloat16",
        help="KV-cache dtype used by the Levanter inference engine.",
    )
    parser.add_argument(
        "--levanter-trainer-mp",
        default="f32",
        help=(
            "JMP mixed-precision policy for loading/running the Levanter model. "
            "Use bf16 for smaller-TPU same-dtype diagnostics; default f32 preserves the high-accuracy reference."
        ),
    )
    parser.add_argument(
        "--levanter-tpu-inference-out-dtype",
        choices=["bfloat16", "float32"],
        default=None,
        help=(
            "Override tpu-inference RPA output/accumulator dtype. "
            "float32 is slower but can reduce attention numerics drift."
        ),
    )
    parser.add_argument(
        "--levanter-preserve-attention-output-dtype",
        action="store_true",
        help="Do not cast paged-attention output back to the residual dtype before the attention output projection.",
    )
    parser.add_argument(
        "--levanter-sampler-top-k-mode",
        choices=[mode.value for mode in SamplerTopKMode],
        default=SamplerTopKMode.CANDIDATE.value,
        help="Levanter sampler top-k implementation used when --top-k is set.",
    )
    parser.add_argument("--batch-timeout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--return-logprobs",
        action="store_true",
        help="Request generated-token logprobs from both OpenAI-compatible backends, matching RL rollout collection.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vllm-port", type=int, default=None)
    parser.add_argument("--levanter-port", type=int, default=None)
    parser.add_argument("--vllm-extra-arg", action="append", default=[])
    parser.add_argument(
        "--dump-levanter-kernels",
        action="store_true",
        help="Write Levanter generation/prefill JAXPR and HLO artifacts under output-dir/levanter_hlo.",
    )
    parser.add_argument(
        "--profile-levanter",
        action="store_true",
        help="Capture JAX profiler traces for measured Levanter rounds under output-dir/levanter_profiles.",
    )
    parser.add_argument(
        "--check-levanter-reference-logits",
        action="store_true",
        help=(
            "Before serving, compare Levanter paged-decode logits against full causal Levanter logits for the "
            "fixed prompt corpus and write levanter_reference_logits.{json,md}."
        ),
    )
    parser.add_argument("--reference-logit-atol", type=float, default=0.5)
    parser.add_argument("--reference-logit-rtol", type=float, default=0.05)
    parser.add_argument(
        "--reference-logit-decode-backend",
        action="append",
        choices=[
            TpuPagedAttentionBackend.TPU_INFERENCE.value,
            TpuPagedAttentionBackend.JAX_RPA.value,
            TpuPagedAttentionBackend.REFERENCE.value,
        ],
        default=[],
        help=(
            "Concrete decode backend to check against full causal logits. Repeat to run a diagnostic matrix. "
            "Defaults to the configured Levanter serving backend."
        ),
    )
    parser.add_argument(
        "--reference-logit-cache-dtype",
        action="append",
        choices=REFERENCE_LOGIT_CACHE_DTYPE_POLICIES,
        default=[],
        help=(
            "KV-cache dtype policy for reference-logit checks. Repeat to run dtype diagnostics. "
            "Defaults to auto, which uses bf16 for tpu-inference and trainer compute dtype otherwise."
        ),
    )
    parser.add_argument(
        "--reference-logit-max-prompts",
        type=int,
        default=None,
        help="Limit reference-logit checking to the first N distinct prompts. Defaults to every distinct prompt.",
    )
    parser.add_argument(
        "--reference-logit-only",
        action="store_true",
        help="Run Levanter reference-logit checks, log artifacts immediately, then exit without serving benchmarks.",
    )
    parser.add_argument(
        "--levanter-diagnose-without-lm-head",
        action="store_true",
        help="Also time a Levanter-only diagnostic that skips LM-head projection and sampling after prefill.",
    )
    parser.add_argument(
        "--levanter-diagnose-lm-head-no-sampling",
        action="store_true",
        help="Also time a Levanter-only diagnostic that computes LM-head logits but skips sampling after prefill.",
    )
    parser.add_argument(
        "--levanter-streaming-greedy-lm-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use block-streaming LM-head argmax/logprob computation for greedy Levanter decode requests. "
            "Pass --no-levanter-streaming-greedy-lm-head to benchmark the legacy materialized-logits path."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")

    cases = matrix_cases(args.matrix)
    if args.case:
        selected = set(args.case)
        cases = [case for case in cases if case.name in selected]
        missing = selected - {case.name for case in cases}
        if missing:
            raise ValueError(f"Unknown case(s): {sorted(missing)}")

    if args.list_cases:
        for case in cases:
            print(json.dumps(dataclasses.asdict(case), sort_keys=True))
        return 0

    stress_enabled = args.stress_duration_seconds > 0.0 or args.stress_max_requests is not None
    if args.warmup_rounds < 0 or args.measure_rounds < 0:
        raise ValueError("--warmup-rounds and --measure-rounds must be >= 0")
    if args.measure_rounds < 1 and not stress_enabled:
        raise ValueError("--measure-rounds must be >= 1 unless stress mode is enabled")
    if args.stress_duration_seconds < 0.0:
        raise ValueError("--stress-duration-seconds must be non-negative")
    if args.stress_concurrent_requests is not None and args.stress_concurrent_requests < 1:
        raise ValueError("--stress-concurrent-requests must be positive when set")
    if args.stress_metrics_interval_seconds <= 0.0:
        raise ValueError("--stress-metrics-interval-seconds must be positive")
    if args.stress_max_requests is not None and args.stress_max_requests < 1:
        raise ValueError("--stress-max-requests must be positive when set")
    if args.temperature < 0.0:
        raise ValueError("--temperature must be non-negative")
    if not (0.0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in the interval (0, 1]")
    if args.top_k is not None and args.top_k < 1:
        raise ValueError("--top-k must be positive when set")
    if args.return_logprobs and (args.levanter_diagnose_without_lm_head or args.levanter_diagnose_lm_head_no_sampling):
        raise ValueError("Levanter no-sampling diagnostics do not produce generated-token logprobs")
    if args.check_levanter_reference_logits and "levanter" not in selected_backends(args.backend):
        raise ValueError("--check-levanter-reference-logits requires the Levanter backend")
    if args.reference_logit_atol < 0.0 or args.reference_logit_rtol < 0.0:
        raise ValueError("--reference-logit-atol and --reference-logit-rtol must be non-negative")
    if args.reference_logit_max_prompts is not None and args.reference_logit_max_prompts < 1:
        raise ValueError("--reference-logit-max-prompts must be positive when set")
    if args.reference_logit_only and not args.check_levanter_reference_logits:
        raise ValueError("--reference-logit-only requires --check-levanter-reference-logits")
    if not args.reference_logit_cache_dtype:
        args.reference_logit_cache_dtype = ["auto"]
    if "vllm" in selected_backends(args.backend) and args.temperature == 0.0:
        greedy_multi_generation = [case.name for case in cases if case.n > 1]
        if greedy_multi_generation:
            raise ValueError(
                "vLLM rejects n > 1 with greedy sampling. "
                f"Use --temperature > 0 for sampled clone-prefix cases or omit {greedy_multi_generation}."
            )
    if args.max_prefill_size is not None and args.max_prefill_size < 1:
        raise ValueError("--max-prefill-size must be positive when set")
    if args.tensor_parallel_size < 1:
        raise ValueError("--tensor-parallel-size must be positive")
    if args.max_rounds < 1:
        raise ValueError("--max-rounds must be positive")
    if args.max_tokens_per_round is not None and args.max_tokens_per_round < 1:
        raise ValueError("--max-tokens-per-round must be positive when set")
    for name in ("rpa_num_kv_pages_per_block", "rpa_num_queries_per_block", "rpa_vmem_limit_bytes"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive when set")
    max_pages_per_seq = (args.max_model_len + 127) // 128
    if args.rpa_num_kv_pages_per_block is not None and args.rpa_num_kv_pages_per_block > max_pages_per_seq:
        raise ValueError(
            "--rpa-num-kv-pages-per-block must be <= ceil(max_model_len / 128); "
            f"got {args.rpa_num_kv_pages_per_block} > {max_pages_per_seq}"
        )

    output_dir = Path(args.output_dir or f"qwen3_tpu_inference_parity_{int(time.time())}")
    checkpoint = args.levanter_checkpoint or args.model
    tokenizer_name = args.tokenizer or args.model
    tokenizer = load_tokenizer(tokenizer_name)
    prompts = {case.name: _prompt_for_token_count(tokenizer, case.input_tokens) for case in cases}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_prompt_corpus(output_dir, cases, prompts, tokenizer)

    results: list[CaseResult] = []
    stress_results: list[StressResult] = []
    backend_envs: dict[str, dict[str, Any]] = {}
    backend_commands: dict[str, list[str] | None] = {}
    env = {
        "model": args.model,
        "levanter_checkpoint": checkpoint,
        "tokenizer": tokenizer_name,
        "matrix": args.matrix,
        "argv": sys.argv[1:] if argv is None else argv,
        "backend": args.backend,
        "backend_envs": backend_envs,
        "backend_commands": backend_commands,
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS"),
        "warmup_rounds": args.warmup_rounds,
        "measure_rounds": args.measure_rounds,
        "stress_duration_seconds": args.stress_duration_seconds,
        "stress_concurrent_requests": args.stress_concurrent_requests,
        "stress_metrics_interval_seconds": args.stress_metrics_interval_seconds,
        "stress_max_requests": args.stress_max_requests,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "levanter_sampler_top_k_mode": args.levanter_sampler_top_k_mode,
        "levanter_compute_dtype": args.levanter_compute_dtype,
        "levanter_trainer_mp": args.levanter_trainer_mp,
        "levanter_tpu_inference_out_dtype": args.levanter_tpu_inference_out_dtype,
        "levanter_preserve_attention_output_dtype": args.levanter_preserve_attention_output_dtype,
        "profile_levanter": args.profile_levanter,
        "check_levanter_reference_logits": args.check_levanter_reference_logits,
        "reference_logit_atol": args.reference_logit_atol,
        "reference_logit_rtol": args.reference_logit_rtol,
        "reference_logit_max_prompts": args.reference_logit_max_prompts,
        "reference_logit_decode_backend": args.reference_logit_decode_backend,
        "reference_logit_cache_dtype": args.reference_logit_cache_dtype,
        "reference_logit_only": args.reference_logit_only,
    }
    for backend in selected_backends(args.backend):
        handle = start_backend(
            args,
            backend,
            output_dir=output_dir,
            checkpoint=checkpoint,
            tokenizer_name=tokenizer_name,
            cases=cases,
            prompts=prompts,
        )
        try:
            if args.reference_logit_only:
                backend_envs[handle.name] = _runtime_env_snapshot(include_jax_devices=backend == "levanter")
                backend_commands[handle.name] = handle.command
                write_outputs(output_dir, results, env, stress_results)
                logger.info("Benchmark reference-logit-only final artifacts:")
                log_output_artifacts(output_dir)
                continue
            if args.measure_rounds > 0:
                results.extend(
                    run_cases_for_backend(
                        args=args,
                        handle=handle,
                        cases=cases,
                        prompts=prompts,
                        output_dir=output_dir,
                    )
                )
            if stress_enabled:
                stress_results.extend(
                    run_stress_cases_for_backend(
                        args=args,
                        handle=handle,
                        cases=cases,
                        prompts=prompts,
                    )
                )
            backend_envs[handle.name] = _runtime_env_snapshot(include_jax_devices=backend == "levanter")
            backend_commands[handle.name] = handle.command
            write_outputs(output_dir, results, env, stress_results)
            logger.info("Benchmark partial outputs after %s backend:", handle.name)
            log_output_artifacts(output_dir)
        finally:
            handle.close()

    write_outputs(output_dir, results, env, stress_results)
    log_output_artifacts(output_dir)
    logger.info("Wrote benchmark outputs to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
