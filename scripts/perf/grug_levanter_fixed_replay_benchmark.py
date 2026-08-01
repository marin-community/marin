#!/usr/bin/env python3
"""Benchmark one native Grug update on the fixed rollout replay.

This is an evidence driver, not a new training entry point. It verifies the
content-addressed replay, loads the June step-630 native checkpoint, and runs
one warmed update on a compact Grug mesh. The 4,096-sequence replay is repacked
as 128 global microbatches of 32 sequences, with one sequence per H100, and one
optimizer boundary after exact gradient accumulation.

The two objectives deliberately have different boundaries:

* ``operational`` uses the current native BF16 policy, fused CE with z-loss,
  ring expert parallelism, and a fresh AdamH optimizer.
* ``matched_ce`` applies the same pending router bias but uses only token-mean
  next-token CE through forward and backward. It has no optimizer.

Compilation, checkpoint loading, replay preparation, and the warmup execution
all occur before any reported timing.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import gc
import hashlib
import json
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlparse

import equinox as eqx
import fsspec
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
from haliax.partitioning import set_mesh
from iris.cluster.client.job_info import get_job_info
from iris.runtime.jax_init import initialize_jax
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.checkpoint import load_checkpoint
from levanter.grug.attention import AttentionMask
from levanter.grug.attention._fa4_cute_backend import cutlass_cute_available
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import compact_grug_mesh
from safetensors import safe_open

from experiments.june_tpu_67b_a2b.moe.model import Transformer
from experiments.june_tpu_67b_a2b.moe.sft_67b_a2b_2stage import _model, _optimizer
from experiments.june_tpu_67b_a2b.moe.train import GrugTrainState, _apply_qb_betas
from scripts.perf.grug_fixed_replay import build_loss_weight, repacked_operational_micro_loss

CHUNK_BYTES = 16 * 1024 * 1024
EXPECTED_FIELDS = (
    "action_log_probs",
    "advantages",
    "attention_mask",
    "loss_mask",
    "response_mask",
    "returns",
    "sequences",
)
EXPECTED_NONE_FIELDS = ("base_action_log_probs", "is_last_step", "rollout_logprobs", "values")
MP_POLICY = "params=float32,compute=bfloat16,output=bfloat16"
_BATCH_AXES = ("replica_dcn", "data", "expert")


class ReplayArrays(NamedTuple):
    tokens: jax.Array
    loss_weight: jax.Array
    segment_ids: jax.Array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-s3-uri", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--logical-batch-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--objective", choices=("operational", "matched_ce"), required=True)
    parser.add_argument("--mode", choices=("preflight", "headline"), required=True)
    parser.add_argument("--result-s3-uri", required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument(
        "--diagnostic-microbatches",
        type=int,
        help=(
            "Run only this many replay microbatches through matched CE while retaining the full logical-batch "
            "loss denominator. Diagnostic runs are never headline eligible."
        ),
    )
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--profile-s3-prefix")
    return parser.parse_args()


def split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"not an S3 object URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def numpy_bytes(value: np.ndarray) -> memoryview:
    return memoryview(np.ascontiguousarray(value)).cast("B")


def _download_verified(record: Mapping[str, Any], path: Path) -> None:
    if path.exists() and path.stat().st_size == int(record["bytes"]) and sha256_file(path) == record["sha256"]:
        return
    uri = str(record["s3_uri"])
    split_s3_uri(uri)
    temporary = path.with_suffix(path.suffix + ".part")
    with fsspec.open(uri, "rb") as source, temporary.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=CHUNK_BYTES)
    if temporary.stat().st_size != int(record["bytes"]):
        raise RuntimeError(f"{record['s3_uri']}: byte count mismatch")
    actual = sha256_file(temporary)
    if actual != record["sha256"]:
        raise RuntimeError(f"{record['s3_uri']}: expected sha256 {record['sha256']}, got {actual}")
    temporary.replace(path)


def _verify_shard_metadata(path: Path, record: Mapping[str, Any], logical_sha256: str) -> None:
    with safe_open(path, framework="numpy") as handle:
        metadata = handle.metadata() or {}
        fields = set(handle.keys())
        row_counts = {int(handle.get_slice(name).get_shape()[0]) for name in fields}
    expected_metadata = {
        "logical_batch_sha256": logical_sha256,
        "rank": str(record["rank"]),
        "start": str(record["start"]),
        "end": str(record["end"]),
    }
    if metadata != expected_metadata:
        raise RuntimeError(f"{path.name}: safetensors metadata mismatch: {metadata}")
    if fields != set(EXPECTED_FIELDS):
        raise RuntimeError(f"{path.name}: unexpected fields {sorted(fields)}")
    expected_rows = int(record["end"]) - int(record["start"])
    if row_counts != {expected_rows}:
        raise RuntimeError(f"{path.name}: row counts {row_counts} disagree with {expected_rows}")


def verify_replay_directory(manifest: Mapping[str, Any], paths: Sequence[Path], expected_logical_sha256: str) -> None:
    """Reconstruct the recovery tool's global canonical digest."""

    canonical = hashlib.sha256()
    for name in sorted(manifest["batch"]["fields"]):
        evidence = manifest["batch"]["fields"][name]
        if evidence.get("value", object()) is None:
            if name not in EXPECTED_NONE_FIELDS:
                raise RuntimeError(f"unexpected null replay field {name}")
            canonical.update(f"{name}:none\n".encode())
            continue
        if name not in EXPECTED_FIELDS:
            raise RuntimeError(f"unexpected tensor replay field {name}")
        header = {key: evidence[key] for key in ("dtype", "shape", "stride", "numel", "nbytes", "sha256")}
        canonical.update(json.dumps({"name": name, **header}, sort_keys=True).encode())
        canonical.update(b"\n")
        field_digest = hashlib.sha256()
        rows = 0
        for path in paths:
            with safe_open(path, framework="numpy") as handle:
                value = handle.get_tensor(name)
            rows += int(value.shape[0])
            raw = numpy_bytes(value)
            field_digest.update(raw)
            canonical.update(raw)
        if rows != int(evidence["shape"][0]):
            raise RuntimeError(f"{name}: reconstructed {rows} rows, expected {evidence['shape'][0]}")
        if field_digest.hexdigest() != evidence["sha256"]:
            raise RuntimeError(f"{name}: global field digest mismatch")
    actual = canonical.hexdigest()
    if actual != expected_logical_sha256:
        raise RuntimeError(f"logical replay digest mismatch: expected {expected_logical_sha256}, got {actual}")


def load_verified_manifest(
    manifest_uri: str,
    expected_manifest_sha256: str,
    expected_logical_sha256: str,
    directory: Path,
) -> tuple[dict[str, Any], list[Path]]:
    split_s3_uri(manifest_uri)
    with fsspec.open(manifest_uri, "rb") as source:
        payload = source.read()
    actual_manifest_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError(
            f"manifest digest mismatch: expected {expected_manifest_sha256}, got {actual_manifest_sha256}"
        )
    manifest = json.loads(payload)
    if manifest["schema_version"] != 1:
        raise RuntimeError(f"unsupported replay schema {manifest['schema_version']}")
    if manifest["logical_batch_sha256"] != expected_logical_sha256:
        raise RuntimeError("manifest names a different logical replay")
    if manifest["batch"]["batch_size"] != 4096 or len(manifest["shards"]) != 32:
        raise RuntimeError("fixed headline replay must contain 4,096 rows in 32 shards")
    expected_start = 0
    for rank, record in enumerate(manifest["shards"]):
        if int(record["rank"]) != rank or int(record["start"]) != expected_start:
            raise RuntimeError(f"non-contiguous replay shard record: {record}")
        expected_start = int(record["end"])
    if expected_start != 4096:
        raise RuntimeError(f"replay shards end at row {expected_start}, expected 4096")

    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / record["filename"] for record in manifest["shards"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(_download_verified, record, path) for record, path in zip(manifest["shards"], paths, strict=True)
        ]
        for future in futures:
            future.result()
    for record, path in zip(manifest["shards"], paths, strict=True):
        _verify_shard_metadata(path, record, expected_logical_sha256)
    verify_replay_directory(manifest, paths, expected_logical_sha256)
    return manifest, paths


def load_process_replay(
    paths: Sequence[Path], *, process_index: int, process_count: int, rows_per_shard: int
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    local_device_count = jax.local_device_count()
    shard_count = process_count * local_device_count
    if shard_count not in (8, 32):
        raise RuntimeError(f"benchmark requires 8 or 32 H100s, found {shard_count}")
    if rows_per_shard not in (1, 128):
        raise RuntimeError(f"rows_per_shard must be 1 or 128, got {rows_per_shard}")
    if shard_count == 32 and rows_per_shard != 128:
        raise RuntimeError("the 32-H100 headline must consume all 128 rows per shard")
    if shard_count == 8 and rows_per_shard != 1:
        raise RuntimeError("the one-node preflight must consume one row per shard")

    selected: dict[str, list[np.ndarray]] = {name: [] for name in EXPECTED_FIELDS}
    rank_start = process_index * local_device_count
    for rank in range(rank_start, rank_start + local_device_count):
        with safe_open(paths[rank], framework="numpy") as handle:
            for name in EXPECTED_FIELDS:
                selected[name].append(handle.get_tensor(name)[:rows_per_shard])

    # [local_device, micro, ...] -> [micro, local_device, ...]. Device/rank r
    # therefore consumes replay row r*128+t at microstep t.
    arrays = {name: np.swapaxes(np.stack(values, axis=0), 0, 1) for name, values in selected.items()}
    sequences = arrays["sequences"].astype(np.int32, copy=False)
    attention_mask = arrays["attention_mask"]
    loss_weight = build_loss_weight(
        arrays["loss_mask"].reshape(-1, arrays["loss_mask"].shape[-1]),
        int(sequences.shape[-1]),
    ).reshape(sequences.shape)
    segment_ids = np.where(attention_mask != 0, 0, -1).astype(np.int32)
    if not np.array_equal(arrays["loss_mask"] != 0, arrays["response_mask"] != 0):
        raise RuntimeError("loss_mask and response_mask select different action tokens")
    if np.any((loss_weight != 0) & (attention_mask == 0)):
        raise RuntimeError("loss weights include padding positions")
    counts = {
        "logical_sequences": int(sequences.shape[0] * local_device_count),
        "allocated_tokens": int(sequences.size),
        "nonpadding_tokens": int(np.count_nonzero(attention_mask)),
        "loss_tokens": int(np.count_nonzero(loss_weight)),
        "microbatches": int(sequences.shape[0]),
        "global_microbatch_size": shard_count,
        "sequence_length": int(sequences.shape[-1]),
    }
    return {"tokens": sequences, "loss_weight": loss_weight, "segment_ids": segment_ids}, counts


def aggregate_process_counts(local_counts: Mapping[str, int]) -> dict[str, int]:
    """Sum data-dependent replay counts and verify common structural counts."""

    summed_names = ("logical_sequences", "allocated_tokens", "nonpadding_tokens", "loss_tokens")
    local_summed = np.asarray([local_counts[name] for name in summed_names], dtype=np.int64)
    gathered_summed = np.asarray(multihost_utils.process_allgather(local_summed)).reshape(-1, len(summed_names))

    common_names = ("microbatches", "global_microbatch_size", "sequence_length")
    local_common = np.asarray([local_counts[name] for name in common_names], dtype=np.int64)
    gathered_common = np.asarray(multihost_utils.process_allgather(local_common)).reshape(-1, len(common_names))
    if not np.all(gathered_common == gathered_common[0]):
        raise RuntimeError(f"replay structure differs across processes: {gathered_common.tolist()}")

    counts = dict(local_counts)
    counts.update(zip(summed_names, np.sum(gathered_summed, axis=0).tolist(), strict=True))
    counts.update(zip(common_names, gathered_common[0].tolist(), strict=True))
    return {name: int(value) for name, value in counts.items()}


def make_global_replay(mesh, local: Mapping[str, np.ndarray], counts: Mapping[str, int]) -> ReplayArrays:
    microbatches = int(counts["microbatches"])
    global_batch = int(counts["global_microbatch_size"])
    sequence_length = int(counts["sequence_length"])
    sharding = NamedSharding(mesh, P(None, _BATCH_AXES, None))

    def put(name: str, shape: tuple[int, ...]) -> jax.Array:
        return jax.make_array_from_process_local_data(sharding, local[name], shape)

    shape = (microbatches, global_batch, sequence_length)
    return ReplayArrays(
        tokens=put("tokens", shape),
        loss_weight=put("loss_weight", shape),
        segment_ids=put("segment_ids", shape),
    )


def _tree_zeros_like(tree):
    return jax.tree.map(lambda value: jnp.zeros_like(value) if eqx.is_inexact_array(value) else None, tree)


def _tree_add(left, right):
    return jax.tree.map(
        lambda a, b: None if a is None else a + b,
        left,
        right,
        is_leaf=lambda value: value is None,
    )


@jax.jit
def _tree_finite_details(leaves: tuple[jax.Array, ...]) -> tuple[jax.Array, jax.Array, jax.Array]:
    finite = jnp.stack([jnp.all(jnp.isfinite(value)) for value in leaves])
    nonfinite_counts = jnp.stack([jnp.sum(jnp.logical_not(jnp.isfinite(value)), dtype=jnp.int64) for value in leaves])
    finite_maxima = jnp.stack(
        [
            jnp.max(jnp.where(jnp.isfinite(value), jnp.abs(value.astype(jnp.float32)), 0.0))
            if value.size
            else jnp.array(0.0, dtype=jnp.float32)
            for value in leaves
        ]
    )
    return finite, nonfinite_counts, finite_maxima


def tree_finite_evidence(tree) -> dict[str, Any]:
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(tree, is_leaf=lambda value: value is None)
    selected = [
        (jax.tree_util.keystr(path), value)
        for path, value in path_leaves
        if value is not None and eqx.is_inexact_array(value)
    ]
    if not selected:
        return {
            "checked_arrays": 0,
            "checked_elements": 0,
            "nonfinite_arrays": 0,
            "nonfinite_elements": 0,
            "max_finite_abs": 0.0,
            "leaves": [],
        }

    paths, leaves = zip(*selected, strict=True)
    # Several stacked Grug parameter leaves contain more than 2**31 elements.
    # Keep the wider integer scope local to this post-timing evidence pass.
    with jax.enable_x64():
        finite, nonfinite_counts, finite_maxima = _tree_finite_details(leaves)
    finite = np.asarray(finite)
    nonfinite_counts = np.asarray(nonfinite_counts)
    finite_maxima = np.asarray(finite_maxima)
    leaf_evidence = [
        {
            "path": path,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "elements": int(value.size),
            "finite": bool(is_finite),
            "nonfinite_elements": int(nonfinite_count),
            "max_finite_abs": float(maximum),
        }
        for path, value, is_finite, nonfinite_count, maximum in zip(
            paths, leaves, finite, nonfinite_counts, finite_maxima, strict=True
        )
    ]
    return {
        "checked_arrays": len(leaves),
        "checked_elements": sum(int(value.size) for value in leaves),
        "nonfinite_arrays": int(np.count_nonzero(np.logical_not(finite))),
        "nonfinite_elements": sum(int(value) for value in nonfinite_counts),
        "max_finite_abs": float(np.max(finite_maxima)),
        "leaves": leaf_evidence,
    }


def validate_output(objective: str, output) -> tuple[dict[str, float | int], dict[str, Any]]:
    if objective == "matched_ce":
        loss, grads = output
        values: dict[str, float | int] = {"loss": float(loss)}
        finite_evidence = tree_finite_evidence(grads)
    else:
        next_state, metric_values = output
        values = {
            "loss": float(metric_values[0]),
            "next_qb_beta_sum": float(metric_values[1]),
            "next_step": int(next_state.step),
        }
        finite_evidence = tree_finite_evidence(next_state)
        if values["next_step"] != 1:
            raise RuntimeError(f"optimizer boundary ended at step {values['next_step']}")
    if not all(np.isfinite(value) for value in values.values()) or finite_evidence["nonfinite_arrays"] != 0:
        raise RuntimeError(f"non-finite benchmark output: values={values}, finite_evidence={finite_evidence}")
    return values, finite_evidence


def _matched_cross_entropy_sum(
    model: Transformer,
    tokens: jax.Array,
    loss_weight: jax.Array,
    mask: AttentionMask,
) -> jax.Array:
    """Pure next-token CE, without native router or z-loss terms."""

    hidden, _ = model(tokens, mask=mask)
    labels = jnp.concatenate([tokens[:, 1:], tokens[:, :1] * 0], axis=1).astype(jnp.int32)
    return fused_linear_softmax_cross_entropy_loss(
        hidden,
        model.output_proj,
        labels,
        weight=loss_weight,
        reduction="sum",
        logsumexp_weight=None,
        dtype=jnp.float32,
        implementation=model.config.ce_implementation,
    )


def _logical_gradients(
    params: Transformer,
    replay: ReplayArrays,
    *,
    mp: jmp.Policy,
    global_loss_tokens: int,
    logsumexp_weight: float | None,
    include_operational_terms: bool,
):
    grad0 = _tree_zeros_like(params)
    beta0 = jnp.zeros((params.config.num_layers, params.config.num_experts), dtype=jnp.float32)

    def micro_loss(model, tokens, loss_weight, segment_ids):
        compute_model = mp.cast_to_compute(model)
        mask = AttentionMask.causal().with_segment_ids(segment_ids)
        if include_operational_terms:
            _, metrics = compute_model.next_token_loss(
                tokens,
                loss_weight,
                mask=mask,
                reduction="sum",
                logsumexp_weight=logsumexp_weight,
                return_router_metrics=True,
            )
            loss = repacked_operational_micro_loss(
                metrics["train/cross_entropy_loss"],
                metrics["train/router/aux_loss_weighted"],
                global_loss_tokens=global_loss_tokens,
                microbatch_count=replay.tokens.shape[0],
            )
            return loss, metrics["qb_beta_per_layer"]
        loss_sum = _matched_cross_entropy_sum(
            compute_model,
            tokens,
            loss_weight,
            mask,
        )
        return loss_sum / global_loss_tokens, beta0

    # The per-microbatch reverse pass lives inside the scan. This retains one
    # sequence per device worth of activations while accumulating FP32
    # gradients, instead of retaining 128 microbatches for a reverse scan.
    def body(carry, microbatch):
        grads, loss, beta = carry
        tokens, loss_weight, segment_ids = microbatch
        (micro_loss_value, micro_beta), micro_grads = jax.value_and_grad(micro_loss, has_aux=True)(
            params, tokens, loss_weight, segment_ids
        )
        return (
            _tree_add(grads, micro_grads),
            loss + micro_loss_value,
            beta + micro_beta,
        ), None

    (grads, loss, beta_sum), _ = jax.lax.scan(
        body,
        (grad0, jnp.array(0.0, dtype=jnp.float32), beta0),
        replay,
    )
    return grads, loss, beta_sum / replay.tokens.shape[0]


def make_matched_step(mp: jmp.Policy, global_loss_tokens: int):
    def step(params: Transformer, pending_qb_betas: jax.Array, replay: ReplayArrays):
        params = _apply_qb_betas(params, pending_qb_betas)
        grads, loss, _ = _logical_gradients(
            params,
            replay,
            mp=mp,
            global_loss_tokens=global_loss_tokens,
            logsumexp_weight=None,
            include_operational_terms=False,
        )
        return loss, grads

    return jax.jit(step, donate_argnums=(0,))


def make_operational_step(optimizer: optax.GradientTransformation, mp: jmp.Policy, global_loss_tokens: int):
    one = jnp.array(1, dtype=jnp.int32)

    def step(state: GrugTrainState, replay: ReplayArrays):
        params = _apply_qb_betas(state.params, state.pending_qb_betas)
        grads, loss, next_qb_betas = _logical_gradients(
            params,
            replay,
            mp=mp,
            global_loss_tokens=global_loss_tokens,
            logsumexp_weight=1e-4,
            include_operational_terms=True,
        )
        updates, opt_state = optimizer.update(grads, state.opt_state, params)
        next_params = optax.apply_updates(params, updates)
        next_state = dataclasses.replace(
            state,
            step=state.step + one,
            params=next_params,
            opt_state=opt_state,
            pending_qb_betas=next_qb_betas,
        )
        metrics = jnp.stack([loss, jnp.sum(next_qb_betas.astype(jnp.float32))])
        return next_state, metrics

    return jax.jit(step, donate_argnums=(0,))


def initialize_params(model_config, mp: jmp.Policy, checkpoint: str, mesh):
    @jax.jit
    def init(key):
        return mp.cast_to_param(Transformer.init(model_config, key=key))

    params = init(jax.random.PRNGKey(0))
    pending = jnp.zeros((model_config.num_layers, model_config.num_experts), dtype=jnp.float32)
    loaded = load_checkpoint(
        {"params": params, "pending_qb_betas": pending},
        checkpoint,
        mesh=mesh,
        allow_partial=True,
    )
    return loaded["params"], loaded["pending_qb_betas"]


def model_fingerprint(params: Transformer, pending_qb_betas: jax.Array) -> list[float]:
    leaves = [value for value in jax.tree.leaves(params) if eqx.is_inexact_array(value)]

    @jax.jit
    def fingerprint(model_leaves, betas):
        # Index global corners directly. Flattening a partitioned array can require
        # an illegal replicated sharding (and a model-sized temporary allocation).
        firsts = [value[(0,) * value.ndim].astype(jnp.float32) for value in model_leaves]
        lasts = [value[tuple(size - 1 for size in value.shape)].astype(jnp.float32) for value in model_leaves]
        return jnp.stack(
            [
                sum(firsts, start=jnp.array(0.0, dtype=jnp.float32)),
                sum(lasts, start=jnp.array(0.0, dtype=jnp.float32)),
                jnp.sum(betas.astype(jnp.float32)),
                jnp.sum(jnp.square(betas.astype(jnp.float32))),
            ]
        )

    return [float(value) for value in fingerprint(leaves, pending_qb_betas)]


def nvidia_smi_rows() -> list[dict[str, Any]]:
    query = "uuid,name,memory.total,memory.used,pci.bus_id"
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in completed.stdout.splitlines():
        uuid, name, total, used, pci_bus = [part.strip() for part in line.split(",", 4)]
        rows.append(
            {
                "uuid": uuid,
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "pci_bus_id": pci_bus,
            }
        )
    return rows


class MemorySampler:
    def __init__(self, interval_seconds: float = 0.1):
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[list[dict[str, Any]]] = []

    def start(self) -> None:
        self.samples.append(nvidia_smi_rows())

        def sample() -> None:
            while not self._stop.wait(self.interval_seconds):
                self.samples.append(nvidia_smi_rows())

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.samples.append(nvidia_smi_rows())
        first = self.samples[0]
        by_uuid: dict[str, list[int]] = {row["uuid"]: [] for row in first}
        for sample in self.samples:
            for row in sample:
                by_uuid[row["uuid"]].append(int(row["memory_used_mib"]))
        return {
            "sample_count": len(self.samples),
            "interval_seconds": self.interval_seconds,
            "baseline_used_mib": {row["uuid"]: row["memory_used_mib"] for row in first},
            "peak_used_mib": {uuid: max(values) for uuid, values in by_uuid.items()},
        }


def gather_json(value: Any) -> list[Any]:
    encoded = json.dumps(value, sort_keys=True).encode()
    lengths = np.asarray(multihost_utils.process_allgather(np.asarray([len(encoded)], dtype=np.int32))).reshape(-1)
    maximum = int(lengths.max())
    padded = np.zeros((maximum,), dtype=np.uint8)
    padded[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    gathered = np.asarray(multihost_utils.process_allgather(padded)).reshape(jax.process_count(), maximum)
    return [json.loads(bytes(row[: int(length)])) for row, length in zip(gathered, lengths, strict=True)]


def upload_json(uri: str, result: dict[str, Any]) -> tuple[str, str]:
    unsigned = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["result_sha256"] = hashlib.sha256(unsigned).hexdigest()
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    split_s3_uri(uri)
    with fsspec.open(uri, "wb") as destination:
        destination.write(payload)
    return result["result_sha256"], hashlib.sha256(payload).hexdigest()


def upload_profile_directory(directory: Path, prefix_uri: str, process_index: int) -> list[dict[str, Any]]:
    """Upload one process's JAX trace before the ephemeral Iris task exits."""

    split_s3_uri(prefix_uri.rstrip("/") + "/placeholder")
    uploaded = []
    for path in sorted(candidate for candidate in directory.rglob("*") if candidate.is_file()):
        relative = path.relative_to(directory).as_posix()
        uri = f"{prefix_uri.rstrip('/')}/process-{process_index:03d}/{relative}"
        with path.open("rb") as source, fsspec.open(uri, "wb") as destination:
            shutil.copyfileobj(source, destination, length=CHUNK_BYTES)
        uploaded.append(
            {
                "s3_uri": uri,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not uploaded:
        raise RuntimeError(f"JAX profiler created no files under {directory}")
    return uploaded


def run_sample(
    callable_: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    label: str,
    profile_dir: Path | None,
) -> tuple[float, Any, dict[str, Any]]:
    multihost_utils.sync_global_devices(f"{label}-ready")
    sampler = MemorySampler()
    sampler.start()
    if profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=True)
        jax.profiler.start_trace(str(profile_dir))
    started = time.perf_counter()
    output = callable_(*args)
    jax.block_until_ready(output)
    elapsed = time.perf_counter() - started
    if profile_dir is not None:
        jax.profiler.stop_trace()
    memory = sampler.stop()
    return elapsed, output, memory


def _config_evidence(
    model_config,
    mesh,
    *,
    objective: str,
    executed_microbatches: int,
    global_microbatch_size: int,
    global_loss_tokens: int,
    diagnostic_microbatches: int | None,
) -> dict[str, Any]:
    return {
        "mixed_precision": MP_POLICY,
        "attention_implementation": model_config.attention_implementation,
        "attention_cutlass_cute_importable": cutlass_cute_available(),
        "ce_implementation": model_config.ce_implementation,
        "moe_implementation": model_config.moe_implementation or "ring",
        "remat_mode": model_config.remat_mode,
        "array_stacked_blocks": model_config.use_array_stacked_blocks,
        "mesh_shape": dict(mesh.shape),
        "mesh_axis_names": list(mesh.axis_names),
        "repacking": (
            f"{executed_microbatches} executed global microbatches x {global_microbatch_size} sequences; "
            "one sequence per H100; one gradient boundary"
        ),
        "global_loss_tokens_denominator": global_loss_tokens,
        "diagnostic_microbatches": diagnostic_microbatches,
        "qb_beta_repacking": (
            "pending step-630 QB beta is applied exactly; matched CE does not compute a next-step beta"
            if objective == "matched_ce"
            else "pending step-630 QB beta is applied exactly before the update; the next-step beta is the "
            f"arithmetic mean of the {executed_microbatches} per-microbatch native QB betas because the full "
            "token-by-expert statistic does not fit"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.diagnostic_microbatches is not None:
        if args.diagnostic_microbatches <= 0:
            raise ValueError("--diagnostic-microbatches must be positive")
        if args.objective != "matched_ce" or args.mode != "headline":
            raise ValueError("--diagnostic-microbatches is restricted to matched_ce headline diagnostics")
    if (args.profile_dir is None) != (args.profile_s3_prefix is None):
        raise ValueError("--profile-dir and --profile-s3-prefix must be supplied together")
    expected_world = 8 if args.mode == "preflight" else 32
    expected_rows = 1 if args.mode == "preflight" else 128

    initialize_jax()
    if jax.device_count() != expected_world or jax.local_device_count() != 8:
        raise RuntimeError(
            f"{args.mode} requires {expected_world} global and 8 local devices, got "
            f"{jax.device_count()} and {jax.local_device_count()}"
        )
    if jax.default_backend() != "gpu" or any("H100" not in device.device_kind for device in jax.local_devices()):
        raise RuntimeError(f"benchmark requires local H100s, got {[d.device_kind for d in jax.local_devices()]}")

    replay_dir = Path(tempfile.gettempdir()) / "grug-fixed-replay" / args.logical_batch_sha256
    manifest, paths = load_verified_manifest(
        args.manifest_s3_uri,
        args.manifest_sha256,
        args.logical_batch_sha256,
        replay_dir,
    )
    local_replay, counts = load_process_replay(
        paths,
        process_index=jax.process_index(),
        process_count=jax.process_count(),
        rows_per_shard=expected_rows,
    )
    counts = aggregate_process_counts(counts)
    if args.mode == "headline":
        manifest_counts = manifest["batch"]["counts"]
        expected_counts = {
            "allocated_tokens": int(manifest_counts["allocated_positions"]),
            "nonpadding_tokens": int(manifest_counts["attention_mask_nonzero"]),
            "loss_tokens": int(manifest_counts["loss_mask_nonzero"]),
        }
        if {name: counts[name] for name in expected_counts} != expected_counts:
            raise RuntimeError(f"logical accounting mismatch: {counts} != {expected_counts}")

    model_config = dataclasses.replace(_model, max_seq_len=counts["sequence_length"])
    if model_config.attention_implementation != "gpu_fa4_cute":
        raise RuntimeError(f"unexpected native attention path {model_config.attention_implementation}")
    if model_config.moe_implementation not in (None, "ring"):
        raise RuntimeError(f"unexpected native expert path {model_config.moe_implementation}")
    mp = jmp.get_policy(MP_POLICY)
    mesh = compact_grug_mesh(expert_axis_size=8, replica_axis_size=1, model_axis_size=1)
    with set_mesh(mesh):
        replay = make_global_replay(mesh, local_replay, counts)
        jax.block_until_ready(replay)
        del local_replay
        executed_counts = counts
        if args.diagnostic_microbatches is not None:
            if args.diagnostic_microbatches > replay.tokens.shape[0]:
                raise ValueError(
                    f"--diagnostic-microbatches={args.diagnostic_microbatches} exceeds "
                    f"the {replay.tokens.shape[0]} replay microbatches"
                )
            replay = jax.tree.map(lambda value: value[: args.diagnostic_microbatches], replay)
            executed_counts = {
                "logical_sequences": int(replay.tokens.shape[0] * replay.tokens.shape[1]),
                "allocated_tokens": int(replay.tokens.size),
                "nonpadding_tokens": int(jnp.sum(replay.segment_ids >= 0)),
                "loss_tokens": int(jnp.sum(replay.loss_weight != 0)),
                "microbatches": int(replay.tokens.shape[0]),
                "global_microbatch_size": int(replay.tokens.shape[1]),
                "sequence_length": int(replay.tokens.shape[2]),
            }

        optimizer = _optimizer.build(400_000)

        @jax.jit
        def initialize_optimizer(model):
            return optimizer.init(model)

        def load_start():
            params, pending = initialize_params(model_config, mp, args.checkpoint, mesh)
            if args.objective == "matched_ce":
                return params, pending

            state = GrugTrainState(
                step=jnp.array(0, dtype=jnp.int32),
                params=params,
                opt_state=initialize_optimizer(params),
                ema_params=None,
                pending_qb_betas=pending,
            )
            return state

        start = load_start()
        if args.objective == "matched_ce":
            params, pending = start
            start_fingerprint = model_fingerprint(params, pending)
            jitted = make_matched_step(mp, counts["loss_tokens"])
            compile_started = time.perf_counter()
            compiled = jitted.lower(params, pending, replay).compile()
            compile_seconds = time.perf_counter() - compile_started
            warm_seconds, warm_output, _ = run_sample(
                compiled,
                (params, pending, replay),
                label="matched-warmup",
                profile_dir=None,
            )
        else:
            state = start
            start_fingerprint = model_fingerprint(state.params, state.pending_qb_betas)
            jitted = make_operational_step(optimizer, mp, counts["loss_tokens"])
            compile_started = time.perf_counter()
            compiled = jitted.lower(state, replay).compile()
            compile_seconds = time.perf_counter() - compile_started
            warm_seconds, warm_output, _ = run_sample(
                compiled,
                (state, replay),
                label="operational-warmup",
                profile_dir=None,
            )
        warmup_values, warmup_finite_evidence = validate_output(args.objective, warm_output)
        del warm_output, start
        gc.collect()

        samples: list[dict[str, Any]] = []
        restored_fingerprints: list[list[float]] = []
        for sample_index in range(args.samples):
            fresh = load_start()
            if args.objective == "matched_ce":
                params, pending = fresh
                restored_fingerprint = model_fingerprint(params, pending)
                call_args = (params, pending, replay)
            else:
                state = fresh
                restored_fingerprint = model_fingerprint(state.params, state.pending_qb_betas)
                if int(state.step) != 0:
                    raise RuntimeError(f"fresh operational state has step {int(state.step)}, expected 0")
                call_args = (state, replay)
            if restored_fingerprint != start_fingerprint:
                raise RuntimeError(
                    f"checkpoint restore changed the timed start: {restored_fingerprint} != {start_fingerprint}"
                )
            restored_fingerprints.append(restored_fingerprint)
            profile_dir = args.profile_dir if sample_index == 0 and jax.process_index() == 0 else None
            elapsed, output, memory = run_sample(
                compiled,
                call_args,
                label=f"{args.objective}-sample-{sample_index}",
                profile_dir=profile_dir,
            )
            values, finite_evidence = validate_output(args.objective, output)
            samples.append(
                {
                    "local_elapsed_seconds": elapsed,
                    "memory": memory,
                    "values": values,
                    "finite_evidence": finite_evidence,
                }
            )
            del output, fresh
            gc.collect()

    local_profile_artifacts = (
        upload_profile_directory(args.profile_dir, args.profile_s3_prefix, jax.process_index())
        if args.profile_dir is not None and jax.process_index() == 0
        else []
    )

    local_hardware = {
        "hostname": socket.gethostname(),
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "devices": nvidia_smi_rows(),
        "jax_devices": [
            {
                "id": device.id,
                "process_index": device.process_index,
                "device_kind": device.device_kind,
                "platform": device.platform,
            }
            for device in jax.local_devices()
        ],
    }
    gathered_samples = gather_json(samples)
    gathered_hardware = gather_json(local_hardware)
    gathered_profile_artifacts = gather_json(local_profile_artifacts)
    job_info = get_job_info()

    if jax.process_index() == 0:
        wall_samples = [
            max(float(process_samples[index]["local_elapsed_seconds"]) for process_samples in gathered_samples)
            for index in range(args.samples)
        ]
        peak_hbm = max(
            int(used)
            for process_samples in gathered_samples
            for sample in process_samples
            for used in sample["memory"]["peak_used_mib"].values()
        )
        result = {
            "schema_version": 1,
            "created_utc": dt.datetime.now(dt.UTC).isoformat(),
            "benchmark": f"levanter_grug_fixed_replay_{args.objective}",
            "objective": args.objective,
            "mode": args.mode,
            "source_revision": args.source_revision,
            "image": args.image,
            "checkpoint": args.checkpoint,
            "job": str(job_info.job_id) if job_info is not None else None,
            "manifest_s3_uri": args.manifest_s3_uri,
            "manifest_sha256": args.manifest_sha256,
            "logical_batch_sha256": args.logical_batch_sha256,
            "manifest_batch": manifest["batch"],
            "manifest_batch_metadata": manifest["batch_metadata"],
            "world_size": expected_world,
            "counts": counts,
            "executed_counts": executed_counts,
            "config": _config_evidence(
                model_config,
                mesh,
                objective=args.objective,
                executed_microbatches=int(replay.tokens.shape[0]),
                global_microbatch_size=int(replay.tokens.shape[1]),
                global_loss_tokens=counts["loss_tokens"],
                diagnostic_microbatches=args.diagnostic_microbatches,
            ),
            "checkpoint_start_fingerprint": start_fingerprint,
            "restored_start_fingerprints": restored_fingerprints,
            "compile_seconds_excluded": compile_seconds,
            "warmup_seconds_excluded": warm_seconds,
            "warmup_values": warmup_values,
            "warmup_finite_evidence": warmup_finite_evidence,
            "profile_in_timed_sample": args.profile_dir is not None,
            "profile_artifacts": gathered_profile_artifacts,
            "wall_samples_seconds": wall_samples,
            "wall_median_seconds": float(np.median(wall_samples)),
            "wall_min_seconds": min(wall_samples),
            "wall_max_seconds": max(wall_samples),
            "wall_spread_seconds": max(wall_samples) - min(wall_samples),
            "gpu_seconds_per_logical_sequence": (
                float(np.median(wall_samples)) * expected_world / executed_counts["logical_sequences"]
            ),
            "logical_sequences_per_second": executed_counts["logical_sequences"] / float(np.median(wall_samples)),
            "allocated_tokens_per_second": executed_counts["allocated_tokens"] / float(np.median(wall_samples)),
            "nonpadding_tokens_per_second": executed_counts["nonpadding_tokens"] / float(np.median(wall_samples)),
            "allocated_tokens_per_second_per_gpu": (
                executed_counts["allocated_tokens"] / float(np.median(wall_samples)) / expected_world
            ),
            "nonpadding_tokens_per_second_per_gpu": (
                executed_counts["nonpadding_tokens"] / float(np.median(wall_samples)) / expected_world
            ),
            "peak_hbm_used_mib": peak_hbm,
            "per_process_samples": gathered_samples,
            "hardware": gathered_hardware,
            "timing_boundary": (
                "after replay verification/staging, checkpoint load, compilation, one exact-shape warmup, "
                "fresh-state restore, and synchronization; includes native BF16 forward, fused token-mean "
                "CE plus z-loss, backward, ring-EP/FSDP collectives, gradient accumulation, and AdamH"
                if args.objective == "operational"
                else "after replay verification/staging, checkpoint load, compilation, one exact-shape warmup, "
                "fresh-state restore, and synchronization; includes native BF16 forward, token-mean next-token "
                "CE, backward, ring-EP/FSDP collectives, and gradient accumulation; excludes optimizer"
            ),
        }
        if args.diagnostic_microbatches is not None:
            result["headline_eligible"] = False
            result["headline_exclusion_reason"] = (
                f"diagnostic run executed {args.diagnostic_microbatches} of {counts['microbatches']} replay microbatches"
            )
        elif args.profile_dir is not None:
            result["headline_eligible"] = False
            result["headline_exclusion_reason"] = "profiler was active in timed sample zero"
        else:
            result["headline_eligible"] = True
        result_sha256, payload_sha256 = upload_json(args.result_s3_uri, result)
        print(
            "GRUG_BENCHMARK_RESULT="
            + json.dumps(
                {
                    "result_s3_uri": args.result_s3_uri,
                    "result_sha256": result_sha256,
                    "payload_sha256": payload_sha256,
                    "wall_samples_seconds": wall_samples,
                    "wall_median_seconds": result["wall_median_seconds"],
                    "gpu_seconds_per_logical_sequence": result["gpu_seconds_per_logical_sequence"],
                    "peak_hbm_used_mib": peak_hbm,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    multihost_utils.sync_global_devices("result-uploaded")


if __name__ == "__main__":
    main()
