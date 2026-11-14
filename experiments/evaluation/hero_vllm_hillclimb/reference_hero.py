# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the hill climb's frozen histories with the native Hero JAX implementation.

The reference restores the same SFT checkpoint the served export was written from, with the export's
pending router-bias update (``restore_bf16``), and runs the training model with the dropless MoE
backend the native sampler uses. Each history is one batch row; the requested positions are gathered
from a single forward pass and projected one position at a time.

Outputs (process zero), under ``--output``:

* ``reference.npz``: flat rows ``row_case`` (index into ``cases``), ``row_position``, ``target_id``
  (next token, -1 at a history's final position), ``target_logprob``, ``top_ids`` [N, K] and
  ``top_logprobs`` [N, K], all from an FP32 log-softmax over the full vocabulary.
* ``reference.json``: case ids, per-row history SHA-256 (``tokens[: position + 1]`` as int32), padding,
  model config, checkpoint, router-bias check, devices and timings.

Position ``p`` predicts token ``p + 1`` from ``tokens[: p + 1]``; it matches index ``p`` of
``probes.Client.score(...)["positions"]``.

Launch with ``submit_reference.py``.
"""

import argparse
import dataclasses
import hashlib
import io
import json
import logging
import time

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import tensorstore as ts
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.cutlass_kernel_cache import cutlass_kernel_cache, install
from levanter.distributed import DistributedConfig
from levanter.grug.sharding import compact_grug_mesh
from levanter.tensorstore_serialization import ARRAY_DRIVER, KVSTORE_DRIVER, build_kvstore_spec
from levanter.utils.jax_utils import leaf_key_paths
from rigging.filesystem.factory import open_url
from rigging.log_setup import configure_logging

from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer
from experiments.grug.moe_hero_ep.train import DEFAULT_DROPLESS_MOE_IMPLEMENTATION
from experiments.grug.moe_hero_ep.weights import (
    PENDING_QB_BETAS_KEY,
    checkpoint_weight_path,
    validate_checkpoint_metadata,
)

logger = logging.getLogger(__name__)

ROUTER_BIAS_PATH = "stacked_blocks/stacked/mlp/router_bias"
READ_CONCURRENCY = 32
PAD_TOKEN = 0
LENGTH_QUANTUM = 128
LONG_TAIL_POSITIONS = 512
LONG_STRIDE = 256
_DROP_STATS = ("capacity_overflow_per_layer", "skipped_assignments_per_layer")


@dataclasses.dataclass(frozen=True)
class Row:
    case_id: str
    tokens: tuple[int, ...]
    positions: tuple[int, ...]


def gate_rows(cases: dict) -> list[Row]:
    return [Row(case["case_id"], tuple(case["tokens"]), tuple(range(len(case["tokens"])))) for case in cases["gate"]]


def long_rows(cases: dict, lengths: tuple[int, ...]) -> list[Row]:
    """One row per requested length, shortest first.

    A length with no case of exactly that length scores the longest case's prefix as ``long{length}``;
    the long cases are prefixes of one book, so this matches what ``prepare.py`` would have written.
    """
    by_length = {len(case["tokens"]): case for case in cases["long"]}
    longest = by_length[max(by_length)]
    rows = []
    for length in sorted(lengths):
        if length in by_length:
            case_id, tokens = by_length[length]["case_id"], by_length[length]["tokens"]
        elif length < len(longest["tokens"]):
            case_id, tokens = f"long{length}", longest["tokens"][:length]
        else:
            raise ValueError(f"No long case reaches {length} tokens; the longest has {len(longest['tokens'])}")
        positions = sorted(set(range(length - LONG_TAIL_POSITIONS, length)) | set(range(0, length, LONG_STRIDE)))
        rows.append(Row(case_id, tuple(tokens), tuple(positions)))
    return rows


def gate_batches(rows: list[Row], batch_size: int) -> list[list[Row]]:
    """Group similar lengths; a short final batch repeats its last row, and the copies are discarded."""
    ordered = sorted(rows, key=lambda row: len(row.tokens))
    return [ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)]


def batch_arrays(rows: list[Row], batch_size: int, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = max(len(row.positions) for row in rows)
    filled = rows + [rows[-1]] * (batch_size - len(rows))
    tokens = np.full((batch_size, length), PAD_TOKEN, dtype=np.int32)
    positions = np.zeros((batch_size, width), dtype=np.int32)
    targets = np.zeros((batch_size, width), dtype=np.int32)
    for index, row in enumerate(filled):
        tokens[index, : len(row.tokens)] = row.tokens
        positions[index, : len(row.positions)] = row.positions
        for column, position in enumerate(row.positions):
            targets[index, column] = row.tokens[position + 1] if position + 1 < len(row.tokens) else 0
    return tokens, positions, targets


@eqx.filter_jit
def score_positions(model: Transformer, tokens: jax.Array, positions: jax.Array, targets: jax.Array, top_k: int):
    hidden, stats = model(tokens)
    selected = hidden.at[jnp.arange(tokens.shape[0])[:, None], positions].get(out_sharding=P())
    targets = jax.sharding.reshard(targets, P())

    def project(inputs):
        state, target = inputs
        logits = jnp.einsum("bh,hv->bv", state, model.output_proj, preferred_element_type=jnp.float32)
        values = jax.nn.log_softmax(jax.sharding.reshard(logits, P()), axis=-1)
        top_values, top_ids = jax.lax.top_k(values, top_k)
        chosen = jnp.take_along_axis(values, target[:, None], axis=-1)[:, 0]
        return chosen, top_ids, top_values

    chosen, top_ids, top_values = jax.lax.map(project, (jnp.swapaxes(selected, 0, 1), jnp.swapaxes(targets, 0, 1)))
    dropped = sum(
        (jnp.sum(stats[key]).astype(jnp.int32) for key in _DROP_STATS if key in stats), jnp.zeros((), jnp.int32)
    )
    return (
        jnp.swapaxes(chosen, 0, 1),
        jnp.swapaxes(top_ids, 0, 1),
        jnp.swapaxes(top_values, 0, 1),
        dropped,
        tuple(sorted(stats)),
    )


def _open_store(checkpoint: str, path: str, context: ts.Context) -> ts.TensorStore:
    spec = {
        "driver": ARRAY_DRIVER,
        "kvstore": {"driver": KVSTORE_DRIVER, "base": build_kvstore_spec(checkpoint), "path": path},
    }
    return ts.open(spec, open=True, read=True, context=context).result()


def applied_router_bias(pending: np.ndarray) -> np.ndarray:
    """The export's and ``apply_qb_betas``'s bias: negated, centered QB betas in FP32."""
    bias = -pending.astype(np.float32)
    return bias - bias.mean(axis=-1, keepdims=True)


def restore_bf16(checkpoint: str, metadata_digest: str, config: GrugModelConfig, mesh) -> tuple[Transformer, dict]:
    """Restore authoritative weights directly into BF16 shards, then apply the pending router bias.

    ``restore_weights`` materializes the FP32 masters first (128 GiB per GPU on 16 GPUs), which does
    not fit beside the BF16 copy. Reading each shard through a TensorStore cast avoids the FP32 copy and
    yields the same BF16 values as the compute-policy cast.
    """
    validate_checkpoint_metadata(checkpoint, metadata_digest)
    weights = checkpoint_weight_path(checkpoint)
    parent = weights.rpartition("/")[0]
    context = ts.Context(
        {
            "cache_pool": {"total_bytes_limit": 0},
            "data_copy_concurrency": {"limit": READ_CONCURRENCY},
            "s3_request_concurrency": {"limit": READ_CONCURRENCY},
        }
    )
    pending = _open_store(checkpoint, f"{parent}/{PENDING_QB_BETAS_KEY}" if parent else PENDING_QB_BETAS_KEY, context)
    applied = applied_router_bias(pending.read().result())
    template = eqx.filter_eval_shape(Transformer.init, config, key=jax.random.PRNGKey(0))
    paths = [path.replace(".", "/") for path in jax.tree.leaves(leaf_key_paths(template))]
    leaves = []
    stored_bias = None
    for path, exemplar in zip(paths, jax.tree.leaves(template), strict=True):
        assert isinstance(exemplar.sharding, NamedSharding), (path, exemplar.sharding)
        sharding = NamedSharding(mesh, exemplar.sharding.spec)
        store = _open_store(checkpoint, f"{weights}/{path}", context)
        if tuple(store.shape) != tuple(exemplar.shape):
            raise ValueError(f"Checkpoint shape differs from model config: {path}: {store.shape} != {exemplar.shape}")
        if path == ROUTER_BIAS_PATH:
            stored_bias = store.read().result().astype(np.float32)
            values = applied.astype(jnp.bfloat16)

            def read(index, values=values):
                return values[index]

        else:

            def read(index, store=store):
                return ts.cast(store[index], ts.bfloat16).read().result()

        started = time.time()
        leaves.append(jax.make_array_from_callback(exemplar.shape, sharding, read, dtype=jnp.bfloat16))
        logger.info("Restored %s %s %s in %.1fs", path, exemplar.shape, sharding.spec, time.time() - started)
    assert stored_bias is not None, "checkpoint has no router bias"
    model = jax.tree.unflatten(jax.tree.structure(template), leaves)
    jax.block_until_ready(model)
    check = {
        "max_abs_stored_minus_applied": float(np.abs(stored_bias - applied).max()),
        "max_abs_stored": float(np.abs(stored_bias).max()),
        "max_abs_applied": float(np.abs(applied).max()),
    }
    return model, check


def peak_device_gib() -> float:
    """The largest allocator peak on any device of any process so far, in GiB."""
    local = max(device.memory_stats()["peak_bytes_in_use"] for device in jax.local_devices())
    return float(np.max(multihost_utils.process_allgather(np.asarray(local, dtype=np.int64)))) / 2**30


def history_sha256(tokens: tuple[int, ...], position: int) -> str:
    return hashlib.sha256(np.asarray(tokens[: position + 1], dtype=np.int32).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--export-request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--long-lengths",
        default="4096,32768,65536",
        help="Long histories to score; a length between case lengths scores a prefix of the longest case",
    )
    parser.add_argument(
        "--context-parallel",
        type=int,
        default=1,
        help="Context-axis size; each batch holds device_count / context_parallel rows",
    )
    parser.add_argument("--long-only", action="store_true", help="Skip the gate histories")
    parser.add_argument("--top-k", type=int, default=64)
    args = parser.parse_args()

    started = time.time()
    DistributedConfig().initialize()
    configure_logging(logging.INFO if jax.process_index() == 0 else logging.WARNING)
    install(cutlass_kernel_cache())
    if jax.default_backend() != "gpu":
        raise ValueError("The Hero reference requires the JAX GPU backend")
    with open_url(args.cases, "r") as handle:
        cases_bytes = handle.read().encode()
    cases = json.loads(cases_bytes)
    with open_url(args.export_request, "r") as handle:
        request = json.load(handle)["config"]
    config = draccus.decode(GrugModelConfig, request["model"])
    # The native sampler's dropless backend; it cannot drop another row's assignments.
    config = dataclasses.replace(config, moe_implementation=DEFAULT_DROPLESS_MOE_IMPLEMENTATION, expert_chunks=1)
    checkpoint = request["checkpoint"]

    # Rows split over data, each row's sequence over context: at context_parallel = device_count a
    # batch is one history, sharded along the sequence on every device.
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1, context_axis_size=args.context_parallel)
    batch_size = mesh.shape["data"]
    sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert")))
    long_lengths = tuple(int(value) for value in args.long_lengths.split(",") if value)
    gate = [] if args.long_only else gate_rows(cases)
    long = long_rows(cases, long_lengths)
    # Long histories run at exact length, one history replicated on every data row.
    batches = [(rows, max(len(row.tokens) for row in rows)) for rows in gate_batches(gate, batch_size)]
    batches = [(rows, -(-length // LENGTH_QUANTUM) * LENGTH_QUANTUM) for rows, length in batches]
    batches += [([row], len(row.tokens)) for row in long]

    outputs = {
        "row_case": [],
        "row_position": [],
        "target_id": [],
        "target_logprob": [],
        "top_ids": [],
        "top_logprobs": [],
    }
    case_ids = [row.case_id for row in gate + long]
    row_meta = []
    timings = {}
    peak_gib = {}
    overflow_total = 0
    stat_keys: tuple[str, ...] = ()
    with jax.set_mesh(mesh):
        restore_started = time.time()
        model, bias_check = restore_bf16(checkpoint, request["metadata_digest"], config, mesh)
        timings["restore_seconds"] = time.time() - restore_started
        logger.info("Restored BF16 weights in %.1fs; router bias check: %s", timings["restore_seconds"], bias_check)
        for rows, length in batches:
            tokens, positions, targets = batch_arrays(rows, batch_size, length)
            arrays = [
                jax.make_array_from_callback(value.shape, sharding, lambda index, value=value: value[index])
                for value in (tokens, positions, targets)
            ]
            batch_started = time.time()
            chosen, top_ids, top_values, overflow, stat_keys = score_positions(model, *arrays, args.top_k)
            chosen, top_ids, top_values = (np.asarray(value) for value in (chosen, top_ids, top_values))
            overflow_total += int(overflow)
            label = f"{rows[0].case_id}..{rows[-1].case_id}@{length}"
            timings[label] = time.time() - batch_started
            peak_gib[label] = peak_device_gib()
            logger.info(
                "Scored %s in %.1fs (rows=%d, dropped=%d, peak %.1f GiB)",
                label,
                timings[label],
                len(rows),
                int(overflow),
                peak_gib[label],
            )
            for index, row in enumerate(rows):
                case_index = case_ids.index(row.case_id)
                for column, position in enumerate(row.positions):
                    final = position + 1 >= len(row.tokens)
                    outputs["row_case"].append(case_index)
                    outputs["row_position"].append(position)
                    outputs["target_id"].append(-1 if final else row.tokens[position + 1])
                    outputs["target_logprob"].append(np.nan if final else float(chosen[index, column]))
                    outputs["top_ids"].append(top_ids[index, column])
                    outputs["top_logprobs"].append(top_values[index, column])
                    row_meta.append(history_sha256(row.tokens, position))
                assert np.isfinite(top_values[index, : len(row.positions)]).all(), row.case_id

    if overflow_total:
        raise AssertionError(f"Reference dropped {overflow_total} MoE assignments")
    if jax.process_index() == 0:
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            row_case=np.asarray(outputs["row_case"], dtype=np.int32),
            row_position=np.asarray(outputs["row_position"], dtype=np.int32),
            target_id=np.asarray(outputs["target_id"], dtype=np.int32),
            target_logprob=np.asarray(outputs["target_logprob"], dtype=np.float32),
            top_ids=np.stack(outputs["top_ids"]).astype(np.int32),
            top_logprobs=np.stack(outputs["top_logprobs"]).astype(np.float32),
        )
        with open_url(f"{args.output}/reference.npz", "wb") as handle:
            handle.write(buffer.getvalue())
        padding = {}
        for rows, length in batches:
            for row in rows:
                padding[row.case_id] = length - len(row.tokens)
        metadata = {
            "cases_uri": args.cases,
            "cases_sha256": hashlib.sha256(cases_bytes).hexdigest(),
            "case_ids": case_ids,
            "row_history_sha256": row_meta,
            "padding_tokens": padding,
            "pad_token": PAD_TOKEN,
            "batch_size": batch_size,
            "mesh": dict(mesh.shape),
            "peak_device_gib_after_batch": peak_gib,
            "top_k": args.top_k,
            "checkpoint": checkpoint,
            "metadata_digest": request["metadata_digest"],
            "export_request": args.export_request,
            "model": draccus.encode(config),
            "router_bias_check": bias_check,
            "moe_overflow_total": overflow_total,
            "model_stat_keys": list(stat_keys),
            "compute_policy": "BF16 weights (TensorStore cast of FP32 masters); logits fp32; log_softmax fp32",
            "jax": jax.__version__,
            "device_kind": sorted({device.device_kind for device in jax.devices()}),
            "process_count": jax.process_count(),
            "timings": timings,
            "total_seconds": time.time() - started,
        }
        with open_url(f"{args.output}/reference.json", "w") as handle:
            json.dump(metadata, handle)
        logger.info("Wrote %d reference rows to %s", len(row_meta), args.output)


if __name__ == "__main__":
    main()
