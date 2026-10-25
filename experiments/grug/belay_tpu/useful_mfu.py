# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure complete-update windows with a conservative Ckpt 1 FLOP ledger."""

import hashlib
import json
import os
import time
from pathlib import Path

import draccus
import jax
import jax.numpy as jnp
import levanter.tracker as tracker
import numpy as np
from jax.experimental import multihost_utils

# Selected dense matrices in the frozen 501,789,462-parameter Ckpt 1 model.
# The full leaf-by-leaf ledger is attached to research issue 253.
DENSE_FORWARD_PER_QUERY = 44_564_480
BF16_PEAK_PER_V4_CHIP = 275e12
WINDOW_STARTS = (200, 500, 800)
WINDOW_STEPS = 300
MODEL_CONFIG_SHA256 = "fba5f5d8c6c296ee3c01510b01ff553ac7440a48454800ceeadeae5af23e4d34"


@jax.jit
def batch_counts(loss_weight, segment_ids):
    """Count binary-weight queries and causal pairs within packed documents."""
    positions = jnp.broadcast_to(jnp.arange(loss_weight.shape[1], dtype=jnp.int32), loss_weight.shape)
    starts = jnp.concatenate(
        [jnp.ones_like(segment_ids[:, :1], bool), segment_ids[:, 1:] != segment_ids[:, :-1]], axis=1
    )
    document_positions = positions - jax.lax.cummax(jnp.where(starts, positions, 0), axis=1)
    valid = loss_weight > 0
    # At B128/S4096 even the unsegmented pair count fits signed int32.
    return (
        jnp.sum(valid, dtype=jnp.int32),
        jnp.sum(jnp.where(valid, document_positions + 1, 0), dtype=jnp.int32),
        jnp.all((loss_weight == 0) | (loss_weight == 1)),
    )


def useful_flops(valid_queries: int, causal_pairs: int, drops: int) -> int:
    total_tokens = 128 * 4096
    total_assignments = total_tokens * 6 * 10
    if not 0 <= valid_queries <= total_tokens or not 0 <= drops <= total_assignments:
        raise ValueError("Invalid Ckpt 1 query or drop count")
    # Removing every possible assignment at excluded queries can only undercount
    # useful work, even when those assignments already appear in the drop count.
    accepted = max(0, total_assignments - drops - (total_tokens - valid_queries) * 6 * 10)
    forward = (
        valid_queries * DENSE_FORWARD_PER_QUERY + accepted * 2 * (256 * 384 + 384 * 256) + causal_pairs * 2 * 4 * 512
    )
    # Forward plus both matrix derivatives; omit KDA recurrence, router, Inkling,
    # convolutions, gates, norms, activations, softmax, embeddings and rematerialization.
    return 3 * forward


class UsefulMfuWindows:
    """Opt-in instrumentation; never changes parameters, gradients or batches."""

    @classmethod
    def create(cls, config, *, initial_step=0):
        if os.environ.get("BELAY_MEASURE_MFU") != "1":
            return None
        model = config.model
        model_dict = draccus.encode(model)
        # Rematerialization changes storage/extra execution, not useful model FLOPs.
        # Its extra execution remains included in the measured duration.
        model_dict["attn_res_remat_attention"] = False
        # Local-token execution changes transport and rounding, not the frozen
        # model's useful FLOP count. Preserve the pre-backend configuration hash.
        model_dict.pop("moe_local_gathered_experts", None)
        model_json = json.dumps(model_dict, sort_keys=True, separators=(",", ":"))
        if (
            hashlib.sha256(model_json.encode()).hexdigest() != MODEL_CONFIG_SHA256
            or config.trainer.trainer.train_batch_size != 128
            or model.max_seq_len != 4096
            or config.trainer.pipeline_depth != 0
            or config.trainer.grad_accum_microbatches != 1
            or jax.default_backend() != "tpu"
            or jax.device_count() not in (16, 32)
            or any("v4" not in device.device_kind.lower() for device in jax.devices())
            or os.environ.get("GRUG_SKIP_OPTIMIZER")
        ):
            raise ValueError("Useful MFU measurement requires the frozen Ckpt 1 recipe on v4-32 or v4-64")
        return cls(initial_step=initial_step)

    def __init__(self, *, initial_step=0):
        self.window_starts = tuple(initial_step + offset for offset in WINDOW_STARTS)
        self.started = None
        self.first_step = None
        self.records = []
        self.output = Path(os.environ["IRIS_OUTPUT_DIR"]) / f"useful-mfu-rank{jax.process_index()}.jsonl"

    def write(self, record):
        with self.output.open("a") as file:
            file.write(json.dumps(record) + "\n")

    def begin_if_due(self, step, state):
        if step not in self.window_starts:
            return
        if self.started is not None:
            raise RuntimeError("Overlapping MFU windows")
        jax.block_until_ready(state)
        multihost_utils.sync_global_devices(f"useful-mfu-start-{step}")
        self.first_step = step
        self.records = []
        self.started = time.perf_counter()

    def observe(self, step, state, batch, metrics):
        if self.started is None and step != 0:
            return
        if state is None or batch.tokens.shape != (128, 4096) or batch.attn_mask.segment_ids is None:
            raise ValueError("MFU windows require complete state and packed Ckpt 1 batches")
        queries, pairs, binary = jax.device_get(batch_counts(batch.loss_weight, batch.attn_mask.segment_ids[0]))
        if not bool(binary):
            raise ValueError("FLOP ledger requires binary loss weights")
        if self.started is None:
            # Compile the counter during warmup, outside every measured window.
            return
        if step != self.first_step + len(self.records):
            raise RuntimeError("MFU window has a missing or repeated step")
        drops = int(np.asarray(jax.device_get(metrics["moe/dropped_assignments"]), dtype=np.int64).sum())
        record = {
            "stage": "useful_mfu_step",
            "step": step,
            "valid_queries": int(queries),
            "causal_pairs": int(pairs),
            "dropped_assignments": drops,
            "useful_flops_lower_bound": useful_flops(int(queries), int(pairs), drops),
        }
        self.records.append(record)
        self.write(record)
        if len(self.records) != WINDOW_STEPS:
            return
        jax.block_until_ready(state)
        multihost_utils.sync_global_devices(f"useful-mfu-end-{step}")
        seconds = time.perf_counter() - self.started
        host_seconds = np.asarray(multihost_utils.process_allgather(np.asarray(seconds))).reshape(-1).tolist()
        elapsed = max(host_seconds)
        flops = sum(item["useful_flops_lower_bound"] for item in self.records)
        peak = BF16_PEAK_PER_V4_CHIP * jax.device_count()
        summary = {
            "stage": "useful_mfu_window",
            "rank": jax.process_index(),
            "first_step": self.first_step,
            "last_step": step,
            "updates": WINDOW_STEPS,
            "host_seconds": host_seconds,
            "seconds": elapsed,
            "useful_flops_lower_bound": flops,
            "physical_bf16_peak_flops_per_second": peak,
            "useful_mfu_lower_bound": flops / elapsed / peak,
            "input_tokens_per_second": WINDOW_STEPS * 128 * 4096 / elapsed,
            "memory_stats": [
                {
                    key: None if stats.get(key) is None else int(stats[key])
                    for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
                }
                for device in jax.local_devices()
                for stats in [device.memory_stats() or {}]
            ],
        }
        self.write(summary)
        print(json.dumps(summary), flush=True)
        tracker.log(
            {
                "useful_mfu/lower_bound": summary["useful_mfu_lower_bound"],
                "useful_mfu/window_seconds": elapsed,
                "useful_mfu/input_tokens_per_second": summary["input_tokens_per_second"],
            },
            step=step,
        )
        self.started = None
