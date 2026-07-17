# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test the Blackwell all-entry destination-local source-push transport."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from levanter.grug._moe.source_push_inbox import AXIS, PushInboxConfig
from levanter.grug._moe.source_push_forward import make_source_push_forward_source_plan_inputs
from levanter.grug._moe.source_push_inbox_blackwell import sharded_destination_local_x_transport
from levanter.grug._moe.source_push_plan import source_push_destination_local_x_jax


def _destination_live_row_mask(host_inputs, hidden_rows_per_rank: int) -> np.ndarray:
    mask = np.zeros((host_inputs.send_meta.shape[0], hidden_rows_per_rank), dtype=np.bool_)
    valid_mask = np.asarray(host_inputs.plan.valid_mask)
    for src, dst_ord, entry, row in np.argwhere(valid_mask):
        dst = (src + dst_ord) % host_inputs.send_meta.shape[0]
        row_start = host_inputs.send_meta[src, dst_ord, entry, 2]
        if host_inputs.use_exact_expert_major:
            expert = host_inputs.send_meta[src, dst_ord, entry, 1]
            row_start += host_inputs.expert_base[dst, expert] + host_inputs.src_base_by_expert[dst, src, expert]
        mask[dst, row_start + row] = True
    return mask


def _require_blackwell_gpus(ep_size: int) -> str:
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"Blackwell transport smoke requires a GPU backend, got {jax.default_backend()!r}")
    devices = jax.devices("gpu")
    if len(devices) < ep_size:
        raise RuntimeError(f"Blackwell transport smoke requires {ep_size} visible GPUs, got {len(devices)}")
    device_kind = getattr(devices[0], "device_kind", "")
    if any(name in device_kind for name in ("B200", "B300", "GB200", "GB300")):
        return device_kind
    compute_capability = getattr(devices[0], "compute_capability", None)
    if compute_capability is not None:
        try:
            if float(compute_capability) >= 10.0:
                return device_kind
        except (TypeError, ValueError):
            pass
    raise RuntimeError(f"Blackwell transport smoke requires Blackwell GPUs, got {device_kind!r}")


def _config(args: argparse.Namespace) -> PushInboxConfig:
    return PushInboxConfig(
        ep_size=args.ep_size,
        entries_per_rank=args.entries_per_rank,
        inbox_slots=args.inbox_slots,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        block_m=args.block_m,
        block_k=args.block_k,
        block_n=args.block_n,
        experts_per_rank=args.experts_per_rank,
        send_worker_programs_per_peer=args.send_worker_programs_per_peer,
        worker_programs_per_peer=args.worker_programs_per_peer,
        send_pipeline_depth=1,
        n_groups_per_job=1,
        routing=args.routing,
        tokens_per_rank=args.tokens_per_rank,
        topk=args.topk,
        capacity_factor=args.capacity_factor,
    )


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    device_kind = _require_blackwell_gpus(args.ep_size)
    config = _config(args)
    config.validate()
    host_inputs = make_source_push_forward_source_plan_inputs(config)
    devices = np.asarray(jax.devices("gpu")[: config.ep_size])
    mesh = Mesh(devices, (AXIS,))
    transport = jax.jit(
        sharded_destination_local_x_transport(
            mesh,
            config,
            use_exact_expert_major=host_inputs.use_exact_expert_major,
        )
    )
    packed_x = jnp.asarray(host_inputs.x, dtype=jnp.bfloat16)
    send_meta = jnp.asarray(host_inputs.send_meta, dtype=jnp.int32)
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    src_base_by_expert = jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32)

    start = time.perf_counter()
    observed = transport(packed_x, send_meta, expert_base, src_base_by_expert)
    observed.block_until_ready()
    first_call_time = time.perf_counter() - start
    expected = source_push_destination_local_x_jax(
        packed_x,
        host_inputs.plan,
        send_meta,
        expert_base,
        src_base_by_expert,
        hidden_rows_per_rank=config.hidden_rows_per_rank,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    )
    observed_host = np.asarray(jax.device_get(observed), dtype=np.float32)
    expected_host = np.asarray(jax.device_get(expected), dtype=np.float32)
    diff = np.abs(observed_host - expected_host)
    live_row_mask = _destination_live_row_mask(host_inputs, config.hidden_rows_per_rank)
    live_diff = diff[live_row_mask]
    unused_diff = np.abs(observed_host[~live_row_mask])
    row = {
        "suite": "blackwell_destination_local_x_transport",
        "device_kind": device_kind,
        "config": asdict(config),
        "first_call_time": first_call_time,
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "nonzero_diff_count": int(np.count_nonzero(diff)),
        "live_max_abs_diff": float(np.max(live_diff)) if live_diff.size else 0.0,
        "live_mean_abs_diff": float(np.mean(live_diff)) if live_diff.size else 0.0,
        "live_nonzero_diff_count": int(np.count_nonzero(live_diff)),
        "unused_observed_max_abs": float(np.max(unused_diff)) if unused_diff.size else 0.0,
        "live_rows_total": int(np.count_nonzero(live_row_mask)),
        "observed_shape": list(observed.shape),
        "expected_shape": list(expected.shape),
        "dropped_routes": int(jax.device_get(host_inputs.plan.dropped_routes)),
        "use_exact_expert_major": host_inputs.use_exact_expert_major,
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    if row["live_max_abs_diff"] != 0.0:
        raise RuntimeError(f"transport mismatch: {row}")
    return row


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--entries-per-rank", type=int, default=4)
    parser.add_argument("--inbox-slots", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--intermediate-dim", type=int, default=128)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--experts-per-rank", type=int, default=2)
    parser.add_argument("--send-worker-programs-per-peer", type=int, default=1)
    parser.add_argument("--worker-programs-per-peer", type=int, default=4)
    parser.add_argument("--routing", choices=("balanced", "roughly_balanced"), default="balanced")
    parser.add_argument("--tokens-per-rank", type=int, default=64)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    run_smoke(parse_args(argv))


if __name__ == "__main__":
    main()
