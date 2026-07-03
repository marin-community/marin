# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI wrapper for the package-private source-push inbox MGPU benchmark."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from typing import Any

from levanter.grug._moe.source_push_inbox import (
    ROUTING_MODES,
    PushInboxConfig,
    run_source_push_inbox,
    run_source_push_inbox_compact_routing,
    run_source_push_inbox_source_plan,
)
from levanter.grug._moe.source_push_inbox_profiles import SOURCE_PUSH_PROFILES, source_push_profile_defaults


def _parse_int_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in value.split(",") if part)
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    return values


def _source_push_profile_defaults(argv: Sequence[str] | None = None) -> dict[str, Any]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--source-push-profile", choices=SOURCE_PUSH_PROFILES, default="none")
    args, _ = pre_parser.parse_known_args(argv)
    return source_push_profile_defaults(args.source_push_profile)


def parse_source_push_inbox_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse source-push inbox benchmark arguments with profile defaults applied."""
    profile_defaults = _source_push_profile_defaults(argv)

    def default(name: str, fallback: Any) -> Any:
        return profile_defaults.get(name, fallback)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-push-profile", choices=SOURCE_PUSH_PROFILES, default="none")
    parser.add_argument("--ep-size", type=int, default=default("ep_size", 8))
    parser.add_argument("--entries-per-rank", type=int, default=default("entries_per_rank", 2))
    parser.add_argument("--sweep-entries-per-rank", type=_parse_int_csv, default=None)
    parser.add_argument("--inbox-slots", type=int, default=default("inbox_slots", 2))
    parser.add_argument("--sweep-inbox-slots", type=_parse_int_csv, default=None)
    parser.add_argument("--hidden-dim", type=int, default=default("hidden_dim", 2560))
    parser.add_argument("--intermediate-dim", type=int, default=default("intermediate_dim", 1280))
    parser.add_argument("--block-m", type=int, default=default("block_m", 64))
    parser.add_argument("--sweep-block-m", type=_parse_int_csv, default=None)
    parser.add_argument("--block-n", type=int, default=default("block_n", 128))
    parser.add_argument("--sweep-block-n", type=_parse_int_csv, default=None)
    parser.add_argument("--block-k", type=int, default=default("block_k", 128))
    parser.add_argument("--sweep-block-k", type=_parse_int_csv, default=None)
    parser.add_argument("--n-group", type=int, default=default("n_group", 1))
    parser.add_argument("--sweep-n-groups", type=_parse_int_csv, default=None)
    parser.add_argument("--experts-per-rank", type=int, default=default("experts_per_rank", 32))
    parser.add_argument(
        "--send-worker-programs-per-peer",
        type=int,
        default=default("send_worker_programs_per_peer", 4),
        help="Logical sender programs in each peer-phase grid slice.",
    )
    parser.add_argument("--sweep-send-worker-programs-per-peer", type=_parse_int_csv, default=None)
    parser.add_argument(
        "--worker-programs-per-peer",
        type=int,
        default=default("worker_programs_per_peer", 16),
        help="Total logical programs launched in each peer-phase grid slice.",
    )
    parser.add_argument("--sweep-worker-programs-per-peer", type=_parse_int_csv, default=None)
    parser.add_argument("--send-pipeline-depth", type=int, default=default("send_pipeline_depth", 1))
    parser.add_argument("--sweep-send-pipeline-depth", type=_parse_int_csv, default=None)
    parser.add_argument("--n-groups-per-job", type=int, default=default("n_groups_per_job", 1))
    parser.add_argument("--sweep-n-groups-per-job", type=_parse_int_csv, default=None)
    parser.add_argument("--routing", choices=ROUTING_MODES, default=default("routing", "balanced"))
    parser.add_argument("--tokens-per-rank", type=int, default=default("tokens_per_rank", 32768))
    parser.add_argument("--topk", type=int, default=default("topk", 4))
    parser.add_argument("--routing-seed", type=int, default=default("routing_seed", 0))
    parser.add_argument("--capacity-factor", type=float, default=default("capacity_factor", 1.25))
    parser.add_argument(
        "--input-mode",
        choices=("synthetic_blocks", "compact_routing", "source_push_plan"),
        default=default("input_mode", "synthetic_blocks"),
    )
    parser.add_argument("--warmup", type=int, default=default("warmup", 1))
    parser.add_argument("--steps", type=int, default=default("steps", 5))
    parser.add_argument("--repeat-runs", type=int, default=default("repeat_runs", 1))
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=default("check", True))
    parser.add_argument(
        "--debug-exceptions", action=argparse.BooleanOptionalAction, default=default("debug_exceptions", False)
    )
    parser.add_argument(
        "--separate-compile", action=argparse.BooleanOptionalAction, default=default("separate_compile", False)
    )
    parser.add_argument(
        "--progress-events", action=argparse.BooleanOptionalAction, default=default("progress_events", False)
    )
    parser.add_argument("--git-sha", type=str, default=None)
    parser.add_argument("--jsonl", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_source_push_inbox_args(argv)
    entries_per_rank_values = args.sweep_entries_per_rank or (args.entries_per_rank,)
    inbox_slots_values = args.sweep_inbox_slots or (args.inbox_slots,)
    block_m_values = args.sweep_block_m or (args.block_m,)
    block_n_values = args.sweep_block_n or (args.block_n,)
    block_k_values = args.sweep_block_k or (args.block_k,)
    send_worker_programs_per_peer_values = args.sweep_send_worker_programs_per_peer or (
        args.send_worker_programs_per_peer,
    )
    worker_programs_per_peer_values = args.sweep_worker_programs_per_peer or (args.worker_programs_per_peer,)
    send_pipeline_depth_values = args.sweep_send_pipeline_depth or (args.send_pipeline_depth,)
    n_group_values = args.sweep_n_groups or (args.n_group,)
    n_groups_per_job_values = args.sweep_n_groups_per_job or (args.n_groups_per_job,)
    if args.jsonl:
        jsonl_dir = os.path.dirname(args.jsonl)
        if jsonl_dir:
            os.makedirs(jsonl_dir, exist_ok=True)

    for entries_per_rank in entries_per_rank_values:
        for inbox_slots in inbox_slots_values:
            for block_m in block_m_values:
                for block_n in block_n_values:
                    for block_k in block_k_values:
                        for send_worker_programs_per_peer in send_worker_programs_per_peer_values:
                            for worker_programs_per_peer in worker_programs_per_peer_values:
                                for send_pipeline_depth in send_pipeline_depth_values:
                                    for n_group in n_group_values:
                                        for n_groups_per_job in n_groups_per_job_values:
                                            config = PushInboxConfig(
                                                ep_size=args.ep_size,
                                                entries_per_rank=entries_per_rank,
                                                inbox_slots=inbox_slots,
                                                hidden_dim=args.hidden_dim,
                                                intermediate_dim=args.intermediate_dim,
                                                block_m=block_m,
                                                block_n=block_n,
                                                block_k=block_k,
                                                n_group=n_group,
                                                n_groups_per_job=n_groups_per_job,
                                                experts_per_rank=args.experts_per_rank,
                                                send_worker_programs_per_peer=send_worker_programs_per_peer,
                                                worker_programs_per_peer=worker_programs_per_peer,
                                                send_pipeline_depth=send_pipeline_depth,
                                                routing=args.routing,
                                                tokens_per_rank=args.tokens_per_rank,
                                                topk=args.topk,
                                                routing_seed=args.routing_seed,
                                                capacity_factor=args.capacity_factor,
                                            )
                                            run_fn = {
                                                "synthetic_blocks": run_source_push_inbox,
                                                "compact_routing": run_source_push_inbox_compact_routing,
                                                "source_push_plan": run_source_push_inbox_source_plan,
                                            }[args.input_mode]
                                            rows = run_fn(
                                                config,
                                                warmup=args.warmup,
                                                steps=args.steps,
                                                repeat_runs=args.repeat_runs,
                                                check=args.check,
                                                debug_exceptions=args.debug_exceptions,
                                                separate_compile=args.separate_compile,
                                                progress_events=args.progress_events,
                                            )
                                            for row in rows:
                                                if args.git_sha is not None:
                                                    row["git_sha"] = args.git_sha
                                                line = json.dumps(row, sort_keys=True)
                                                print(line, flush=True)
                                                if args.jsonl:
                                                    with open(
                                                        args.jsonl,
                                                        "a",
                                                        encoding="utf-8",
                                                    ) as f:
                                                        print(line, file=f, flush=True)


if __name__ == "__main__":
    main()
