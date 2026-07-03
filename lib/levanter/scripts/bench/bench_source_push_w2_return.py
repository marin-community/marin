# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI wrapper for the package-private source-push W2 return MGPU benchmark."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from typing import Any

from levanter.grug._moe.source_push_inbox import PushInboxConfig
from levanter.grug._moe.source_push_inbox_profiles import SOURCE_PUSH_PROFILES, source_push_profile_defaults
from levanter.grug._moe.source_push_w2_return import (
    W2_HIDDEN_INPUT_MODES,
    W2_HIDDEN_INPUT_SYNTHETIC,
    W2_RETURN_MODE_DESTINATION_LOCAL,
    W2_RETURN_MODES,
    run_source_push_w2_return_source_plan,
)


def _profile_defaults(argv: Sequence[str] | None = None) -> dict[str, Any]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--source-push-profile", choices=SOURCE_PUSH_PROFILES, default="none")
    args, _ = pre_parser.parse_known_args(argv)
    return source_push_profile_defaults(args.source_push_profile)


def parse_source_push_w2_return_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse source-push W2 return benchmark arguments with profile defaults applied."""

    profile_defaults = _profile_defaults(argv)

    def default(name: str, fallback: Any) -> Any:
        return profile_defaults.get(name, fallback)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-push-profile", choices=SOURCE_PUSH_PROFILES, default="none")
    parser.add_argument("--ep-size", type=int, default=default("ep_size", 8))
    parser.add_argument("--entries-per-rank", type=int, default=default("entries_per_rank", 2))
    parser.add_argument("--inbox-slots", type=int, default=default("inbox_slots", 2))
    parser.add_argument("--hidden-dim", type=int, default=default("hidden_dim", 2560))
    parser.add_argument("--intermediate-dim", type=int, default=default("intermediate_dim", 1280))
    parser.add_argument("--block-m", type=int, default=default("block_m", 64))
    parser.add_argument("--block-n", type=int, default=default("block_n", 128))
    parser.add_argument("--block-k", type=int, default=default("block_k", 128))
    parser.add_argument("--n-group", type=int, default=default("n_group", 1))
    parser.add_argument("--n-groups-per-job", type=int, default=default("n_groups_per_job", 1))
    parser.add_argument("--experts-per-rank", type=int, default=default("experts_per_rank", 32))
    parser.add_argument(
        "--send-worker-programs-per-peer",
        type=int,
        default=default("send_worker_programs_per_peer", 4),
    )
    parser.add_argument(
        "--worker-programs-per-peer",
        type=int,
        default=default("worker_programs_per_peer", 16),
    )
    parser.add_argument("--send-pipeline-depth", type=int, default=default("send_pipeline_depth", 1))
    parser.add_argument("--routing", default=default("routing", "balanced"))
    parser.add_argument("--tokens-per-rank", type=int, default=default("tokens_per_rank", 32768))
    parser.add_argument("--topk", type=int, default=default("topk", 4))
    parser.add_argument("--routing-seed", type=int, default=default("routing_seed", 0))
    parser.add_argument("--capacity-factor", type=float, default=default("capacity_factor", 1.25))
    parser.add_argument(
        "--hidden-input-mode",
        choices=W2_HIDDEN_INPUT_MODES,
        default=default("hidden_input_mode", W2_HIDDEN_INPUT_SYNTHETIC),
    )
    parser.add_argument(
        "--return-mode",
        choices=W2_RETURN_MODES,
        default=default("return_mode", W2_RETURN_MODE_DESTINATION_LOCAL),
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
    args = parse_source_push_w2_return_args(argv)
    if args.jsonl:
        jsonl_dir = os.path.dirname(args.jsonl)
        if jsonl_dir:
            os.makedirs(jsonl_dir, exist_ok=True)

    config = PushInboxConfig(
        ep_size=args.ep_size,
        entries_per_rank=args.entries_per_rank,
        inbox_slots=args.inbox_slots,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
        n_group=args.n_group,
        n_groups_per_job=args.n_groups_per_job,
        experts_per_rank=args.experts_per_rank,
        send_worker_programs_per_peer=args.send_worker_programs_per_peer,
        worker_programs_per_peer=args.worker_programs_per_peer,
        send_pipeline_depth=args.send_pipeline_depth,
        routing=args.routing,
        tokens_per_rank=args.tokens_per_rank,
        topk=args.topk,
        routing_seed=args.routing_seed,
        capacity_factor=args.capacity_factor,
    )
    rows = run_source_push_w2_return_source_plan(
        config,
        warmup=args.warmup,
        steps=args.steps,
        repeat_runs=args.repeat_runs,
        check=args.check,
        debug_exceptions=args.debug_exceptions,
        separate_compile=args.separate_compile,
        progress_events=args.progress_events,
        hidden_input_mode=args.hidden_input_mode,
        return_mode=args.return_mode,
    )
    for row in rows:
        if args.git_sha is not None:
            row["git_sha"] = args.git_sha
        line = json.dumps(row, sort_keys=True)
        print(line, flush=True)
        if args.jsonl:
            with open(args.jsonl, "a", encoding="utf-8") as f:
                print(line, file=f, flush=True)


if __name__ == "__main__":
    main()
