# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep Blackwell source-push copy/release diagnostics for staged MoE."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict, replace
from statistics import median
from typing import Any, TextIO

from levanter.grug._moe.source_push_inbox import (
    DIAGNOSTIC_INPUT_MODE_SOURCE_PUSH_PLAN,
    DIAGNOSTIC_VARIANT_COPY_RELEASE_ONLY,
    DIAGNOSTIC_VARIANT_SEMAPHORE_ONLY,
    PushInboxConfig,
    SourcePushInboxRunSettings,
    run_source_push_inbox_diagnostic,
    source_push_inbox_profile,
)
from levanter.grug._moe.source_push_inbox_profiles import SOURCE_PUSH_PROFILE_BLACKWELL_65K_D3072_I3072


_SWEEP_VARIANTS = (DIAGNOSTIC_VARIANT_COPY_RELEASE_ONLY, DIAGNOSTIC_VARIANT_SEMAPHORE_ONLY)
_SUMMARY_FIELDS = (
    "steady_state_time",
    "send_gbps_per_rank",
    "useful_w13_tflops_per_rank",
    "rounded_w13_tflops_per_rank",
    "compile_time",
    "lower_compile_time",
    "first_run_time",
)


def _parse_int_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in value.split(",") if part)
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    return values


def _parse_variant_csv(value: str) -> tuple[str, ...]:
    variants = tuple(part for part in value.split(",") if part)
    unknown = set(variants) - set(_SWEEP_VARIANTS)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown variants: {sorted(unknown)}")
    if not variants:
        raise argparse.ArgumentTypeError("expected a comma-separated list of variants")
    return variants


def _median(rows: Iterable[dict[str, Any]], field: str) -> float | int | None:
    values = [row[field] for row in rows if row.get(field) is not None]
    if not values:
        return None
    return median(values)


def _write_row(row: dict[str, Any], jsonl_file: TextIO | None) -> None:
    line = json.dumps(row, sort_keys=True)
    print(line, flush=True)
    if jsonl_file is not None:
        print(line, file=jsonl_file, flush=True)


def _config_key(config: PushInboxConfig) -> dict[str, Any]:
    return {
        "entries_per_rank": config.entries_per_rank,
        "inbox_slots": config.inbox_slots,
        "send_worker_programs_per_peer": config.send_worker_programs_per_peer,
        "worker_programs_per_peer": config.worker_programs_per_peer,
        "send_pipeline_depth": config.send_pipeline_depth,
        "n_groups_per_job": config.n_groups_per_job,
    }


def _summary_row(
    repeat_rows: list[dict[str, Any]],
    *,
    config: PushInboxConfig,
    variant: str,
    git_sha: str | None,
) -> dict[str, Any]:
    valid_rows = [
        row for row in repeat_rows if row.get("error_type") is None and row.get("steady_state_time") is not None
    ]
    summary: dict[str, Any] = {
        "row_type": "summary",
        "suite": "blackwell_source_push_copy_sweep",
        "variant": variant,
        "diagnostic_variant": variant,
        "source_push_profile": SOURCE_PUSH_PROFILE_BLACKWELL_65K_D3072_I3072,
        "git_sha": git_sha,
        "repeat_rows": len(valid_rows),
        "error_rows": len(repeat_rows) - len(valid_rows),
        "config": asdict(config),
        **_config_key(config),
    }
    for field in _SUMMARY_FIELDS:
        summary[f"median_{field}"] = _median(valid_rows, field)
    if valid_rows:
        first = valid_rows[0]
        for field in (
            "dropped_routes",
            "dropped_rows_total",
            "dropped_entries_total",
            "plan_useful_rows_total",
            "plan_padded_rows_total",
            "plan_row_efficiency",
            "send_entries_total",
            "live_entries_total",
            "tail_entries_total",
            "send_masked_row_fraction",
        ):
            if field in first:
                summary[field] = first[field]
    else:
        summary["errors"] = [row.get("error") for row in repeat_rows if row.get("error") is not None]
    return summary


def _candidate_configs(base_config: PushInboxConfig, args: argparse.Namespace) -> Iterable[PushInboxConfig]:
    for entries_per_rank in args.entries_per_rank:
        for inbox_slots in args.inbox_slots:
            for send_worker_programs_per_peer in args.send_worker_programs_per_peer:
                for worker_programs_per_peer in args.worker_programs_per_peer:
                    for send_pipeline_depth in args.send_pipeline_depth:
                        for n_groups_per_job in args.n_groups_per_job:
                            yield replace(
                                base_config,
                                entries_per_rank=entries_per_rank,
                                inbox_slots=inbox_slots,
                                send_worker_programs_per_peer=send_worker_programs_per_peer,
                                worker_programs_per_peer=worker_programs_per_peer,
                                send_pipeline_depth=send_pipeline_depth,
                                n_groups_per_job=n_groups_per_job,
                            )


def _settings(base_settings: SourcePushInboxRunSettings, args: argparse.Namespace) -> SourcePushInboxRunSettings:
    return replace(
        base_settings,
        warmup=args.warmup,
        steps=args.steps,
        repeat_runs=args.repeat_runs,
        check=False,
        separate_compile=args.separate_compile,
        progress_events=args.progress_events,
        debug_exceptions=args.debug_exceptions,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=_parse_variant_csv, default=(DIAGNOSTIC_VARIANT_COPY_RELEASE_ONLY,))
    parser.add_argument("--entries-per-rank", type=_parse_int_csv, default=(576,))
    parser.add_argument("--inbox-slots", type=_parse_int_csv, default=(12,))
    parser.add_argument("--send-worker-programs-per-peer", type=_parse_int_csv, default=(2,))
    parser.add_argument("--worker-programs-per-peer", type=_parse_int_csv, default=(32,))
    parser.add_argument("--send-pipeline-depth", type=_parse_int_csv, default=(1,))
    parser.add_argument("--n-groups-per-job", type=_parse_int_csv, default=(2,))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--separate-compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-events", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-exceptions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--git-sha", type=str, default=None)
    parser.add_argument("--jsonl", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    base_config, base_settings = source_push_inbox_profile(SOURCE_PUSH_PROFILE_BLACKWELL_65K_D3072_I3072)
    settings = _settings(base_settings, args)
    jsonl_file = None
    if args.jsonl:
        jsonl_dir = os.path.dirname(args.jsonl)
        if jsonl_dir:
            os.makedirs(jsonl_dir, exist_ok=True)
        jsonl_file = open(args.jsonl, "a", encoding="utf-8")

    try:
        for config in _candidate_configs(base_config, args):
            try:
                config.validate()
            except Exception as exc:
                for variant in args.variants:
                    _write_row(
                        {
                            "row_type": "summary",
                            "suite": "blackwell_source_push_copy_sweep",
                            "variant": variant,
                            "source_push_profile": SOURCE_PUSH_PROFILE_BLACKWELL_65K_D3072_I3072,
                            "git_sha": args.git_sha,
                            "error_rows": 1,
                            "repeat_rows": 0,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "config": asdict(config),
                            **_config_key(config),
                        },
                        jsonl_file,
                    )
                continue
            for variant in args.variants:
                rows = run_source_push_inbox_diagnostic(
                    config,
                    diagnostic_variant=variant,
                    warmup=settings.warmup,
                    steps=settings.steps,
                    repeat_runs=settings.repeat_runs,
                    debug_exceptions=settings.debug_exceptions,
                    separate_compile=settings.separate_compile,
                    progress_events=settings.progress_events,
                    input_mode=DIAGNOSTIC_INPUT_MODE_SOURCE_PUSH_PLAN,
                )
                repeat_rows = [
                    {
                        **row,
                        "row_type": "repeat",
                        "suite": "blackwell_source_push_copy_sweep",
                        "variant": variant,
                        "source_push_profile": SOURCE_PUSH_PROFILE_BLACKWELL_65K_D3072_I3072,
                        "git_sha": args.git_sha,
                        **_config_key(config),
                    }
                    for row in rows
                ]
                for row in repeat_rows:
                    _write_row(row, jsonl_file)
                _write_row(_summary_row(repeat_rows, config=config, variant=variant, git_sha=args.git_sha), jsonl_file)
    finally:
        if jsonl_file is not None:
            jsonl_file.close()


if __name__ == "__main__":
    main()
