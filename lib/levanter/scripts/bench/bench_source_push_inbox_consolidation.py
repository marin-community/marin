# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidation benchmarks for the stable Hopper source-push inbox profile."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from dataclasses import replace
from statistics import median
from typing import Any, TextIO

from levanter.grug._moe.source_push_inbox import (
    PushInboxConfig,
    SourcePushInboxRunSettings,
    run_source_push_inbox,
    run_source_push_inbox_compact_routing,
    source_push_inbox_profile,
)
from levanter.grug._moe.source_push_inbox_profiles import SOURCE_PUSH_PROFILE_STABLE_216, SOURCE_PUSH_PROFILES


SUMMARY_METRICS = (
    "steady_state_time",
    "w13_tflops_per_rank",
    "rounded_w13_tflops_per_rank",
    "useful_w13_tflops_per_rank",
    "send_gbps_per_rank",
    "max_abs_diff",
    "metadata_mismatches",
    "hidden_max_abs_diff",
)


def _write_row(row: dict[str, Any], jsonl_file: TextIO | None) -> None:
    line = json.dumps(row, sort_keys=True)
    print(line, flush=True)
    if jsonl_file is not None:
        print(line, file=jsonl_file, flush=True)


def _median_field(rows: Iterable[dict[str, Any]], field: str) -> float | int | None:
    values = [row[field] for row in rows if row.get(field) is not None]
    if not values:
        return None
    return median(values)


def _summary_row(
    rows: list[dict[str, Any]],
    *,
    suite: str,
    variant: str,
    axis: str,
    source_push_profile: str,
) -> dict[str, Any]:
    valid_rows = [row for row in rows if row.get("error_type") is None and row.get("steady_state_time") is not None]
    errors = [row.get("error") for row in rows if row.get("error_type") is not None]
    summary: dict[str, Any] = {
        "row_type": "summary",
        "suite": suite,
        "variant": variant,
        "axis": axis,
        "source_push_profile": source_push_profile,
        "repeat_rows": len(valid_rows),
        "error_rows": len(errors),
        "errors": errors,
    }
    for metric in SUMMARY_METRICS:
        summary[f"median_{metric}"] = _median_field(valid_rows, metric)
    if valid_rows:
        first = valid_rows[0]
        summary["implementation"] = first.get("implementation")
        summary["config"] = first.get("config")
        summary["queue_stats"] = first.get("queue_stats")
        for field in (
            "dropped_entries_total",
            "dropped_rows_total",
            "live_entries_total",
            "payload_send_entries_total",
            "masked_rows_total",
            "send_masked_row_fraction",
            "direct_self_entries_total",
            "num_compute_jobs_per_entry",
            "send_pipeline_depth",
            "n_groups_per_job",
            "input_mode",
        ):
            if field in first:
                summary[field] = first[field]
    return summary


def _emit_benchmark_rows(
    rows: list[dict[str, Any]],
    *,
    suite: str,
    variant: str,
    axis: str,
    source_push_profile: str,
    jsonl_file: TextIO | None,
) -> dict[str, Any]:
    tagged_rows = []
    for row in rows:
        tagged = {
            **row,
            "row_type": "repeat",
            "suite": suite,
            "variant": variant,
            "axis": axis,
            "source_push_profile": source_push_profile,
        }
        tagged_rows.append(tagged)
        _write_row(tagged, jsonl_file)
    summary = _summary_row(
        tagged_rows,
        suite=suite,
        variant=variant,
        axis=axis,
        source_push_profile=source_push_profile,
    )
    _write_row(summary, jsonl_file)
    return summary


def _run_variant(
    config: PushInboxConfig,
    settings: SourcePushInboxRunSettings,
    *,
    suite: str,
    variant: str,
    axis: str,
    source_push_profile: str,
    jsonl_file: TextIO | None,
    compact_routing: bool = False,
) -> dict[str, Any]:
    runner = run_source_push_inbox_compact_routing if compact_routing else run_source_push_inbox
    rows = runner(
        config,
        warmup=settings.warmup,
        steps=settings.steps,
        repeat_runs=settings.repeat_runs,
        check=settings.check,
        debug_exceptions=settings.debug_exceptions,
        separate_compile=settings.separate_compile,
        progress_events=settings.progress_events,
    )
    return _emit_benchmark_rows(
        rows,
        suite=suite,
        variant=variant,
        axis=axis,
        source_push_profile=source_push_profile,
        jsonl_file=jsonl_file,
    )


def _ablation_variants(config: PushInboxConfig) -> tuple[tuple[str, str, PushInboxConfig], ...]:
    return (
        ("winning", "baseline", config),
        ("n_groups_per_job=1", "n_groups_per_job", replace(config, n_groups_per_job=1)),
        ("send_pipeline_depth=2", "send_pipeline_depth", replace(config, send_pipeline_depth=2)),
    )


def _integration_smoke_config(config: PushInboxConfig) -> PushInboxConfig:
    return replace(
        config,
        tokens_per_rank=256,
        hidden_dim=128,
        intermediate_dim=128,
        experts_per_rank=4,
        entries_per_rank=4,
        inbox_slots=4,
        block_m=64,
        block_k=64,
        block_n=64,
        n_groups_per_job=2,
        send_worker_programs_per_peer=2,
        worker_programs_per_peer=16,
    )


def _integration_smoke_settings(settings: SourcePushInboxRunSettings) -> SourcePushInboxRunSettings:
    return replace(settings, warmup=1, steps=2, repeat_runs=3, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-push-profile",
        choices=SOURCE_PUSH_PROFILES,
        default=SOURCE_PUSH_PROFILE_STABLE_216,
    )
    parser.add_argument("--suite", choices=("ablation", "integration", "all"), default="all")
    parser.add_argument("--repeat-runs", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--separate-compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progress-events", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--debug-exceptions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--jsonl", type=str, default=None)
    return parser.parse_args()


def _apply_setting_overrides(
    settings: SourcePushInboxRunSettings,
    args: argparse.Namespace,
) -> SourcePushInboxRunSettings:
    overrides: dict[str, Any] = {}
    for name in ("repeat_runs", "warmup", "steps", "check", "separate_compile", "progress_events", "debug_exceptions"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    if not overrides:
        return settings
    return replace(settings, **overrides)


def main() -> None:
    args = _parse_args()
    config, settings = source_push_inbox_profile(args.source_push_profile)
    settings = _apply_setting_overrides(settings, args)
    jsonl_file = None
    if args.jsonl:
        jsonl_dir = os.path.dirname(args.jsonl)
        if jsonl_dir:
            os.makedirs(jsonl_dir, exist_ok=True)
        jsonl_file = open(args.jsonl, "a", encoding="utf-8")

    try:
        if args.suite in ("ablation", "all"):
            for variant, axis, variant_config in _ablation_variants(config):
                _run_variant(
                    variant_config,
                    settings,
                    suite="ablation",
                    variant=variant,
                    axis=axis,
                    source_push_profile=args.source_push_profile,
                    jsonl_file=jsonl_file,
                )

        if args.suite in ("integration", "all"):
            _run_variant(
                _integration_smoke_config(config),
                _integration_smoke_settings(settings),
                suite="integration",
                variant="compact_routing_smoke",
                axis="input_mode",
                source_push_profile=args.source_push_profile,
                jsonl_file=jsonl_file,
                compact_routing=True,
            )
    finally:
        if jsonl_file is not None:
            jsonl_file.close()


if __name__ == "__main__":
    main()
