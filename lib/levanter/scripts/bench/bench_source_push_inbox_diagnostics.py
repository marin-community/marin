# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic decomposition benchmark for the source-push inbox MGPU kernel."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import replace
from statistics import median
from typing import Any, TextIO

from levanter.grug._moe.source_push_inbox import (
    DIAGNOSTIC_VARIANTS,
    PushInboxConfig,
    SourcePushInboxRunSettings,
    run_source_push_inbox_diagnostic,
    source_push_inbox_profile,
)
from levanter.grug._moe.source_push_inbox_profiles import SOURCE_PUSH_PROFILE_STABLE_216, SOURCE_PUSH_PROFILES


SUMMARY_METRICS = (
    "steady_state_time",
    "w13_tflops_per_rank",
    "rounded_w13_tflops_per_rank",
    "useful_w13_tflops_per_rank",
    "send_gbps_per_rank",
    "compile_time",
    "lower_compile_time",
    "first_run_time",
)


def _parse_variant_csv(value: str) -> tuple[str, ...]:
    variants = tuple(part for part in value.split(",") if part)
    if not variants:
        raise argparse.ArgumentTypeError("expected a comma-separated list of diagnostic variants")
    unknown = set(variants) - set(DIAGNOSTIC_VARIANTS)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown diagnostic variants: {sorted(unknown)}")
    return variants


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
    diagnostic_variant: str,
    source_push_profile: str,
) -> dict[str, Any]:
    valid_rows = [row for row in rows if row.get("error_type") is None and row.get("steady_state_time") is not None]
    errors = [row.get("error") for row in rows if row.get("error_type") is not None]
    summary: dict[str, Any] = {
        "row_type": "summary",
        "suite": "source_push_inbox_diagnostics",
        "variant": diagnostic_variant,
        "diagnostic_variant": diagnostic_variant,
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
            "input_mode",
            "live_entries_total",
            "payload_send_entries_total",
            "masked_rows_total",
            "send_masked_row_fraction",
            "num_compute_jobs_per_entry",
            "slot_empty_waits",
            "slot_full_waits",
            "send_pipeline_depth",
            "n_groups_per_job",
        ):
            if field in first:
                summary[field] = first[field]
    return summary


def _emit_rows(
    rows: list[dict[str, Any]],
    *,
    diagnostic_variant: str,
    source_push_profile: str,
    jsonl_file: TextIO | None,
) -> dict[str, Any]:
    tagged_rows = []
    for row in rows:
        tagged = {
            **row,
            "row_type": "repeat",
            "suite": "source_push_inbox_diagnostics",
            "variant": diagnostic_variant,
            "source_push_profile": source_push_profile,
        }
        tagged_rows.append(tagged)
        _write_row(tagged, jsonl_file)
    summary = _summary_row(
        tagged_rows,
        diagnostic_variant=diagnostic_variant,
        source_push_profile=source_push_profile,
    )
    _write_row(summary, jsonl_file)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-push-profile",
        choices=SOURCE_PUSH_PROFILES,
        default=SOURCE_PUSH_PROFILE_STABLE_216,
    )
    parser.add_argument("--variants", type=_parse_variant_csv, default=DIAGNOSTIC_VARIANTS)
    parser.add_argument("--compact-routing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--repeat-runs", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--separate-compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progress-events", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--debug-exceptions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--jsonl", type=str, default=None)
    return parser.parse_args(argv)


def _apply_setting_overrides(
    settings: SourcePushInboxRunSettings,
    args: argparse.Namespace,
) -> SourcePushInboxRunSettings:
    overrides: dict[str, Any] = {"check": False}
    for name in ("repeat_runs", "warmup", "steps", "separate_compile", "progress_events", "debug_exceptions"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    return replace(settings, **overrides)


def run_diagnostics(
    config: PushInboxConfig,
    settings: SourcePushInboxRunSettings,
    *,
    variants: Sequence[str],
    compact_routing: bool,
    source_push_profile: str,
    jsonl_file: TextIO | None,
) -> list[dict[str, Any]]:
    summaries = []
    for diagnostic_variant in variants:
        rows = run_source_push_inbox_diagnostic(
            config,
            diagnostic_variant=diagnostic_variant,
            warmup=settings.warmup,
            steps=settings.steps,
            repeat_runs=settings.repeat_runs,
            debug_exceptions=settings.debug_exceptions,
            separate_compile=settings.separate_compile,
            progress_events=settings.progress_events,
            compact_routing=compact_routing,
        )
        summaries.append(
            _emit_rows(
                rows,
                diagnostic_variant=diagnostic_variant,
                source_push_profile=source_push_profile,
                jsonl_file=jsonl_file,
            )
        )
    return summaries


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config, settings = source_push_inbox_profile(args.source_push_profile)
    settings = _apply_setting_overrides(settings, args)
    jsonl_file = None
    if args.jsonl:
        jsonl_dir = os.path.dirname(args.jsonl)
        if jsonl_dir:
            os.makedirs(jsonl_dir, exist_ok=True)
        jsonl_file = open(args.jsonl, "a", encoding="utf-8")

    try:
        run_diagnostics(
            config,
            settings,
            variants=args.variants,
            compact_routing=args.compact_routing,
            source_push_profile=args.source_push_profile,
            jsonl_file=jsonl_file,
        )
    finally:
        if jsonl_file is not None:
            jsonl_file.close()


if __name__ == "__main__":
    main()
