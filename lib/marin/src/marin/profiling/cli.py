# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI for profile ingestion, querying, and before/after comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marin.profiling.ingest import (
    DEFAULT_ARTIFACT_ALIAS,
    download_latest_profile_artifact_for_run,
    download_wandb_profile_artifact,
    summarize_profile_artifact,
    summarize_trace,
)
from marin.profiling.compare_bundle import run_profile_comparison_bundle
from marin.profiling.publish import publish_profile_summary_artifact
from marin.profiling.query import compare_profile_summaries, query_profile_summary
from marin.profiling.report import build_markdown_report
from marin.profiling.schema import profile_summary_from_dict
from marin.profiling.tracking import (
    RegressionThresholds,
    append_regression_record,
    assess_profile_regression,
    make_regression_record,
    summarize_regression_history,
)

_BREAKDOWN_MODES = ("exclusive_per_track", "exclusive_global")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest and query JAX/xprof profile artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize", help="Build a normalized summary from a profile artifact or trace.")
    summarize.add_argument(
        "--artifact",
        help="Optional W&B artifact reference (entity/project/name:v0). If set, the artifact is downloaded first.",
    )
    summarize.add_argument(
        "--run-target",
        help=(
            "Optional W&B run target (run id, entity/project/run_id, or run URL). "
            "Downloads the selected jax_profile artifact for that run."
        ),
    )
    summarize.add_argument("--entity", help="W&B entity when --run-target is a bare run id.")
    summarize.add_argument("--project", help="W&B project when --run-target is a bare run id.")
    summarize.add_argument(
        "--alias",
        default=DEFAULT_ARTIFACT_ALIAS,
        help=f"Artifact alias preference when using --run-target. Defaults to '{DEFAULT_ARTIFACT_ALIAS}'.",
    )
    summarize.add_argument("--download-root", type=Path, help="Optional root directory for downloaded artifacts.")
    summarize.add_argument("--profile-dir", type=Path, help="Path to a downloaded jax_profile artifact directory.")
    summarize.add_argument("--trace-file", type=Path, help="Path to an explicit trace JSON(.gz) file.")
    summarize.add_argument("--warmup-steps", type=int, default=5, help="Initial steps ignored for steady-state stats.")
    summarize.add_argument("--hot-op-limit", type=int, default=25, help="Maximum number of hot ops in the summary.")
    summarize.add_argument(
        "--breakdown-mode",
        choices=_BREAKDOWN_MODES,
        default="exclusive_per_track",
        help="Time-breakdown attribution mode.",
    )
    summarize.add_argument("--output", type=Path, help="Optional output JSON path. Defaults to stdout.")

    query = subparsers.add_parser("query", help="Run a structured query against a summary JSON.")
    query.add_argument("--summary", type=Path, required=True, help="Path to a profile summary JSON.")
    query.add_argument("--question", required=True, help="Question to answer.")
    query.add_argument("--top-k", type=int, default=10, help="Maximum number of rows for list-style answers.")

    compare = subparsers.add_parser("compare", help="Compare two profile summary JSON files.")
    compare.add_argument("--before", type=Path, required=True, help="Baseline profile summary JSON.")
    compare.add_argument("--after", type=Path, required=True, help="Candidate profile summary JSON.")
    compare.add_argument("--top-k", type=int, default=10, help="Maximum number of improved/regressed ops to report.")
    compare.add_argument(
        "--strict-provenance",
        action="store_true",
        help="Fail if provenance checks indicate before/after are likely the same trace.",
    )

    track = subparsers.add_parser(
        "track",
        help="Compare two profile summaries, classify pass/warn/fail, and optionally append to a history JSONL file.",
    )
    track.add_argument("--before", type=Path, required=True, help="Baseline profile summary JSON.")
    track.add_argument("--after", type=Path, required=True, help="Candidate profile summary JSON.")
    track.add_argument("--top-k", type=int, default=10, help="Maximum number of improved/regressed ops to report.")
    track.add_argument(
        "--strict-provenance",
        action="store_true",
        help="Fail if provenance checks indicate before/after are likely the same trace.",
    )
    track.add_argument(
        "--max-step-median-regression-pct",
        type=float,
        default=5.0,
        help="Fail threshold for steady-state median step-time regression percentage.",
    )
    track.add_argument(
        "--max-step-p90-regression-pct",
        type=float,
        default=10.0,
        help="Fail threshold for steady-state p90 step-time regression percentage.",
    )
    track.add_argument(
        "--max-communication-share-regression-abs",
        type=float,
        default=0.05,
        help="Warn threshold for communication-share absolute increase.",
    )
    track.add_argument(
        "--max-stall-share-regression-abs",
        type=float,
        default=0.05,
        help="Warn threshold for stall-share absolute increase.",
    )
    track.add_argument("--label", help="Optional label attached to history records.")
    track.add_argument("--history", type=Path, help="Optional JSONL file to append regression tracking records.")
    track.add_argument("--output", type=Path, help="Optional output JSON path. Defaults to stdout.")

    report = subparsers.add_parser("report", help="Render a deterministic markdown root-cause report from a summary.")
    report.add_argument("--summary", type=Path, required=True, help="Path to a profile summary JSON.")
    report.add_argument("--top-k", type=int, default=10, help="Maximum number of hot ops/collectives in the report.")
    report.add_argument("--output", type=Path, help="Optional markdown output path. Defaults to stdout.")

    history = subparsers.add_parser("history", help="Summarize a regression tracking JSONL history file.")
    history.add_argument("--history", type=Path, required=True, help="Path to regression history JSONL.")
    history.add_argument("--tail", type=int, default=20, help="Number of recent records to include.")
    history.add_argument("--output", type=Path, help="Optional output JSON path. Defaults to stdout.")

    bundle = subparsers.add_parser(
        "bundle",
        help="Run a one-shot comparison bundle: summarize -> compare -> track -> reports.",
    )
    bundle.add_argument("--before-summary", type=Path, help="Optional existing baseline summary JSON.")
    bundle.add_argument("--after-summary", type=Path, help="Optional existing candidate summary JSON.")
    bundle.add_argument("--before-run-target", help="Optional baseline run target for auto summarization.")
    bundle.add_argument("--after-run-target", help="Optional candidate run target for auto summarization.")
    bundle.add_argument("--entity", help="W&B entity when run targets are bare run ids.")
    bundle.add_argument("--project", help="W&B project when run targets are bare run ids.")
    bundle.add_argument(
        "--alias",
        default=DEFAULT_ARTIFACT_ALIAS,
        help=f"Artifact alias preference when summarizing from run targets. Defaults to '{DEFAULT_ARTIFACT_ALIAS}'.",
    )
    bundle.add_argument("--download-root", type=Path, help="Optional root directory for downloaded artifacts.")
    bundle.add_argument("--warmup-steps", type=int, default=5, help="Warmup steps ignored in summary generation.")
    bundle.add_argument("--hot-op-limit", type=int, default=25, help="Maximum hot ops in generated summaries.")
    bundle.add_argument(
        "--breakdown-mode",
        choices=_BREAKDOWN_MODES,
        default="exclusive_per_track",
        help="Time-breakdown attribution mode for generated summaries.",
    )
    bundle.add_argument("--top-k", type=int, default=10, help="Top-k rows for compare/reports.")
    bundle.add_argument(
        "--strict-provenance",
        action="store_true",
        help="Fail if provenance checks indicate before/after are likely the same trace.",
    )
    bundle.add_argument(
        "--max-step-median-regression-pct",
        type=float,
        default=5.0,
        help="Fail threshold for steady-state median step-time regression percentage.",
    )
    bundle.add_argument(
        "--max-step-p90-regression-pct",
        type=float,
        default=10.0,
        help="Fail threshold for steady-state p90 step-time regression percentage.",
    )
    bundle.add_argument(
        "--max-communication-share-regression-abs",
        type=float,
        default=0.05,
        help="Warn threshold for communication-share absolute increase.",
    )
    bundle.add_argument(
        "--max-stall-share-regression-abs",
        type=float,
        default=0.05,
        help="Warn threshold for stall-share absolute increase.",
    )
    bundle.add_argument("--label", help="Optional label attached to tracking records.")
    bundle.add_argument("--history", type=Path, help="Optional regression history JSONL path.")
    bundle.add_argument("--output-dir", type=Path, required=True, help="Directory for bundle outputs.")
    bundle.add_argument("--output", type=Path, help="Optional bundle-manifest JSON path. Defaults to stdout.")

    publish = subparsers.add_parser("publish", help="Publish summary/report as a W&B profile_summary artifact.")
    publish.add_argument("--summary", type=Path, required=True, help="Path to profile summary JSON.")
    publish.add_argument("--report", type=Path, help="Optional markdown report path to include.")
    publish.add_argument("--entity", default=None, help="W&B entity. Defaults to WANDB_ENTITY.")
    publish.add_argument("--project", default=None, help="W&B project. Defaults to WANDB_PROJECT.")
    publish.add_argument("--artifact-name", help="Optional artifact name override.")
    publish.add_argument(
        "--alias",
        action="append",
        dest="aliases",
        help="Artifact alias. Repeat for multiple aliases. Defaults to 'latest'.",
    )
    publish.add_argument("--dry-run", action="store_true", help="Print publication metadata without uploading.")
    publish.add_argument("--output", type=Path, help="Optional output JSON path. Defaults to stdout.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "summarize":
        summary = _handle_summarize(args)
        output_json = summary.to_json()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output_json + "\n", encoding="utf-8")
            print(str(args.output))
        else:
            print(output_json)
        return

    if args.command == "query":
        summary = _load_summary(args.summary)
        response = query_profile_summary(summary, args.question, top_k=args.top_k)
        print(json.dumps(response, indent=2, sort_keys=True))
        return

    if args.command == "compare":
        before = _load_summary(args.before)
        after = _load_summary(args.after)
        response = compare_profile_summaries(before, after, top_k=args.top_k)
        _enforce_provenance_policy(response, strict=args.strict_provenance)
        print(json.dumps(response, indent=2, sort_keys=True))
        return

    if args.command == "track":
        before = _load_summary(args.before)
        after = _load_summary(args.after)
        thresholds = RegressionThresholds(
            max_step_median_regression_pct=args.max_step_median_regression_pct,
            max_step_p90_regression_pct=args.max_step_p90_regression_pct,
            max_communication_share_regression_abs=args.max_communication_share_regression_abs,
            max_stall_share_regression_abs=args.max_stall_share_regression_abs,
        )
        assessment = assess_profile_regression(before, after, thresholds=thresholds, top_k=args.top_k)
        _enforce_provenance_policy(assessment["comparison"], strict=args.strict_provenance)
        record = make_regression_record(
            before=before,
            after=after,
            assessment=assessment,
            label=args.label,
        )
        if args.history:
            append_regression_record(args.history, record)

        response_json = json.dumps(record, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(response_json + "\n", encoding="utf-8")
            print(str(args.output))
        else:
            print(response_json)
        return

    if args.command == "report":
        summary = _load_summary(args.summary)
        markdown = build_markdown_report(summary, top_k=args.top_k)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(markdown, encoding="utf-8")
            print(str(args.output))
        else:
            print(markdown)
        return

    if args.command == "history":
        summary = summarize_regression_history(args.history, tail=args.tail)
        output_json = json.dumps(summary, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output_json + "\n", encoding="utf-8")
            print(str(args.output))
        else:
            print(output_json)
        return

    if args.command == "bundle":
        before = _resolve_bundle_summary(
            summary_path=args.before_summary,
            run_target=args.before_run_target,
            entity=args.entity,
            project=args.project,
            alias=args.alias,
            download_root=args.download_root,
            warmup_steps=args.warmup_steps,
            hot_op_limit=args.hot_op_limit,
            breakdown_mode=args.breakdown_mode,
        )
        after = _resolve_bundle_summary(
            summary_path=args.after_summary,
            run_target=args.after_run_target,
            entity=args.entity,
            project=args.project,
            alias=args.alias,
            download_root=args.download_root,
            warmup_steps=args.warmup_steps,
            hot_op_limit=args.hot_op_limit,
            breakdown_mode=args.breakdown_mode,
        )

        thresholds = RegressionThresholds(
            max_step_median_regression_pct=args.max_step_median_regression_pct,
            max_step_p90_regression_pct=args.max_step_p90_regression_pct,
            max_communication_share_regression_abs=args.max_communication_share_regression_abs,
            max_stall_share_regression_abs=args.max_stall_share_regression_abs,
        )
        preflight_comparison = compare_profile_summaries(before, after, top_k=args.top_k)
        _enforce_provenance_policy(preflight_comparison, strict=args.strict_provenance)
        result = run_profile_comparison_bundle(
            before=before,
            after=after,
            output_dir=args.output_dir,
            thresholds=thresholds,
            top_k=args.top_k,
            label=args.label,
            history_path=args.history,
        )
        payload = result.to_dict()
        output_json = json.dumps(payload, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output_json + "\n", encoding="utf-8")
            print(str(args.output))
        else:
            print(output_json)
        return

    if args.command == "publish":
        kwargs = {
            "summary_path": args.summary,
            "report_path": args.report,
            "artifact_name": args.artifact_name,
            "aliases": args.aliases,
            "dry_run": args.dry_run,
        }
        if args.entity is not None:
            kwargs["entity"] = args.entity
        if args.project is not None:
            kwargs["project"] = args.project
        response = publish_profile_summary_artifact(**kwargs)
        output_json = json.dumps(response, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output_json + "\n", encoding="utf-8")
            print(str(args.output))
        else:
            print(output_json)
        return

    raise ValueError(f"Unhandled command: {args.command}")


def _handle_summarize(args: argparse.Namespace):
    if args.trace_file:
        return summarize_trace(
            args.trace_file,
            warmup_steps=args.warmup_steps,
            hot_op_limit=args.hot_op_limit,
            breakdown_mode=args.breakdown_mode,
        )

    if args.artifact:
        downloaded = download_wandb_profile_artifact(args.artifact)
        return summarize_profile_artifact(
            downloaded.artifact_dir,
            run_metadata=downloaded.run_metadata,
            warmup_steps=args.warmup_steps,
            hot_op_limit=args.hot_op_limit,
            breakdown_mode=args.breakdown_mode,
        )

    if args.run_target:
        downloaded = download_latest_profile_artifact_for_run(
            args.run_target,
            entity=args.entity,
            project=args.project,
            alias=args.alias,
            download_root=args.download_root,
        )
        return summarize_profile_artifact(
            downloaded.artifact_dir,
            run_metadata=downloaded.run_metadata,
            warmup_steps=args.warmup_steps,
            hot_op_limit=args.hot_op_limit,
            breakdown_mode=args.breakdown_mode,
        )

    if args.profile_dir:
        return summarize_profile_artifact(
            args.profile_dir,
            warmup_steps=args.warmup_steps,
            hot_op_limit=args.hot_op_limit,
            breakdown_mode=args.breakdown_mode,
        )

    raise ValueError("Specify one of --trace-file, --profile-dir, --artifact, or --run-target.")


def _load_summary(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in summary '{path}'.")
    return profile_summary_from_dict(data)


def _resolve_bundle_summary(
    *,
    summary_path: Path | None,
    run_target: str | None,
    entity: str | None,
    project: str | None,
    alias: str,
    download_root: Path | None,
    warmup_steps: int,
    hot_op_limit: int,
    breakdown_mode: str,
):
    if summary_path is not None:
        return _load_summary(summary_path)
    if run_target is not None:
        downloaded = download_latest_profile_artifact_for_run(
            run_target,
            entity=entity,
            project=project,
            alias=alias,
            download_root=download_root,
        )
        return summarize_profile_artifact(
            downloaded.artifact_dir,
            run_metadata=downloaded.run_metadata,
            warmup_steps=warmup_steps,
            hot_op_limit=hot_op_limit,
            breakdown_mode=breakdown_mode,
        )
    raise ValueError("Bundle requires either a summary path or a run target for both before and after.")


def _enforce_provenance_policy(comparison: dict, *, strict: bool) -> None:
    if not strict:
        return
    checks = comparison.get("provenance_checks")
    if not isinstance(checks, dict):
        return
    if checks.get("status") != "fail":
        return
    messages = checks.get("messages")
    if isinstance(messages, list) and messages:
        details = " ".join(str(message) for message in messages)
        raise ValueError(f"Provenance checks failed. {details}")
    raise ValueError("Provenance checks failed.")


if __name__ == "__main__":
    main()
