# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI for profile ingestion, querying, and before/after comparison."""

import argparse
import json
from pathlib import Path
from typing import Any

from marin.profiling.compare_bundle import run_profile_comparison_bundle
from marin.profiling.ingest import (
    download_profile_dir_for_run,
    download_wandb_profile_artifact,
    summarize_profile_artifact,
    summarize_trace,
)
from marin.profiling.publish import publish_profile_summary_artifact
from marin.profiling.query import compare_profile_summaries, query_profile_summary
from marin.profiling.report import build_markdown_report
from marin.profiling.schema import ProfileSummary, profile_summary_from_dict
from marin.profiling.tracking import (
    RegressionThresholds,
    append_regression_record,
    assess_profile_regression,
    make_regression_record,
    summarize_regression_history,
)
from marin.profiling.xplane import summarize_xplane
from marin.utilities.wandb_utils import WANDB_ENTITY, WANDB_PROJECT

_BREAKDOWN_MODES = ("exclusive_per_track", "exclusive_global")


def _add_output_option(parser: argparse.ArgumentParser, kind: str = "JSON") -> None:
    parser.add_argument("--output", type=Path, help=f"Optional {kind} output path. Defaults to stdout.")


def _add_top_k_option(parser: argparse.ArgumentParser, items: str) -> None:
    parser.add_argument("--top-k", type=int, default=10, help=f"Maximum number of {items}.")


def _add_summary_pair_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--before", type=Path, required=True, help="Baseline profile summary JSON.")
    parser.add_argument("--after", type=Path, required=True, help="Candidate profile summary JSON.")
    parser.add_argument(
        "--strict-provenance",
        action="store_true",
        help="Fail if provenance checks indicate before/after are likely the same trace.",
    )


def _add_summarization_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--warmup-steps", type=int, default=5, help="Initial steps ignored for steady-state stats.")
    parser.add_argument("--hot-op-limit", type=int, default=25, help="Maximum number of hot ops in a summary.")
    parser.add_argument(
        "--breakdown-mode",
        choices=_BREAKDOWN_MODES,
        default="exclusive_per_track",
        help="Time-breakdown attribution mode.",
    )


def _add_wandb_lookup_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--entity", help="W&B entity when a run target is a bare run id.")
    parser.add_argument("--project", help="W&B project when a run target is a bare run id.")
    parser.add_argument("--download-root", type=Path, help="Optional root directory for downloaded artifacts.")


def _add_threshold_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-step-median-regression-pct",
        type=float,
        default=5.0,
        help="Fail threshold for steady-state median step-time regression percentage.",
    )
    parser.add_argument(
        "--max-step-p90-regression-pct",
        type=float,
        default=10.0,
        help="Fail threshold for steady-state p90 step-time regression percentage.",
    )
    parser.add_argument(
        "--max-communication-share-regression-abs",
        type=float,
        default=0.05,
        help="Warn threshold for communication-share absolute increase.",
    )
    parser.add_argument(
        "--max-stall-share-regression-abs",
        type=float,
        default=0.05,
        help="Warn threshold for stall-share absolute increase.",
    )
    parser.add_argument("--label", help="Optional label attached to regression history records.")
    parser.add_argument("--history", type=Path, help="Optional JSONL file to append regression tracking records.")


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
            "Downloads the profiler directory for that run."
        ),
    )
    _add_wandb_lookup_options(summarize)
    summarize.add_argument("--profile-dir", type=Path, help="Path to a downloaded jax_profile artifact directory.")
    summarize.add_argument("--trace-file", type=Path, help="Path to an explicit trace JSON(.gz) file.")
    summarize.add_argument(
        "--xplane-file",
        type=Path,
        help="Path to an explicit *.xplane.pb protobuf profile. Parsed directly; xprof tables augment when available.",
    )
    summarize.add_argument(
        "--xplane-output-dir",
        type=Path,
        help="Optional directory for xprof table JSON exported while summarizing --xplane-file. Requires xprof.",
    )
    summarize.add_argument(
        "--xplane-count-trace-events",
        action="store_true",
        help="Ask xprof for trace_viewer event count while summarizing --xplane-file. This can be slower.",
    )
    _add_summarization_options(summarize)
    _add_output_option(summarize)

    query = subparsers.add_parser("query", help="Run a structured query against a summary JSON.")
    query.add_argument("--summary", type=Path, required=True, help="Path to a profile summary JSON.")
    query.add_argument("--question", required=True, help="Question to answer.")
    _add_top_k_option(query, "rows for list-style answers")

    compare = subparsers.add_parser("compare", help="Compare two profile summary JSON files.")
    _add_summary_pair_options(compare)
    _add_top_k_option(compare, "improved/regressed ops to report")

    track = subparsers.add_parser(
        "track",
        help="Compare two profile summaries, classify pass/warn/fail, and optionally append to a history JSONL file.",
    )
    _add_summary_pair_options(track)
    _add_top_k_option(track, "improved/regressed ops to report")
    _add_threshold_options(track)
    _add_output_option(track)

    report = subparsers.add_parser("report", help="Render a deterministic markdown root-cause report from a summary.")
    report.add_argument("--summary", type=Path, required=True, help="Path to a profile summary JSON.")
    _add_top_k_option(report, "hot ops/collectives in the report")
    _add_output_option(report, kind="markdown")

    history = subparsers.add_parser("history", help="Summarize a regression tracking JSONL history file.")
    history.add_argument("--history", type=Path, required=True, help="Path to regression history JSONL.")
    history.add_argument("--tail", type=int, default=20, help="Number of recent records to include.")
    _add_output_option(history)

    bundle = subparsers.add_parser(
        "bundle",
        help="Run a one-shot comparison bundle: summarize -> compare -> track -> reports.",
    )
    bundle.add_argument("--before-summary", type=Path, help="Optional existing baseline summary JSON.")
    bundle.add_argument("--after-summary", type=Path, help="Optional existing candidate summary JSON.")
    bundle.add_argument("--before-run-target", help="Optional baseline run target for auto summarization.")
    bundle.add_argument("--after-run-target", help="Optional candidate run target for auto summarization.")
    bundle.add_argument(
        "--strict-provenance",
        action="store_true",
        help="Fail if provenance checks indicate before/after are likely the same trace.",
    )
    _add_wandb_lookup_options(bundle)
    _add_summarization_options(bundle)
    _add_top_k_option(bundle, "rows for compare/reports")
    _add_threshold_options(bundle)
    bundle.add_argument("--output-dir", type=Path, required=True, help="Directory for bundle outputs.")
    _add_output_option(bundle, kind="bundle-manifest JSON")

    publish = subparsers.add_parser("publish", help="Publish summary/report as a W&B profile_summary artifact.")
    publish.add_argument("--summary", type=Path, required=True, help="Path to profile summary JSON.")
    publish.add_argument("--report", type=Path, help="Optional markdown report path to include.")
    publish.add_argument("--entity", default=WANDB_ENTITY, help="W&B entity (default: %(default)s).")
    publish.add_argument("--project", default=WANDB_PROJECT, help="W&B project (default: %(default)s).")
    publish.add_argument("--artifact-name", help="Optional artifact name override.")
    publish.add_argument(
        "--alias",
        action="append",
        dest="aliases",
        help="Artifact alias. Repeat for multiple aliases. Defaults to 'latest'.",
    )
    publish.add_argument("--dry-run", action="store_true", help="Print publication metadata without uploading.")
    _add_output_option(publish)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "summarize":
        _emit(_handle_summarize(args).to_json(), args.output)
        return

    if args.command == "query":
        summary = _load_summary(args.summary)
        print(_json_text(query_profile_summary(summary, args.question, top_k=args.top_k)))
        return

    if args.command == "compare":
        comparison = compare_profile_summaries(_load_summary(args.before), _load_summary(args.after), top_k=args.top_k)
        _enforce_provenance_policy(comparison, strict=args.strict_provenance)
        print(_json_text(comparison))
        return

    if args.command == "track":
        before = _load_summary(args.before)
        after = _load_summary(args.after)
        assessment = assess_profile_regression(before, after, thresholds=_thresholds_from_args(args), top_k=args.top_k)
        _enforce_provenance_policy(assessment["comparison"], strict=args.strict_provenance)
        record = make_regression_record(before=before, after=after, assessment=assessment, label=args.label)
        if args.history:
            append_regression_record(args.history, record)
        _emit(_json_text(record), args.output)
        return

    if args.command == "report":
        _emit(build_markdown_report(_load_summary(args.summary), top_k=args.top_k), args.output)
        return

    if args.command == "history":
        _emit(_json_text(summarize_regression_history(args.history, tail=args.tail)), args.output)
        return

    if args.command == "bundle":
        before = _resolve_bundle_summary(args, summary_path=args.before_summary, run_target=args.before_run_target)
        after = _resolve_bundle_summary(args, summary_path=args.after_summary, run_target=args.after_run_target)
        # Check provenance before doing the (much more expensive) bundle work.
        _enforce_provenance_policy(
            compare_profile_summaries(before, after, top_k=args.top_k), strict=args.strict_provenance
        )
        result = run_profile_comparison_bundle(
            before=before,
            after=after,
            output_dir=args.output_dir,
            thresholds=_thresholds_from_args(args),
            top_k=args.top_k,
            label=args.label,
            history_path=args.history,
        )
        _emit(_json_text(result.to_dict()), args.output)
        return

    if args.command == "publish":
        response = publish_profile_summary_artifact(
            summary_path=args.summary,
            report_path=args.report,
            entity=args.entity,
            project=args.project,
            artifact_name=args.artifact_name,
            aliases=args.aliases,
            dry_run=args.dry_run,
        )
        _emit(_json_text(response), args.output)
        return

    raise ValueError(f"Unhandled command: {args.command}")


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _emit(text: str, output: Path | None) -> None:
    """Write *text* to *output* and echo the path, or print *text* to stdout."""
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    print(str(output))


def _thresholds_from_args(args: argparse.Namespace) -> RegressionThresholds:
    return RegressionThresholds(
        max_step_median_regression_pct=args.max_step_median_regression_pct,
        max_step_p90_regression_pct=args.max_step_p90_regression_pct,
        max_communication_share_regression_abs=args.max_communication_share_regression_abs,
        max_stall_share_regression_abs=args.max_stall_share_regression_abs,
    )


def _handle_summarize(args: argparse.Namespace) -> ProfileSummary:
    if args.trace_file:
        return summarize_trace(
            args.trace_file,
            warmup_steps=args.warmup_steps,
            hot_op_limit=args.hot_op_limit,
            breakdown_mode=args.breakdown_mode,
        )

    if args.xplane_file:
        return summarize_xplane(
            args.xplane_file,
            output_dir=args.xplane_output_dir,
            warmup_steps=args.warmup_steps,
            hot_op_limit=args.hot_op_limit,
            count_trace_events=args.xplane_count_trace_events,
            breakdown_mode=args.breakdown_mode,
        )

    if args.artifact:
        downloaded = download_wandb_profile_artifact(args.artifact, download_root=args.download_root)
        return summarize_profile_artifact(
            downloaded.artifact_dir,
            run_metadata=downloaded.run_metadata,
            warmup_steps=args.warmup_steps,
            hot_op_limit=args.hot_op_limit,
            breakdown_mode=args.breakdown_mode,
        )

    if args.run_target:
        return _summarize_run_target(args, args.run_target)

    if args.profile_dir:
        return summarize_profile_artifact(
            args.profile_dir,
            warmup_steps=args.warmup_steps,
            hot_op_limit=args.hot_op_limit,
            breakdown_mode=args.breakdown_mode,
        )

    raise ValueError("Specify one of --trace-file, --xplane-file, --profile-dir, --artifact, or --run-target.")


def _summarize_run_target(args: argparse.Namespace, run_target: str) -> ProfileSummary:
    downloaded = download_profile_dir_for_run(
        run_target,
        entity=args.entity,
        project=args.project,
        download_root=args.download_root,
    )
    return summarize_profile_artifact(
        downloaded.profile_dir,
        run_metadata=downloaded.run_metadata,
        warmup_steps=args.warmup_steps,
        hot_op_limit=args.hot_op_limit,
        breakdown_mode=args.breakdown_mode,
    )


def _load_summary(path: Path) -> ProfileSummary:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in summary '{path}'.")
    return profile_summary_from_dict(data)


def _resolve_bundle_summary(
    args: argparse.Namespace, *, summary_path: Path | None, run_target: str | None
) -> ProfileSummary:
    if summary_path is not None:
        return _load_summary(summary_path)
    if run_target is not None:
        return _summarize_run_target(args, run_target)
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
