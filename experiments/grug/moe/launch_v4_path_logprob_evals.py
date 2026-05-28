# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch missing Grug-MoE v4 path logprob evals.

This launcher is intentionally rerunnable. It discovers successful
``grug_moe_mix_v4_path_r1_*`` training checkpoints, skips existing
``evaluation/grug_logprob`` result cells, and submits only missing evals. Run it
again as larger path-test checkpoints finish.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import fsspec
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main
from marin.utils import fsspec_glob

from experiments.grug.moe.eval_logprob import (
    GRUG_LOGPROB_TASKS,
    LEGACY_MOE_FLAT_CHECKPOINT_LAYOUT,
    CheckpointLayout,
    build_grug_logprob_eval_step,
    task_key,
)

GCS_ROOT = "gs://marin-us-east5"
GCS_GRUG_PREFIX = f"{GCS_ROOT}/grug"
GCS_EVAL_PREFIX = f"{GCS_ROOT}/evaluation/grug_logprob"
RUN_ID_PREFIX = "grug_moe_mix_v4_path_r1"
DEFAULT_MAX_CONCURRENT = 64
DEFAULT_RETRY_ATTEMPT = "numeric_retry1"
REFERENCE_OUTPUTS_DIR = Path("experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs")
OUTPUT_DIR = REFERENCE_OUTPUTS_DIR / "grug_moe_v4_path_eval_20260522"
METRIC_SUFFIX = ",none"
TASK_HASH_RE = re.compile(r"-[0-9a-f]{6}$")
OUTPUT_ATTEMPT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
REQUIRED_NUMERIC_METRICS_BY_TASK_ALIAS = {
    "logprob_gsm8k_5shot": frozenset({"bpb", "nll"}),
    "logprob_humaneval_10shot": frozenset({"bpb", "nll"}),
}
RUN_ID_RE = re.compile(
    r"""
    ^(grug_moe_mix_v4_path_r1_t\d{3}_d(?P<hidden_dim>\d+)-(?P<budget>[0-9.]+e[+-]\d+))
    -[0-9a-f]+$
    """,
    re.VERBOSE,
)
BASE_RUN_ID_RE = re.compile(
    r"""
    ^(grug_moe_mix(?:_v\d+)?_d(?P<hidden_dim>\d+)-(?P<budget>[0-9.]+e[+-]\d+))
    -[0-9a-f]+$
    """,
    re.VERBOSE,
)


class ExistingResultStatus(StrEnum):
    VALID = "valid_existing_result"
    INVALID = "invalid_existing_result"


@dataclass(frozen=True)
class TrainingCheckpoint:
    run_id: str
    root: str
    checkpoint_subpath: str
    hidden_dim: int
    budget: float
    target_steps: int | None
    checkpoint_layout: CheckpointLayout = LEGACY_MOE_FLAT_CHECKPOINT_LAYOUT


@dataclass(frozen=True)
class EvalCandidate:
    run_id: str
    hidden_dim: int
    budget: float
    checkpoint_subpath: str
    task_alias: str
    checkpoint_layout: CheckpointLayout
    output_prefix: str
    output_attempt: str | None
    action: str
    reason: str


def _read_text(path: str, *, allow_missing: bool = False) -> str:
    try:
        with fsspec.open(path, "rt") as handle:
            return handle.read()
    except FileNotFoundError:
        if allow_missing:
            return ""
        raise


def _glob_gs(pattern: str) -> list[str]:
    if not pattern.startswith("gs://"):
        raise ValueError(f"Expected gs:// pattern, got {pattern}")
    return sorted(path.rstrip("/") for path in fsspec_glob(pattern))


def _validate_output_attempt(output_attempt: str) -> None:
    if not OUTPUT_ATTEMPT_RE.match(output_attempt):
        raise ValueError(f"Invalid output attempt path segment: {output_attempt!r}")


def _root_name(root: str) -> str:
    return root.rstrip("/").rsplit("/", 1)[-1]


def _checkpoint_subpath(root: str) -> str:
    if not root.startswith(f"{GCS_ROOT}/"):
        raise ValueError(f"Checkpoint root is not east5-local: {root}")
    return f"{root.removeprefix(f'{GCS_ROOT}/')}/checkpoints"


def _training_checkpoint_from_root(root: str) -> TrainingCheckpoint | None:
    name = _root_name(root)
    match = RUN_ID_RE.match(name)
    if not match:
        match = BASE_RUN_ID_RE.match(name)
    if not match:
        return None
    status = _read_text(f"{root}/.executor_status", allow_missing=True).strip()
    if status != "SUCCESS":
        return None
    info = json.loads(_read_text(f"{root}/.executor_info"))
    config = info.get("config", {})
    return TrainingCheckpoint(
        run_id=name.rsplit("-", 1)[0],
        root=root,
        checkpoint_subpath=_checkpoint_subpath(root),
        hidden_dim=int(match.group("hidden_dim")),
        budget=float(match.group("budget")),
        target_steps=config.get("steps") if isinstance(config.get("steps"), int) else None,
    )


def discover_successful_path_checkpoints(
    only_run_ids: frozenset[str] = frozenset(),
    checkpoint_roots: tuple[str, ...] = (),
) -> list[TrainingCheckpoint]:
    if checkpoint_roots:
        roots = [root.rstrip("/") for root in checkpoint_roots]
    elif only_run_ids:
        status_paths = [
            status_path
            for run_id in sorted(only_run_ids)
            for status_path in _glob_gs(f"{GCS_GRUG_PREFIX}/{run_id}-*/.executor_status")
        ]
        roots = [path.rsplit("/", 1)[0] for path in status_paths]
    else:
        status_paths = _glob_gs(f"{GCS_GRUG_PREFIX}/{RUN_ID_PREFIX}_*/.executor_status")
        roots = [path.rsplit("/", 1)[0] for path in status_paths]
    checkpoints = [_training_checkpoint_from_root(root) for root in roots]
    resolved = [checkpoint for checkpoint in checkpoints if checkpoint is not None]
    return sorted(resolved, key=lambda item: (item.run_id, item.hidden_dim))


def parse_result_path(path: str) -> tuple[str, str]:
    rel = path.split(f"{GCS_EVAL_PREFIX}/", maxsplit=1)[1]
    parts = rel.rstrip("/").split("/")
    if len(parts) < 3:
        raise ValueError(f"Expected eval result path under run/task/results.json, got {path}")
    return parts[0], TASK_HASH_RE.sub("", parts[1])


def _result_patterns(
    *,
    only_run_ids: frozenset[str],
    only_task_aliases: frozenset[str],
) -> list[str]:
    if only_run_ids and only_task_aliases:
        patterns = []
        for run_id in sorted(only_run_ids):
            for task_alias in sorted(only_task_aliases):
                patterns.append(f"{GCS_EVAL_PREFIX}/{run_id}/{task_alias}*/results.json")
                patterns.append(f"{GCS_EVAL_PREFIX}/{run_id}/{task_alias}/**/results.json")
        return patterns
    if only_run_ids:
        patterns = []
        for run_id in sorted(only_run_ids):
            patterns.append(f"{GCS_EVAL_PREFIX}/{run_id}/*/results.json")
            patterns.append(f"{GCS_EVAL_PREFIX}/{run_id}/*/**/results.json")
        return patterns
    return [
        f"{GCS_EVAL_PREFIX}/{RUN_ID_PREFIX}_*/*/results.json",
        f"{GCS_EVAL_PREFIX}/{RUN_ID_PREFIX}_*/*/**/results.json",
    ]


def _aggregate_result_key(task_alias: str, results: dict[str, Any]) -> str:
    if len(results) == 1:
        return next(iter(results))
    candidates = [
        task_alias,
        re.sub(r"_(?:0|5|10)shot$", "", task_alias),
        task_alias.replace("_5shot", ""),
        task_alias.replace("_0shot", ""),
    ]
    for candidate in candidates:
        if candidate in results:
            return candidate
    for key, value in results.items():
        if isinstance(value, dict) and not value.get("alias", "").startswith(" - "):
            return key
    return next(iter(results))


def _numeric_metric_names(metric_dict: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for key, value in metric_dict.items():
        if not key.endswith(METRIC_SUFFIX):
            continue
        metric = key[: -len(METRIC_SUFFIX)]
        if metric.endswith("_stderr"):
            continue
        if isinstance(value, (int, float)):
            names.add(metric)
    return names


def result_has_required_numeric_metrics(path: str, task_alias: str) -> bool:
    """Return whether ``results.json`` has usable numeric metrics for a task."""
    try:
        data = json.loads(_read_text(path))
        results = data.get("results", {})
        if not isinstance(results, dict) or not results:
            return False
        result_key = _aggregate_result_key(task_alias, results)
        metric_dict = results[result_key]
        if not isinstance(metric_dict, dict):
            return False
    except (KeyError, json.JSONDecodeError, TypeError):
        return False

    metric_names = _numeric_metric_names(metric_dict)
    required_metrics = REQUIRED_NUMERIC_METRICS_BY_TASK_ALIAS.get(task_alias)
    if required_metrics is not None:
        return required_metrics.issubset(metric_names)
    return bool(metric_names)


def discover_existing_result_statuses(
    only_run_ids: frozenset[str] = frozenset(),
    only_task_aliases: frozenset[str] = frozenset(),
) -> dict[tuple[str, str], ExistingResultStatus]:
    """Return validity status for existing ``(run_id, task_alias)`` results."""
    statuses: dict[tuple[str, str], ExistingResultStatus] = {}
    result_paths = sorted(
        {
            path
            for pattern in _result_patterns(only_run_ids=only_run_ids, only_task_aliases=only_task_aliases)
            for path in _glob_gs(pattern)
        }
    )
    for result_path in result_paths:
        run_id, task_alias = parse_result_path(result_path)
        key = (run_id, task_alias)
        if only_run_ids and run_id not in only_run_ids:
            continue
        if only_task_aliases and task_alias not in only_task_aliases:
            continue
        if result_has_required_numeric_metrics(result_path, task_alias):
            statuses[key] = ExistingResultStatus.VALID
        elif statuses.get(key) != ExistingResultStatus.VALID:
            statuses[key] = ExistingResultStatus.INVALID
    return statuses


def build_eval_candidates(
    *,
    force_existing: bool,
    only_run_ids: frozenset[str] = frozenset(),
    only_task_aliases: frozenset[str] = frozenset(),
    checkpoint_roots: tuple[str, ...] = (),
    assume_missing: bool = False,
    retry_attempt: str = DEFAULT_RETRY_ATTEMPT,
) -> list[EvalCandidate]:
    candidates: list[EvalCandidate] = []
    _validate_output_attempt(retry_attempt)
    checkpoints = discover_successful_path_checkpoints(only_run_ids, checkpoint_roots)
    discovered_run_ids = frozenset(checkpoint.run_id for checkpoint in checkpoints)
    status_run_ids = only_run_ids or discovered_run_ids
    existing_results = (
        {}
        if assume_missing
        else discover_existing_result_statuses(status_run_ids, only_task_aliases)
    )
    for checkpoint in checkpoints:
        if only_run_ids and checkpoint.run_id not in only_run_ids:
            continue
        for task in GRUG_LOGPROB_TASKS:
            alias = task_key(task)
            if only_task_aliases and alias not in only_task_aliases:
                continue
            existing_status = existing_results.get((checkpoint.run_id, alias))
            action = "skip" if existing_status == ExistingResultStatus.VALID and not force_existing else "launch"
            if force_existing and existing_status is not None:
                reason = "force_existing"
            elif existing_status == ExistingResultStatus.VALID:
                reason = ExistingResultStatus.VALID.value
            elif existing_status == ExistingResultStatus.INVALID:
                reason = ExistingResultStatus.INVALID.value
            else:
                reason = "missing"
            output_attempt = retry_attempt if action == "launch" and existing_status is not None else None
            output_prefix = f"{GCS_EVAL_PREFIX}/{checkpoint.run_id}/{alias}"
            if output_attempt:
                output_prefix = f"{output_prefix}/{output_attempt}"
            candidates.append(
                EvalCandidate(
                    run_id=checkpoint.run_id,
                    hidden_dim=checkpoint.hidden_dim,
                    budget=checkpoint.budget,
                    checkpoint_subpath=checkpoint.checkpoint_subpath,
                    task_alias=alias,
                    checkpoint_layout=checkpoint.checkpoint_layout,
                    output_prefix=output_prefix,
                    output_attempt=output_attempt,
                    action=action,
                    reason=str(reason),
                )
            )
    return candidates


def build_eval_steps(candidates: list[EvalCandidate], *, max_eval_instances: int | None) -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    seen_names: set[str] = set()
    for candidate in candidates:
        if candidate.action != "launch":
            continue
        task = next(task for task in GRUG_LOGPROB_TASKS if task_key(task) == candidate.task_alias)
        step = build_grug_logprob_eval_step(
            run_id=candidate.run_id,
            hidden_dim=candidate.hidden_dim,
            budget=candidate.budget,
            checkpoint_subpath=candidate.checkpoint_subpath,
            task=task,
            max_eval_instances=max_eval_instances,
            checkpoint_layout=candidate.checkpoint_layout,
            output_attempt=candidate.output_attempt,
        )
        if step.name in seen_names:
            raise ValueError(f"Duplicate eval step name: {step.name}")
        seen_names.add(step.name)
        steps.append(step)
    return steps


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_artifacts(
    candidates: list[EvalCandidate],
    steps: list[ExecutorStep],
    *,
    dry_run: bool,
    max_concurrent: int,
    only_run_ids: list[str],
    only_task_aliases: list[str],
    checkpoint_roots: list[str],
    assume_missing: bool,
    retry_attempt: str,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    invalid_existing_count = sum(candidate.reason == ExistingResultStatus.INVALID.value for candidate in candidates)
    valid_existing_count = sum(candidate.reason == ExistingResultStatus.VALID.value for candidate in candidates)
    _write_csv(
        OUTPUT_DIR / "eval_manifest.csv",
        [asdict(candidate) for candidate in candidates],
        [
            "run_id",
            "hidden_dim",
            "budget",
            "checkpoint_subpath",
            "task_alias",
            "checkpoint_layout",
            "output_prefix",
            "output_attempt",
            "action",
            "reason",
        ],
    )
    summary = {
        "description": "Grug-MoE v4 path logprob eval completion.",
        "run_id_prefix": RUN_ID_PREFIX,
        "num_tasks": len(GRUG_LOGPROB_TASKS),
        "num_successful_training_checkpoints": len({candidate.run_id for candidate in candidates}),
        "num_candidate_cells": len(candidates),
        "num_launch_cells": len(steps),
        "num_skipped_cells": sum(candidate.action == "skip" for candidate in candidates),
        "num_invalid_existing_cells": invalid_existing_count,
        "num_valid_existing_cells": valid_existing_count,
        "checkpoint_layouts": sorted({str(candidate.checkpoint_layout) for candidate in candidates}),
        "max_concurrent": max_concurrent,
        "only_run_ids": only_run_ids,
        "only_task_aliases": only_task_aliases,
        "checkpoint_roots": checkpoint_roots,
        "assume_missing": assume_missing,
        "retry_attempt": retry_attempt,
        "dry_run": dry_run,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT_DIR / "step_names.txt").write_text("\n".join(step.name for step in steps) + ("\n" if steps else ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Write manifests and print counts without submitting.")
    parser.add_argument("--force-existing", action="store_true", help="Launch even when a matching results.json exists.")
    parser.add_argument(
        "--only-run-id",
        action="append",
        default=[],
        help="Restrict to one successful training run ID. Repeatable; useful for canaries.",
    )
    parser.add_argument(
        "--only-task-alias",
        action="append",
        default=[],
        help="Restrict to one task alias from GRUG_LOGPROB_TASKS. Repeatable; useful for canaries.",
    )
    parser.add_argument(
        "--checkpoint-root",
        action="append",
        default=[],
        help="Use exact successful training root. Repeatable; useful for canaries.",
    )
    parser.add_argument(
        "--assume-missing",
        action="store_true",
        help="Skip result discovery and treat filtered cells as missing. Use only for exact canaries/retries.",
    )
    parser.add_argument(
        "--max-eval-instances",
        type=int,
        default=None,
        help="Optional lm-eval limit for canaries. Leave unset for full evals.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help="Maximum concurrent eval cells for executor submission.",
    )
    parser.add_argument(
        "--retry-attempt",
        default=DEFAULT_RETRY_ATTEMPT,
        help="Path segment used for invalid existing result retries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = build_eval_candidates(
        force_existing=args.force_existing,
        only_run_ids=frozenset(args.only_run_id),
        only_task_aliases=frozenset(args.only_task_alias),
        checkpoint_roots=tuple(args.checkpoint_root),
        assume_missing=args.assume_missing,
        retry_attempt=args.retry_attempt,
    )
    steps = build_eval_steps(candidates, max_eval_instances=args.max_eval_instances)
    write_artifacts(
        candidates,
        steps,
        dry_run=args.dry_run,
        max_concurrent=args.max_concurrent,
        only_run_ids=args.only_run_id,
        only_task_aliases=args.only_task_alias,
        checkpoint_roots=args.checkpoint_root,
        assume_missing=args.assume_missing,
        retry_attempt=args.retry_attempt,
    )

    launched = len(steps)
    skipped = sum(candidate.action == "skip" for candidate in candidates)
    invalid_existing_count = sum(candidate.reason == ExistingResultStatus.INVALID.value for candidate in candidates)
    valid_existing_count = sum(candidate.reason == ExistingResultStatus.VALID.value for candidate in candidates)
    print(
        json.dumps(
            {
                "successful_training_checkpoints": len({candidate.run_id for candidate in candidates}),
                "tasks": len(GRUG_LOGPROB_TASKS),
                "launch_cells": launched,
                "skipped_cells": skipped,
                "invalid_existing_cells": invalid_existing_count,
                "valid_existing_cells": valid_existing_count,
                "checkpoint_layouts": sorted({str(candidate.checkpoint_layout) for candidate in candidates}),
                "max_concurrent": args.max_concurrent,
                "only_run_ids": args.only_run_id,
                "only_task_aliases": args.only_task_alias,
                "checkpoint_roots": args.checkpoint_root,
                "assume_missing": args.assume_missing,
                "retry_attempt": args.retry_attempt,
                "output_dir": str(OUTPUT_DIR),
            },
            indent=2,
        )
    )
    if args.dry_run:
        return
    if not steps:
        print("No missing Grug path eval cells to launch.")
        return
    executor_main(
        ExecutorMainConfig(prefix=GCS_ROOT, max_concurrent=args.max_concurrent),
        steps=steps,
        description="Grug-MoE v4 proportional-to-v4 path logprob eval completion.",
    )


if __name__ == "__main__":
    main()
