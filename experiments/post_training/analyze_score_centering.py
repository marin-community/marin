# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize completion-aware quality from SkyRL's saved in-run evaluations.

Example:
    python -m experiments.post_training.analyze_score_centering \
        --run age4=s3://bucket/path/to/exports \
        --run age8=s3://bucket/other/exports \
        --output /tmp/score-centering-evals.csv

The script reads every held-out response. SkyRL's completed-stop score metric is
a signed reward contribution, so it cannot stand in for completed correctness.
Repeat --iris-log for a run continued under another Iris parent, in chronological
job order. Identical log lines are deduplicated; later attempts and jobs replace
their repeated optimizer steps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fsspec
from fsspec.spec import AbstractFileSystem
from rigging.filesystem.storage_path import prefix_join

ACCEPTED_STOPS = frozenset({"complete", "end_turn", "eos", "stop"})
CORE_MATH_DATASETS = ("val-gsm8k", "val-math500")
FIELDS = (
    "run",
    "step",
    "eval_dump_written_utc",
    "dataset",
    "questions",
    "completed_correct",
    "completed_correct_rate",
    "correct_any_stop",
    "raw_reward_mean",
    "completed_fraction",
    "length_stop_fraction",
    "response_tokens_mean",
    "membership_sha256",
)
METRIC_FIELDS = (
    "run",
    "step",
    "iris_job_index",
    "iris_attempt",
    "consumed_tokens",
    "cumulative_consumed_tokens",
    "consumed_sequences",
    "informative_group_fraction",
    "step_seconds",
    "cumulative_cycle_seconds",
    "nominal_cycle_gpu_hours",
    "age_mean",
    "age_p90",
    "age_at_least_four_fraction",
    "stale_rejected",
    "rejected_count",
    "rejected_rate",
    "mismatch_log_ratio_abs_mean",
    "tis_capped_fraction",
    "correction_abs_mean",
    "policy_entropy",
    "response_bytes_mean",
    "length_stop_fraction",
)


def _filesystem(path: str, s3_endpoint: str) -> tuple[AbstractFileSystem, str]:
    if path.startswith("s3://"):
        return (
            fsspec.filesystem(
                "s3",
                client_kwargs={"endpoint_url": s3_endpoint},
                config_kwargs={"s3": {"addressing_style": "virtual"}},
            ),
            path.removeprefix("s3://").rstrip("/"),
        )
    return fsspec.filesystem("file"), str(Path(path).resolve())


def _read_jsonl(fs: AbstractFileSystem, path: str) -> list[dict[str, Any]]:
    with fs.open(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _membership_hash(rows: list[dict[str, Any]]) -> str:
    questions = sorted((row["input_prompt"], row["env_extras"]["reward_spec"]["ground_truth"]) for row in rows)
    return hashlib.sha256(json.dumps(questions, ensure_ascii=False).encode()).hexdigest()


def _modified_utc(fs: AbstractFileSystem, path: str) -> str:
    info = fs.info(path)
    modified = info.get("LastModified") or info.get("mtime")
    if isinstance(modified, datetime):
        return modified.astimezone(UTC).isoformat()
    if isinstance(modified, (int, float)):
        return datetime.fromtimestamp(modified, tz=UTC).isoformat()
    raise ValueError(f"evaluation aggregate has no usable modification time: {path}")


def _summarize(
    run: str,
    step: int,
    dataset: str,
    rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    membership_hash: str,
    eval_dump_written_utc: str,
    response_tokens_mean: float | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"{run} step {step} dataset {dataset} has no responses")
    scores = [row["score"] for row in rows]
    if any(type(score) not in (int, float) or not math.isfinite(score) for score in scores):
        raise ValueError(f"{run} step {step} dataset {dataset} has a nonfinite or missing score")
    completed = [row["stop_reason"] in ACCEPTED_STOPS for row in rows]
    count = len(rows)
    completed_correct = sum(score > 0 and done for score, done in zip(scores, completed, strict=True))
    if response_tokens_mean is None:
        prefix = "eval/all" if dataset == "all" else f"eval/{dataset}"
        response_tokens_mean = aggregate[f"{prefix}/response_tokens_mean"]
    return {
        "run": run,
        "step": step,
        "eval_dump_written_utc": eval_dump_written_utc,
        "dataset": dataset,
        "questions": count,
        "completed_correct": completed_correct,
        "completed_correct_rate": completed_correct / count,
        "correct_any_stop": sum(score > 0 for score in scores),
        "raw_reward_mean": sum(scores) / count,
        "completed_fraction": sum(completed) / count,
        "length_stop_fraction": sum(row["stop_reason"] == "length" for row in rows) / count,
        "response_tokens_mean": response_tokens_mean,
        "membership_sha256": membership_hash,
    }


def summarize_run(label: str, export_path: str, s3_endpoint: str, *, core_math: bool = False) -> list[dict[str, Any]]:
    fs, root = _filesystem(export_path, s3_endpoint)
    sessions = fs.glob(prefix_join(prefix_join(root, "dumped_evals"), "global_step_*_evals"))
    if not sessions:
        raise ValueError(f"{label}: no in-run evaluation dumps under {export_path}")
    output: list[dict[str, Any]] = []
    for session in sessions:
        step = int(session.rsplit("/global_step_", 1)[1].removesuffix("_evals"))
        aggregate_path = prefix_join(session, "aggregated_results.jsonl")
        aggregate_rows = _read_jsonl(fs, aggregate_path)
        if len(aggregate_rows) != 1:
            raise ValueError(f"{label} step {step}: expected one aggregate metrics row")
        aggregate = aggregate_rows[0]
        eval_dump_written_utc = _modified_utc(fs, aggregate_path)
        all_rows: list[dict[str, Any]] = []
        by_dataset: dict[str, list[dict[str, Any]]] = {}
        for path in sorted(fs.glob(prefix_join(session, "*.jsonl"))):
            if path.endswith("/aggregated_results.jsonl"):
                continue
            dataset = path.rsplit("/", 1)[1].removesuffix(".jsonl")
            if dataset in by_dataset:
                raise ValueError(f"{label} step {step}: duplicate dataset {dataset}")
            rows = _read_jsonl(fs, path)
            by_dataset[dataset] = rows
            all_rows.extend(rows)
            output.append(
                _summarize(label, step, dataset, rows, aggregate, _membership_hash(rows), eval_dump_written_utc)
            )
        output.append(
            _summarize(label, step, "all", all_rows, aggregate, _membership_hash(all_rows), eval_dump_written_utc)
        )
        if core_math:
            if any(dataset not in by_dataset for dataset in CORE_MATH_DATASETS):
                raise ValueError(f"{label} step {step}: missing core math dataset")
            core_rows = [row for dataset in CORE_MATH_DATASETS for row in by_dataset[dataset]]
            core_tokens = sum(
                len(by_dataset[dataset]) * aggregate[f"eval/{dataset}/response_tokens_mean"]
                for dataset in CORE_MATH_DATASETS
            ) / len(core_rows)
            output.append(
                _summarize(
                    label,
                    step,
                    "core-math",
                    core_rows,
                    aggregate,
                    _membership_hash(core_rows),
                    eval_dump_written_utc,
                    core_tokens,
                )
            )
    return sorted(output, key=lambda row: (row["step"], row["dataset"]))


def verify_membership(rows: list[dict[str, Any]]) -> str:
    hashes = {row["membership_sha256"] for row in rows if row["dataset"] == "all"}
    if len(hashes) != 1:
        raise ValueError("held-out prompt and ground-truth membership differs between evaluations")
    return hashes.pop()


def summarize_iris_logs(label: str, paths: list[Path]) -> list[dict[str, Any]]:
    """Read mirrors, preferring later Iris jobs and then later attempts for repeated steps."""
    if not paths:
        raise ValueError(f"{label}: no Iris log paths")
    marker = "WANDB_MIRROR kind=train step="
    steps: dict[int, tuple[int | str, int | str, dict[str, Any]]] = {}
    for job_index, path in enumerate(paths):
        with path.open() as stream:
            for line in stream:
                line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
                if marker not in line:
                    continue
                step_text, payload = line.split(marker, 1)[1].split(" metrics=", 1)
                step = int(step_text)
                attempt_match = re.search(r"\battempt=(\d+)\b", line.split(marker, 1)[0])
                attempt = int(attempt_match.group(1)) if attempt_match else 0
                metrics = json.loads(payload)
                previous = steps.get(step)
                if previous is not None and previous[:2] == (job_index, attempt):
                    if previous[2] != metrics:
                        raise ValueError(f"{label}: conflicting Iris mirror step {step} within attempt {attempt}")
                    continue
                if previous is None or (job_index, attempt) > previous[:2]:
                    steps[step] = (job_index, attempt, metrics)
    if not steps:
        raise ValueError(f"{label}: no training metrics in {paths}")
    return _summarize_train_steps(label, steps)


def _summarize_train_steps(
    label: str, steps: dict[int, tuple[int | str, int | str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    cumulative_tokens = 0
    cumulative_seconds = 0.0
    for step, (job_index, attempt, metrics) in sorted(steps.items()):
        if metrics["trainer/global_step"] != step:
            raise ValueError(f"{label}: training record step {step} disagrees with trainer/global_step")
        tokens = metrics["async/performance/consumed_loss_tokens"]
        seconds = metrics["timing/step"]
        gpus = (
            metrics["async/performance/configured_policy_gpus"] + metrics["async/performance/configured_inference_gpus"]
        )
        cumulative_tokens += tokens
        cumulative_seconds += seconds
        result.append(
            {
                "run": label,
                "step": step,
                "iris_job_index": job_index,
                "iris_attempt": attempt,
                "consumed_tokens": tokens,
                "cumulative_consumed_tokens": cumulative_tokens,
                "consumed_sequences": metrics.get("consumed/sequences"),
                "informative_group_fraction": metrics.get("reward/informative_group_fraction"),
                "step_seconds": seconds,
                "cumulative_cycle_seconds": cumulative_seconds,
                "nominal_cycle_gpu_hours": cumulative_seconds * gpus / 3600,
                "age_mean": metrics.get("async/consumed_token_age_mean"),
                "age_p90": metrics.get("async/consumed_token_age_p90"),
                "age_at_least_four_fraction": metrics.get("async/consumed_token_age_at_least_four_fraction"),
                "stale_rejected": metrics.get("async/rejected_count/stale"),
                "rejected_count": metrics.get("async/rejected_count"),
                "rejected_rate": metrics.get("async/rejected_rate"),
                "mismatch_log_ratio_abs_mean": metrics.get("policy/mismatch/pooled/log_ratio_abs_mean"),
                "tis_capped_fraction": metrics.get("policy/tis/imp_ratio_capped_fraction"),
                "correction_abs_mean": metrics.get("policy/score_centering/correction_abs_mean"),
                "policy_entropy": metrics.get("policy/policy_entropy"),
                "response_bytes_mean": metrics.get("inference_bridge/response_bytes/mean"),
                "length_stop_fraction": metrics.get("consumed/length_stop_fraction"),
            }
        )
    return result


def summarize_iris_log(label: str, path: Path) -> list[dict[str, Any]]:
    """Read one Iris job's mirrors, preferring its later retry for repeated steps."""
    return summarize_iris_logs(label, [path])


def summarize_wandb_history(label: str, path: Path) -> list[dict[str, Any]]:
    """Read retained W&B training history when the Iris pod log is unavailable."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row["run"] == label]
    if len({row["wandb_run_id"] for row in rows}) != 1:
        raise ValueError(f"{label}: expected one W&B run identity in {path}")
    steps: dict[int, tuple[int | str, int | str, dict[str, Any]]] = {}
    for row in rows:
        step = row["trainer/global_step"]
        if type(step) is not int or row["_step"] != step or step in steps:
            raise ValueError(f"{label}: duplicate or inconsistent W&B training step {step}")
        steps[step] = ("", "", row)
    if not steps:
        raise ValueError(f"{label}: no W&B training rows in {path}")
    if sorted(steps) != list(range(1, max(steps) + 1)):
        raise ValueError(f"{label}: W&B training history has missing optimizer steps")
    return _summarize_train_steps(label, steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=EXPORT_PATH")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iris-log", action="append", default=[], metavar="LABEL=LOG_FILE")
    parser.add_argument("--wandb-history", action="append", default=[], metavar="LABEL=JSONL")
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--s3-endpoint", default="https://cwobject.com")
    parser.add_argument("--core-math", action="store_true", help="also summarize the shared GSM8K and Math500 subset")
    args = parser.parse_args()

    result: list[dict[str, Any]] = []
    labels: set[str] = set()
    for item in args.run:
        label, separator, path = item.partition("=")
        if not separator or not label or not path or label in labels:
            parser.error(f"invalid or duplicate --run {item!r}; expected unique LABEL=EXPORT_PATH")
        labels.add(label)
        result.extend(summarize_run(label, path, args.s3_endpoint, core_math=args.core_math))
    membership_hash = verify_membership(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)
    print(f"Wrote {len(result)} rows to {args.output}; held-out membership SHA-256: {membership_hash}")
    if args.iris_log or args.wandb_history:
        if args.metrics_output is None:
            parser.error("--metrics-output is required with --iris-log or --wandb-history")
        metrics: list[dict[str, Any]] = []
        log_paths: dict[str, list[Path]] = {}
        for item in args.iris_log:
            label, separator, path = item.partition("=")
            if not separator or label not in labels or not path:
                parser.error(f"invalid --iris-log {item!r}; its label must match a --run")
            log_paths.setdefault(label, []).append(Path(path))
        for label, paths in log_paths.items():
            metrics.extend(summarize_iris_logs(label, paths))
        history_labels: set[str] = set()
        for item in args.wandb_history:
            label, separator, path = item.partition("=")
            if not separator or label not in labels or not path or label in log_paths or label in history_labels:
                parser.error(f"invalid --wandb-history {item!r}; its label must match one --run")
            history_labels.add(label)
            metrics.extend(summarize_wandb_history(label, Path(path)))
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, METRIC_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(metrics)
        print(f"Wrote {len(metrics)} training metric rows to {args.metrics_output}")


if __name__ == "__main__":
    main()
