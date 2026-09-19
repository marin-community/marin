"""Summarize completion-aware quality from SkyRL's saved in-run evaluations.

Example:
    python -m experiments.post_training.analyze_score_centering \
        --run age4=s3://bucket/path/to/exports \
        --run age8=s3://bucket/other/exports \
        --output /tmp/score-centering-evals.csv

The script reads every held-out response. SkyRL's completed-stop score metric is
a signed reward contribution, so it cannot stand in for completed correctness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import fsspec

ACCEPTED_STOPS = frozenset({"complete", "end_turn", "eos", "stop"})
FIELDS = (
    "run",
    "step",
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
    "consumed_tokens",
    "cumulative_consumed_tokens",
    "step_seconds",
    "cumulative_cycle_seconds",
    "nominal_cycle_gpu_hours",
    "age_mean",
    "age_p90",
    "age_at_least_four_fraction",
    "stale_rejected",
    "rejected_rate",
    "mismatch_log_ratio_abs_mean",
    "tis_capped_fraction",
    "correction_abs_mean",
    "response_bytes_mean",
    "length_stop_fraction",
)


def _filesystem(path: str, s3_endpoint: str) -> tuple[Any, str]:
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


def _read_jsonl(fs: Any, path: str) -> list[dict[str, Any]]:
    with fs.open(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _membership_hash(rows: list[dict[str, Any]]) -> str:
    questions = sorted((row["input_prompt"], row["env_extras"]["reward_spec"]["ground_truth"]) for row in rows)
    return hashlib.sha256(json.dumps(questions, ensure_ascii=False).encode()).hexdigest()


def _summarize(
    run: str,
    step: int,
    dataset: str,
    rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    membership_hash: str,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"{run} step {step} dataset {dataset} has no responses")
    scores = [row["score"] for row in rows]
    if any(type(score) not in (int, float) or not math.isfinite(score) for score in scores):
        raise ValueError(f"{run} step {step} dataset {dataset} has a nonfinite or missing score")
    completed = [row["stop_reason"] in ACCEPTED_STOPS for row in rows]
    count = len(rows)
    completed_correct = sum(score > 0 and done for score, done in zip(scores, completed, strict=True))
    prefix = "eval/all" if dataset == "all" else f"eval/{dataset}"
    response_tokens_mean = aggregate[f"{prefix}/response_tokens_mean"]
    return {
        "run": run,
        "step": step,
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


def summarize_run(label: str, export_path: str, s3_endpoint: str) -> list[dict[str, Any]]:
    fs, root = _filesystem(export_path, s3_endpoint)
    sessions = fs.glob(f"{root}/dumped_evals/global_step_*_evals")
    if not sessions:
        raise ValueError(f"{label}: no in-run evaluation dumps under {export_path}")
    output: list[dict[str, Any]] = []
    for session in sessions:
        step = int(session.rsplit("/global_step_", 1)[1].removesuffix("_evals"))
        aggregate_rows = _read_jsonl(fs, f"{session}/aggregated_results.jsonl")
        if len(aggregate_rows) != 1:
            raise ValueError(f"{label} step {step}: expected one aggregate metrics row")
        aggregate = aggregate_rows[0]
        all_rows: list[dict[str, Any]] = []
        for path in sorted(fs.glob(f"{session}/*.jsonl")):
            if path.endswith("/aggregated_results.jsonl"):
                continue
            dataset = path.rsplit("/", 1)[1].removesuffix(".jsonl")
            rows = _read_jsonl(fs, path)
            all_rows.extend(rows)
            output.append(_summarize(label, step, dataset, rows, aggregate, _membership_hash(rows)))
        output.append(_summarize(label, step, "all", all_rows, aggregate, _membership_hash(all_rows)))
    return sorted(output, key=lambda row: (row["step"], row["dataset"]))


def verify_membership(rows: list[dict[str, Any]]) -> str:
    hashes = {row["membership_sha256"] for row in rows if row["dataset"] == "all"}
    if len(hashes) != 1:
        raise ValueError("held-out prompt and ground-truth membership differs between evaluations")
    return hashes.pop()


def summarize_iris_log(label: str, path: Path) -> list[dict[str, Any]]:
    """Read the durable per-step stdout mirror when a W&B run ends without its final flush."""
    marker = "WANDB_MIRROR kind=train step="
    steps: dict[int, dict[str, Any]] = {}
    with path.open() as stream:
        for line in stream:
            if marker not in line:
                continue
            step_text, payload = line.split(marker, 1)[1].split(" metrics=", 1)
            step = int(step_text)
            if step in steps:
                raise ValueError(f"{label}: duplicate Iris mirror step {step}; inspect retries before analysis")
            steps[step] = json.loads(payload)
    if not steps:
        raise ValueError(f"{label}: no training metrics in {path}")
    result: list[dict[str, Any]] = []
    cumulative_tokens = 0
    cumulative_seconds = 0.0
    for step, metrics in sorted(steps.items()):
        if metrics["trainer/global_step"] != step:
            raise ValueError(f"{label}: Iris mirror step {step} disagrees with trainer/global_step")
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
                "consumed_tokens": tokens,
                "cumulative_consumed_tokens": cumulative_tokens,
                "step_seconds": seconds,
                "cumulative_cycle_seconds": cumulative_seconds,
                "nominal_cycle_gpu_hours": cumulative_seconds * gpus / 3600,
                "age_mean": metrics.get("async/consumed_token_age_mean"),
                "age_p90": metrics.get("async/consumed_token_age_p90"),
                "age_at_least_four_fraction": metrics.get("async/consumed_token_age_at_least_four_fraction"),
                "stale_rejected": metrics.get("async/rejected_count/stale"),
                "rejected_rate": metrics.get("async/rejected_rate"),
                "mismatch_log_ratio_abs_mean": metrics.get("policy/mismatch/pooled/log_ratio_abs_mean"),
                "tis_capped_fraction": metrics.get("policy/tis/imp_ratio_capped_fraction"),
                "correction_abs_mean": metrics.get("policy/score_centering/correction_abs_mean"),
                "response_bytes_mean": metrics.get("inference_bridge/response_bytes/mean"),
                "length_stop_fraction": metrics.get("consumed/length_stop_fraction"),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=EXPORT_PATH")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iris-log", action="append", default=[], metavar="LABEL=LOG_FILE")
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--s3-endpoint", default="https://cwobject.com")
    args = parser.parse_args()

    result: list[dict[str, Any]] = []
    labels: set[str] = set()
    for item in args.run:
        label, separator, path = item.partition("=")
        if not separator or not label or not path or label in labels:
            parser.error(f"invalid or duplicate --run {item!r}; expected unique LABEL=EXPORT_PATH")
        labels.add(label)
        result.extend(summarize_run(label, path, args.s3_endpoint))
    membership_hash = verify_membership(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        writer.writerows(result)
    print(f"Wrote {len(result)} rows to {args.output}; held-out membership SHA-256: {membership_hash}")
    if args.iris_log:
        if args.metrics_output is None:
            parser.error("--metrics-output is required with --iris-log")
        metrics: list[dict[str, Any]] = []
        for item in args.iris_log:
            label, separator, path = item.partition("=")
            if not separator or label not in labels or not path:
                parser.error(f"invalid --iris-log {item!r}; its label must match a --run")
            metrics.extend(summarize_iris_log(label, Path(path)))
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, METRIC_FIELDS)
            writer.writeheader()
            writer.writerows(metrics)
        print(f"Wrote {len(metrics)} Iris mirror rows to {args.metrics_output}")


if __name__ == "__main__":
    main()
