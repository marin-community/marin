# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze compact item responses and a manifest for a BioTasks solver calibration."""

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

import fsspec


def _read(uri: str) -> tuple[bytes, list[dict[str, Any]]]:
    with fsspec.open(uri, "rb") as stream:
        data = stream.read()
    return data, [json.loads(line) for line in data.splitlines() if line]


def _write_new(uri: str, data: bytes) -> None:
    fs, path = fsspec.core.url_to_fs(uri)
    if fs.exists(path):
        raise FileExistsError(f"refusing to overwrite immutable artifact: {uri}")
    with fs.open(path, "wb") as stream:
        stream.write(data)


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows)


def _percentile(values: list[int], fraction: float) -> int:
    return sorted(values)[round((len(values) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--taskspecs", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--iris-job", required=True)
    args = parser.parse_args()

    result_bytes, results = _read(args.results)
    task_bytes, task_rows = _read(args.taskspecs)
    task_records = {row["task_id"]: row for row in task_rows}
    if len(results) != len(task_records) or {row["task_id"] for row in results} != set(task_records):
        raise ValueError("calibration results must cover every frozen TaskSpec exactly once")

    compact = []
    sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grade_statuses: Counter[str] = Counter()
    ungraded_causes: Counter[str] = Counter()
    for row in results:
        trial = row["trial"]
        task_record = task_records[row["task_id"]]
        source_campaign = task_record["source_campaign"]
        sources[source_campaign].append(row)
        grade_status = "ungraded"
        if row["attempts"]:
            grade_status = row["attempts"][-1].get("grade", {}).get("status", "ungraded")
        if trial["status"] == "graded":
            grade_statuses[grade_status] += 1
        else:
            ungraded_causes[str(trial.get("cause", "unknown"))] += 1
        compact.append(
            {
                "task_id": row["task_id"],
                "frozen_task_digest": row["frozen_task_digest"],
                "execution_task_digest": row["execution_task_digest"],
                "source_campaign": source_campaign,
                "task_family": row["task_family"],
                "model": row["model"],
                "policy_digest": row["policy_digest"],
                "score": 1 if trial["correct"] is True else 0,
                "trial_status": trial["status"],
                "grade_status": grade_status,
                "reward": trial["reward"],
                "stop_reason": trial["stop_reason"],
                "cause": trial.get("cause"),
                "tokens_in": trial["tokens_in"],
                "tokens_out": trial["tokens_out"],
                "turns": trial["turns"],
                "elapsed_seconds": row["elapsed_seconds"],
            }
        )
    compact.sort(key=lambda row: row["task_id"])
    compact_bytes = _jsonl(compact)

    tokens_in = [row["trial"]["tokens_in"] for row in results]
    tokens_out = [row["trial"]["tokens_out"] for row in results]
    correct = sum(row["trial"]["correct"] is True for row in results)
    graded = sum(row["trial"]["status"] == "graded" for row in results)
    summary = {
        "task_count": len(results),
        "correct": correct,
        "incorrect_or_ungraded": len(results) - correct,
        "accuracy": correct / len(results),
        "graded": graded,
        "ungraded": len(results) - graded,
        "graded_accuracy": correct / graded,
        "graded_result_statuses": dict(sorted(grade_statuses.items())),
        "ungraded_causes": dict(sorted(ungraded_causes.items())),
        "stop_reasons": dict(sorted(Counter(row["trial"]["stop_reason"] for row in results).items())),
        "tokens": {
            "input_total": sum(tokens_in),
            "output_total": sum(tokens_out),
            "input_median": statistics.median(tokens_in),
            "output_median": statistics.median(tokens_out),
            "input_p95": _percentile(tokens_in, 0.95),
            "output_p95": _percentile(tokens_out, 0.95),
        },
        "source_campaigns": {
            source: {
                "task_count": len(rows),
                "correct": sum(row["trial"]["correct"] is True for row in rows),
                "accuracy": sum(row["trial"]["correct"] is True for row in rows) / len(rows),
            }
            for source, rows in sorted(sources.items())
        },
    }
    summary_bytes = json.dumps(summary, indent=2, sort_keys=True).encode() + b"\n"
    artifacts = {
        "item-responses.jsonl": compact_bytes,
        "summary.json": summary_bytes,
    }
    for name, data in artifacts.items():
        _write_new(f"{args.output_root.rstrip('/')}/{name}", data)

    manifest = {
        "schema_version": "1",
        "campaign_id": results[0]["campaign_id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code_commit": args.code_commit,
        "iris_job": args.iris_job,
        "release_status": "internal_unreviewed_do_not_publish",
        "scoring_rule": "score=1 only for exact reward 1; graded failures and model/runtime terminations score 0",
        "execution_note": "ShellSim and gVisor results must remain separate populations",
        "inputs": {
            "results": {
                "uri": args.results,
                "bytes": len(result_bytes),
                "sha256": hashlib.sha256(result_bytes).hexdigest(),
            },
            "taskspecs": {
                "uri": args.taskspecs,
                "bytes": len(task_bytes),
                "sha256": hashlib.sha256(task_bytes).hexdigest(),
            },
        },
        "artifacts": {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in artifacts.items()
        },
        "summary": summary,
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    _write_new(f"{args.output_root.rstrip('/')}/manifest.json", manifest_bytes)
    print(manifest_bytes.decode(), end="")


if __name__ == "__main__":
    main()
