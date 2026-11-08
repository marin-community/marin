# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze a compact item-response view of the immutable OpenRouter first wave.

Failed upstream 429 attempts are retained; a separate recovery may supersede
only those missing trials. Ungraded trials remain unscored until adjudicated.
"""

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import fsspec
from taskforge.content_hash import digest

from experiments.post_training.capability_driven_envs.biotasks_library.finalize_combined_v1 import _write_new
from experiments.post_training.capability_driven_envs.biotasks_library.recover_openrouter_first_wave import (
    _is_rate_limited,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_first_wave import (
    OUTPUT_ROOT,
    _model_path,
    load_eligible,
    load_panel,
)

SNAPSHOT = OUTPUT_ROOT + "snapshots/complete-1/"


def _read(uri: str) -> tuple[list[dict[str, Any]], dict[str, object]]:
    with fsspec.open(uri, "rb") as source:
        data = source.read()
    return [json.loads(line) for line in data.splitlines() if line], {
        "uri": uri,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows)


def freeze(output: str, code_commit: str) -> dict[str, Any]:
    tasks, task_sha = load_eligible(
        "s3://marin-us-east-02a/marin4sc/taskforge/task-library/"
        "campaigns/biotasks/combined/v2/snapshots/verified-corrections-1/taskspecs.jsonl"
    )
    digests = {task.id: task for task in tasks}
    models, panel_sha = load_panel()
    responses = []
    sources = []
    for model in models:
        prefix = f"{OUTPUT_ROOT}attempts/{_model_path(model)}/"
        original, source = _read(prefix + "results.jsonl")
        sources.append(source)
        by_id = {row["task_id"]: row for row in original}
        if len(original) != len(tasks) or set(by_id) != set(digests):
            raise ValueError(f"Incomplete original records: {model.model}")
        failed_ids = {task_id for task_id, row in by_id.items() if _is_rate_limited(row)}
        recovery_uri = prefix + "rate-limit-recovery1.jsonl"
        fs, path = fsspec.core.url_to_fs(recovery_uri)
        recovery = {}
        if failed_ids:
            if not fs.exists(path):
                raise ValueError(f"Missing recovery for {model.model}: {len(failed_ids)} upstream 429s")
            recovered, source = _read(recovery_uri)
            sources.append(source)
            recovery = {row["task_id"]: row for row in recovered}
            if len(recovery) != len(recovered) or set(recovery) != failed_ids:
                raise ValueError(f"Recovery changed the set of missing pairs for {model.model}")
        backfill = {}
        for filename in ("route-metadata-backfill.jsonl", "route-metadata-backfill-rate-limit-recovery1.jsonl"):
            uri = prefix + filename
            b_fs, b_path = fsspec.core.url_to_fs(uri)
            if b_fs.exists(b_path):
                rows, source = _read(uri)
                sources.append(source)
                for row in rows:
                    if row["response_id"] in backfill:
                        raise ValueError("Duplicate generation metadata backfill")
                    backfill[row["response_id"]] = row
        for task in tasks:
            first = by_id[task.id]
            selected = recovery.get(task.id, first)
            if (
                selected["frozen_task_digest"] != digest(task)
                or first["frozen_task_digest"] != digest(task)
                or selected["model"] != model.model
                or selected.get("policy_digest") != first.get("policy_digest")
                or selected.get("execution_backend") != first.get("execution_backend")
            ):
                raise ValueError(f"Mismatched recovery digest/model: {task.id}")
            evidence = selected.get("provider_evidence", [])
            if selected.get("trial", {}).get("status") == "graded" and not evidence:
                raise ValueError(f"Graded trial lacks routed generation evidence: {model.model} / {task.id}")
            for entry in evidence:
                if entry.get("route_matches"):
                    continue
                supplement = backfill.get(entry.get("response_id"))
                if supplement is None or supplement["provider_name"] != model.provider_name:
                    raise ValueError(f"Unverified route for {model.model} / {task.id}")
            trial = selected.get("trial")
            if not isinstance(trial, dict):
                raise ValueError(f"No trial result for {model.model} / {task.id}")
            attempts = selected.get("attempts", [])
            failure = (attempts[-1].get("rollout", {}).get("failure") or {}) if attempts else {}
            failure_type = failure.get("exception_type")
            missing_infrastructure = bool(
                selected.get("infrastructure_error")
                or failure_type in {"RateLimitError", "APITimeoutError", "APIConnectionError"}
                or (failure_type == "ValueError" and "no token usage" in str(failure.get("diagnostics", {})))
            )
            score = trial.get("correct") if trial.get("status") == "graded" and not missing_infrastructure else None
            responses.append(
                {
                    "task_id": task.id,
                    "task_spec_digest": selected["frozen_task_digest"],
                    "model": model.model,
                    "provider_tag": model.provider_tag,
                    "provider_name": model.provider_name,
                    "selected_attempt": "rate-limit-recovery1" if task.id in recovery else "initial",
                    "initial_rate_limited": task.id in failed_ids,
                    "score": int(score) if isinstance(score, bool) else None,
                    "trial_status": trial.get("status"),
                    "stop_reason": trial.get("stop_reason"),
                    "cause": trial.get("cause"),
                    "failure_type": failure_type,
                    "missing_infrastructure": missing_infrastructure,
                    "route_verified": bool(evidence)
                    and all(
                        entry.get("route_matches")
                        or backfill.get(entry.get("response_id"), {}).get("provider_name") == model.provider_name
                        for entry in evidence
                    ),
                    "response_ids": [entry.get("response_id") for entry in evidence],
                    "record_uri": recovery_uri if task.id in recovery else prefix + "results.jsonl",
                }
            )
    if len(responses) != 12 * len(tasks) or len({(r["model"], r["task_id"]) for r in responses}) != len(responses):
        raise ValueError("Incomplete or repeated first-wave response matrix")
    responses.sort(key=lambda row: (row["task_id"], row["model"]))
    counts = Counter(
        (
            "missing_infrastructure"
            if row["missing_infrastructure"]
            else "ungraded"
            if row["score"] is None
            else "correct"
            if row["score"]
            else "incorrect"
        )
        for row in responses
    )
    response_bytes = _jsonl(responses)
    manifest = {
        "schema_version": "1",
        "campaign_id": "biotasks/combined/v2/calibration/openrouter-first-wave-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code_commit": code_commit,
        "status": "internal_diagnostic_not_irt_validated",
        "warning": (
            "245 items have only model-audit confirmations; 58 disputed v2 items were excluded. "
            "Ungraded and infrastructure trials remain separate from scored zeros."
        ),
        "task_snapshot_sha256": task_sha,
        "panel_manifest_sha256": panel_sha,
        "summary": {
            "item_count": len(tasks),
            "model_count": len(models),
            "pair_count": len(responses),
            "outcomes": dict(sorted(counts.items())),
            "original_upstream_rate_limits": sum(row["initial_rate_limited"] for row in responses),
            "remaining_upstream_rate_limits": sum(row["failure_type"] == "RateLimitError" for row in responses),
        },
        "inputs": sources,
        "artifacts": {
            "item-responses.jsonl": {"sha256": hashlib.sha256(response_bytes).hexdigest(), "bytes": len(response_bytes)}
        },
    }
    _write_new(output.rstrip("/") + "/item-responses.jsonl", response_bytes)
    _write_new(output.rstrip("/") + "/manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=SNAPSHOT)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output, args.code_commit), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
