# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the v3 response matrix, regrading reusable v2 generations offline.

Never rewrite source rows. A response is reused only under the generation
identity checks in run_openrouter_first_wave_v3.load_cohort; it is separately
scored against the new, frozen v3 TaskSpec and answer format.
"""

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import fsspec
from taskcompendium.grading import grade_answer
from taskcompendium.grading_result import Outcome
from taskcompendium.models import ConversationTrace, GradingAttempt, TaskSpec, TextMessage
from taskforge.content_hash import digest

from experiments.post_training.capability_driven_envs.biotasks_library.finalize_combined_v1 import _write_new
from experiments.post_training.capability_driven_envs.biotasks_library.freeze_combined_v3 import V3_ROOT
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_first_wave import (
    _model_path,
    load_panel,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_first_wave_v3 import (
    OUTPUT_ROOT,
    _source_responses,
    load_cohort,
)

SNAPSHOT = OUTPUT_ROOT + "snapshots/complete-1/"


def _read(uri: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with fsspec.open(uri, "rb") as source:
        data = source.read()
    return [json.loads(line) for line in data.splitlines() if line], {
        "uri": uri,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _score(task: TaskSpec, raw: dict[str, Any]) -> tuple[int | None, str]:
    """Apply only the v3 verifier to an already-observed terminal response."""
    if raw.get("infrastructure_error"):
        return None, "missing_infrastructure"
    trial = raw["trial"]
    attempts = raw.get("attempts", [])
    failure = (attempts[-1].get("rollout") or {}).get("failure") or {} if attempts else {}
    failure_type = failure.get("exception_type")
    if failure_type in {"RateLimitError", "APITimeoutError", "APIConnectionError"} or (
        failure_type == "ValueError" and "no token usage" in str(failure.get("diagnostics", {}))
    ):
        return None, "missing_infrastructure"
    if trial["status"] != "graded":
        return None, "ungraded"
    if len(attempts) != 1:
        raise ValueError(f"Expected one attempt for {task.id}")
    conversations = (attempts[0].get("rollout") or {}).get("conversations") or []
    messages = [message for conversation in conversations for message in conversation.get("messages", [])]
    final = messages[-1] if messages else None
    if final is None or final.get("role") != "assistant" or final.get("tool_calls"):
        return 0, "submission_failure"
    content = final.get("content") or ""
    if not isinstance(content, str):
        raise ValueError(f"Unexpected final response content: {task.id}")
    attempt = GradingAttempt(
        conversation=ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=content)))
    )
    result = grade_answer(task, attempt)
    if result.status not in {Outcome.GRADED, Outcome.SUBMISSION_FAILURE} or result.reward is None:
        raise ValueError(f"v3 verifier failed for {task.id}: {result.status} {result.error}")
    return int(result.reward == 1.0), result.status.value


def freeze(output: str, commit: str) -> dict[str, Any]:
    cohort = load_cohort()
    models, panel_sha = load_panel()
    prior = _source_responses()
    tasks = {r["task_id"]: TaskSpec.model_validate(r["task"]) for r in cohort.rows}
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for model in models:
        prefix = f"{OUTPUT_ROOT}attempts/{_model_path(model)}/"
        smoke, source = _read(prefix + "smoke.jsonl")
        inputs.append(source)
        complete, source = _read(prefix + "results.jsonl")
        inputs.append(source)
        new = {r["task_id"]: (r, prefix + "smoke.jsonl") for r in smoke}
        if any(r["task_id"] in new for r in complete):
            raise ValueError("Duplicated smoke and model batch trial")
        new.update({r["task_id"]: (r, prefix + "results.jsonl") for r in complete})
        backfill = {}
        for name in ("route-metadata-backfill-smoke.jsonl", "route-metadata-backfill.jsonl"):
            overlay_uri = prefix + name
            overlay_fs, overlay_path = fsspec.core.url_to_fs(overlay_uri)
            if overlay_fs.exists(overlay_path):
                overlay_rows, source = _read(overlay_uri)
                inputs.append(source)
                for entry in overlay_rows:
                    if entry["response_id"] in backfill:
                        raise ValueError("Duplicate metadata backfill ID")
                    backfill[entry["response_id"]] = entry
        original_uri = (
            V3_ROOT.replace("v3/snapshots/reference-path-review-1/", "v2/calibration/openrouter-first-wave-v1/")
            + f"attempts/{_model_path(model)}/results.jsonl"
        )
        original, source = _read(original_uri)
        inputs.append(source)
        old = {r["task_id"]: r for r in original}
        recovery_uri = original_uri.replace("results.jsonl", "rate-limit-recovery1.jsonl")
        fs, path = fsspec.core.url_to_fs(recovery_uri)
        recovered = {}
        if fs.exists(path):
            recovery, source = _read(recovery_uri)
            inputs.append(source)
            recovered = {r["task_id"]: r for r in recovery}
        for spec in cohort.rows:
            task = tasks[spec["task_id"]]
            old_summary = prior[(model.model, spec["v2_task_id"])]
            if task.id in new:
                raw, source_uri = new[task.id]
                if raw["frozen_task_digest"] != spec["task_digest"]:
                    raise ValueError("New task digest mismatch")
                reused = False
            else:
                if (
                    task.id not in cohort.reusable_ids
                    or old_summary["missing_infrastructure"]
                    or not old_summary["route_verified"]
                ):
                    raise ValueError(f"Unjustified reuse or missing changed trial for {model.model}/{task.id}")
                raw = recovered.get(spec["v2_task_id"], old[spec["v2_task_id"]])
                source_uri = recovery_uri if spec["v2_task_id"] in recovered else original_uri
                if raw["frozen_task_digest"] != spec["v2_task_digest"]:
                    raise ValueError("Reused task digest mismatch")
                reused = True
            if raw.get("model") != model.model or raw.get("requested_provider_tag") != model.provider_tag:
                raise ValueError("Response route changed")
            if not reused:
                # The runner already checks each generation ID and writes any
                # eventual-consistency overlay. Refuse a new scored trial
                # without a verified route; absent evidence is missing.
                evidence = raw.get("provider_evidence", [])
                if any(
                    not e.get("route_matches")
                    and backfill.get(e.get("response_id"), {}).get("provider_name") != model.provider_name
                    for e in evidence
                ):
                    raise ValueError("Unverified or mismatched provider route")
            score, grade_status = _score(task, raw)
            if not reused and score is not None and not raw.get("provider_evidence"):
                raise ValueError("Scored trial lacks provider evidence")
            rows.append(
                {
                    "task_id": task.id,
                    "task_digest": digest(task),
                    "model": model.model,
                    "provider_tag": model.provider_tag,
                    "provider_name": model.provider_name,
                    "reused_identical_generation": reused,
                    "source_v2_task_id": spec["v2_task_id"] if reused else None,
                    "prior_frozen_score": old_summary["score"] if reused else None,
                    "score": score,
                    "grade_status": grade_status,
                    "original_trial_status": raw.get("trial", {}).get("status"),
                    "stop_reason": raw.get("trial", {}).get("stop_reason"),
                    "response_ids": [e.get("response_id") for e in raw.get("provider_evidence", [])],
                    "source_uri": source_uri,
                }
            )
        expected_new = {r["task_id"] for r in rows if r["model"] == model.model and not r["reused_identical_generation"]}
        if set(new) != expected_new or len(smoke) != 1 or len(new) != len(smoke) + len(complete):
            raise ValueError(f"Incomplete or duplicated new model batch for {model.model}")
    if len(rows) != 230 * 12 or len({(r["model"], r["task_id"]) for r in rows}) != len(rows):
        raise ValueError("Incomplete v3 response matrix")
    rows.sort(key=lambda row: (row["task_id"], row["model"]))
    data = b"".join(json.dumps(r, sort_keys=True, separators=(",", ":")).encode() + b"\n" for r in rows)
    outcomes = Counter(
        "correct" if r["score"] == 1 else "incorrect" if r["score"] == 0 else r["grade_status"] for r in rows
    )
    manifest = {
        "schema_version": "1",
        "campaign_id": "biotasks/combined/v3/calibration/openrouter-first-wave-v3",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code_commit": commit,
        "status": "internal_provisional_not_irt_validated",
        "task_snapshot_sha256": "b893bd40fb1c6676619b32cd11a180969605b2b69a7a3b5b6df1974ae4bbb8d6",
        "panel_manifest_sha256": panel_sha,
        "summary": {
            "items": 230,
            "models": 12,
            "pairs": len(rows),
            "reused_generation_pairs": sum(r["reused_identical_generation"] for r in rows),
            "new_generation_pairs": sum(not r["reused_identical_generation"] for r in rows),
            "outcomes": dict(outcomes),
        },
        "inputs": inputs,
        "artifacts": {"item-responses.jsonl": {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}},
    }
    _write_new(output.rstrip("/") + "/item-responses.jsonl", data)
    _write_new(output.rstrip("/") + "/manifest.json", json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=SNAPSHOT)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output, args.code_commit), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
