# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline, in-region diagnostics of the frozen first-wave solver trajectories.

Only aggregates and task IDs are printed. Does not call any model or change a grade.
"""

import json
from collections import Counter, defaultdict
from typing import Any

import fsspec

from experiments.post_training.capability_driven_envs.biotasks_library.freeze_combined_v2 import V2_ROOT
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_first_wave import (
    OUTPUT_ROOT,
    _model_path,
    load_panel,
)


def _rows(uri: str) -> list[dict[str, Any]]:
    with fsspec.open(uri, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _candidate_values(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    result = []
    for idx, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[idx:])
        except ValueError:
            continue
        if isinstance(value, (dict, list)) and value not in result:
            result.append(value)
    return result


def audit() -> dict[str, Any]:
    tasks = {
        row["task_id"]: row["task"]
        for row in _rows(V2_ROOT + "taskspecs.jsonl")
        if row["audit_disposition"] != "audit_disputed_unresolved"
    }
    models, _ = load_panel()
    per_model = {}
    task_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for model in models:
        prefix = OUTPUT_ROOT + f"attempts/{_model_path(model)}/"
        originals = {row["task_id"]: row for row in _rows(prefix + "results.jsonl")}
        uri = prefix + "rate-limit-recovery1.jsonl"
        fs, path = fsspec.core.url_to_fs(uri)
        recovery = {row["task_id"]: row for row in _rows(uri)} if fs.exists(path) else {}
        counts: Counter[str] = Counter()
        for task_id, original in originals.items():
            row = recovery.get(task_id, original)
            if row.get("infrastructure_error"):
                counts["infrastructure_error"] += 1
                continue
            trial = row["trial"]
            counts[f"trial:{trial['status']}:{trial.get('stop_reason')}"] += 1
            attempts = row.get("attempts", [])
            if not attempts:
                counts["no_attempt_record"] += 1
                continue
            attempt = attempts[-1]
            grade = attempt.get("grade") or {}
            counts[f"grade:{grade.get('status')}:{grade.get('error')}"] += 1
            failure = (attempt.get("rollout") or {}).get("failure") or {}
            if failure:
                counts[f"failure:{failure.get('exception_type')}:{failure.get('diagnostics', {}).get('operation')}"] += 1
            conversations = (attempt.get("rollout") or {}).get("conversations") or []
            messages = [message for conversation in conversations for message in conversation.get("messages", [])]
            tools = [message for message in messages if message.get("role") == "tool"]
            if tools:
                counts["tool_used"] += 1
            path_errors = [message for message in tools if "/workspace/workspace/" in str(message.get("content", ""))]
            if path_errors:
                counts["bad_workspace_prefix_error"] += 1
                if trial["status"] == "graded" and trial["correct"]:
                    counts["bad_workspace_prefix_then_correct"] += 1
            assistant = [message for message in messages if message.get("role") == "assistant"]
            final = assistant[-1] if assistant else {}
            text = final.get("content") or ""
            if not isinstance(text, str):
                text = ""
            counts["final_empty" if not text.strip() else "final_nonempty"] += 1
            expected = tasks[task_id]["grader"]["parameters"].get("expected")
            candidates = _candidate_values(text)
            if candidates:
                counts["embedded_json_candidate"] += 1
                exact = any(value == expected for value in candidates)
                if exact:
                    counts["embedded_exact_reference"] += 1
                if grade.get("status") == "submission_failure":
                    counts["malformed_but_json_candidate"] += 1
                    if exact:
                        counts["malformed_but_exact_reference"] += 1
                task_candidates[task_id].append(
                    {
                        "model": model.model,
                        "exact": exact,
                        "valid_submission": grade.get("status") == "graded",
                        "score": trial.get("correct"),
                    }
                )
        per_model[model.model] = dict(sorted(counts.items()))
    priorities: list[dict[str, Any]] = []
    for task_id, candidates in task_candidates.items():
        exact = sum(item["exact"] for item in candidates)
        wrong = sum(not item["exact"] for item in candidates)
        recoverable = sum(item["exact"] and not item["valid_submission"] for item in candidates)
        if recoverable or wrong >= 3:
            priorities.append(
                {
                    "task_id": task_id,
                    "exact_candidates": exact,
                    "nonexact_candidates": wrong,
                    "exact_but_malformed": recoverable,
                }
            )
    priorities.sort(key=lambda item: (-item["exact_but_malformed"], -item["nonexact_candidates"], item["task_id"]))
    return {
        "tasks": len(tasks),
        "model_counts": per_model,
        "candidate_diagnostics": priorities[:30],
        "candidate_diagnostic_total": len(priorities),
    }


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True, indent=2))
