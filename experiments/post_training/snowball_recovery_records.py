# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Strict records for recovering saved step-50 Snowball results without new inference."""

import base64
import hashlib
import json
import math
from collections.abc import Callable
from urllib.parse import urlsplit

RUNTIME_COMMIT = "e18e6bc00ed17e34cde7f74eaa550a02f4948639"
LIBRARY_COMMIT = "6e2fe1839ea067be147f1ce06aced63a96670f7a"
FAILURE_COUNTS = {"sft": 2, "drdoom": 1}
TOKENIZER_REVISIONS = {
    "sft": "fe666478454e79fe6027a0bab0388d9abd68f51a",
    "drdoom": "5b5705e8345bb111de03b3f0f9a00c8041e57db4",
}
RECEIPT_FIELDS = ("task_id", "answer_sha256", "task_rollout_spec_sha256")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def pinned_bytes(data: bytes, expected: str) -> bytes:
    if sha256(data) != expected:
        raise ValueError("Recovery input differs from its immutable SHA-256")
    return data


def require_disjoint(destination: str, original_roots: list[str]) -> None:
    """Reject ancestor, descendant and equivalent paths in any original output tree."""

    def location(uri: str) -> tuple[str, str, tuple[str, ...]]:
        parsed = urlsplit(uri)
        parts = tuple(part for part in parsed.path.split("/") if part)
        if any(part in {".", ".."} for part in parts):
            raise ValueError("Recovery destinations must use canonical paths")
        return parsed.scheme, parsed.netloc, parts

    target = location(destination)
    for root in original_roots:
        source = location(root)
        shared = min(len(target[2]), len(source[2]))
        if target[:2] == source[:2] and target[2][:shared] == source[2][:shared]:
            raise ValueError("Recovery output overlaps an original output or control tree")


def require_source(runtime_commit: str, revisions: dict[str, str]) -> None:
    if (
        runtime_commit != RUNTIME_COMMIT
        or set(revisions.values()) != {LIBRARY_COMMIT}
        or set(revisions) != {"marin-taskcompendium", "marin-rolloutengine", "marin-verifyit"}
    ):
        raise ValueError("Recovery requires the original e18 runtime and all three 6e2 libraries")


def saved_answer(response_ids: list[int], eos: int, decode: Callable[..., str], extract: Callable[[str], str]) -> str:
    """Reproduce e18's authoritative token-based reasoning extraction path only."""
    if not response_ids:
        raise ValueError("Saved completion has no response tokens")
    semantic_ids = response_ids[:-1] if response_ids[-1] == eos else response_ids
    text = decode(semantic_ids, skip_special_tokens=False)
    if not any(
        marker in text
        for marker in ("<think>", "</think>", "<thinking>", "</thinking>", "<|start_think|>", "<|end_think|>")
    ):
        raise ValueError("Saved tokens do not prove e18's token-based final-answer extraction path")
    return extract(text)


def prepare_records(
    arm: str,
    report: dict,
    probe: dict,
    panel: list[dict],
    dumps: list[tuple[dict, str]],
    expected_failed_rows: dict[str, str],
    decode: Callable[..., str],
    eos: int,
    extract: Callable[[str], str],
) -> dict:
    """Freeze only the reviewed operational failures and their original complete task specs."""
    tasks = report["tasks"]
    if report["step"] != 50 or report["protocol"] != {
        "max_new_tokens": 8192,
        "request_window_tokens": 24576,
        "temperature": 0.0,
    }:
        raise ValueError("Recovery requires the original step50 matched8k protocol")
    failed = {row["task_id"]: row for row in tasks if row["score"] is None}
    if (
        len(tasks) != 910
        or len({row["task_id"] for row in tasks}) != 910
        or report["complete"]
        or len(failed) != FAILURE_COUNTS[arm]
        or set(failed) != set(expected_failed_rows)
    ):
        raise ValueError("Original report does not match the reviewed fixed-910 failure set")
    pairing = [(row["task_id"], row["evaluation_family"], row["contract_id"]) for row in tasks]
    identity = "sha256:" + sha256(json.dumps(pairing, separators=(",", ":")).encode())
    if identity != report["pairing_identity"]:
        raise ValueError("Original report pairing identity differs")
    for row in tasks:
        score = row["score"]
        if score is not None and (isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Original scored result is not a finite normalized score")
    by_id = {row["task_id"]: row for row in panel}
    admitted = probe["validation_tasks"]
    uid_map = {row["dataset_uid"]: row["task_id"] for row in admitted}
    if (
        probe["status"] != "passed"
        or probe["validation_sha256"] != report["panel_sha256"]
        or len(by_id) != 910
        or len(uid_map) != 910
        or set(uid_map.values()) != set(by_id)
        or set(by_id) != {row["task_id"] for row in tasks}
    ):
        raise ValueError("Panel, admission report and original task pairing differ")
    for row in tasks:
        if by_id[row["task_id"]]["source_contract_id"] != row["contract_id"]:
            raise ValueError("Original source contract changed")
    selected = []
    for row, raw_row_sha256 in dumps:
        task_id = uid_map[row["uid"]]
        if task_id not in failed:
            continue
        verification = row["verification"]
        if (
            raw_row_sha256 != expected_failed_rows[task_id]
            or verification["status"] not in {"error", "unavailable"}
            or (verification.get("diagnostics") or {}).get("operation") not in {None, "grade"}
            or row["server_error"] is not None
            or row["exception_type"] != failed[task_id]["failure_type"]
        ):
            raise ValueError("Only original grading failures may be recovered")
        raw = by_id[task_id]["task_rollout_spec"]
        spec = json.loads(raw)
        task = spec["task"]
        if (
            task["id"] != task_id
            or task["answer_type"] != "text"
            or task["answer_format"] != {"kind": "plain_text"}
            or any(task.get(key) for key in ("final_tools", "interaction_tools", "tool_bindings", "output_paths"))
            or spec["limits"]["max_turns"] != 1
            or task["grader"]["kind"] != "script"
            or spec["task_machine"] is not None
            or spec["verifier_machine"]["backend"] != "daytona"
        ):
            raise ValueError("Recovery requires the audited one-turn, text-only original script contract")
        if (
            decode(row["prompt_token_ids"]) != row["input_prompt"]
            or decode(row["response_token_ids"]) != row["output_response"]
        ):
            raise ValueError("Tokenizer does not reproduce saved token evidence")
        answer = saved_answer(row["response_token_ids"], eos, decode, extract).encode()
        # Retain complete immutable resource bytes; no URI-backed resource may drift during a regrade.
        for resources in task["resources"].values():
            for resource in resources:
                if resource["source"]["kind"] != "inline_file":
                    raise ValueError("Original grading resources must be inline and immutable")
                base64.b64decode(resource["source"]["content_base64"], validate=True)
        selected.append(
            {
                "task_id": task_id,
                "task_rollout_spec": raw,
                "task_rollout_spec_sha256": sha256(raw.encode()),
                "answer_utf8_base64": base64.b64encode(answer).decode(),
                "answer_sha256": sha256(answer),
                "saved_row_sha256": raw_row_sha256,
                "original_error_row": failed[task_id],
                "stop_reason": row["stop_reason"],
            }
        )
    if len(selected) != len(failed) or {row["task_id"] for row in selected} != set(failed):
        raise ValueError("Missing or duplicate original failed completion")
    return {
        "arm": arm,
        "runtime_commit": RUNTIME_COMMIT,
        "original_report_content_sha256": sha256(encoded(report)),
        "requests": selected,
        "regenerated_completions": 0,
        "original_training_job_success": False,
    }


def join_receipts(
    report: dict,
    prepared: dict,
    receipts: list[dict],
    evidence: dict[str, bytes],
    summarize: Callable[[list[dict]], dict],
) -> dict:
    """Create a derived report while keeping every original scored row unchanged."""
    if prepared["original_report_content_sha256"] != sha256(encoded(report)):
        raise ValueError("Prepared answers belong to another original report")
    expected = {row["task_id"]: row for row in prepared["requests"]}
    failed = {row["task_id"] for row in report["tasks"] if row["score"] is None}
    if (
        set(expected) != failed
        or len(receipts) != FAILURE_COUNTS[prepared["arm"]]
        or {row["task_id"] for row in receipts} != failed
    ):
        raise ValueError("Missing or repeated original-failure receipts")
    replacements = {}
    for receipt in receipts:
        request = expected[receipt["task_id"]]
        if any(receipt[key] != request[key] for key in RECEIPT_FIELDS):
            raise ValueError("Receipt changed the saved answer or original task/machine/limits")
        raw = evidence[receipt["execution_evidence_sha256"]]
        if sha256(raw) != receipt["execution_evidence_sha256"]:
            raise ValueError("Grading execution evidence changed")
        execution = json.loads(raw)
        if any(execution[key] != receipt[key] for key in (*RECEIPT_FIELDS, "score", "source")):
            raise ValueError("Receipt differs from its original execution evidence")
        require_source(receipt["source"]["runtime_commit"], receipt["source"]["library_revisions"])
        score = receipt["score"]
        if (
            receipt["attempt_count"] != 1
            or execution["grade"]["status"] != "graded"
            or isinstance(score, bool)
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("Receipt is not one usable original-grader attempt")
        grade = execution["grade"]
        if (
            not all(math.isfinite(grade[key]) for key in ("reward", "score_min", "score_max"))
            or grade["score_max"] <= grade["score_min"]
        ):
            raise ValueError("Invalid original grader score range")
        normalized = min(
            1.0, max(0.0, (grade["reward"] - grade["score_min"]) / (grade["score_max"] - grade["score_min"]))
        )
        if normalized != score:
            raise ValueError("Receipt score differs from original grader normalization")
        replacements[receipt["task_id"]] = score
    rows = [
        {
            **row,
            "score": replacements.get(row["task_id"], row["score"]),
            "verification_status": "verified" if row["task_id"] in replacements else row["verification_status"],
            "failure_type": None if row["task_id"] in replacements else row["failure_type"],
        }
        for row in report["tasks"]
    ]
    summary = summarize(rows)
    if not summary["complete"] or summary["pairing_identity"] != report["pairing_identity"]:
        raise ValueError("Derived recovery failed complete fixed-910 pairing")
    unchanged = [row for row in report["tasks"] if row["task_id"] not in replacements]
    if unchanged != [row for row in summary["tasks"] if row["task_id"] not in replacements]:
        raise ValueError("Recovery changed an originally scored row")
    return {
        "kind": "derived-regraded-saved-completions",
        "original_training_job_success": False,
        "original_report_content_sha256": sha256(encoded(report)),
        "original_error_rows": [row for row in report["tasks"] if row["task_id"] in replacements],
        "unchanged_scored_rows": len(unchanged),
        "regenerated_completions": 0,
        "protocol": report["protocol"],
        "panel_sha256": report["panel_sha256"],
        "summary": summary,
    }
