# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish native SkyRL evaluation rows through FineStore and EvalDash's record contract."""

import hashlib
import json
from collections import defaultdict

from evalrecords.records import EvalRunRecord, TaskCoverage, write_record
from finestore.eval import EvalSample, EvaluationStore, Grading, SampleKind
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.long_loop.archive import digest, write_once

# The sample-row contract this writer fills. Bumped when the fields a row carries change, so an
# already-published archive can be deliberately re-published (its rows superseded) without its
# unchanged inputs looking drifted.
ROW_VERSION = 2

# Rows stay bounded: output and extracted are display payload, not training data.
MAX_OUTPUT_CHARS = 4000
MAX_EXTRACTED_CHARS = 600

_THINK_OPEN_MARKERS = ("<|start_think|>", "<think>")
_THINK_CLOSE_MARKERS = ("<|end_think|>", "</think>")
_END_OF_TURN_TOKENS = ("<|eot_id|>", "<|end_of_text|>", "<|im_end|>")


def _strip_end_of_turn_tokens(text: str) -> str:
    """Drop the decoded end-of-turn special tokens a response can end with."""
    stripped = text.rstrip()
    changed = True
    while changed:
        changed = False
        for token in _END_OF_TURN_TOKENS:
            if stripped.endswith(token):
                stripped = stripped[: -len(token)].rstrip()
                changed = True
    return stripped


def visible_response(text: object) -> str | None:
    """The final response after the think block, or None when there is none.

    A response whose thinking never closed has no final response; the sample row records a failure
    reason instead. A response with no think markers at all is all final response.
    """
    if not isinstance(text, str) or not text:
        return None
    starts = [text.rfind(marker) + len(marker) for marker in _THINK_CLOSE_MARKERS if text.rfind(marker) >= 0]
    if starts:
        body = text[max(starts) :]
    elif any(marker in text for marker in _THINK_OPEN_MARKERS):
        return None
    else:
        body = text
    return _strip_end_of_turn_tokens(body).strip() or None


def _message_text(content: object) -> str:
    """Flatten one message's content: a string, or a list of ``{type, text}`` parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part if isinstance(part, str) else part.get("text", "") if isinstance(part, dict) else "" for part in content
        ]
        return "\n".join(part for part in parts if part)
    return ""


def comparable_answer(messages: object) -> str | None:
    """The final assistant text the verifier read, or None when it said nothing.

    The evidence messages are what the grader scored against; their last assistant turn is the
    model's comparable answer. An empty final turn (a truncated rollout) is no answer.
    """
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return _message_text(message.get("content")).strip() or None
    return None


def bounded(text: str | None, limit: int) -> str | None:
    """None stays None; empty text becomes None; long text is cut with a visible marker."""
    if text is None:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + " …[truncated]"


def no_answer_reason(verification: object, stop_reason: object) -> str | None:
    """A visible reason for a row that produced no comparable answer."""
    if isinstance(stop_reason, str) and stop_reason and stop_reason != "stop":
        return f"no answer (stop_reason={stop_reason})"
    if isinstance(verification, dict):
        reason = verification.get("reason")
        if isinstance(reason, str) and reason.strip():
            return reason.strip()
    return None


def publish_evaluation(*, source: str, record: EvalRunRecord, records_prefix: str) -> str:
    """Keep every native row, including failed trials; index scores only when verified."""
    files = sorted((StoragePath(source) / "*.jsonl").glob(), key=str)
    if not files:
        raise ValueError("No native evaluation files to archive")
    signature = hashlib.sha256(
        json.dumps(
            {
                "record": record.model_dump(mode="json", by_alias=True),
                "sources": [(str(path), hashlib.sha256(path.read_bytes()).hexdigest()) for path in files],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    marker_path = StoragePath(record.results_path) / "publication.json"
    if marker_path.exists():
        marker = json.loads(marker_path.read_bytes())
        if marker["signature"] != signature:
            raise ValueError("Evaluation archive inputs changed")
        if marker.get("row_version") == ROW_VERSION:
            if digest(marker["record_path"])[1] != marker["record_sha256"]:
                raise ValueError("Published evaluation record changed")
            if digest(prefix_join(record.results_path, "HEAD"))[1] != marker["head_sha256"]:
                raise ValueError("Published evaluation archive HEAD changed")
            return marker["record_path"]
        # The inputs are unchanged (same signature) but the marker predates the current row
        # contract: deliberately re-archive so its rows carry the current fields.
    counts = defaultdict(lambda: {"attempted": 0, "scored": 0, "correct": 0})
    scores = defaultdict(list)
    store = EvaluationStore.open(record.results_path, writer_id="long-loop-archive")
    try:
        for path in files:
            raw = path.read_bytes()
            filename = path.name
            store.add_source_artifact(filename, raw, content_type="application/x-ndjson")
            if filename == "aggregated_results.jsonl":
                continue
            for line_number, line in enumerate(raw.splitlines()):
                if not line.strip():
                    continue
                row = json.loads(line)
                task, uid = row["data_source"], row["uid"]
                trial_id = f"{filename}:{line_number}"
                name = hashlib.sha256(trial_id.encode()).hexdigest() + ".json"
                trajectory = store.add_artifact(name, line, metadata={"source": filename, "line": line_number})
                verification = row["verification"]
                verified = verification is not None and verification["status"] == "verified"
                score = verification["score"] if verified else None
                counts[task]["attempted"] += 1
                counts[task]["scored"] += int(verified)
                counts[task]["correct"] += int(verified and verification["passed"] is True)
                if verified:
                    scores[task].append(score)
                answer = comparable_answer(row.get("messages"))
                reason = None if answer else no_answer_reason(verification, row.get("stop_reason"))
                store.add_sample(
                    EvalSample(
                        task=task,
                        doc_id=uid,
                        kind=SampleKind.AGENTIC if row["env_class"] == "harbor_trial" else SampleKind.GENERATION,
                        trajectory_uri=trajectory,
                        output=bounded(visible_response(row.get("output_response")), MAX_OUTPUT_CHARS),
                        extracted=bounded(answer if answer is not None else reason, MAX_EXTRACTED_CHARS),
                        grading=Grading(
                            method="native-verifier",
                            metric="reward",
                            score=score,
                            passed=verification["passed"] if verified else None,
                            detail=json.dumps(verification),
                        ),
                        metrics={"reward": score} if verified else {},
                        doc=json.dumps(
                            {
                                "uid": uid,
                                "data_source": task,
                                "native_source": filename,
                                "source_line": line_number,
                                "source_sha256": hashlib.sha256(raw).hexdigest(),
                            }
                        ),
                    ),
                    trial_id=trial_id,
                )
            store.flush()
        store.seal()
    finally:
        store.close()
    updated = record.model_copy(
        update={
            "metrics": {task: {"reward": sum(values) / len(values)} for task, values in scores.items()},
            "coverage": {
                task: TaskCoverage(n_attempted=c["attempted"], n_scored=c["scored"], n_correct=c["correct"])
                for task, c in counts.items()
            },
        }
    )
    path = write_record(updated, records_prefix)
    marker = json.dumps(
        {
            "signature": signature,
            "row_version": ROW_VERSION,
            "record_path": path,
            "record_sha256": digest(path)[1],
            "head_sha256": digest(prefix_join(record.results_path, "HEAD"))[1],
        },
        sort_keys=True,
    ).encode()
    if marker_path.exists():
        marker_path.write_bytes(marker)
    else:
        write_once(prefix_join(record.results_path, "publication.json"), marker)
    return path
