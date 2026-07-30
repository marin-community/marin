# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LLM failure-mode review over a run's per-sample export.

Samples up to ``n`` rows of one run/task through the shared samples reader, renders each into a compact
text digest (prompt, the model's answer, the gold answer, grader detail, and -- for agentic samples --
the resolved trajectory), and asks one Claude call to bucket them into a fixed failure-mode rubric with
a short narrative.

Every non-happy path degrades to ``{"available": false, "reason": ...}`` rather than raising, mirroring
the logs and artifact endpoints: the ``anthropic`` SDK not being importable, ``ANTHROPIC_API_KEY``
unset, no samples for the task, or a model reply that is not valid JSON. The SDK is imported inside the
call so the server runs without it.
"""

from __future__ import annotations

import json
import logging
import os

import samples
from marin.evaluation.samples import EvalSample, SampleKind

logger = logging.getLogger(__name__)

# The review model is cheap and fast by design; override with EVALDASH_REVIEW_MODEL.
REVIEW_MODEL = os.environ.get("EVALDASH_REVIEW_MODEL", "claude-haiku-4-5-20251001")
REVIEW_MAX_TOKENS = 1500

# Per-sample digest caps so the prompt stays bounded regardless of how large any one field is.
MAX_PROMPT_CHARS = 1200
MAX_ANSWER_CHARS = 800
MAX_TRAJECTORY_CHARS = 2000
MAX_DETAIL_CHARS = 600
MAX_DOC_IDS_PER_CATEGORY = 3

_SYSTEM_PROMPT = (
    "You are an eval failure-mode analyst. You are given graded samples from one model on one "
    "evaluation task and must bucket them by why each turned out the way it did. Prefer these fixed "
    "category labels, and only invent a new short label when none fit:\n"
    "- misread the question\n"
    "- arithmetic error\n"
    "- format/extraction mismatch\n"
    "- refused/empty\n"
    "- off-by-one option\n"
    "- correct\n"
    "- other\n\n"
    "Reply with ONLY a JSON object, no prose and no markdown fences, matching exactly:\n"
    '{"categories": [{"label": string, "count": integer, "doc_ids": [string]}], "narrative": string}\n'
    "List at most 3 representative doc_ids per category, the counts should sum to the number of samples "
    "provided, and narrative is 2-4 sentences on the dominant failure modes."
)


def _unavailable(model: str | None, task: str, sample_filter: str, reason: str, n_reviewed: int = 0) -> dict:
    return {
        "available": False,
        "reason": reason,
        "model": model,
        "task": task,
        "filter": sample_filter,
        "n_reviewed": n_reviewed,
        "summary": None,
    }


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _prompt_text(sample: EvalSample) -> str:
    if sample.prompt_text:
        return sample.prompt_text
    if sample.prompt_messages:
        return "\n".join(f"{message.role}: {message.content}" for message in sample.prompt_messages)
    return ""


def _choice_label(sample: EvalSample, index: int | None) -> str:
    if index is None or not sample.choices or index < 0 or index >= len(sample.choices):
        return "(none)"
    choice = sample.choices[index]
    return _clip(f"{choice.label}. {choice.text}", MAX_ANSWER_CHARS)


def _outcome(sample: EvalSample) -> str:
    if sample.correct is True:
        return "correct"
    if sample.correct is False:
        return "incorrect"
    return "ungraded"


def _trajectory_text(results_path: str | None, sample: EvalSample) -> str | None:
    """Resolve one agentic sample's trajectory to text, reusing the size-capped artifact reader."""
    if not sample.trajectory_uri or not results_path:
        return None
    artifact = samples.fetch_artifact(results_path, sample.trajectory_uri)
    return artifact.text if artifact.available else None


def _sample_digest(results_path: str | None, sample: EvalSample) -> str:
    """One sample rendered as a bounded text block the review model can categorize."""
    lines = [f"doc_id: {sample.doc_id}", f"outcome: {_outcome(sample)}"]
    prompt = _prompt_text(sample)
    if prompt:
        lines.append(f"prompt: {_clip(prompt, MAX_PROMPT_CHARS)}")
    if sample.kind == SampleKind.MULTIPLE_CHOICE:
        lines.append(f"model_choice: {_choice_label(sample, sample.model_choice)}")
        lines.append(f"gold_choice: {_choice_label(sample, sample.target_choice)}")
    elif sample.kind == SampleKind.GENERATION:
        lines.append(f"model_answer: {_clip(sample.extracted or sample.output or '', MAX_ANSWER_CHARS)}")
        if sample.target_text:
            lines.append(f"gold: {_clip(sample.target_text, MAX_ANSWER_CHARS)}")
    elif sample.kind == SampleKind.AGENTIC:
        if sample.target_text:
            lines.append(f"gold: {_clip(sample.target_text, MAX_ANSWER_CHARS)}")
        trajectory = _trajectory_text(results_path, sample)
        if trajectory:
            lines.append(f"trajectory: {_clip(trajectory, MAX_TRAJECTORY_CHARS)}")
    if sample.grading and sample.grading.detail and sample.grading.detail != "{}":
        lines.append(f"grading_detail: {_clip(sample.grading.detail, MAX_DETAIL_CHARS)}")
    return "\n".join(lines)


def _render_prompt(task: str, sample_filter: str, digests: list[str]) -> str:
    header = f"Task: {task}\nFilter: {sample_filter} ({len(digests)} samples)\n\nSamples:"
    blocks = [f"--- sample {i + 1} ---\n{digest}" for i, digest in enumerate(digests)]
    return "\n".join([header, *blocks])


def _between_braces(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else None


def _extract_json(text: str) -> dict | None:
    """Parse the reply as a JSON object, tolerating leading/trailing prose or a code fence."""
    for candidate in (text, _between_braces(text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _shape_summary(raw: dict) -> dict | None:
    """Coerce a parsed reply into the ``summary`` contract, or None when it has neither expected key."""
    if "categories" not in raw and "narrative" not in raw:
        return None
    categories = []
    for entry in raw.get("categories") or []:
        if not isinstance(entry, dict):
            continue
        count = entry.get("count")
        doc_ids = entry.get("doc_ids") or []
        categories.append(
            {
                "label": str(entry.get("label", "")),
                "count": int(count) if isinstance(count, (int, float)) and not isinstance(count, bool) else 0,
                "doc_ids": [str(doc_id) for doc_id in doc_ids][:MAX_DOC_IDS_PER_CATEGORY],
            }
        )
    return {"categories": categories, "narrative": str(raw.get("narrative", ""))}


def review_run_samples(
    results_path: str | None,
    model: str | None,
    task: str,
    sample_filter: str,
    n: int,
) -> dict:
    """Review up to ``n`` of a run's ``task`` samples for failure modes with one Claude call.

    ``sample_filter`` is ``all``/``correct``/``incorrect`` (the samples reader's correctness filter).
    Degrades to ``{"available": false, "reason": ...}`` -- never raising -- when the ``anthropic`` SDK is
    missing, ``ANTHROPIC_API_KEY`` is unset, the task has no matching samples, or the model does not
    return valid JSON.
    """
    try:
        import anthropic  # noqa: PLC0415  guarded optional dep -- absence degrades gracefully
    except ImportError:
        return _unavailable(model, task, sample_filter, "anthropic SDK is not installed")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _unavailable(model, task, sample_filter, "ANTHROPIC_API_KEY is not set")

    page = samples.fetch_samples(results_path, task, offset=0, limit=n, correct=sample_filter)
    if not page.available:
        return _unavailable(model, task, sample_filter, page.error or "samples are unavailable")
    if not page.rows:
        return _unavailable(model, task, sample_filter, f"no {sample_filter} samples for task {task!r}")

    digests = [_sample_digest(results_path, row) for row in page.rows]
    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=REVIEW_MODEL,
            max_tokens=REVIEW_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _render_prompt(task, sample_filter, digests)}],
        )
    except Exception as exc:
        # The review model is best-effort: a transport or API error degrades like the logs view rather
        # than 500-ing the sample browser.
        logger.info("review model call failed for task %s: %s", task, exc)
        return _unavailable(model, task, sample_filter, f"review model call failed: {exc}"[:400], len(digests))

    text = "".join(block.text for block in response.content if block.type == "text")
    parsed = _extract_json(text)
    summary = _shape_summary(parsed) if parsed is not None else None
    if summary is None:
        return _unavailable(model, task, sample_filter, "model did not return valid JSON", len(digests))

    return {
        "available": True,
        "reason": None,
        "model": model,
        "task": task,
        "filter": sample_filter,
        "n_reviewed": len(digests),
        "summary": summary,
    }
