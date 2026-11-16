"""Opt-in clients for existing verifyit graders; source extraction stays in callers."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def _grade_reasoning_entry(task: str, entry: dict[str, Any], answer: str) -> float:
    from verifyit.grade import Status, run
    from verifyit.spec import ReasoningGymSpec, render_spec

    with TemporaryDirectory(prefix="skyrl-verifyit-") as directory:
        root = Path(directory)
        (root / "entry.json").write_text(json.dumps(entry, allow_nan=False))
        (root / "answer.txt").write_text(answer)
        spec = root / "verifier.toml"
        spec.write_text(render_spec(ReasoningGymSpec(dataset=task)))
        verdict = run(spec, root)

    if verdict.status is not Status.SCORED or verdict.detail.get("reason") == "scorer_error":
        raise RuntimeError("Reasoning Gym verification failed")
    return verdict.reward


def _grade_mcqa_option(gold: str, prediction: str | None, allowed: set[str]) -> float:
    from verifyit.modes.grade_mcq import grade_mcq_candidate
    from verifyit.spec import McqSpec

    # Map a possibly noncontiguous source option set independently to a contiguous core set.
    ordered = sorted(allowed)
    if gold not in ordered or not ordered or len(ordered) > 26:
        raise ValueError("MCQA reference is outside declared options")
    expected = chr(ord("A") + ordered.index(gold))
    candidate = chr(ord("A") + ordered.index(prediction)) if prediction in ordered else ""
    return grade_mcq_candidate(McqSpec(expected=expected, options=len(ordered)), candidate).reward


def grade_reasoning_entry(task: str, entry: dict[str, Any], answer: str) -> float:
    try:
        return _grade_reasoning_entry(task, entry, answer)
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError("Reasoning Gym verification failed") from error


def grade_mcqa_option(gold: str, prediction: str | None, allowed: set[str]) -> float:
    try:
        return _grade_mcqa_option(gold, prediction, allowed)
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError("MCQA verification failed") from error
