"""Shared helpers for Gemma log-prob consistency checks."""

from __future__ import annotations

import json
import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from fsspec.core import url_to_fs

DEFAULT_PROMPT = (
    "Marin is building shared infrastructure for open models.\n"
    "Consistent log probabilities across hardware backends are a basic sanity check."
)


def add_eos_if_missing(token_ids: Sequence[int], eos_id: int | None) -> list[int]:
    """Return a copy of token_ids that always ends with eos_id when available."""
    token_ids = list(token_ids)
    if eos_id is None:
        return token_ids
    if not token_ids or token_ids[-1] != eos_id:
        token_ids.append(eos_id)
    return token_ids


@dataclass(slots=True)
class LogProbResult:
    backend: str
    model_id: str
    revision: str | None
    prompt: str
    token_ids: list[int]
    predicted_token_ids: list[int]
    predicted_tokens: list[str]
    per_token_logprobs: list[float]

    @property
    def total_logprob(self) -> float:
        return float(sum(self.per_token_logprobs))

    def to_json_dict(self) -> dict:
        return {
            "backend": self.backend,
            "model_id": self.model_id,
            "revision": self.revision,
            "prompt": self.prompt,
            "token_ids": self.token_ids,
            "predicted_token_ids": self.predicted_token_ids,
            "predicted_tokens": self.predicted_tokens,
            "per_token_logprobs": self.per_token_logprobs,
            "total_logprob": self.total_logprob,
        }


def save_result(result: LogProbResult, path: str | Path) -> None:
    """Serialize the result to JSON."""
    _write_json(path, result.to_json_dict())


def load_result(path: str | Path) -> LogProbResult:
    """Parse a previously saved result."""
    raw = _read_json(path)
    return LogProbResult(
        backend=raw["backend"],
        model_id=raw["model_id"],
        revision=raw.get("revision"),
        prompt=raw["prompt"],
        token_ids=list(raw["token_ids"]),
        predicted_token_ids=list(raw["predicted_token_ids"]),
        predicted_tokens=list(raw["predicted_tokens"]),
        per_token_logprobs=[float(x) for x in raw["per_token_logprobs"]],
    )


def compare_results(
    candidate: LogProbResult,
    reference: LogProbResult,
    *,
    tolerance: float = 5e-5,
) -> dict[str, float]:
    """Compare two runs and raise ValueError when they diverge beyond tolerance."""
    if candidate.model_id != reference.model_id:
        raise ValueError(f"Model id mismatch: {candidate.model_id} vs {reference.model_id}")
    if candidate.revision != reference.revision:
        raise ValueError(f"Revision mismatch: {candidate.revision} vs {reference.revision}")
    if candidate.prompt != reference.prompt:
        raise ValueError("Prompts differ; ensure both runs evaluate the same text.")
    if candidate.token_ids != reference.token_ids:
        raise ValueError("Token sequences differ; tokenizer settings may not match.")

    per_token_diff = _max_abs_diff(candidate.per_token_logprobs, reference.per_token_logprobs)
    total_diff = abs(candidate.total_logprob - reference.total_logprob)

    if total_diff > tolerance:
        raise ValueError(f"Total log-prob diff {total_diff:.2e} exceeds tolerance {tolerance:.2e}")
    if per_token_diff > tolerance:
        raise ValueError(f"Per-token log-prob diff {per_token_diff:.2e} exceeds tolerance {tolerance:.2e}")

    return {"total_diff": total_diff, "per_token_diff": per_token_diff}


def _max_abs_diff(lhs: Iterable[float], rhs: Iterable[float]) -> float:
    return max((abs(float(a) - float(b)) for a, b in zip(lhs, rhs)), default=0.0)


def _write_json(path: str | Path, payload: dict) -> None:
    fs, fs_path = url_to_fs(str(path))
    dir_path = posixpath.dirname(fs_path)
    if dir_path:
        fs.makedirs(dir_path, exist_ok=True)
    with fs.open(fs_path, "w") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True))
        handle.write("\n")


def _read_json(path: str | Path) -> dict:
    fs, fs_path = url_to_fs(str(path))
    with fs.open(fs_path) as handle:
        return json.load(handle)
