"""Stored per-benchmark prompt lengths and the generation budget they imply.

The longest prompt in a benchmark is a property of its dataset and prompt
template, so it is measured once by ``scripts/benchmarks/compute_prompt_lengths.py``
and stored in ``prompt_lengths.json`` (see ``prompt_lengths.md`` for how to
refresh it). At run time a benchmark reserves that many tokens plus a margin for
its prompt and spends the rest of the context window on the response, instead of
taking a single global ``--max_tokens`` that leaves most of the window unused.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.limits import DEFAULT_CONTEXT_SAFETY_TOKENS

REFERENCE_TOKENIZER = "Qwen/Qwen3-8B"
"""Tokenizer and chat template every stored prompt length is measured against."""

PROMPT_LENGTHS_PATH = Path(__file__).parent / "prompt_lengths.json"
PROMPT_LENGTHS_SCHEMA_VERSION = 1

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_MARGIN_TOKENS = 256
"""Headroom over the measured longest prompt, for chat-template and tokenizer drift."""


class PromptLengthError(ValueError):
    """Raised when stored prompt lengths are missing, malformed, or unusable."""


@dataclass(frozen=True)
class BenchmarkPromptLength:
    """One benchmark's measured prompt corpus."""

    distinct_prompt_count: int
    longest_prompt_chars: int
    longest_prompt_tokens: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "distinct_prompt_count": self.distinct_prompt_count,
            "longest_prompt_chars": self.longest_prompt_chars,
            "longest_prompt_tokens": self.longest_prompt_tokens,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class PromptLengths:
    """Every benchmark's stored prompt length, measured against one tokenizer."""

    reference_tokenizer: str
    prompt_margin_tokens: int
    benchmarks: Mapping[str, BenchmarkPromptLength]
    unmeasured: Mapping[str, str]

    def max_prompt_tokens(self, task_name: str) -> int | None:
        """Return the prompt tokens to reserve for a benchmark, or ``None`` when unmeasured."""
        measured = self.benchmarks.get(task_name)
        if measured is None:
            return None
        return measured.longest_prompt_tokens + self.prompt_margin_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROMPT_LENGTHS_SCHEMA_VERSION,
            "reference_tokenizer": self.reference_tokenizer,
            "prompt_margin_tokens": self.prompt_margin_tokens,
            "benchmarks": {name: self.benchmarks[name].to_dict() for name in sorted(self.benchmarks)},
            "unmeasured": {name: self.unmeasured[name] for name in sorted(self.unmeasured)},
        }


def load_prompt_lengths(path: Path = PROMPT_LENGTHS_PATH) -> PromptLengths:
    """Read and validate the stored prompt-length metadata.

    An absent file means no benchmark has a stored length yet, so every task
    keeps its own output default. That is the state the measuring tool starts
    from; ``tests/contracts/test_prompt_length.py`` is what requires the
    committed file to cover every registered benchmark. A file that exists but
    does not satisfy the schema is an error rather than a fallback.
    """
    if not path.exists():
        return PromptLengths(REFERENCE_TOKENIZER, DEFAULT_PROMPT_MARGIN_TOKENS, {}, {})
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromptLengthError(f"prompt lengths are unreadable: {path}") from exc
    if not isinstance(payload, Mapping):
        raise PromptLengthError("prompt lengths must be a JSON object")
    if payload.get("schema_version") != PROMPT_LENGTHS_SCHEMA_VERSION:
        raise PromptLengthError(f"prompt lengths schema_version must be {PROMPT_LENGTHS_SCHEMA_VERSION}")

    reference_tokenizer = payload.get("reference_tokenizer")
    if not isinstance(reference_tokenizer, str) or not reference_tokenizer:
        raise PromptLengthError("prompt lengths must declare a reference_tokenizer")
    margin = payload.get("prompt_margin_tokens")
    if not isinstance(margin, int) or isinstance(margin, bool) or margin < 0:
        raise PromptLengthError("prompt_margin_tokens must be a non-negative integer")

    entries = payload.get("benchmarks")
    if not isinstance(entries, Mapping):
        raise PromptLengthError("prompt lengths must contain a benchmarks object")
    benchmarks = {name: _entry(name, value) for name, value in entries.items()}

    unmeasured = payload.get("unmeasured", {})
    if not isinstance(unmeasured, Mapping) or not all(
        isinstance(name, str) and isinstance(reason, str) and name and reason for name, reason in unmeasured.items()
    ):
        raise PromptLengthError("unmeasured must map a benchmark name to a non-empty reason")
    overlap = sorted(set(benchmarks) & set(unmeasured))
    if overlap:
        raise PromptLengthError(f"benchmarks cannot be both measured and unmeasured: {overlap}")

    return PromptLengths(reference_tokenizer, margin, benchmarks, dict(unmeasured))


def resolve_task_max_tokens(
    task_name: str,
    *,
    context_length: int | None,
    requested_max_tokens: int | None,
    prompt_lengths: PromptLengths | None = None,
    safety_tokens: int = DEFAULT_CONTEXT_SAFETY_TOKENS,
) -> int | None:
    """Return one benchmark's generation cap for this run.

    An explicit ``--max_tokens`` wins, because a caller asking for a specific
    output budget means it. Otherwise the cap is what the context window has
    left once the benchmark's stored prompt length and the safety margin are
    reserved, which is how a benchmark with short prompts gets a long response
    budget. Returns ``None`` when neither input is available, leaving the
    benchmark's own default in place.
    """
    if requested_max_tokens is not None:
        return requested_max_tokens
    if context_length is None:
        return None
    lengths = prompt_lengths if prompt_lengths is not None else load_prompt_lengths()
    prompt_tokens = lengths.max_prompt_tokens(task_name)
    if prompt_tokens is None:
        return None
    budget = context_length - prompt_tokens - safety_tokens
    if budget <= 0:
        # Some prompts in this benchmark cannot fit the window at all. Deriving a
        # budget from the longest one would cap every request, including the ones
        # that do fit, so leave the benchmark's default and let the per-request
        # preflight reject the oversized prompts.
        logger.warning(
            "%s: longest prompt reserves %s of %s context tokens; keeping the benchmark's output default",
            task_name,
            prompt_tokens,
            context_length,
        )
        return None
    return budget


def _entry(task_name: Any, value: Any) -> BenchmarkPromptLength:
    if not isinstance(task_name, str) or not task_name:
        raise PromptLengthError("benchmark names must be non-empty strings")
    if not isinstance(value, Mapping):
        raise PromptLengthError(f"{task_name}: prompt length entry must be an object")
    fields = {}
    for field in ("distinct_prompt_count", "longest_prompt_chars", "longest_prompt_tokens"):
        count = value.get(field)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise PromptLengthError(f"{task_name}: {field} must be a positive integer")
        fields[field] = count
    digest = value.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise PromptLengthError(f"{task_name}: sha256 must be a 64-character digest")
    return BenchmarkPromptLength(sha256=digest, **fields)
