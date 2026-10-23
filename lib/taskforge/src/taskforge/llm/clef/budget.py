# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copied from the WS1 clefclient package, phase11/ws1-clef-client/clefclient/budget.py in the WS1 scratch
# workspace (2026-10-15). That package has no repository; the source file's sha256 is
# 274289315151d472837155d274214f2069443e25ea863fabf0a0977f8587034f.
# Changed: its clefclient imports now name taskforge.llm.clef, and fit_request narrows one type.

"""Exact token accounting for Clef reads, and a state budget that truncates loudly.

``input_tokens`` mirrors Cloudflare's reference ``encode_record`` (text-only records): the
system prefix, the rendered state, one schema block per question, and the assistant suffix,
each piece tokenized separately with ``add_special_tokens=False``. Clef's server cuts an
overlong state from the TAIL without an error; ``fit_request`` instead cuts a caller-chosen
elastic field from the middle or the head, marks the cut in the text, and reports it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from taskforge.llm.clef.schema import (
    ChoiceQuestion,
    ClefRequestError,
    JsonValue,
    NoulQuestion,
    Question,
    SystemOneRequest,
)

logger = logging.getLogger(__name__)

CLEF_REPO = "Cloudflare/clef"
CLEF_REVISION = "0b331204bb13fbd2ca93a64df1956f5b55478ce5"
MAX_LENGTH = 131072  # the Clef service maximum (Mark, 2026-10-15); pass 16384 to target a 16k engine

# Verbatim from joint_schema_model.py at CLEF_REVISION.
SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
PREFIX = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
SUFFIX = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
SCHEMA_HEADER = "\n\nSCHEMA FIELDS:\n"
NOUL_DEFAULTS = {
    "true": "The proposition is true or the answer is yes.",
    "false": "The proposition is false or the answer is no.",
}


@dataclass(frozen=True)
class ClefTokenizer:
    """Clef's ``tokenizer.json`` through ``tokenizers``: what the server and the reference count with."""

    tokenizer: Tokenizer
    source: str  # repo@revision actually loaded

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)


def load_tokenizer() -> ClefTokenizer:
    """Load Clef's pinned tokenizer; propagate download failures so budgets use the server's vocabulary."""
    path = hf_hub_download(CLEF_REPO, "tokenizer.json", revision=CLEF_REVISION)
    return ClefTokenizer(Tokenizer.from_file(path), f"{CLEF_REPO}@{CLEF_REVISION[:12]}")


def render(value: JsonValue) -> str:
    """The state as Clef reads it: strings verbatim; JSON compact, sorted keys, non-ASCII kept."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _options(question: Question) -> list[tuple[str, str | None]]:
    if isinstance(question, NoulQuestion):
        criteria = dict(NOUL_DEFAULTS)
        criteria.update({k: v for k, v in (("true", question.true), ("false", question.false)) if v is not None})
        return [(k, criteria[k]) for k in ("true", "false")]
    if isinstance(question, ChoiceQuestion):
        return sorted(question.options)
    return [(str(i), level) for i, level in enumerate(question.levels)]


def fixed_tokens(tok: ClefTokenizer, questions: Mapping[str, Question]) -> int:
    """Tokens a read costs before any state: prefix, every question's schema block, suffix."""
    n = tok.count(PREFIX) + tok.count(SUFFIX) + tok.count(SCHEMA_HEADER)
    for index, (qid, question) in enumerate(questions.items()):
        kind = question.to_json()["type"]
        n += tok.count(f"\nFIELD {index + 1}\nID: {qid}\nTYPE: {kind}\nINSTRUCTION: ")
        n += tok.count(render(question.instructions or qid))
        n += tok.count("\nALLOWED OPTIONS:\n")
        for o, (option_id, description) in enumerate(_options(question)):
            semantics: dict[str, Any] = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            n += tok.count(f"OPTION {o + 1}: ") + tok.count(render(semantics)) + tok.count("\n")
        n += tok.count("END FIELD\n")
    return n


def input_tokens(tok: ClefTokenizer, request: SystemOneRequest, max_length: int = MAX_LENGTH) -> int:
    """``usage.input_tokens`` Clef will report: fixed + state, the state cut at max_length."""
    return min(max_length, fixed_tokens(tok, request.questions) + tok.count(render(request.state)))


class TruncateFrom(StrEnum):
    MIDDLE = "middle"  # keep the head and the tail (a transcript's setup and its ending)
    HEAD = "head"  # drop the oldest text, keep the end


@dataclass(frozen=True)
class StateBudget:
    """What fitting a request cost. ``tokens_cut`` > 0 means the elastic field was truncated."""

    max_length: int
    fixed_tokens: int
    state_tokens: int  # after fitting
    elastic_tokens_before: int
    elastic_tokens_after: int  # including the cut marker
    tokens_cut: int
    truncate_from: TruncateFrom

    @property
    def state_limit(self) -> int:
        """The most state tokens this read can carry: max_length - fixed (scaffolding + schema)."""
        return self.max_length - self.fixed_tokens

    @property
    def input_tokens(self) -> int:
        return self.fixed_tokens + self.state_tokens

    @property
    def truncated(self) -> bool:
        return self.tokens_cut > 0


def _get(state: JsonValue, path: Sequence[str]) -> JsonValue:
    for key in path:
        if not isinstance(state, dict):
            raise ClefRequestError(f"elastic path {list(path)} does not resolve in the state")
        state = state[key]
    return state


def _set(state: JsonValue, path: Sequence[str], value: str) -> JsonValue:
    if not path:
        return value
    if not isinstance(state, dict):
        raise ClefRequestError(f"elastic path {list(path)} does not resolve in the state")
    return {**state, path[0]: _set(state[path[0]], path[1:], value)}


def _cut(tok: ClefTokenizer, ids: list[int], keep: int, how: TruncateFrom) -> str:
    marker = f"[... {len(ids) - keep} tokens truncated by the Clef state budget ...]"
    if how is TruncateFrom.HEAD:
        return f"{marker}\n{tok.decode(ids[len(ids) - keep :])}"
    head = keep - keep // 2
    return f"{tok.decode(ids[:head])}\n{marker}\n{tok.decode(ids[len(ids) - (keep - head) :])}"


def fit_request(
    tok: ClefTokenizer,
    request: SystemOneRequest,
    elastic: Sequence[str],
    truncate_from: TruncateFrom,
    max_length: int = MAX_LENGTH,
) -> tuple[SystemOneRequest, StateBudget]:
    """Shrink the elastic string field at ``elastic`` (a key path; ``()`` = the state itself) until
    the whole read fits ``max_length``. Raises if the questions, or the non-elastic state, cannot fit."""
    fixed = fixed_tokens(tok, request.questions)
    available = max_length - fixed
    if available <= 0:
        raise ClefRequestError(f"questions alone need {fixed} tokens; max_length is {max_length}")
    text = _get(request.state, elastic)
    if not isinstance(text, str):
        raise ClefRequestError(f"elastic field {list(elastic)} must be a string, got {type(text).__name__}")
    ids = tok.encode(text)
    state_tokens = tok.count(render(request.state))
    keep, state = len(ids), request.state
    if state_tokens > available:
        # Every count is taken on the RENDERED state: JSON escaping (quotes, newlines) can make a raw
        # elastic token cost several rendered tokens, so the raw overflow is not the amount to cut.
        # Model rendered = floor + rate * keep; floor is the state with only the cut marker left.
        floor = tok.count(render(_set(request.state, elastic, _cut(tok, ids, 0, truncate_from))))
        if floor > available:
            raise ClefRequestError(
                f"state does not fit: {fixed} question tokens + non-elastic state ({floor} rendered tokens) "
                f"exceed {max_length} even with {list(elastic)} emptied"
            )
        while state_tokens > available:
            rate = max(state_tokens - floor, 1) / keep
            keep = min(keep - 1, int((available - floor) / rate))
            state = _set(request.state, elastic, _cut(tok, ids, keep, truncate_from))
            state_tokens = tok.count(render(state))
    kept = _get(state, elastic)
    assert isinstance(kept, str)  # _set wrote a string there
    after = len(ids) if keep == len(ids) else tok.count(kept)
    budget = StateBudget(max_length, fixed, state_tokens, len(ids), after, len(ids) - keep, truncate_from)
    if budget.truncated:
        logger.info("clef state budget: cut %d of %d elastic tokens (%s)", budget.tokens_cut, len(ids), truncate_from)
    return replace(request, state=state), budget
