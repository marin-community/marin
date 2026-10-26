# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copied from the WS1 clefclient package, phase11/ws1-clef-client/clefclient/budget.py in the WS1 scratch
# workspace (2026-10-15). That package has no repository; the source file's sha256 is
# 274289315151d472837155d274214f2069443e25ea863fabf0a0977f8587034f.
# Changed: imports name taskforge.llm.clef; only full-input accounting is retained.

"""Count the complete Clef request using its pinned tokenizer and wire rendering.

The server can silently truncate overlong state. Callers compare this count with the
supported context window before sending anything; no part of the state is shortened.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from taskforge.llm.clef.schema import (
    ChoiceQuestion,
    JsonValue,
    NoulQuestion,
    Question,
    SystemOneRequest,
)

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

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)


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


def input_tokens(tok: ClefTokenizer, request: SystemOneRequest) -> int:
    """Tokens in the complete request, before any server-side truncation."""
    return fixed_tokens(tok, request.questions) + tok.count(render(request.state))
