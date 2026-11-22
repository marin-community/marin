# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared comparison primitives for informal problems and Lean theorem statements.

This module normalizes prose and theorem types, creates stable row fingerprints and token n-grams,
and computes Jaccard similarity. ``numina_autoinformalization_manifest.py`` and ``eval_overlap.py`` share
these operations so their text and Lean comparisons follow the same rules. It does not decide which
rows to remove or how to split them.
"""

import hashlib
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from experiments.post_training.task_curation.datasets.lean.lean_grade import TARGET_NAME, without_comments

NGRAM_SIZE = 5
DECLARATION = re.compile(rf"^[ \t]*(?:theorem|lemma)\s+(?P<name>{TARGET_NAME.pattern})", re.MULTILINE)
PROOF_START = re.compile(r"(?s)^(?P<type>.*?):=")
BINDER_NAMES = re.compile(r"[({\[⦃]\s*(?P<names>[^:()\[\]{}⦃⦄]+?)\s*:")
IDENTIFIER = re.compile(r"[^\W\d]\w*'*")
TEXT_TOKEN = re.compile(r"\\[a-z]+|[^\W_]+|[=+*/<>^-]")
LEAN_TOKEN = re.compile(r"#\d+|[^\W\d]\w*'*|\d+(?:\.\d+)?|:=|↔|→|∀|∃|[=+*/<>^:-]")
LATEX_LAYOUT = re.compile(r"\\(?:left|right|displaystyle|quad|qquad|textbf|mathrm|textrm|text|operatorname)\b")


def row_key(row: Mapping[str, Any]) -> str | None:
    """Identify one formal-statement/problem pair while retaining its source UUID."""
    uuid, formal, problem = (row.get(name) for name in ("uuid", "formal_statement", "problem"))
    if not all(isinstance(value, str) and value.strip() for value in (uuid, formal, problem)):
        return None
    digest = hashlib.sha256(f"{formal}\0{problem}".encode()).hexdigest()
    index = row.get("_numina_index")
    location = f"{index}:" if isinstance(index, int) else ""
    return f"{location}{uuid}:{digest[:12]}"


def statement_key(declaration: str) -> str | None:
    """Normalize the last theorem's type, including consistent bound-variable renaming."""
    body = without_comments(declaration)
    found = list(DECLARATION.finditer(body))
    match = PROOF_START.match(body[found[-1].end() :]) if found else None
    if match is None:
        return None
    text = match["type"]
    names = [name for group in BINDER_NAMES.finditer(text) for name in group["names"].split()]
    for index, name in enumerate(dict.fromkeys(name for name in names if IDENTIFIER.fullmatch(name))):
        text = re.sub(rf"(?<![\w.']){re.escape(name)}(?![\w'])", f"#{index}", text)
    return "".join(text.split())


def normalized_text(text: str) -> str:
    """Normalize prose and LaTeX layout while retaining numbers and supported operators."""
    normalized = unicodedata.normalize("NFKC", text).casefold().replace(r"\dfrac", r"\frac")
    return " ".join(TEXT_TOKEN.findall(LATEX_LAYOUT.sub(" ", normalized)))


def ngrams(text: str, *, formal: bool = False) -> frozenset[tuple[str, ...]]:
    """Return five-token n-grams from normalized prose or a normalized Lean theorem type."""
    tokens = LEAN_TOKEN.findall(text) if formal else normalized_text(text).split()
    return frozenset(zip(*(tokens[index:] for index in range(NGRAM_SIZE)), strict=False))


def statement_ngrams(declaration: str) -> frozenset[tuple[str, ...]]:
    key = statement_key(declaration)
    return ngrams(key, formal=True) if key is not None else frozenset()


def jaccard(left: frozenset, right: frozenset) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
