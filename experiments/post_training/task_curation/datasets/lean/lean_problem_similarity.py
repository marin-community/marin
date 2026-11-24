# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared comparison primitives for informal problems and Lean theorem statements.

This module normalizes prose and theorem types, creates token n-grams, and computes Jaccard
similarity. ``eval_overlap.py`` uses these operations to match NuminaMath-LEAN rows against Marin's
frozen Lean evaluation problems. It does not decide which rows to remove.
"""

import re
import unicodedata

from experiments.post_training.task_curation.datasets.lean.lean_grade import TARGET_NAME, without_comments

NGRAM_SIZE = 5
DECLARATION = re.compile(rf"^[ \t]*(?:theorem|lemma)\s+(?P<name>{TARGET_NAME.pattern})", re.MULTILINE)
PROOF_START = re.compile(r"(?s)^(?P<type>.*?):=")
BINDER_NAMES = re.compile(r"[({\[⦃]\s*(?P<names>[^:()\[\]{}⦃⦄]+?)\s*:")
IDENTIFIER = re.compile(r"[^\W\d]\w*'*")
TEXT_TOKEN = re.compile(r"\\[a-z]+|[^\W_]+|[=+*/<>^-]")
LEAN_TOKEN = re.compile(r"#\d+|[^\W\d]\w*'*|\d+(?:\.\d+)?|:=|↔|→|∀|∃|[=+*/<>^:-]")
LATEX_LAYOUT = re.compile(r"\\(?:left|right|displaystyle|quad|qquad|textbf|mathrm|textrm|text|operatorname)\b")


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
