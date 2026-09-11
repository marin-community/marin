# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

r"""Mode exact: the output compared to ``expected`` as strings, after normalization.

Two candidates are tried, and either one matching scores 1: the content of the last ``\boxed{}``
when the output has one, and the whole output.

A single expected entry must equal the candidate. Several expected entries make the candidate a
list: it is split on newlines and commas, empty items are dropped, and the items must match the
expected entries in order when ``ordered``, otherwise as a multiset.

``ignore_case`` casefolds both sides. ``ignore_whitespace`` collapses every run of whitespace to a
single space; the candidate's outer whitespace is stripped either way, so a trailing newline in the
answer file never decides the reward.
"""

import re
from collections import Counter
from pathlib import Path

from tasktrove_verify.modes.extract import collapse_whitespace, extract_boxed
from tasktrove_verify.grade import read_output
from tasktrove_verify.grade import InvalidTask, Reward, scored
from tasktrove_verify.spec import ExactSpec

ITEM_SEPARATOR = re.compile(r"[\n,]")
MAX_DETAIL_CHARS = 400


def _normalize(text: str, spec: ExactSpec) -> str:
    value = collapse_whitespace(text) if spec.ignore_whitespace else text.strip()
    return value.casefold() if spec.ignore_case else value


def _items(text: str, spec: ExactSpec) -> list[str]:
    items = (_normalize(part, spec) for part in ITEM_SEPARATOR.split(text))
    return [item for item in items if item]


def _matches(candidate: str, spec: ExactSpec) -> bool:
    if len(spec.expected) == 1:
        return _normalize(candidate, spec) == _normalize(spec.expected[0], spec)
    expected = [_normalize(entry, spec) for entry in spec.expected]
    items = _items(candidate, spec)
    if spec.ordered:
        return items == expected
    return Counter(items) == Counter(expected)


def grade(spec: ExactSpec, tests_dir: Path, workspace: Path) -> Reward:
    if not spec.expected:
        raise InvalidTask("exact expects at least one expected string")

    text = read_output(spec, workspace)
    if text is None:
        return scored(0.0, reason="no_output")
    boxed = extract_boxed(text)
    candidates = [boxed, text] if boxed is not None else [text]
    detail = {"extracted": candidates[0].strip()[:MAX_DETAIL_CHARS], "expected": list(spec.expected)}
    match = any(_matches(candidate, spec) for candidate in candidates)
    return scored(float(match), **detail)
