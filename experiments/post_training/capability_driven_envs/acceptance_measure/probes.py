# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The deterministic probe battery (lever L2): candidates generated from a reference answer and a stated contract.

``battery(reference_reply, reference_files, contract, witnesses)`` returns ``Probe`` records, each with a class, a
candidate (reply and files) and an expectation derived only from the ``Contract`` fields, never from the author.
The classes follow d3-v2's ``grader-probes.md`` table (kit/PROVENANCE.json). ``met(probe, status, reward,
ceiling)`` decides a graded probe. Nothing here grades: the session tool (``author.py``) grades each probe on the
real grader machine with the draft's grader.

A JSON reference (framing ``exact``/``fenced``/``last``) gets structural mutations; a ``text`` reference gets the
text-level ones. For the ``file`` channel the mutations apply to the answer file at ``answer_path`` and the reply
stays empty.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from taskcompendium.grading_result import Outcome

FULL_CREDIT_EPSILON = 1e-9
DEEP_NESTING = 100_000
"""Brackets in the adversarial nesting probe: a 200 KB reply (a 164 KB one crashed a campaign grader)."""


class Channel(StrEnum):
    REPLY = "reply"
    FILE = "file"


class Framing(StrEnum):
    EXACT = "exact"
    FENCED = "fenced"
    LAST = "last"
    TEXT = "text"


class Noise(StrEnum):
    STRIP = "strip"
    REJECT = "reject"


class Contract(BaseModel):
    """The answer contract the author states, as the battery reads it (``text/probe-battery.md``)."""

    model_config = ConfigDict(extra="forbid")

    channel: Channel = Channel.REPLY
    answer_path: str | None = None
    framing: Framing = Framing.EXACT
    required_keys: list[str] = Field(default_factory=list)
    prose_allowed: bool = False
    transport_noise: Noise = Noise.STRIP
    closed_schema: bool = True
    labels: str = Field(default="exact", pattern="^(exact|aliases)$")
    integers: str = Field(default="strict", pattern="^(strict|integral_float_ok)$")
    extra_digits_allowed: bool = True

    def model_post_init(self, context: Any) -> None:
        if (self.channel is Channel.FILE) != (self.answer_path is not None):
            raise ValueError("answer_path is given exactly when the channel is file")
        if self.answer_path is not None and not self.answer_path.startswith("/"):
            raise ValueError("answer_path is absolute, as the grader reads it")
        if self.framing is Framing.LAST and not self.required_keys:
            raise ValueError("framing last needs required_keys")


class Expect(StrEnum):
    FULL = "full"
    AT_MOST_CEILING = "<=ceiling"
    BELOW_FULL = "<full"


@dataclass(frozen=True)
class Probe:
    name: str
    probe_class: str
    expect: Expect
    reply: str
    files: dict[str, str] = field(default_factory=dict)
    """Answer files by absolute machine path."""


@dataclass(frozen=True)
class Witness:
    """A defect submission from an earlier round, replayed as a regression control."""

    id: str
    role: str
    """``false_accept`` or ``false_reject``."""
    reply: str
    files: dict[str, str]


def met(expect: Expect, status: Outcome, reward: float | None, ceiling: float) -> bool:
    """Whether a graded probe met its expectation. A submission failure counts as reward 0."""
    value = 0.0 if status is Outcome.SUBMISSION_FAILURE else reward
    if value is None:
        return False
    match expect:
        case Expect.FULL:
            return value >= 1.0 - FULL_CREDIT_EPSILON
        case Expect.AT_MOST_CEILING:
            return value <= ceiling + FULL_CREDIT_EPSILON
        case Expect.BELOW_FULL:
            return value < 1.0 - FULL_CREDIT_EPSILON


def _json_reference(text: str) -> Any:
    """The reference's JSON value: the whole text, the fenced block, or (LAST) the final object."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.rfind("{")
        while start >= 0:
            try:
                return json.loads(stripped[start:])
            except json.JSONDecodeError:
                start = stripped.rfind("{", 0, start)
        raise ValueError("the reference is not JSON under the stated framing") from None


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _first_key(value: Any, predicate) -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if predicate(item):
                return key
    return None


def _wrong(value: Any) -> Any:
    """A value of the same type that differs: a substantively wrong field."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value * 1.5 + 1.0
    if isinstance(value, str):
        return value + "_x" if value else "x"
    if isinstance(value, list):
        return value[:-1] if value else ["x"]
    if isinstance(value, dict):
        return {**value, "__wrong__": 1}
    return "x"


def _structural(value: Any, contract: Contract) -> list[tuple[str, str, Expect, str]]:
    """(name, class, expectation, text) mutations of a JSON reference value."""
    ceiling = Expect.AT_MOST_CEILING
    out: list[tuple[str, str, Expect, str]] = []
    if isinstance(value, dict) and len(value) > 1:
        out.append(("keys_reordered", "equivalent_form", Expect.FULL, _dump(dict(reversed(list(value.items()))))))
    if isinstance(value, dict) and value:
        key = next(iter(value))
        wrong, right = _dump({key: _wrong(value[key])})[1:-1], _dump(value)[1:]
        out.append(("duplicate_key_wrong_first", "duplicate_json_keys", ceiling, "{" + wrong + ", " + right))
        out.append(("one_field_wrong", "substance", ceiling, _dump({**value, key: _wrong(value[key])})))
        out.append(("bare_values", "keyword_soup", ceiling, " ".join(_dump(v) for v in value.values())))
        if contract.closed_schema:
            out.append(("extra_key", "open_schema", ceiling, _dump({**value, "notes": "see above"})))
        number = _first_key(value, lambda v: isinstance(v, (int, float)) and not isinstance(v, bool))
        if number is not None:
            out.append(("number_as_string", "type_coercion", ceiling, _dump({**value, number: str(value[number])})))
            out.append(("nan", "type_coercion", ceiling, _dump(value).replace(_dump(value[number]), "NaN", 1)))
            if isinstance(value[number], float) and contract.extra_digits_allowed:
                padded = _dump(value).replace(_dump(value[number]), repr(value[number]) + "000", 1)
                out.append(("trailing_zeros", "equivalent_form", Expect.FULL, padded))
        integer = _first_key(value, lambda v: isinstance(v, int) and not isinstance(v, bool))
        if integer is not None and contract.integers == "strict":
            text = _dump(value).replace(f'"{integer}": {value[integer]}', f'"{integer}": {value[integer]}.0', 1)
            out.append(("integral_float", "type_coercion", ceiling, text))
        label = _first_key(value, lambda v: isinstance(v, str) and 0 < len(v) <= 40 and v != v.upper())
        if label is not None and contract.labels == "exact":
            out.append(("label_case", "label_normalisation", ceiling, _dump({**value, label: value[label].upper()})))
    if isinstance(value, list) and value:
        out.append(("list_duplicate_entry", "list_handling", ceiling, _dump([*value, value[0]])))
        out.append(("list_missing_entry", "list_handling", ceiling, _dump(value[:-1])))
    return out


def _framing_probes(text: str, contract: Contract) -> list[tuple[str, str, Expect, str]]:
    """Probes on how the answer is framed: fences, prose and several answers."""
    ceiling = Expect.AT_MOST_CEILING
    fenced = f"```json\n{text}\n```"
    prose = f"Here is my answer.\n\n{text}\n\nI checked every value."
    out: list[tuple[str, str, Expect, str]] = []
    if contract.framing is Framing.FENCED:
        out.append(("fenced", "equivalent_form", Expect.FULL, fenced))
    elif contract.framing is Framing.EXACT:
        out.append(("fenced", "text_around_answer", ceiling, fenced))
    if contract.framing is Framing.LAST or contract.prose_allowed:
        out.append(("prose_before", "equivalent_form", Expect.FULL, f"I computed the values below.\n\n{text}"))
    else:
        out.append(("prose_around", "text_around_answer", ceiling, prose))
    return out


def _text_probes(text: str) -> list[tuple[str, str, Expect, str]]:
    ceiling = Expect.AT_MOST_CEILING
    words = sorted({w for w in text.replace("\n", " ").split(" ") if w})
    return [
        ("prose_around", "text_around_answer", ceiling, f"My answer follows.\n{text}\nThat is all."),
        ("hedged", "text_around_answer", ceiling, f"{text}\n\nAlternatively:\n{text[::-1]}"),
        ("keyword_soup", "keyword_soup", ceiling, " ".join(reversed(words))),
    ]


def _noise(text: str, contract: Contract) -> list[tuple[str, str, Expect, str]]:
    expect = Expect.FULL if contract.transport_noise is Noise.STRIP else Expect.AT_MOST_CEILING
    kind = "equivalent_form" if contract.transport_noise is Noise.STRIP else "transport_noise_policy"
    out = [("bom", kind, expect, "﻿" + text), ("trailing_whitespace", kind, expect, text + "  \n\n")]
    if "\n" in text:
        out.append(("crlf", kind, expect, text.replace("\n", "\r\n")))
    return out


def battery(
    reference_reply: str,
    reference_files: dict[str, str],
    contract: Contract,
    witnesses: Sequence[Witness] = (),
) -> list[Probe]:
    """The battery for a draft whose reference answer is ``reference_reply`` and ``reference_files``.

    ``reference_files`` maps absolute machine paths to text. For the ``file`` channel the reference text is the
    file at ``contract.answer_path``, which must be among them.

    Raises:
        ValueError: the reference does not parse under the stated framing, or the answer file is missing.
    """
    if contract.channel is Channel.FILE:
        assert contract.answer_path is not None
        if contract.answer_path not in reference_files:
            raise ValueError(f"the reference has no answer file at {contract.answer_path}")
        reference = reference_files[contract.answer_path]
    else:
        reference = reference_reply
    mutations: list[tuple[str, str, Expect, str]] = [("reference", "reference", Expect.FULL, reference)]
    if contract.framing is Framing.TEXT:
        mutations += _text_probes(reference)
    else:
        value = _json_reference(reference)
        canonical = reference.strip()
        mutations += _structural(value, contract)
        mutations += _framing_probes(_dump(value), contract)
        wrong_first = _dump(_wrong(value)) + "\n" + canonical
        mutations.append(("wrong_copy_first", "text_around_answer", Expect.AT_MOST_CEILING, wrong_first))
    mutations += _noise(reference, contract)
    mutations += [
        ("blank", "empty", Expect.AT_MOST_CEILING, ""),
        ("whitespace", "empty", Expect.AT_MOST_CEILING, " \n\t "),
        ("deep_nesting", "adversarial_input", Expect.AT_MOST_CEILING, "[" * DEEP_NESTING),
    ]
    # A mutation that reproduces the reference (a one-word answer's keyword soup) cannot expect less than full credit.
    mutations = [m for m in mutations if m[0] == "reference" or m[2] is Expect.FULL or m[3] != reference]
    probes = [
        _candidate(name, cls, expect, text, reference_reply, reference_files, contract)
        for name, cls, expect, text in mutations
    ]
    other_files = {p: t for p, t in reference_files.items() if p != contract.answer_path}
    if contract.channel is Channel.FILE:
        probes.append(Probe("reply_channel_only", "channel_fallback", Expect.AT_MOST_CEILING, reference, other_files))
    elif reference_files:
        probes.append(
            Probe("files_without_reply", "channel_fallback", Expect.AT_MOST_CEILING, "", dict(reference_files))
        )
    for witness in witnesses:
        expect = Expect.BELOW_FULL if witness.role == "false_accept" else Expect.FULL
        probes.append(
            Probe(f"witness_{witness.id}", f"witness_{witness.role}", expect, witness.reply, dict(witness.files))
        )
    return probes


def _candidate(
    name: str,
    probe_class: str,
    expect: Expect,
    text: str,
    reference_reply: str,
    reference_files: dict[str, str],
    contract: Contract,
) -> Probe:
    if contract.channel is Channel.FILE:
        assert contract.answer_path is not None
        return Probe(name, probe_class, expect, "", {**reference_files, contract.answer_path: text})
    return Probe(name, probe_class, expect, text, dict(reference_files))
