# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generated PPL probes and plain-text training data for low-level circuit skills."""

from __future__ import annotations

import csv
import io
import json
import os
import posixpath
import random
import re
import textwrap
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import fsspec
from fray.types import ResourceConfig
from levanter.utils import fsspec_utils
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, supervised_text_dataset
from marin.execution.executor import ExecutorStep
from marin.execution.types import this_output_path

EPIC_5005 = 5005
PPL_CIRCUIT_COVERAGE_V2_ISSUE = 6070
PPL_CIRCUIT_COVERAGE_V2_SOURCE = "generated_ppl_circuit_coverage_v2"
PPL_CIRCUIT_COVERAGE_V2_SEED = 6203
EXAMPLES_PER_CONFIG = 1000
LOCAL_SAMPLE_EXAMPLES_PER_CONFIG = 4
RAW_SHARD_SUFFIX = ".jsonl.gz"
PLAIN_TEXT_PRETRAINING_TARGET_TOKENS = 1_000_000_000
PLAIN_TEXT_PRETRAINING_SHARDS = 256
PLAIN_TEXT_PRETRAINING_SOURCE = "generated_ppl_circuit_coverage_v2_plain_text"
CHARS_PER_TOKEN_ESTIMATE = 4


class PplCircuitCoverageV2Family(StrEnum):
    STRING_BYTE_TRANSFORMS = "string_byte_transforms"
    INDEXING_POSITION_TRACKING = "indexing_position_tracking"
    ARITHMETIC = "arithmetic"
    STRUCTURED_SERIALIZATION = "structured_serialization"
    STATE_MACHINES = "state_machines"
    FORMAT_STYLE_INSTRUCTION = "format_style_instruction"


@dataclass(frozen=True)
class PplCircuitCoverageV2Slice:
    family: PplCircuitCoverageV2Family
    task_name: str

    @property
    def registry_key(self) -> str:
        return posixpath.join("ppl_circuit_coverage_v2", self.family.value, self.task_name)

    @property
    def output_name(self) -> str:
        return f"{self.family.value}_{self.task_name}"

    @property
    def tags(self) -> tuple[str, ...]:
        return (
            "ppl_circuit_coverage_v2",
            f"epic:{EPIC_5005}",
            f"issue:{PPL_CIRCUIT_COVERAGE_V2_ISSUE}",
            f"family:{self.family.value}",
            f"task:{self.task_name}",
            f"seed:{PPL_CIRCUIT_COVERAGE_V2_SEED}",
            f"examples:{EXAMPLES_PER_CONFIG}",
            f"source:{PPL_CIRCUIT_COVERAGE_V2_SOURCE}",
            "loss:target_only",
            "eval_only",
        )

    def to_raw_text_dataset(self, *, raw_step: ExecutorStep) -> RawTextEvaluationDataset:
        return supervised_text_dataset(
            raw_step.cd(f"{self.output_name}{RAW_SHARD_SUFFIX}"),
            input_key="input",
            target_key="target",
            tags=self.tags,
        )


def _slice(
    *,
    family: PplCircuitCoverageV2Family,
    task_name: str,
) -> PplCircuitCoverageV2Slice:
    return PplCircuitCoverageV2Slice(family=family, task_name=task_name)


PPL_CIRCUIT_COVERAGE_V2_SLICES: tuple[PplCircuitCoverageV2Slice, ...] = (
    _slice(family=PplCircuitCoverageV2Family.STRING_BYTE_TRANSFORMS, task_name="string_slicing"),
    _slice(family=PplCircuitCoverageV2Family.STRING_BYTE_TRANSFORMS, task_name="string_reversal"),
    _slice(family=PplCircuitCoverageV2Family.STRING_BYTE_TRANSFORMS, task_name="string_rotation"),
    _slice(family=PplCircuitCoverageV2Family.STRING_BYTE_TRANSFORMS, task_name="escape_unescape"),
    _slice(family=PplCircuitCoverageV2Family.STRING_BYTE_TRANSFORMS, task_name="chars_vs_bytes"),
    _slice(family=PplCircuitCoverageV2Family.STRING_BYTE_TRANSFORMS, task_name="unicode_normalization"),
    _slice(family=PplCircuitCoverageV2Family.STRING_BYTE_TRANSFORMS, task_name="unicode_casefolding"),
    _slice(family=PplCircuitCoverageV2Family.INDEXING_POSITION_TRACKING, task_name="character_at_index"),
    _slice(family=PplCircuitCoverageV2Family.INDEXING_POSITION_TRACKING, task_name="all_indices"),
    _slice(family=PplCircuitCoverageV2Family.INDEXING_POSITION_TRACKING, task_name="line_column_offsets"),
    _slice(family=PplCircuitCoverageV2Family.INDEXING_POSITION_TRACKING, task_name="bracket_matching"),
    _slice(family=PplCircuitCoverageV2Family.ARITHMETIC, task_name="carry_addition"),
    _slice(family=PplCircuitCoverageV2Family.ARITHMETIC, task_name="borrow_subtraction"),
    _slice(family=PplCircuitCoverageV2Family.ARITHMETIC, task_name="modular_arithmetic"),
    _slice(family=PplCircuitCoverageV2Family.ARITHMETIC, task_name="base_conversion"),
    _slice(family=PplCircuitCoverageV2Family.ARITHMETIC, task_name="digit_checksum"),
    _slice(family=PplCircuitCoverageV2Family.STRUCTURED_SERIALIZATION, task_name="python_repr_containers"),
    _slice(family=PplCircuitCoverageV2Family.STRUCTURED_SERIALIZATION, task_name="json_field_reemit"),
    _slice(family=PplCircuitCoverageV2Family.STRUCTURED_SERIALIZATION, task_name="csv_tsv_transforms"),
    _slice(family=PplCircuitCoverageV2Family.STATE_MACHINES, task_name="stack_push_pop"),
    _slice(family=PplCircuitCoverageV2Family.STATE_MACHINES, task_name="finite_automata"),
    _slice(family=PplCircuitCoverageV2Family.STATE_MACHINES, task_name="turtle_commands"),
    _slice(family=PplCircuitCoverageV2Family.STATE_MACHINES, task_name="regex_lite"),
    _slice(family=PplCircuitCoverageV2Family.STATE_MACHINES, task_name="brainfuck_lite"),
    _slice(family=PplCircuitCoverageV2Family.FORMAT_STYLE_INSTRUCTION, task_name="markdown_table_padding"),
    _slice(family=PplCircuitCoverageV2Family.FORMAT_STYLE_INSTRUCTION, task_name="line_wrapping"),
    _slice(family=PplCircuitCoverageV2Family.FORMAT_STYLE_INSTRUCTION, task_name="outline_indentation"),
)


def ppl_circuit_coverage_v2_raw_validation_sets(
    *, raw_step: ExecutorStep | None = None
) -> dict[str, RawTextEvaluationDataset]:
    resolved_step = ppl_circuit_coverage_v2_raw_executor if raw_step is None else raw_step
    return {
        slice_.registry_key: slice_.to_raw_text_dataset(raw_step=resolved_step)
        for slice_ in PPL_CIRCUIT_COVERAGE_V2_SLICES
    }


def _record(
    *,
    slice_: PplCircuitCoverageV2Slice,
    row_index: int,
    seed: int,
    input_text: str,
    target: str,
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "id": f"{slice_.output_name}_{row_index:05d}",
        "input": input_text,
        "target": target if target.endswith("\n") else f"{target}\n",
        "subset": slice_.output_name,
        "task": slice_.task_name,
        "seed": seed,
        "tags": list(slice_.tags),
        "metadata": {
            "family": slice_.family.value,
            "row_index": row_index,
            "generator": PPL_CIRCUIT_COVERAGE_V2_SOURCE,
            "eval_only": True,
            **metadata,
        },
    }


def _json_string(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _few_shot_block(examples: tuple[tuple[str, str], ...]) -> str:
    lines = ["worked examples:"]
    for query, answer in examples:
        lines.append(query)
        lines.append(f"answer: {answer}")
    lines.append("held-out query:")
    return "\n".join(lines) + "\n"


def _ascii_piece(rng: random.Random, *, min_length: int = 3, max_length: int = 8) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(min_length, max_length)))


def _string_text(rng: random.Random) -> str:
    unicode_pieces = (
        "\u00e9",
        "\u03bb",
        "\u03c0",
        "\u4e2d",
        "\u65e5",
        "\u00df",
        "\U0001f642",
        "\u0301",
        "\u212b",
        "\ufb01",
    )
    parts = [_ascii_piece(rng), rng.choice(("_", "-", "::", "/")), _ascii_piece(rng, min_length=2)]
    if rng.random() < 0.7:
        parts.insert(rng.randrange(len(parts) + 1), rng.choice(unicode_pieces))
    if rng.random() < 0.4:
        parts.append(str(rng.randint(10, 99)))
    return "".join(parts)


def _generate_string_slicing(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    text = _string_text(rng)
    start = rng.randint(0, max(0, len(text) - 2))
    stop = rng.randint(start + 1, len(text))
    step = rng.choice((1, 2))
    result = text[start:stop:step]
    input_text = (
        _few_shot_block(
            (
                ('text = "abcdef"; text[1:5:2] ->', "'bd'"),
                ('text = "marin"; text[0:4:1] ->', "'mari'"),
            )
        )
        + f"text = {_json_string(text)}; text[{start}:{stop}:{step}] ->\n"
    )
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=repr(result),
        metadata={"operation": "slice", "text": text, "start": start, "stop": stop, "step": step},
    )


def _generate_string_reversal(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    text = _string_text(rng)
    input_text = (
        _few_shot_block(
            (
                ('reverse("abc-12") ->', "'21-cba'"),
                ('reverse("lam") ->', "'mal'"),
            )
        )
        + f"reverse({_json_string(text)}) ->\n"
    )
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=repr(text[::-1]),
        metadata={"operation": "reverse", "text": text},
    )


def _generate_string_rotation(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    text = _string_text(rng)
    amount = rng.randint(1, max(1, len(text) - 1))
    direction = rng.choice(("left", "right"))
    if direction == "left":
        result = text[amount:] + text[:amount]
    else:
        result = text[-amount:] + text[:-amount]
    input_text = (
        _few_shot_block(
            (
                ('rotate_left("abcdef", 2) ->', "'cdefab'"),
                ('rotate_right("abcdef", 2) ->', "'efabcd'"),
            )
        )
        + f"rotate_{direction}({_json_string(text)}, {amount}) ->\n"
    )
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=repr(result),
        metadata={"operation": f"rotate_{direction}", "text": text, "amount": amount},
    )


def _escape_text(rng: random.Random) -> str:
    pieces = (
        'quote "inside"',
        "backslash \\ path",
        "line\nbreak",
        "tab\tcell",
        "snowman \u2603",
        "emoji \U0001f642",
    )
    return rng.choice(pieces) + " " + _ascii_piece(rng)


def _generate_escape_unescape(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    text = _escape_text(rng)
    escaped = json.dumps(text, ensure_ascii=False)
    if row_index % 2 == 0:
        input_text = (
            _few_shot_block(
                (
                    ("json.dumps('a\"b') ->", '"a\\"b"'),
                    ("json.dumps('line\\nbreak') ->", '"line\\nbreak"'),
                )
            )
            + f"json.dumps({text!r}) ->\n"
        )
        target = escaped
        operation = "json_escape"
    else:
        input_text = (
            _few_shot_block(
                (
                    ("json.loads('\"a\\\\tb\"') ->", "'a\\tb'"),
                    ('json.loads(\'"quote \\\\"inside\\\\""\') ->', "'quote \"inside\"'"),
                )
            )
            + f"json.loads({escaped!r}) ->\n"
        )
        target = repr(text)
        operation = "json_unescape"
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=target,
        metadata={"operation": operation, "text": text, "escaped": escaped},
    )


def _generate_chars_vs_bytes(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    text = _string_text(rng) + rng.choice(("\u00e9", "\u4e2d", "\U0001f642", "\u0301"))
    counts = {"chars": len(text), "utf8_bytes": len(text.encode("utf-8"))}
    input_text = (
        _few_shot_block(
            (
                ('char_and_utf8_byte_count("abc") ->', '{"chars":3,"utf8_bytes":3}'),
                ('char_and_utf8_byte_count("éλ") ->', '{"chars":2,"utf8_bytes":4}'),
            )
        )
        + f"char_and_utf8_byte_count({_json_string(text)}) ->\n"
    )
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=json.dumps(counts, separators=(",", ":")),
        metadata={"operation": "chars_vs_utf8_bytes", "text": text, **counts},
    )


def _normalization_text(rng: random.Random) -> str:
    samples = (
        "Cafe\u0301",
        "\u212bngstrom",
        "\ufb01le",
        "Noe\u0308l",
        "\u2460 item",
        "A\u030a",
    )
    return rng.choice(samples) + rng.choice(("", " " + _ascii_piece(rng, min_length=2, max_length=4)))


def _generate_unicode_normalization(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    text = _normalization_text(rng)
    form = rng.choice(("NFC", "NFD", "NFKC", "NFKD"))
    result = unicodedata.normalize(form, text)
    input_text = (
        _few_shot_block(
            (
                ('unicodedata.normalize("NFC", "Café") ->', repr(unicodedata.normalize("NFC", "Cafe\u0301"))),
                ('unicodedata.normalize("NFKC", "① item") ->', repr(unicodedata.normalize("NFKC", "\u2460 item"))),
            )
        )
        + f"unicodedata.normalize({form!r}, {_json_string(text)}) ->\n"
    )
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=repr(result),
        metadata={"operation": "unicode_normalize", "form": form, "text": text, "result": result},
    )


def _casefold_text(rng: random.Random) -> str:
    samples = (
        "Stra\u00dfe",
        "\u0130stanbul",
        "CAF\u00c9",
        "\u039c\u03ac\u03ca\u03bf\u03c2",
        "DATA_\u00df_" + _ascii_piece(rng, min_length=2, max_length=4).upper(),
    )
    return rng.choice(samples)


def _generate_unicode_casefolding(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    text = _casefold_text(rng)
    input_text = (
        _few_shot_block(
            (
                ('"Straße".casefold() ->', repr("Stra\u00dfe".casefold())),
                ('"CAFÉ".casefold() ->', repr("CAF\u00c9".casefold())),
            )
        )
        + f"{_json_string(text)}.casefold() ->\n"
    )
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=repr(text.casefold()),
        metadata={"operation": "casefold", "text": text, "result": text.casefold()},
    )


def _generate_character_at_index(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    text = _string_text(rng)
    index = rng.randrange(len(text))
    input_text = (
        _few_shot_block(
            (
                ('char_at("abcdef", index=2) ->', "'c'"),
                ('char_at("éx🙂", index=2) ->', "'🙂'"),
            )
        )
        + f"char_at({_json_string(text)}, index={index}) ->\n"
    )
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=repr(text[index]),
        metadata={"operation": "character_at_index", "text": text, "index": index},
    )


def _all_indices(text: str, needle: str) -> list[int]:
    return [index for index in range(len(text) - len(needle) + 1) if text[index : index + len(needle)] == needle]


def _generate_all_indices(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    motif = rng.choice(("ana", "aa", "\u03bb", "\u00e9", "xy"))
    spacer = _ascii_piece(rng, min_length=1, max_length=3)
    text = motif + spacer + motif[: max(1, len(motif) - 1)] + motif + rng.choice(("", motif))
    needle = motif if row_index % 2 else rng.choice(tuple(motif))
    input_text = (
        _few_shot_block(
            (
                ('all_indices("banana", "ana") ->', "[1, 3]"),
                ('all_indices("aaaa", "aa") ->', "[0, 1, 2]"),
            )
        )
        + f"all_indices({_json_string(text)}, {_json_string(needle)}) ->\n"
    )
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=repr(_all_indices(text, needle)),
        metadata={"operation": "all_indices", "text": text, "needle": needle, "overlapping": True},
    )


def _lines(rng: random.Random) -> list[str]:
    lines = []
    for _ in range(rng.randint(3, 5)):
        line = _ascii_piece(rng, min_length=2, max_length=7)
        if rng.random() < 0.4:
            line += rng.choice(("\u00e9", "\u03bb", "\U0001f642"))
        lines.append(line)
    return lines


def _line_starts(lines: list[str]) -> list[int]:
    starts = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line) + 1
    return starts


def _generate_line_column_offsets(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    lines = _lines(rng)
    text = "\n".join(lines)
    starts = _line_starts(lines)
    line_index = rng.randrange(len(lines))
    column = rng.randrange(len(lines[line_index]))
    offset = starts[line_index] + column
    if row_index % 2 == 0:
        input_text = (
            _few_shot_block(
                (
                    ("offset_to_line_col('ab\\ncde', offset=4) ->", '{"line":1,"column":1}'),
                    ("offset_to_line_col('xy\\nz', offset=3) ->", '{"line":1,"column":0}'),
                )
            )
            + f"offset_to_line_col(raw text below, offset={offset})\n{text}\nresult:\n"
        )
        target = json.dumps({"line": line_index, "column": column}, separators=(",", ":"))
        operation = "offset_to_line_col"
    else:
        input_text = (
            _few_shot_block(
                (
                    ("line_col_to_offset('ab\\ncde', line=1, column=1) ->", "4"),
                    ("line_col_to_offset('xy\\nz', line=1, column=0) ->", "3"),
                )
            )
            + f"line_col_to_offset(raw text below, line={line_index}, column={column})\n{text}\nresult:\n"
        )
        target = str(offset)
        operation = "line_col_to_offset"
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=target,
        metadata={
            "operation": operation,
            "text": text,
            "line": line_index,
            "column": column,
            "offset": offset,
            "line_starts": starts,
        },
    )


_BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}"}
_CLOSING_TO_OPENING = {closing: opening for opening, closing in _BRACKET_PAIRS.items()}


def _bracket_expression(rng: random.Random, depth: int = 0) -> str:
    if depth >= 3 or rng.random() < 0.25:
        return _ascii_piece(rng, min_length=1, max_length=3)
    opening = rng.choice(tuple(_BRACKET_PAIRS))
    closing = _BRACKET_PAIRS[opening]
    left = _bracket_expression(rng, depth + 1)
    right = _bracket_expression(rng, depth + 1)
    separator = rng.choice(("", ",", ":", "+"))
    return f"{opening}{left}{separator}{right}{closing}"


def _bracket_matches(text: str) -> dict[int, int]:
    stack: list[tuple[str, int]] = []
    matches: dict[int, int] = {}
    for index, char in enumerate(text):
        if char in _BRACKET_PAIRS:
            stack.append((char, index))
            continue
        if char in _CLOSING_TO_OPENING:
            opening, opening_index = stack.pop()
            if opening != _CLOSING_TO_OPENING[char]:
                raise ValueError(f"Unbalanced bracket expression: {text!r}")
            matches[opening_index] = index
            matches[index] = opening_index
    if stack:
        raise ValueError(f"Unbalanced bracket expression: {text!r}")
    return matches


def _generate_bracket_matching(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    text = ""
    matches: dict[int, int] = {}
    while not matches:
        text = _bracket_expression(rng)
        matches = _bracket_matches(text)
    bracket_index = rng.choice(tuple(sorted(matches)))
    input_text = (
        _few_shot_block(
            (
                ('matching_bracket_index("a(b[c])", index=1) ->', "6"),
                ('matching_bracket_index("{x:[y]}", index=3) ->', "5"),
            )
        )
        + f"matching_bracket_index({_json_string(text)}, index={bracket_index}) ->\n"
    )
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=str(matches[bracket_index]),
        metadata={
            "operation": "bracket_matching",
            "text": text,
            "bracket_index": bracket_index,
            "bracket": text[bracket_index],
            "matching_index": matches[bracket_index],
        },
    )


def _digits(value: int) -> list[int]:
    return [int(char) for char in str(abs(value))]


def _addition_carry_positions(left: int, right: int) -> list[int]:
    positions: list[int] = []
    carry = 0
    left_digits = list(reversed(_digits(left)))
    right_digits = list(reversed(_digits(right)))
    for index in range(max(len(left_digits), len(right_digits))):
        total = left_digits[index] if index < len(left_digits) else 0
        total += right_digits[index] if index < len(right_digits) else 0
        total += carry
        carry = total // 10
        if carry:
            positions.append(index)
    return positions


def _subtraction_borrow_positions(left: int, right: int) -> list[int]:
    positions: list[int] = []
    borrow = 0
    left_digits = list(reversed(_digits(left)))
    right_digits = list(reversed(_digits(right)))
    for index in range(max(len(left_digits), len(right_digits))):
        left_digit = (left_digits[index] if index < len(left_digits) else 0) - borrow
        right_digit = right_digits[index] if index < len(right_digits) else 0
        borrow = 1 if left_digit < right_digit else 0
        if borrow:
            positions.append(index)
    return positions


def _generate_carry_addition(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    while True:
        left = rng.randint(1000, 999999)
        right = rng.randint(1000, 999999)
        carry_positions = _addition_carry_positions(left, right)
        if carry_positions:
            break
    input_text = _few_shot_block(((">>> 47 + 58", "105"), (">>> 386 + 275", "661"))) + f">>> {left} + {right}\n"
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=str(left + right),
        metadata={"operation": "addition", "left": left, "right": right, "carry_positions": carry_positions},
    )


def _generate_borrow_subtraction(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    while True:
        left = rng.randint(1000, 999999)
        right = rng.randint(1000, left - 1)
        borrow_positions = _subtraction_borrow_positions(left, right)
        if borrow_positions:
            break
    input_text = _few_shot_block(((">>> 502 - 148", "354"), (">>> 1004 - 879", "125"))) + f">>> {left} - {right}\n"
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=str(left - right),
        metadata={"operation": "subtraction", "left": left, "right": right, "borrow_positions": borrow_positions},
    )


def _generate_modular_arithmetic(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    left = rng.randint(17, 999)
    right = rng.randint(11, 997)
    offset = rng.randint(-500, 500)
    modulus = rng.randint(7, 97)
    expression = f"(({left} * {right}) + {offset}) % {modulus}"
    input_text = (
        _few_shot_block(
            (
                (">>> ((17 * 19) + 5) % 13", "3"),
                (">>> ((41 * 7) + -9) % 20", "18"),
            )
        )
        + f">>> {expression}\n"
    )
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=str((left * right + offset) % modulus),
        metadata={
            "operation": "modular_arithmetic",
            "left": left,
            "right": right,
            "offset": offset,
            "modulus": modulus,
            "expression": expression,
        },
    )


def _to_base(value: int, base: int) -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    if value == 0:
        return "0"
    digits: list[str] = []
    remaining = value
    while remaining:
        remaining, digit = divmod(remaining, base)
        digits.append(alphabet[digit])
    return "".join(reversed(digits))


def _generate_base_conversion(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    source_base, target_base = rng.choice(((10, 2), (10, 16), (16, 10), (2, 10), (8, 16), (16, 8)))
    value = rng.randint(32, 65535)
    source_text = _to_base(value, source_base)
    target_text = _to_base(int(source_text, source_base), target_base)
    input_text = (
        _few_shot_block(
            (
                ('>>> base_convert("255", from_base=10, to_base=16)', "'ff'"),
                ('>>> base_convert("101101", from_base=2, to_base=10)', "'45'"),
            )
        )
        + f'>>> base_convert("{source_text}", from_base={source_base}, to_base={target_base})\n'
    )
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=repr(target_text),
        metadata={
            "operation": "base_conversion",
            "source_text": source_text,
            "source_base": source_base,
            "target_base": target_base,
            "value": value,
        },
    )


def _generate_digit_checksum(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    digits = [rng.randint(0, 9) for _ in range(rng.randint(8, 16))]
    weights = [3 if index % 2 else 1 for index in range(len(digits))]
    checksum = sum(digit * weight for digit, weight in zip(digits, weights, strict=True)) % 10
    text = "".join(str(digit) for digit in digits)
    input_text = (
        _few_shot_block(
            (
                ('>>> weighted_digit_checksum("12345")', "7"),
                ('>>> weighted_digit_checksum("908172")', "3"),
            )
        )
        + f'>>> weighted_digit_checksum("{text}")\n'
    )
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=str(checksum),
        metadata={"operation": "weighted_digit_checksum", "digits": digits, "weights": weights},
    )


def _generate_python_repr_containers(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    value: dict[str, Any] = {
        "id": f"case_{row_index:04d}",
        "items": [_ascii_piece(rng), rng.randint(-20, 20), None],
        "flags": (rng.choice((True, False)), rng.choice((True, False))),
        "counts": {f"k{i}": rng.randint(0, 9) for i in range(3)},
    }
    result = (value["id"], value["items"], value["flags"], value["counts"])
    input_text = (
        _few_shot_block(
            (
                (
                    ">>> record = {'id': 'demo', 'items': ['aa', 2, None], "
                    "'flags': (True, False), 'counts': {'k0': 1}}\n"
                    ">>> (record['id'], record['items'], record['flags'], record['counts'])",
                    "('demo', ['aa', 2, None], (True, False), {'k0': 1})",
                ),
            )
        )
        + f">>> record = {value!r}\n"
        + ">>> (record['id'], record['items'], record['flags'], record['counts'])\n"
    )
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=repr(result),
        metadata={"operation": "python_repr", "value": value, "result": result},
    )


def _generate_json_field_reemit(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    obj: dict[str, Any] = {
        "id": f"doc-{row_index:04d}",
        "score": rng.randint(0, 100),
        "label": rng.choice(("ready", "hold", "drop")),
        "payload": {"x": rng.randint(0, 9), "y": rng.randint(0, 9)},
        "tags": [_ascii_piece(rng, min_length=2, max_length=4), _ascii_piece(rng, min_length=2, max_length=4)],
    }
    fields = tuple(rng.sample(tuple(obj), k=3))
    result = {field: obj[field] for field in fields}
    input_text = (
        _few_shot_block(
            (
                (
                    'object={"id":"demo","score":7,"label":"ready","payload":{"x":1},"tags":["a"]}\n'
                    'fields=["id","label","score"]\njson=',
                    '{"id":"demo","label":"ready","score":7}',
                ),
            )
        )
        + f"object={json.dumps(obj, sort_keys=True, separators=(',', ':'))}\n"
        + f"fields={json.dumps(fields, separators=(',', ':'))}\n"
        + "json="
    )
    target = json.dumps(result, sort_keys=True, separators=(",", ":"))
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=target,
        metadata={"operation": "json_field_reemit", "object": obj, "fields": fields},
    )


def _csv_row(values: Iterable[str], delimiter: str) -> str:
    sink = io.StringIO()
    writer = csv.writer(sink, delimiter=delimiter, lineterminator="")
    writer.writerow(list(values))
    return sink.getvalue()


def _generate_csv_tsv_transforms(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    delimiter = rng.choice((",", "\t"))
    name = "csv" if delimiter == "," else "tsv"
    rows = [
        (f"sku-{row_index:03d}-{index}", rng.randint(1, 9), rng.randint(3, 25)) for index in range(rng.randint(3, 5))
    ]
    input_lines = [_csv_row(("sku", "qty", "unit"), delimiter)]
    input_lines.extend(_csv_row((sku, str(qty), str(unit)), delimiter) for sku, qty, unit in rows)
    output_rows = [(sku, qty * unit) for sku, qty, unit in rows]
    target = "\n".join(_csv_row((sku, str(total)), delimiter) for sku, total in output_rows)
    input_text = (
        _few_shot_block(
            (
                (
                    "sku,qty,unit\n" "demo-0,2,5\n" "demo-1,3,4\n" "csv_totals sku,total",
                    "demo-0,10\ndemo-1,12",
                ),
            )
        )
        + "\n".join(input_lines)
        + f"\n{name}_totals sku,total\n"
    )
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=target,
        metadata={"operation": "delimited_totals", "delimiter": delimiter, "rows": rows},
    )


def _generate_stack_push_pop(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    stack: list[str] = []
    operations: list[str] = []
    for _ in range(rng.randint(6, 12)):
        if stack and rng.random() < 0.35:
            operations.append("pop")
            stack.pop()
            continue
        item = _ascii_piece(rng, min_length=2, max_length=5)
        operations.append(f"push {item}")
        stack.append(item)
    input_text = (
        _few_shot_block((("stack trace:\npush a\npush b\npop\nfinal_stack=", "['a']"),))
        + "\n".join(operations)
        + "\nfinal_stack="
    )
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=repr(stack),
        metadata={"operation": "stack_push_pop", "operations": operations},
    )


def _generate_finite_automata(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    bits = "".join(rng.choice("01") for _ in range(rng.randint(8, 20)))
    ones = bits.count("1")
    zeros = bits.count("0")
    state = f"ones_{ones % 2}_zeros_{zeros % 3}"
    accept = ones % 2 == 0 and zeros % 3 == 0
    input_text = (
        _few_shot_block(
            (
                ('>>> dfa_even_ones_zero_mod3("111000")', '{"state":"ones_1_zeros_0","accept":false}'),
                ('>>> dfa_even_ones_zero_mod3("11000")', '{"state":"ones_0_zeros_0","accept":true}'),
            )
        )
        + f'>>> dfa_even_ones_zero_mod3("{bits}")\n'
    )
    target = json.dumps({"state": state, "accept": accept}, separators=(",", ":"))
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=target,
        metadata={"operation": "dfa_even_ones_zero_mod3", "bits": bits},
    )


def _generate_turtle_commands(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    headings = ("N", "E", "S", "W")
    heading_index = 0
    x = 0
    y = 0
    commands: list[str] = []
    for _ in range(rng.randint(5, 12)):
        command = rng.choice(("L", "R", "F"))
        if command == "L":
            heading_index = (heading_index - 1) % len(headings)
            commands.append("L")
        elif command == "R":
            heading_index = (heading_index + 1) % len(headings)
            commands.append("R")
        else:
            distance = rng.randint(1, 5)
            heading = headings[heading_index]
            x += distance if heading == "E" else -distance if heading == "W" else 0
            y += distance if heading == "N" else -distance if heading == "S" else 0
            commands.append(f"F{distance}")
    target = json.dumps({"x": x, "y": y, "heading": headings[heading_index]}, separators=(",", ":"))
    input_text = (
        _few_shot_block((("turtle start=(0,0,N)\ncommands=F2 R F3\nend=", '{"x":3,"y":2,"heading":"E"}'),))
        + "turtle start=(0,0,N)\ncommands="
        + " ".join(commands)
        + "\nend="
    )
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=target,
        metadata={"operation": "turtle", "commands": commands},
    )


def _generate_regex_lite(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    chunks: list[str] = []
    matches: list[str] = []
    for _ in range(rng.randint(5, 10)):
        if rng.random() < 0.45:
            match = rng.choice(("ab", "xy")) + str(rng.randint(0, 9))
            chunks.append(match)
            matches.append(match)
        else:
            chunks.append(_ascii_piece(rng))
    text = "-".join(chunks)
    pattern = r"(?:ab|xy)\d"
    input_text = (
        _few_shot_block((('>>> regex_lite_findall(r"(?:ab|xy)\\d", "ab1-no-xy7-ab")', "['ab1', 'xy7']"),))
        + f'>>> regex_lite_findall(r"{pattern}", "{text}")\n'
    )
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=repr(re.findall(pattern, text)),
        metadata={"operation": "regex_lite_findall", "pattern": pattern, "text": text},
    )


def _brainfuck_lite_state(program: str, num_cells: int = 4) -> tuple[list[int], int]:
    cells = [0 for _ in range(num_cells)]
    pointer = 0
    for command in program:
        if command == ">":
            pointer = min(pointer + 1, len(cells) - 1)
        elif command == "<":
            pointer = max(pointer - 1, 0)
        elif command == "+":
            cells[pointer] = (cells[pointer] + 1) % 256
        elif command == "-":
            cells[pointer] = (cells[pointer] - 1) % 256
    return cells, pointer


def _generate_brainfuck_lite(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    program = "".join(rng.choice(("+", "+", "-", ">", "<")) for _ in range(rng.randint(12, 28)))
    cells, pointer = _brainfuck_lite_state(program)
    target = json.dumps({"cells": cells, "pointer": pointer}, separators=(",", ":"))
    input_text = (
        _few_shot_block((('brainfuck_lite cells=4 program="++>+<+"\nfinal=', '{"cells":[3,1,0,0],"pointer":0}'),))
        + f'brainfuck_lite cells=4 program="{program}"\n'
        + "final="
    )
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=target,
        metadata={"operation": "brainfuck_lite", "program": program, "num_cells": 4},
    )


def _markdown_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    widths = [
        max(len(headers[column_index]), *(len(row[column_index]) for row in rows))
        for column_index in range(len(headers))
    ]

    def render_row(row: Iterable[str]) -> str:
        return "| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join((render_row(headers), separator, *(render_row(row) for row in rows)))


def _generate_markdown_table_padding(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    headers = ("name", "status", "score")
    rows = [
        (
            rng.choice(("alpha", "beta_long", "gamma", "delta_tool")),
            rng.choice(("ok", "needs review", "blocked", "ready")),
            str(rng.randint(1, 999)),
        )
        for _ in range(rng.randint(3, 5))
    ]
    rough_table = "\n".join("|".join(row) for row in (headers, *rows))
    target = _markdown_table(headers, rows)
    input_text = (
        _few_shot_block(
            (
                (
                    "pad_markdown_table:\nkey|value\nshort|7\nlong_name|42\naligned=",
                    "| key       | value |\n| --------- | ----- |\n| short     | 7     |\n| long_name | 42    |",
                ),
            )
        )
        + "pad_markdown_table:\n"
        + rough_table
        + "\naligned=\n"
    )
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=target,
        metadata={"operation": "markdown_table_padding", "headers": headers, "rows": rows},
    )


def _wrap_source_words(rng: random.Random) -> list[str]:
    vocabulary = (
        "format",
        "columns",
        "stable",
        "alignment",
        "prompt",
        "answer",
        "spacing",
        "deterministic",
        "indent",
        "markdown",
        "record",
        "line",
        "width",
        "style",
        "instruction",
        "serialize",
        "literal",
    )
    return [rng.choice(vocabulary) for _ in range(rng.randint(18, 36))]


def _generate_line_wrapping(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    width = rng.randint(28, 48)
    paragraph = " ".join(_wrap_source_words(rng))
    wrapped = "\n".join(textwrap.wrap(paragraph, width=width, break_long_words=False, break_on_hyphens=False))
    input_text = (
        _few_shot_block(
            (
                (
                    "wrap width=12:\nalpha beta gamma delta\nwrapped=",
                    "alpha beta\ngamma delta",
                ),
                (
                    "wrap width=16:\nformat columns without drift\nwrapped=",
                    "format columns\nwithout drift",
                ),
            )
        )
        + f"wrap width={width}:\n{paragraph}\nwrapped=\n"
    )
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=wrapped,
        metadata={"operation": "line_wrapping", "width": width, "paragraph": paragraph},
    )


def _generate_outline_indentation(
    slice_: PplCircuitCoverageV2Slice, row_index: int, rng: random.Random, seed: int
) -> dict[str, object]:
    labels = [_ascii_piece(rng, min_length=4, max_length=8) for _ in range(rng.randint(5, 9))]
    levels: list[int] = []
    current_level = 0
    for _ in labels:
        current_level = max(0, min(3, current_level + rng.choice((-1, 0, 1))))
        levels.append(current_level)
    entries = list(zip(levels, labels, strict=True))
    target = "\n".join(f"{'  ' * level}- {label}" for level, label in entries)
    compact = "\n".join(f"{level}:{label}" for level, label in entries)
    input_text = (
        _few_shot_block(
            (
                (
                    "indent_outline:\n0:root\n1:child\n2:leaf\n1:sibling\noutline=",
                    "- root\n  - child\n    - leaf\n  - sibling",
                ),
            )
        )
        + "indent_outline:\n"
        + compact
        + "\noutline=\n"
    )
    return _record(
        slice_=slice_,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=target,
        metadata={"operation": "outline_indentation", "entries": entries},
    )


Generator = Callable[[PplCircuitCoverageV2Slice, int, random.Random, int], dict[str, object]]

_GENERATORS: dict[str, Generator] = {
    "string_slicing": _generate_string_slicing,
    "string_reversal": _generate_string_reversal,
    "string_rotation": _generate_string_rotation,
    "escape_unescape": _generate_escape_unescape,
    "chars_vs_bytes": _generate_chars_vs_bytes,
    "unicode_normalization": _generate_unicode_normalization,
    "unicode_casefolding": _generate_unicode_casefolding,
    "character_at_index": _generate_character_at_index,
    "all_indices": _generate_all_indices,
    "line_column_offsets": _generate_line_column_offsets,
    "bracket_matching": _generate_bracket_matching,
    "carry_addition": _generate_carry_addition,
    "borrow_subtraction": _generate_borrow_subtraction,
    "modular_arithmetic": _generate_modular_arithmetic,
    "base_conversion": _generate_base_conversion,
    "digit_checksum": _generate_digit_checksum,
    "python_repr_containers": _generate_python_repr_containers,
    "json_field_reemit": _generate_json_field_reemit,
    "csv_tsv_transforms": _generate_csv_tsv_transforms,
    "stack_push_pop": _generate_stack_push_pop,
    "finite_automata": _generate_finite_automata,
    "turtle_commands": _generate_turtle_commands,
    "regex_lite": _generate_regex_lite,
    "brainfuck_lite": _generate_brainfuck_lite,
    "markdown_table_padding": _generate_markdown_table_padding,
    "line_wrapping": _generate_line_wrapping,
    "outline_indentation": _generate_outline_indentation,
}


def iter_ppl_circuit_coverage_v2_records(
    *,
    slices: Iterable[PplCircuitCoverageV2Slice] = PPL_CIRCUIT_COVERAGE_V2_SLICES,
    examples_per_config: int = EXAMPLES_PER_CONFIG,
    seed: int = PPL_CIRCUIT_COVERAGE_V2_SEED,
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    slice_indices = {slice_: index for index, slice_ in enumerate(PPL_CIRCUIT_COVERAGE_V2_SLICES)}
    for slice_ in slices:
        rng = random.Random(seed + slice_indices[slice_] * 1_000_003)
        generator = _GENERATORS[slice_.task_name]
        for row_index in range(examples_per_config):
            records.append(generator(slice_, row_index, rng, seed))
    return tuple(records)


def materialize_ppl_circuit_coverage_v2_raw(
    output_path: str,
    *,
    examples_per_config: int = EXAMPLES_PER_CONFIG,
    seed: int = PPL_CIRCUIT_COVERAGE_V2_SEED,
) -> None:
    fsspec_utils.mkdirs(output_path)
    records_by_subset: dict[str, list[dict[str, object]]] = {
        slice_.output_name: [] for slice_ in PPL_CIRCUIT_COVERAGE_V2_SLICES
    }
    for record in iter_ppl_circuit_coverage_v2_records(examples_per_config=examples_per_config, seed=seed):
        records_by_subset[str(record["subset"])].append(record)

    for subset, records in records_by_subset.items():
        output_file = os.path.join(output_path, f"{subset}{RAW_SHARD_SUFFIX}")
        with fsspec.open(output_file, "wt", compression="gzip") as sink:
            for record in records:
                sink.write(json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
                sink.write("\n")

    summary = {
        "source": PPL_CIRCUIT_COVERAGE_V2_SOURCE,
        "seed": seed,
        "examples_per_config": examples_per_config,
        "slices": [slice_.registry_key for slice_ in PPL_CIRCUIT_COVERAGE_V2_SLICES],
    }
    with fsspec.open(os.path.join(output_path, "coverage_summary.json"), "wt") as sink:
        sink.write(json.dumps(summary, ensure_ascii=True, sort_keys=True))
        sink.write("\n")


@dataclass(frozen=True)
class PplCircuitCoverageV2RawConfig:
    output_path: object
    source: str = PPL_CIRCUIT_COVERAGE_V2_SOURCE
    examples_per_config: int = EXAMPLES_PER_CONFIG
    seed: int = PPL_CIRCUIT_COVERAGE_V2_SEED
    schema: tuple[str, ...] = ("input", "target", "id", "subset", "task", "seed", "tags", "metadata")


def materialize_ppl_circuit_coverage_v2_raw_from_config(config: PplCircuitCoverageV2RawConfig) -> None:
    materialize_ppl_circuit_coverage_v2_raw(
        str(config.output_path),
        examples_per_config=config.examples_per_config,
        seed=config.seed,
    )


def estimated_plain_text_tokens(text: str) -> int:
    return max(1, (len(text) + CHARS_PER_TOKEN_ESTIMATE - 1) // CHARS_PER_TOKEN_ESTIMATE)


def render_ppl_circuit_coverage_v2_plain_text_document(record: dict[str, object]) -> str:
    metadata = record["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected record metadata dict, got {type(metadata).__name__}")
    family = metadata["family"]
    task = record["task"]
    target = str(record["target"]).rstrip("\n")
    return (
        "### Circuit practice example\n"
        f"Family: {family}\n"
        f"Task: {task}\n"
        "\n"
        "Prompt:\n"
        f"{record['input'].rstrip()}\n"
        "\n"
        "Final answer:\n"
        f"{target}\n"
    )


def _pretraining_weight(slice_: PplCircuitCoverageV2Slice) -> int:
    if slice_.family == PplCircuitCoverageV2Family.FORMAT_STYLE_INSTRUCTION:
        return 5
    if slice_.family in (
        PplCircuitCoverageV2Family.STRING_BYTE_TRANSFORMS,
        PplCircuitCoverageV2Family.INDEXING_POSITION_TRACKING,
    ):
        return 3
    if slice_.family in (
        PplCircuitCoverageV2Family.ARITHMETIC,
        PplCircuitCoverageV2Family.STRUCTURED_SERIALIZATION,
    ):
        return 2
    return 1


def _plain_text_pretraining_slice_cycle() -> tuple[PplCircuitCoverageV2Slice, ...]:
    cycle: list[PplCircuitCoverageV2Slice] = []
    for slice_ in sorted(PPL_CIRCUIT_COVERAGE_V2_SLICES, key=lambda item: _pretraining_weight(item), reverse=True):
        cycle.extend([slice_] * _pretraining_weight(slice_))
    return tuple(cycle)


def iter_ppl_circuit_coverage_v2_plain_text_documents(
    *,
    target_tokens: int = PLAIN_TEXT_PRETRAINING_TARGET_TOKENS,
    seed: int = PPL_CIRCUIT_COVERAGE_V2_SEED,
) -> Iterable[dict[str, object]]:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    slice_cycle = _plain_text_pretraining_slice_cycle()
    slice_indices = {slice_: index for index, slice_ in enumerate(PPL_CIRCUIT_COVERAGE_V2_SLICES)}
    rngs = {
        slice_: random.Random(seed + slice_indices[slice_] * 1_000_003 + 7919)
        for slice_ in PPL_CIRCUIT_COVERAGE_V2_SLICES
    }
    row_indices = {slice_: 0 for slice_ in PPL_CIRCUIT_COVERAGE_V2_SLICES}
    estimated_tokens = 0
    document_index = 0

    while estimated_tokens < target_tokens:
        for slice_ in slice_cycle:
            row_index = row_indices[slice_]
            row_indices[slice_] = row_index + 1
            record = _GENERATORS[slice_.task_name](slice_, row_index, rngs[slice_], seed)
            text = render_ppl_circuit_coverage_v2_plain_text_document(record)
            document_tokens = estimated_plain_text_tokens(text)
            estimated_tokens += document_tokens
            yield {
                "id": f"{PLAIN_TEXT_PRETRAINING_SOURCE}_{document_index:012d}",
                "text": text,
                "estimated_tokens": document_tokens,
                "source": PLAIN_TEXT_PRETRAINING_SOURCE,
                "family": slice_.family.value,
                "task": slice_.task_name,
                "seed": seed,
                "supervised_record_id": record["id"],
            }
            document_index += 1
            if estimated_tokens >= target_tokens:
                return


def materialize_ppl_circuit_coverage_v2_plain_text_pretraining(
    output_path: str,
    *,
    target_tokens: int = PLAIN_TEXT_PRETRAINING_TARGET_TOKENS,
    num_shards: int = PLAIN_TEXT_PRETRAINING_SHARDS,
    seed: int = PPL_CIRCUIT_COVERAGE_V2_SEED,
) -> None:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    fsspec_utils.mkdirs(output_path)
    target_tokens_per_shard = max(1, (target_tokens + num_shards - 1) // num_shards)
    documents = iter_ppl_circuit_coverage_v2_plain_text_documents(target_tokens=target_tokens, seed=seed)
    total_estimated_tokens = 0
    total_documents = 0
    counts_by_task = {slice_.registry_key: 0 for slice_ in PPL_CIRCUIT_COVERAGE_V2_SLICES}

    for shard_index in range(num_shards):
        shard_estimated_tokens = 0
        shard_documents = 0
        output_file = os.path.join(output_path, f"part-{shard_index:05d}-of-{num_shards:05d}.jsonl.gz")
        with fsspec.open(output_file, "wt", compression="gzip") as sink:
            while total_estimated_tokens < target_tokens and (
                shard_documents == 0 or shard_estimated_tokens < target_tokens_per_shard
            ):
                document = next(documents)
                task_key = posixpath.join("ppl_circuit_coverage_v2", str(document["family"]), str(document["task"]))
                counts_by_task[task_key] += 1
                document_tokens = int(document["estimated_tokens"])
                total_estimated_tokens += document_tokens
                shard_estimated_tokens += document_tokens
                total_documents += 1
                shard_documents += 1
                sink.write(json.dumps(document, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
                sink.write("\n")
        if total_estimated_tokens >= target_tokens:
            break

    summary = {
        "source": PLAIN_TEXT_PRETRAINING_SOURCE,
        "seed": seed,
        "target_tokens": target_tokens,
        "estimated_tokens": total_estimated_tokens,
        "chars_per_token_estimate": CHARS_PER_TOKEN_ESTIMATE,
        "documents": total_documents,
        "num_shards_requested": num_shards,
        "slice_weights": {slice_.registry_key: _pretraining_weight(slice_) for slice_ in PPL_CIRCUIT_COVERAGE_V2_SLICES},
        "counts_by_task": counts_by_task,
    }
    with fsspec.open(os.path.join(output_path, "pretraining_summary.json"), "wt") as sink:
        sink.write(json.dumps(summary, ensure_ascii=True, sort_keys=True))
        sink.write("\n")


@dataclass(frozen=True)
class PplCircuitCoverageV2PlainTextPretrainingConfig:
    output_path: object
    source: str = PLAIN_TEXT_PRETRAINING_SOURCE
    seed: int = PPL_CIRCUIT_COVERAGE_V2_SEED
    target_tokens: int = PLAIN_TEXT_PRETRAINING_TARGET_TOKENS
    num_shards: int = PLAIN_TEXT_PRETRAINING_SHARDS
    chars_per_token_estimate: int = CHARS_PER_TOKEN_ESTIMATE
    schema: tuple[str, ...] = (
        "id",
        "text",
        "estimated_tokens",
        "source",
        "family",
        "task",
        "seed",
        "supervised_record_id",
    )


def materialize_ppl_circuit_coverage_v2_plain_text_pretraining_from_config(
    config: PplCircuitCoverageV2PlainTextPretrainingConfig,
) -> None:
    materialize_ppl_circuit_coverage_v2_plain_text_pretraining(
        str(config.output_path),
        target_tokens=config.target_tokens,
        num_shards=config.num_shards,
        seed=config.seed,
    )


ppl_circuit_coverage_v2_raw_executor = ExecutorStep(
    name=os.path.join("raw", "evals", "ppl_circuit_coverage_v2"),
    fn=materialize_ppl_circuit_coverage_v2_raw_from_config,
    config=PplCircuitCoverageV2RawConfig(output_path=this_output_path()),
)

ppl_circuit_coverage_v2_plain_text_pretraining_executor = ExecutorStep(
    name=os.path.join("raw", "pretraining", "ppl_circuit_coverage_v2_plain_text_1b"),
    fn=materialize_ppl_circuit_coverage_v2_plain_text_pretraining_from_config,
    config=PplCircuitCoverageV2PlainTextPretrainingConfig(output_path=this_output_path()),
    resources=ResourceConfig.with_cpu(cpu=16, ram="64g", disk="200g"),
)


def write_local_sample(
    output_path: str | Path,
    *,
    examples_per_config: int = LOCAL_SAMPLE_EXAMPLES_PER_CONFIG,
    seed: int = PPL_CIRCUIT_COVERAGE_V2_SEED,
) -> None:
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for slice_ in PPL_CIRCUIT_COVERAGE_V2_SLICES:
        sample_file = output_dir / f"{slice_.output_name}.jsonl"
        with sample_file.open("w", encoding="utf-8") as sink:
            for row in iter_ppl_circuit_coverage_v2_records(
                slices=(slice_,),
                examples_per_config=examples_per_config,
                seed=seed,
            ):
                sink.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
                sink.write("\n")


if __name__ == "__main__":
    write_local_sample("/tmp/marin_ppl_circuit_coverage_v2_sample")
