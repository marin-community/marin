# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import csv
import io
import json
import re
import textwrap
import unicodedata
from collections.abc import Callable, Iterable

from experiments.evals.ppl_circuit_coverage_v2 import (
    PPL_CIRCUIT_COVERAGE_V2_SLICES,
    PplCircuitCoverageV2TaskTier,
    _brainfuck_lite_state,
    _generate_borrow_subtraction,
    _markdown_table,
    _to_base,
    iter_ppl_circuit_coverage_v2_plain_text_documents,
    iter_ppl_circuit_coverage_v2_records,
    ppl_circuit_coverage_v2_raw_validation_sets,
    render_ppl_circuit_coverage_v2_plain_text_document,
    write_local_sample,
)


def _compact_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_for_task(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


def _all_indices(text: str, needle: str) -> list[int]:
    return [index for index in range(len(text) - len(needle) + 1) if text[index : index + len(needle)] == needle]


def _matching_bracket_index(text: str, bracket_index: int) -> int:
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing_to_opening = {closing: opening for opening, closing in pairs.items()}
    stack: list[tuple[str, int]] = []
    matches: dict[int, int] = {}
    for index, char in enumerate(text):
        if char in pairs:
            stack.append((char, index))
            continue
        if char not in closing_to_opening:
            continue
        opening, opening_index = stack.pop()
        assert opening == closing_to_opening[char]
        matches[opening_index] = index
        matches[index] = opening_index
    return matches[bracket_index]


def _csv_row(values: Iterable[str], delimiter: str) -> str:
    sink = io.StringIO()
    writer = csv.writer(sink, delimiter=delimiter, lineterminator="")
    writer.writerow(list(values))
    return sink.getvalue()


def _final_stack(operations: Iterable[str]) -> list[str]:
    stack: list[str] = []
    for operation in operations:
        if operation == "pop":
            stack.pop()
            continue
        stack.append(operation.split(" ", maxsplit=1)[1])
    return stack


def _turtle_state(commands: Iterable[str]) -> dict[str, object]:
    headings = ("N", "E", "S", "W")
    heading_index = 0
    x = 0
    y = 0
    for command in commands:
        if command == "L":
            heading_index = (heading_index - 1) % len(headings)
            continue
        if command == "R":
            heading_index = (heading_index + 1) % len(headings)
            continue
        distance = int(command[1:])
        heading = headings[heading_index]
        x += distance if heading == "E" else -distance if heading == "W" else 0
        y += distance if heading == "N" else -distance if heading == "S" else 0
    return {"x": x, "y": y, "heading": headings[heading_index]}


def _expected_string_slicing(metadata: dict[str, object]) -> str:
    text = str(metadata["text"])
    return repr(text[int(metadata["start"]) : int(metadata["stop"]) : int(metadata["step"])])


def _expected_string_reversal(metadata: dict[str, object]) -> str:
    return repr(str(metadata["text"])[::-1])


def _expected_string_rotation(metadata: dict[str, object]) -> str:
    text = str(metadata["text"])
    amount = int(metadata["amount"])
    if metadata["operation"] == "rotate_left":
        return repr(text[amount:] + text[:amount])
    return repr(text[-amount:] + text[:-amount])


def _expected_escape_unescape(metadata: dict[str, object]) -> str:
    text = str(metadata["text"])
    if metadata["operation"] == "json_escape":
        return json.dumps(text, ensure_ascii=False)
    return repr(text)


def _expected_chars_vs_bytes(metadata: dict[str, object]) -> str:
    text = str(metadata["text"])
    return _json_for_task({"chars": len(text), "utf8_bytes": len(text.encode("utf-8"))})


def _expected_unicode_normalization(metadata: dict[str, object]) -> str:
    return repr(unicodedata.normalize(str(metadata["form"]), str(metadata["text"])))


def _expected_unicode_casefolding(metadata: dict[str, object]) -> str:
    return repr(str(metadata["text"]).casefold())


def _expected_character_at_index(metadata: dict[str, object]) -> str:
    return repr(str(metadata["text"])[int(metadata["index"])])


def _expected_all_indices(metadata: dict[str, object]) -> str:
    return repr(_all_indices(str(metadata["text"]), str(metadata["needle"])))


def _expected_line_column_offsets(metadata: dict[str, object]) -> str:
    if metadata["operation"] == "offset_to_line_col":
        return _json_for_task({"line": metadata["line"], "column": metadata["column"]})
    line_starts = metadata["line_starts"]
    assert isinstance(line_starts, list)
    return str(int(line_starts[int(metadata["line"])]) + int(metadata["column"]))


def _expected_bracket_matching(metadata: dict[str, object]) -> str:
    return str(_matching_bracket_index(str(metadata["text"]), int(metadata["bracket_index"])))


def _expected_carry_addition(metadata: dict[str, object]) -> str:
    return str(int(metadata["left"]) + int(metadata["right"]))


def _expected_borrow_subtraction(metadata: dict[str, object]) -> str:
    return str(int(metadata["left"]) - int(metadata["right"]))


def _expected_modular_arithmetic(metadata: dict[str, object]) -> str:
    return str((int(metadata["left"]) * int(metadata["right"]) + int(metadata["offset"])) % int(metadata["modulus"]))


def _expected_base_conversion(metadata: dict[str, object]) -> str:
    source_text = str(metadata["source_text"])
    target_base = int(metadata["target_base"])
    source_base = int(metadata["source_base"])
    return repr(_to_base(int(source_text, source_base), target_base))


def _expected_digit_checksum(metadata: dict[str, object]) -> str:
    digits = metadata["digits"]
    weights = metadata["weights"]
    assert isinstance(digits, list)
    assert isinstance(weights, list)
    return str(sum(int(digit) * int(weight) for digit, weight in zip(digits, weights, strict=True)) % 10)


def _expected_python_repr_containers(metadata: dict[str, object]) -> str:
    value = metadata["value"]
    assert isinstance(value, dict)
    return repr((value["id"], value["items"], value["flags"], value["counts"]))


def _expected_json_field_reemit(metadata: dict[str, object]) -> str:
    obj = metadata["object"]
    fields = metadata["fields"]
    assert isinstance(obj, dict)
    assert isinstance(fields, tuple)
    return _compact_json({field: obj[field] for field in fields})


def _expected_csv_tsv_transforms(metadata: dict[str, object]) -> str:
    delimiter = str(metadata["delimiter"])
    rows = metadata["rows"]
    assert isinstance(rows, list)
    return "\n".join(_csv_row((str(sku), str(int(qty) * int(unit))), delimiter) for sku, qty, unit in rows)


def _expected_stack_push_pop(metadata: dict[str, object]) -> str:
    operations = metadata["operations"]
    assert isinstance(operations, list)
    return repr(_final_stack(str(operation) for operation in operations))


def _expected_finite_automata(metadata: dict[str, object]) -> str:
    bits = str(metadata["bits"])
    return _json_for_task(
        {
            "state": f"ones_{bits.count('1') % 2}_zeros_{bits.count('0') % 3}",
            "accept": bits.count("1") % 2 == 0 and bits.count("0") % 3 == 0,
        }
    )


def _expected_turtle_commands(metadata: dict[str, object]) -> str:
    commands = metadata["commands"]
    assert isinstance(commands, list)
    return _json_for_task(_turtle_state(str(command) for command in commands))


def _expected_regex_lite(metadata: dict[str, object]) -> str:
    return repr(re.findall(str(metadata["pattern"]), str(metadata["text"])))


def _expected_brainfuck_lite(metadata: dict[str, object]) -> str:
    cells, pointer = _brainfuck_lite_state(str(metadata["program"]), int(metadata["num_cells"]))
    return _json_for_task({"cells": cells, "pointer": pointer})


def _expected_markdown_table_padding(metadata: dict[str, object]) -> str:
    headers = metadata["headers"]
    rows = metadata["rows"]
    assert isinstance(headers, tuple)
    assert isinstance(rows, list)
    return _markdown_table(headers, rows)


def _expected_line_wrapping(metadata: dict[str, object]) -> str:
    return "\n".join(
        textwrap.wrap(
            str(metadata["paragraph"]),
            width=int(metadata["width"]),
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def _expected_outline_indentation(metadata: dict[str, object]) -> str:
    entries = metadata["entries"]
    assert isinstance(entries, list)
    return "\n".join(f"{'  ' * int(level)}- {label}" for level, label in entries)


ExpectedTarget = Callable[[dict[str, object]], str]

EXPECTED_TARGET_BY_TASK: dict[str, ExpectedTarget] = {
    "string_slicing": _expected_string_slicing,
    "string_reversal": _expected_string_reversal,
    "string_rotation": _expected_string_rotation,
    "escape_unescape": _expected_escape_unescape,
    "chars_vs_bytes": _expected_chars_vs_bytes,
    "unicode_normalization": _expected_unicode_normalization,
    "unicode_casefolding": _expected_unicode_casefolding,
    "character_at_index": _expected_character_at_index,
    "all_indices": _expected_all_indices,
    "line_column_offsets": _expected_line_column_offsets,
    "bracket_matching": _expected_bracket_matching,
    "carry_addition": _expected_carry_addition,
    "borrow_subtraction": _expected_borrow_subtraction,
    "modular_arithmetic": _expected_modular_arithmetic,
    "base_conversion": _expected_base_conversion,
    "digit_checksum": _expected_digit_checksum,
    "python_repr_containers": _expected_python_repr_containers,
    "json_field_reemit": _expected_json_field_reemit,
    "csv_tsv_transforms": _expected_csv_tsv_transforms,
    "stack_push_pop": _expected_stack_push_pop,
    "finite_automata": _expected_finite_automata,
    "turtle_commands": _expected_turtle_commands,
    "regex_lite": _expected_regex_lite,
    "brainfuck_lite": _expected_brainfuck_lite,
    "markdown_table_padding": _expected_markdown_table_padding,
    "line_wrapping": _expected_line_wrapping,
    "outline_indentation": _expected_outline_indentation,
}

EXTENDED_TASKS = {
    "brainfuck_lite",
    "finite_automata",
    "regex_lite",
    "turtle_commands",
}
# PPL circuit prompts should stay as compact few-shot completions, not task-labeled instructions.
BANNED_SCAFFOLD_STRINGS = (
    "Family:",
    "Task:",
    "Prompt:",
    "worked examples",
    "held-out query",
    "answer:",
    "Final answer:",
)


def test_registry_points_all_slices_at_target_only_supervised_fields():
    datasets = ppl_circuit_coverage_v2_raw_validation_sets()

    assert set(datasets) == {slice_.registry_key for slice_ in PPL_CIRCUIT_COVERAGE_V2_SLICES}
    for slice_ in PPL_CIRCUIT_COVERAGE_V2_SLICES:
        dataset = datasets[slice_.registry_key]
        assert dataset.input_key == "input"
        assert dataset.target_key == "output"
        assert "loss:target_only" in dataset.tags
        assert f"family:{slice_.family.value}" in dataset.tags
        assert f"task:{slice_.task_name}" in dataset.tags
        assert f"task_tier:{slice_.task_tier.value}" in dataset.tags


def test_task_tiers_separate_reflexive_core_from_interpreter_stress_tasks():
    tiers_by_task = {slice_.task_name: slice_.task_tier for slice_ in PPL_CIRCUIT_COVERAGE_V2_SLICES}

    assert {task for task, tier in tiers_by_task.items() if tier == PplCircuitCoverageV2TaskTier.EXTENDED} == (
        EXTENDED_TASKS
    )
    assert all(tier in PplCircuitCoverageV2TaskTier for tier in tiers_by_task.values())


def test_records_are_deterministic_target_only_completion_prompts():
    rows = iter_ppl_circuit_coverage_v2_records(examples_per_config=2, seed=2026)
    rows_again = iter_ppl_circuit_coverage_v2_records(examples_per_config=2, seed=2026)

    assert rows == rows_again
    assert len(rows) == 2 * len(PPL_CIRCUIT_COVERAGE_V2_SLICES)
    for row in rows:
        assert str(row["output"]).endswith("\n")
        assert not str(row["input"]).endswith(str(row["output"]))
        assert f"task_tier:{row['metadata']['task_tier']}" in row["tags"]


def test_string_slicing_prompt_keeps_demonstrations_complete_and_query_completion_only():
    slice_ = next(slice_ for slice_ in PPL_CIRCUIT_COVERAGE_V2_SLICES if slice_.task_name == "string_slicing")
    rows = iter_ppl_circuit_coverage_v2_records(slices=(slice_,), examples_per_config=10, seed=2027)

    surfaces = set()
    for row in rows:
        metadata = row["metadata"]
        assert isinstance(metadata, dict)
        surfaces.add(metadata["surface"])
        input_text = str(row["input"])
        output = str(row["output"])
        lines = input_text.splitlines()

        assert output.endswith("\n")
        assert output.count("\n") == 1
        assert len(lines) == 3
        assert lines[-1].endswith(" -> ")
        assert output.strip() not in lines[-1]
        assert not input_text.endswith(output)
        assert output == f"{_expected_string_slicing(metadata)}\n"
        assert all(scaffold not in input_text for scaffold in BANNED_SCAFFOLD_STRINGS)

    assert len(surfaces) >= 5


def test_string_slicing_surfaces_are_used_in_local_samples(tmp_path):
    write_local_sample(tmp_path, seed=2027)
    sample_file = tmp_path / "string_byte_transforms_string_slicing.jsonl"
    rows = (json.loads(line) for line in sample_file.read_text(encoding="utf-8").splitlines())

    surfaces = {row["metadata"]["surface"] for row in rows}

    assert len(surfaces) >= 5


def test_generated_targets_match_independent_task_interpreters():
    assert set(EXPECTED_TARGET_BY_TASK) == {slice_.task_name for slice_ in PPL_CIRCUIT_COVERAGE_V2_SLICES}

    rows = iter_ppl_circuit_coverage_v2_records(examples_per_config=4, seed=2027)

    for row in rows:
        metadata = row["metadata"]
        assert isinstance(metadata, dict)
        expected_target = EXPECTED_TARGET_BY_TASK[str(row["task"])](metadata)
        assert row["output"] == f"{expected_target}\n"


def test_borrow_subtraction_generator_handles_minimum_left_boundary():
    class BoundaryRng:
        def __init__(self) -> None:
            self.values = [1001, 1000, 1010, 1009]

        def randint(self, lower: int, upper: int) -> int:
            if lower > upper:
                raise ValueError(f"empty range for randrange() ({lower}, {upper + 1}, 0)")
            value = self.values.pop(0)
            assert lower <= value <= upper
            return value

    slice_ = next(slice_ for slice_ in PPL_CIRCUIT_COVERAGE_V2_SLICES if slice_.task_name == "borrow_subtraction")

    row = _generate_borrow_subtraction(slice_, row_index=0, rng=BoundaryRng(), seed=6203)

    assert row["metadata"]["left"] == 1010
    assert row["metadata"]["right"] == 1009
    assert row["metadata"]["borrow_positions"]
    assert row["output"] == "1\n"


def test_plain_text_documents_append_the_target():
    row = next(
        row
        for row in iter_ppl_circuit_coverage_v2_records(examples_per_config=1)
        if row["task"] == "markdown_table_padding"
    )

    text = render_ppl_circuit_coverage_v2_plain_text_document(row)

    assert text == row["input"] + row["output"]


def test_plain_text_stream_respects_budget_and_keeps_task_variety():
    documents = tuple(iter_ppl_circuit_coverage_v2_plain_text_documents(target_tokens=2500, seed=2029))

    assert sum(int(document["estimated_tokens"]) for document in documents) >= 2500
    assert len({document["task"] for document in documents}) > 1
    assert {document["task_tier"] for document in documents}.issubset(
        {tier.value for tier in PplCircuitCoverageV2TaskTier}
    )
    assert all(str(document["text"]).endswith("\n") for document in documents)
