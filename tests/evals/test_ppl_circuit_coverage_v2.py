# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import unicodedata

from experiments.evals.ppl_circuit_coverage_v2 import (
    PPL_CIRCUIT_COVERAGE_V2_SLICES,
    PplCircuitCoverageV2Family,
    _generate_borrow_subtraction,
    iter_ppl_circuit_coverage_v2_plain_text_documents,
    iter_ppl_circuit_coverage_v2_records,
    ppl_circuit_coverage_v2_raw_validation_sets,
    render_ppl_circuit_coverage_v2_plain_text_document,
)

STRING_INDEXING_TASK_NAMES = {
    "string_slicing",
    "string_reversal",
    "string_rotation",
    "escape_unescape",
    "chars_vs_bytes",
    "unicode_normalization",
    "unicode_casefolding",
    "character_at_index",
    "all_indices",
    "line_column_offsets",
    "bracket_matching",
}


def test_ppl_circuit_coverage_v2_registry_uses_supervised_target_only_format():
    datasets = ppl_circuit_coverage_v2_raw_validation_sets()

    assert set(datasets) == {slice_.registry_key for slice_ in PPL_CIRCUIT_COVERAGE_V2_SLICES}
    for slice_ in PPL_CIRCUIT_COVERAGE_V2_SLICES:
        dataset = datasets[slice_.registry_key]
        assert dataset.input_key == "input"
        assert dataset.target_key == "target"
        assert "loss:target_only" in dataset.tags
        assert f"family:{slice_.family.value}" in dataset.tags
        assert f"task:{slice_.task_name}" in dataset.tags


def test_ppl_circuit_coverage_v2_slices_include_basic_functionality_families():
    slices_by_family = {
        family: {slice_.task_name for slice_ in PPL_CIRCUIT_COVERAGE_V2_SLICES if slice_.family == family}
        for family in PplCircuitCoverageV2Family
    }

    assert {
        "carry_addition",
        "borrow_subtraction",
        "modular_arithmetic",
        "base_conversion",
        "digit_checksum",
    }.issubset(slices_by_family[PplCircuitCoverageV2Family.ARITHMETIC])
    assert {
        "python_repr_containers",
        "json_field_reemit",
        "csv_tsv_transforms",
    }.issubset(slices_by_family[PplCircuitCoverageV2Family.STRUCTURED_SERIALIZATION])
    assert {
        "stack_push_pop",
        "finite_automata",
        "turtle_commands",
        "regex_lite",
        "brainfuck_lite",
    }.issubset(slices_by_family[PplCircuitCoverageV2Family.STATE_MACHINES])
    assert {
        "markdown_table_padding",
        "line_wrapping",
        "outline_indentation",
    }.issubset(slices_by_family[PplCircuitCoverageV2Family.FORMAT_STYLE_INSTRUCTION])


def test_ppl_circuit_coverage_v2_slices_include_string_and_indexing_families():
    slices_by_family = {
        family: {slice_.task_name for slice_ in PPL_CIRCUIT_COVERAGE_V2_SLICES if slice_.family == family}
        for family in PplCircuitCoverageV2Family
    }

    assert {
        "string_slicing",
        "string_reversal",
        "string_rotation",
        "escape_unescape",
        "chars_vs_bytes",
        "unicode_normalization",
        "unicode_casefolding",
    }.issubset(slices_by_family[PplCircuitCoverageV2Family.STRING_BYTE_TRANSFORMS])
    assert {
        "character_at_index",
        "all_indices",
        "line_column_offsets",
        "bracket_matching",
    }.issubset(slices_by_family[PplCircuitCoverageV2Family.INDEXING_POSITION_TRACKING])


def test_ppl_circuit_coverage_v2_records_are_deterministic_supervised_examples():
    rows = iter_ppl_circuit_coverage_v2_records(examples_per_config=2)
    rows_again = iter_ppl_circuit_coverage_v2_records(examples_per_config=2)

    assert rows == rows_again
    assert len(rows) == 2 * len(PPL_CIRCUIT_COVERAGE_V2_SLICES)
    for row in rows:
        assert set(row) == {"id", "input", "target", "subset", "task", "seed", "tags", "metadata"}
        assert row["input"]
        assert row["target"].endswith("\n")
        assert "loss:target_only" in row["tags"]
        assert row["metadata"]["family"]
        assert row["metadata"]["generator"] == "generated_ppl_circuit_coverage_v2"


def test_new_basic_functionality_slices_use_few_shot_context_without_final_answer_leak():
    new_families = {
        PplCircuitCoverageV2Family.ARITHMETIC.value,
        PplCircuitCoverageV2Family.STRUCTURED_SERIALIZATION.value,
        PplCircuitCoverageV2Family.STATE_MACHINES.value,
        PplCircuitCoverageV2Family.FORMAT_STYLE_INSTRUCTION.value,
    }
    rows = [
        row
        for row in iter_ppl_circuit_coverage_v2_records(examples_per_config=1)
        if row["metadata"]["family"] in new_families
    ]

    assert len(rows) == 16
    for row in rows:
        assert row["input"].startswith("worked examples:\n")
        assert "held-out query:\n" in row["input"]
        held_out_query = row["input"].split("held-out query:\n", maxsplit=1)[1]
        assert held_out_query.rstrip().endswith("answer:")
        assert row["target"].strip() not in {line.strip() for line in held_out_query.splitlines()}
        assert not row["input"].endswith(row["target"])


def test_string_and_indexing_slices_use_few_shot_context_without_final_answer_leak():
    rows = [
        row
        for row in iter_ppl_circuit_coverage_v2_records(examples_per_config=2)
        if row["task"] in STRING_INDEXING_TASK_NAMES
    ]

    assert len(rows) == 2 * len(STRING_INDEXING_TASK_NAMES)
    for row in rows:
        assert row["input"].startswith("worked examples:\n")
        assert "held-out query:\n" in row["input"]
        final_query = row["input"].split("held-out query:\n", maxsplit=1)[1]
        assert final_query.rstrip().endswith("answer:")
        assert not row["input"].endswith(row["target"])


def test_selected_new_basic_functionality_targets_match_metadata():
    rows = {row["task"]: row for row in iter_ppl_circuit_coverage_v2_records(examples_per_config=1)}

    carry = rows["carry_addition"]
    assert carry["target"] == f"{carry['metadata']['left'] + carry['metadata']['right']}\n"
    assert carry["metadata"]["carry_positions"]

    borrow = rows["borrow_subtraction"]
    assert borrow["target"] == f"{borrow['metadata']['left'] - borrow['metadata']['right']}\n"
    assert borrow["metadata"]["borrow_positions"]

    checksum = rows["digit_checksum"]
    checksum_value = (
        sum(
            digit * weight
            for digit, weight in zip(checksum["metadata"]["digits"], checksum["metadata"]["weights"], strict=True)
        )
        % 10
    )
    assert checksum["target"] == f"{checksum_value}\n"

    finite_automata = rows["finite_automata"]
    dfa_target = json.loads(finite_automata["target"])
    bits = finite_automata["metadata"]["bits"]
    assert dfa_target == {
        "state": f"ones_{bits.count('1') % 2}_zeros_{bits.count('0') % 3}",
        "accept": bits.count("1") % 2 == 0 and bits.count("0") % 3 == 0,
    }

    stack = rows["stack_push_pop"]
    expected_stack = []
    for operation in stack["metadata"]["operations"]:
        if operation == "pop":
            expected_stack.pop()
        else:
            expected_stack.append(operation.split(" ", maxsplit=1)[1])
    assert stack["target"] == f"{expected_stack!r}\n"


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
    assert row["target"] == "1\n"


def test_string_transform_targets_match_metadata():
    rows = [
        row
        for row in iter_ppl_circuit_coverage_v2_records(examples_per_config=4, seed=2026)
        if row["metadata"]["family"] == PplCircuitCoverageV2Family.STRING_BYTE_TRANSFORMS.value
    ]

    assert {row["task"] for row in rows} == {
        "string_slicing",
        "string_reversal",
        "string_rotation",
        "escape_unescape",
        "chars_vs_bytes",
        "unicode_normalization",
        "unicode_casefolding",
    }
    for row in rows:
        metadata = row["metadata"]
        task = row["task"]
        target = row["target"]
        text = metadata["text"]

        if task == "string_slicing":
            expected = text[metadata["start"] : metadata["stop"] : metadata["step"]]
            assert target == repr(expected) + "\n"
        elif task == "string_reversal":
            assert target == repr(text[::-1]) + "\n"
        elif task == "string_rotation":
            amount = metadata["amount"]
            if metadata["operation"] == "rotate_left":
                expected = text[amount:] + text[:amount]
            else:
                expected = text[-amount:] + text[:-amount]
            assert target == repr(expected) + "\n"
        elif task == "escape_unescape":
            if metadata["operation"] == "json_escape":
                assert target == json.dumps(text, ensure_ascii=False) + "\n"
            else:
                assert target == repr(text) + "\n"
        elif task == "chars_vs_bytes":
            assert json.loads(target) == {"chars": len(text), "utf8_bytes": len(text.encode("utf-8"))}
        elif task == "unicode_normalization":
            assert target == repr(unicodedata.normalize(metadata["form"], text)) + "\n"
        elif task == "unicode_casefolding":
            assert target == repr(text.casefold()) + "\n"
        else:
            raise AssertionError(f"unexpected string transform task: {task}")


def test_indexing_targets_match_metadata():
    rows = [
        row
        for row in iter_ppl_circuit_coverage_v2_records(examples_per_config=4, seed=2027)
        if row["metadata"]["family"] == PplCircuitCoverageV2Family.INDEXING_POSITION_TRACKING.value
    ]

    assert {row["task"] for row in rows} == {
        "character_at_index",
        "all_indices",
        "line_column_offsets",
        "bracket_matching",
    }
    for row in rows:
        metadata = row["metadata"]
        task = row["task"]
        target = row["target"]
        text = metadata["text"]

        if task == "character_at_index":
            assert target == repr(text[metadata["index"]]) + "\n"
        elif task == "all_indices":
            needle = metadata["needle"]
            expected = [
                index for index in range(len(text) - len(needle) + 1) if text[index : index + len(needle)] == needle
            ]
            assert target == repr(expected) + "\n"
        elif task == "line_column_offsets":
            if metadata["operation"] == "offset_to_line_col":
                assert json.loads(target) == {"line": metadata["line"], "column": metadata["column"]}
            else:
                assert target == f"{metadata['offset']}\n"
        elif task == "bracket_matching":
            assert target == f"{metadata['matching_index']}\n"
        else:
            raise AssertionError(f"unexpected indexing task: {task}")


def test_format_style_instruction_targets_match_metadata():
    rows = [
        row
        for row in iter_ppl_circuit_coverage_v2_records(examples_per_config=3, seed=2028)
        if row["metadata"]["family"] == PplCircuitCoverageV2Family.FORMAT_STYLE_INSTRUCTION.value
    ]

    assert {row["task"] for row in rows} == {
        "markdown_table_padding",
        "line_wrapping",
        "outline_indentation",
    }
    for row in rows:
        metadata = row["metadata"]
        target = row["target"].rstrip("\n")
        if row["task"] == "markdown_table_padding":
            lines = target.splitlines()
            assert len(lines) >= 5
            assert all(line.startswith("| ") and line.endswith(" |") for line in lines)
            assert len({len(line) for line in lines}) == 1
        elif row["task"] == "line_wrapping":
            assert all(len(line) <= metadata["width"] for line in target.splitlines())
            assert " ".join(target.splitlines()) == metadata["paragraph"]
        elif row["task"] == "outline_indentation":
            expected = "\n".join(f"{'  ' * level}- {label}" for level, label in metadata["entries"])
            assert target == expected
        else:
            raise AssertionError(f"unexpected format/style task: {row['task']}")


def test_plain_text_pretraining_documents_use_the_supervised_prompt_template():
    row = next(
        row
        for row in iter_ppl_circuit_coverage_v2_records(examples_per_config=1)
        if row["task"] == "markdown_table_padding"
    )

    text = render_ppl_circuit_coverage_v2_plain_text_document(row)

    assert "### Circuit practice example" in text
    assert "Family: format_style_instruction" in text
    assert "Task: markdown_table_padding" in text
    assert "Final answer:" not in text
    assert row["input"] + row["target"] in text
    assert row["target"].strip() in text
    assert row["target"].strip() not in row["input"]


def test_plain_text_pretraining_stream_reaches_approximate_token_budget():
    documents = tuple(iter_ppl_circuit_coverage_v2_plain_text_documents(target_tokens=2500, seed=2029))

    assert documents
    assert sum(int(document["estimated_tokens"]) for document in documents) >= 2500
    assert {document["family"] for document in documents} >= {
        PplCircuitCoverageV2Family.FORMAT_STYLE_INSTRUCTION.value,
        PplCircuitCoverageV2Family.STRING_BYTE_TRANSFORMS.value,
    }
    for document in documents:
        assert set(document) == {
            "id",
            "text",
            "estimated_tokens",
            "source",
            "family",
            "task",
            "seed",
            "supervised_record_id",
        }
        assert document["text"].endswith("\n")
