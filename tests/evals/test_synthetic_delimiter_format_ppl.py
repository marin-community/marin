# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.evals.synthetic_delimiter_format_ppl import (
    iter_synthetic_delimiter_format_records,
    synthetic_delimiter_format_raw_validation_sets,
)

EXPECTED_DELIMITER_KEYS = {
    "synthetic_delimiter_format_ppl/delimited_fields/tsv_next_field",
    "synthetic_delimiter_format_ppl/delimited_fields/csv_next_field",
    "synthetic_delimiter_format_ppl/delimited_fields/rare_control_delimiters",
    "synthetic_delimiter_format_ppl/table_rows/pipe_rows",
    "synthetic_delimiter_format_ppl/table_rows/fixed_width_rows",
    "synthetic_delimiter_format_ppl/table_rows/markdown_table_rows",
    "synthetic_delimiter_format_ppl/whitespace_control/aligned_space_columns",
    "synthetic_delimiter_format_ppl/whitespace_control/python_indentation_or_makefile_tabs",
}


def test_synthetic_delimiter_format_registry_uses_supervised_target_only_format():
    datasets = synthetic_delimiter_format_raw_validation_sets()

    assert set(datasets) == EXPECTED_DELIMITER_KEYS
    for key, dataset in datasets.items():
        assert dataset.hf_dataset_name == key.rsplit("/", maxsplit=1)[-1]
        assert dataset.input_key == "input"
        assert dataset.target_key == "target"
        assert dataset.split == "validation"
        assert "loss:target_only" in dataset.tags


def test_synthetic_delimiter_format_examples_are_raw_continuations_with_non_tiny_targets():
    rows = iter_synthetic_delimiter_format_records(examples_per_config=2)
    instruction_fragments = ("Complete ", "Continue ", "Task:", "User:", "Assistant:")

    assert len(rows) == 16
    for row in rows:
        assert set(row) == {"id", "input", "target", "subset", "task", "seed", "metadata"}
        assert row["input"]
        assert not any(fragment in row["input"] for fragment in instruction_fragments)
        assert row["target"].endswith("\n")
        assert len(row["target"].encode("utf-8")) >= 16


def test_delimiter_examples_score_whole_rows_not_single_fields():
    rows_by_task = {row["task"]: row for row in iter_synthetic_delimiter_format_records(examples_per_config=1)}

    tsv = rows_by_task["tsv_next_field"]
    assert tsv["input"].startswith("record_id\tbatch\tstatus\tchecksum\nrec-0000-0\t")
    assert str(tsv["target"]).startswith("rec-0000-4\t")
    assert str(tsv["target"]).count("\t") == 3

    pipe = rows_by_task["pipe_rows"]
    assert str(pipe["input"]).startswith("node-0 | ")
    assert str(pipe["target"]).startswith("node-4 | ")
    assert str(pipe["target"]).count(" | ") == 2

    indentation = rows_by_task["python_indentation_or_makefile_tabs"]
    assert str(indentation["input"]).startswith("build_example: src/build_example.py\n\tpython -m pytest")
    assert str(indentation["target"]).startswith("\t")
    assert str(indentation["target"]).count("\n") == 2
