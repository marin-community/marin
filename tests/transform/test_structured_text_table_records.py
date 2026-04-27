# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Byte-preservation and structure tests for the table-record serializers.

These datasets arrive as pre-parsed Parquet records. The serializer's one
correctness property is: every cell value that was a string in the record
must survive into the emitted text verbatim — no whitespace munging,
no float round-trip, no case folding.
"""

import gzip
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import Dataset

from marin.datakit.ingestion_manifest import (
    IdentityTreatment,
    IngestionPolicy,
    IngestionSourceManifest,
    SampleCapConfig,
    SecretRedaction,
    StagingMetadata,
    UsagePolicy,
)
from marin.transform.structured_text.table_records import (
    TABLE_CELL_DELIMITER,
    TableRecordStagingConfig,
    serialize_totto_example,
    serialize_wikitablequestions_example,
    stage_table_record_source,
)

# ---------------------------------------------------------------------------
# ToTTo serializer
# ---------------------------------------------------------------------------


def _totto_fixture(target: str = "Some summary sentence.") -> dict:
    return {
        "table_page_title": "List of tallest buildings",
        "table_section_title": "By year",
        "table": [
            [
                {"value": "Year", "is_header": True, "column_span": 1, "row_span": 1},
                {"value": "Building", "is_header": True, "column_span": 1, "row_span": 1},
                {"value": "Height (m)", "is_header": True, "column_span": 1, "row_span": 1},
            ],
            [
                {"value": "1931", "is_header": False, "column_span": 1, "row_span": 1},
                {"value": "Empire State Building", "is_header": False, "column_span": 1, "row_span": 1},
                {"value": "381.0", "is_header": False, "column_span": 1, "row_span": 1},
            ],
            [
                {"value": "1973", "is_header": False, "column_span": 1, "row_span": 1},
                {"value": "Sears Tower", "is_header": False, "column_span": 1, "row_span": 1},
                {"value": "442.1", "is_header": False, "column_span": 1, "row_span": 1},
            ],
        ],
        "highlighted_cells": [[1, 2]],
        "sentence_annotations": {
            "final_sentence": [target, "An alternate phrasing."],
        },
    }


def test_serialize_totto_preserves_cell_strings():
    example = _totto_fixture()
    text = serialize_totto_example(example)

    # Every cell value must appear verbatim
    for expected in ["Year", "Building", "Height (m)", "1931", "381.0", "Empire State Building", "442.1"]:
        assert expected in text, f"cell {expected!r} must survive into the emitted text"

    # Structural separators
    assert TABLE_CELL_DELIMITER in text
    assert "title: List of tallest buildings" in text
    assert "section: By year" in text


def test_serialize_totto_includes_target_sentence():
    example = _totto_fixture(target="The Empire State Building is 381.0 m tall.")
    text = serialize_totto_example(example)
    assert "The Empire State Building is 381.0 m tall." in text


def test_serialize_totto_skips_empty_target():
    example = _totto_fixture()
    example["sentence_annotations"] = {"final_sentence": ["", "   "]}
    text = serialize_totto_example(example)
    # Title + section + table must still appear; target must not add a
    # trailing empty section.
    assert "381.0" in text
    assert not text.endswith("\n\n")


def test_serialize_totto_preserves_numeric_precision_in_cells():
    # If the source stored a float-like string with lots of digits, none
    # of them must be dropped by our serializer.
    example = _totto_fixture()
    example["table"][1][2]["value"] = "3.141592653589793"
    text = serialize_totto_example(example)
    assert "3.141592653589793" in text


def test_serialize_totto_handles_plain_string_cells():
    # Some loaders flatten the cell dicts into plain strings. We accept both.
    example = {
        "table_page_title": "T",
        "table_section_title": "S",
        "table": [["Year", "Building"], ["1931", "Empire State"]],
        "sentence_annotations": {"final_sentence": ["hello"]},
    }
    text = serialize_totto_example(example)
    assert "Year" in text
    assert "Empire State" in text
    assert "hello" in text


# ---------------------------------------------------------------------------
# WikiTableQuestions serializer
# ---------------------------------------------------------------------------


def _wtq_fixture() -> dict:
    return {
        "question": "Which building is taller?",
        "answers": ["Sears Tower"],
        "table": {
            "header": ["Year", "Building", "Height"],
            "rows": [
                ["1931", "Empire State Building", "381.0"],
                ["1973", "Sears Tower", "442.1"],
            ],
        },
    }


def test_serialize_wtq_emits_header_rows_question_answer():
    text = serialize_wikitablequestions_example(_wtq_fixture())
    assert "Year\tBuilding\tHeight" in text
    assert "1931\tEmpire State Building\t381.0" in text
    assert "Q: Which building is taller?" in text
    assert "A: Sears Tower" in text


def test_serialize_wtq_multiple_answers_joined_with_comma():
    example = _wtq_fixture()
    example["answers"] = ["Sears Tower", "Willis Tower"]
    text = serialize_wikitablequestions_example(example)
    assert "A: Sears Tower, Willis Tower" in text


def test_serialize_wtq_preserves_unicode_and_empty_cells():
    example = {
        "question": "What colour?",
        "answers": ["rouge"],
        "table": {
            "header": ["état", "drapeau"],
            "rows": [
                ["Fráncia", "rouge"],
                ["日本", ""],
            ],
        },
    }
    text = serialize_wikitablequestions_example(example)
    assert "état\tdrapeau" in text
    assert "Fráncia" in text
    assert "日本" in text
    # Empty cell must be preserved as an empty field between tab delimiters.
    assert text.endswith("日本\t") or "日本\t\n" in text


def test_serialize_wtq_preserves_numeric_literal_cells():
    example = _wtq_fixture()
    example["table"]["rows"][0][2] = "3.14159265358979323846"
    text = serialize_wikitablequestions_example(example)
    assert "3.14159265358979323846" in text


# ---------------------------------------------------------------------------
# End-to-end staging via a fake HF iterator
# ---------------------------------------------------------------------------


def _read_staged_records(output_path: Path, filename: str = "staged.jsonl.gz") -> list[dict]:
    with gzip.open(output_path / filename, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _manifest(*, serializer_name: str, source_label: str = "wtq:test") -> IngestionSourceManifest:
    return IngestionSourceManifest(
        dataset_key="Stanford/wikitablequestions",
        slice_key="structured_text/wikitablequestions/validation",
        source_label=source_label,
        source_urls=("https://huggingface.co/datasets/Stanford/wikitablequestions",),
        source_license="CC BY 4.0",
        source_format="huggingface_parquet_table_records",
        surface_form="wikipedia_table_tsv_plus_question_answer_lines",
        policy=IngestionPolicy(
            usage_policy=UsagePolicy.EVAL_ONLY,
            use_policy="Eval-only structured-text probe.",
            requires_sanitization=False,
            identity_treatment=IdentityTreatment.PRESERVE,
            secret_redaction=SecretRedaction.NONE,
            contamination_risk="high: held-out eval contamination if reused in training",
            provenance_notes="Pinned HF revision.",
        ),
        staging=StagingMetadata(
            transform_name="stage_table_record_source",
            serializer_name=serializer_name,
            split="validation",
            metadata={"output_filename": "staged.jsonl.gz"},
        ),
        epic_issue=5005,
        issue_numbers=(5059,),
        sample_caps=SampleCapConfig(max_bytes_per_source=30 * 1024 * 1024),
        source_metadata={"hf_revision": "fac45b3184e0ce9b79eecac454acf17e0a51f94e"},
    )


def test_stage_table_record_source_end_to_end_wtq(tmp_path):
    fixtures = [_wtq_fixture() for _ in range(3)]
    manifest = _manifest(serializer_name="wikitablequestions")
    cfg = TableRecordStagingConfig(
        input_path="fake://wtq",
        output_path=str(tmp_path),
        source_label="wtq:test",
        serializer_name="wikitablequestions",
        source_manifest=manifest,
        content_fingerprint=manifest.fingerprint(),
    )

    with patch(
        "marin.transform.structured_text.table_records._load_hf_iterable",
        return_value=iter(fixtures),
    ):
        result = stage_table_record_source(cfg)

    assert result["record_count"] == 3
    assert result["metadata_file"] == str(tmp_path / "metadata.json")
    records = _read_staged_records(tmp_path)
    assert len(records) == 3
    for record in records:
        assert "Q: Which building is taller?" in record["text"]
        assert record["source"] == "wtq:test"
        assert record["provenance"]["serializer"] == "wikitablequestions"
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_manifest"]["source_label"] == "wtq:test"
    assert metadata["source_manifest"]["staging"]["serializer_name"] == "wikitablequestions"
    assert metadata["source_manifest"]["staging"]["metadata"]["output_filename"] == "staged.jsonl.gz"
    assert metadata["materialized_output"]["record_count"] == 3


def test_stage_table_record_source_loads_downloaded_parquet_split(tmp_path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "staged"
    input_dir.mkdir()
    Dataset.from_list([_wtq_fixture()]).to_parquet(input_dir / "validation-00000-of-00001.parquet")
    Dataset.from_list([_wtq_fixture()]).to_parquet(input_dir / "train-00000-of-00001.parquet")

    cfg = TableRecordStagingConfig(
        input_path=str(input_dir),
        output_path=str(output_dir),
        source_label="wtq:test",
        serializer_name="wikitablequestions",
        split="validation",
    )
    result = stage_table_record_source(cfg)

    assert result["record_count"] == 1
    records = _read_staged_records(output_dir)
    assert len(records) == 1
    assert records[0]["id"] == "wtq:test:validation:00000000"


def test_stage_table_record_source_respects_byte_cap(tmp_path):
    # Use many copies; the cap should stop ingestion before all are written.
    fixtures = [_wtq_fixture() for _ in range(500)]
    cfg = TableRecordStagingConfig(
        input_path="fake://wtq",
        output_path=str(tmp_path),
        source_label="wtq:test",
        serializer_name="wikitablequestions",
        max_bytes_per_source=1000,  # tiny cap
    )

    with patch(
        "marin.transform.structured_text.table_records._load_hf_iterable",
        return_value=iter(fixtures),
    ):
        result = stage_table_record_source(cfg)

    # We must have written at least one record (the stop condition doesn't
    # trigger on the very first one) but fewer than the full 500.
    assert 0 < result["record_count"] < 500
    assert result["bytes_written"] >= 0


def test_stage_table_record_source_rejects_unknown_serializer(tmp_path):
    cfg = TableRecordStagingConfig(
        input_path="fake://x",
        output_path=str(tmp_path),
        source_label="x",
        serializer_name="nonexistent",
    )
    with pytest.raises(ValueError, match="Unknown serializer"):
        stage_table_record_source(cfg)
