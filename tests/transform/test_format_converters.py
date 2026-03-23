# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for format conversion transforms to Dolma."""

from __future__ import annotations

import hashlib
from pathlib import Path

from marin.transform.medical.lavita_to_dolma import LavitaToDolmaConfig, convert_lavita_split_to_dolma


def _read_parquet(path: Path) -> list[dict]:
    """Read records from a parquet file."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pylist()


def test_lavita_pubmedqa_to_dolma(tmp_path: Path, write_jsonl_gz) -> None:
    """Test LaVita pubmed-qa subset to Dolma conversion."""
    input_dir = tmp_path / "input" / "pubmed-qa"
    output_dir = tmp_path / "output"

    # Create sample pubmed-qa records
    pubmedqa_records = [
        {
            "CONTEXTS": [
                "Background: This study examines heart disease.",
                "Methods: We used a cohort study design.",
                "Results: We found significant associations.",
            ],
            "QUESTION": "Is there a link between diet and heart disease?",
            "final_decision": "yes",
            "LONG_ANSWER": (
                "Our study demonstrates a clear association between dietary patterns and cardiovascular outcomes."
            ),
        },
        {
            "CONTEXTS": [
                "Background: Cancer research is critical.",
            ],
            "QUESTION": "Does chemotherapy improve survival?",
            "final_decision": "maybe",
            "LONG_ANSWER": "The evidence is mixed, with some studies showing benefits and others showing no effect.",
        },
    ]

    # Write parquet file
    import pyarrow as pa
    import pyarrow.parquet as pq

    input_file = input_dir / "data-train-0000.parquet"
    input_file.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(pubmedqa_records)
    pq.write_table(table, input_file)

    config = LavitaToDolmaConfig(
        input_path=str(tmp_path / "input"),
        output_path=str(output_dir),
        subset="pubmed-qa",
        split="train",
    )

    convert_lavita_split_to_dolma(config)

    # Read output files
    output_files = list(output_dir.glob("*.parquet"))
    assert len(output_files) > 0, "Expected at least one output file"

    all_records = []
    for output_file in output_files:
        all_records.extend(_read_parquet(output_file))

    assert len(all_records) == 2, "Expected 2 output records"

    # Verify Dolma schema
    for record in all_records:
        assert "id" in record, "Missing 'id' field"
        assert "text" in record, "Missing 'text' field"
        assert "source" in record, "Missing 'source' field"
        assert record["source"] == "lavita/medical-qa-datasets/pubmed-qa"

    # Verify first record
    first = all_records[0]
    assert "Background: This study examines heart disease." in first["text"]
    assert "Is there a link between diet and heart disease?" in first["text"]
    assert "yes" in first["text"]
    assert "cardiovascular outcomes" in first["text"]

    # Verify ID is a hash
    expected_id = hashlib.sha256(
        (
            "\n".join(pubmedqa_records[0]["CONTEXTS"])
            + pubmedqa_records[0]["QUESTION"]
            + pubmedqa_records[0]["final_decision"]
            + pubmedqa_records[0]["LONG_ANSWER"]
        ).encode("utf-8")
    ).hexdigest()
    assert first["id"] == expected_id


def test_lavita_medmcqa_to_dolma(tmp_path: Path) -> None:
    """Test LaVita medmcqa subset to Dolma conversion."""
    input_dir = tmp_path / "input" / "medmcqa"
    output_dir = tmp_path / "output"

    # Create sample medmcqa records
    medmcqa_records = [
        {
            "id": "mcqa-001",
            "question": "What is the most common cause of pneumonia?",
            "opa": "Streptococcus pneumoniae",
            "opb": "Haemophilus influenzae",
            "opc": "Mycoplasma pneumoniae",
            "opd": "Legionella pneumophila",
            "cop": 0,  # Correct option is 'a'
            "exp": "Streptococcus pneumoniae is the most common bacterial cause of community-acquired pneumonia.",
        },
        {
            "id": "mcqa-002",
            "question": "Which vitamin deficiency causes scurvy?",
            "opa": "Vitamin A",
            "opb": "Vitamin B12",
            "opc": "Vitamin C",
            "opd": "Vitamin D",
            "cop": 2,  # Correct option is 'c'
            "exp": None,  # Test null explanation
        },
    ]

    # Write parquet file
    import pyarrow as pa
    import pyarrow.parquet as pq

    input_file = input_dir / "data-train-0000.parquet"
    input_file.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(medmcqa_records)
    pq.write_table(table, input_file)

    config = LavitaToDolmaConfig(
        input_path=str(tmp_path / "input"),
        output_path=str(output_dir),
        subset="medmcqa",
        split="train",
    )

    convert_lavita_split_to_dolma(config)

    # Read output files
    output_files = list(output_dir.glob("*.parquet"))
    assert len(output_files) > 0, "Expected at least one output file"

    all_records = []
    for output_file in output_files:
        all_records.extend(_read_parquet(output_file))

    assert len(all_records) == 2, "Expected 2 output records"

    # Verify Dolma schema
    for record in all_records:
        assert "id" in record, "Missing 'id' field"
        assert "text" in record, "Missing 'text' field"
        assert "source" in record, "Missing 'source' field"
        assert record["source"] == "lavita/medical-qa-datasets/medmcqa"

    # Verify first record
    first = all_records[0]
    assert first["id"] == "mcqa-001"
    assert "What is the most common cause of pneumonia?" in first["text"]
    assert "a. Streptococcus pneumoniae" in first["text"]
    assert "Answer: a" in first["text"]
    assert "most common bacterial cause" in first["text"]

    # Verify second record (with null explanation)
    second = all_records[1]
    assert second["id"] == "mcqa-002"
    assert "Which vitamin deficiency causes scurvy?" in second["text"]
    assert "c. Vitamin C" in second["text"]
    assert "Answer: c" in second["text"]
    # Should not crash with null explanation
    assert len(second["text"]) > 0


def test_lavita_allprocessed_to_dolma(tmp_path: Path) -> None:
    """Test LaVita all-processed subset to Dolma conversion."""
    input_dir = tmp_path / "input" / "all-processed"
    output_dir = tmp_path / "output"

    # Create sample all-processed records
    allprocessed_records = [
        {
            "instruction": "Explain the mechanism of action of aspirin.",
            "input": "Aspirin is a non-steroidal anti-inflammatory drug (NSAID).",
            "output": "Aspirin works by inhibiting cyclooxygenase (COX) enzymes, which reduces prostaglandin synthesis.",
        },
        {
            "instruction": "Describe the symptoms of diabetes.",
            "input": "Type 2 diabetes is a metabolic disorder.",
            "output": "Common symptoms include increased thirst, frequent urination, and unexplained weight loss.",
        },
    ]

    # Write parquet file
    import pyarrow as pa
    import pyarrow.parquet as pq

    input_file = input_dir / "data-train-0000.parquet"
    input_file.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(allprocessed_records)
    pq.write_table(table, input_file)

    config = LavitaToDolmaConfig(
        input_path=str(tmp_path / "input"),
        output_path=str(output_dir),
        subset="all-processed",
        split="train",
    )

    convert_lavita_split_to_dolma(config)

    # Read output files
    output_files = list(output_dir.glob("*.parquet"))
    assert len(output_files) > 0, "Expected at least one output file"

    all_records = []
    for output_file in output_files:
        all_records.extend(_read_parquet(output_file))

    assert len(all_records) == 2, "Expected 2 output records"

    # Verify Dolma schema
    for record in all_records:
        assert "id" in record, "Missing 'id' field"
        assert "text" in record, "Missing 'text' field"
        assert "source" in record, "Missing 'source' field"
        assert record["source"] == "lavita/medical-qa-datasets/all-processed"

    # Verify first record
    first = all_records[0]
    assert "Explain the mechanism of action of aspirin." in first["text"]
    assert "Context:" in first["text"]
    assert "Aspirin is a non-steroidal anti-inflammatory drug" in first["text"]
    assert "Answer:" in first["text"]
    assert "inhibiting cyclooxygenase" in first["text"]

    # Verify ID is a hash
    expected_id = hashlib.sha256(
        (
            allprocessed_records[0]["instruction"] + allprocessed_records[0]["input"] + allprocessed_records[0]["output"]
        ).encode("utf-8")
    ).hexdigest()
    assert first["id"] == expected_id
