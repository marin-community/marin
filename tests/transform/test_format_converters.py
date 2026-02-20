# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for format conversion transforms to Dolma."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from marin.transform.evaluation.eval_to_dolma import ConvertEvalToDolmaConfig, convert_eval_to_dolma
from marin.transform.lingoly.to_dolma import ConvertLingolyToDolmaConfig, convert_lingoly_to_dolma
from marin.transform.medical.lavita_to_dolma import LavitaToDolmaConfig, convert_lavita_split_to_dolma


def _read_jsonl_gz(path: Path) -> list[dict]:
    """Read records from a gzipped JSONL file."""
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _read_jsonl(path: Path) -> list[dict]:
    """Read records from a JSONL file."""
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _read_parquet(path: Path) -> list[dict]:
    """Read records from a parquet file."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pylist()


def test_lingoly_to_dolma(tmp_path: Path, create_zip) -> None:
    """Test LingOly zip to Dolma conversion with chunking."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    # Create sample LingOly data with questions structure
    lingoly_records = [
        {
            "preamble": "This is a preamble about linguistics.",
            "context": "Context: A study on language evolution.",
            "questions": json.dumps(
                [
                    {
                        "prompt": "Question 1: What is the main topic?",
                        "subprompts": [
                            {"question": "Subquestion 1.1: Define linguistics."},
                            {"question": "Subquestion 1.2: What is evolution?"},
                        ],
                    },
                    {
                        "prompt": "Question 2: Why is this important?",
                        "subprompts": [
                            {"question": "Subquestion 2.1: Explain significance."},
                        ],
                    },
                ]
            ),
        },
        {
            "preamble": "Second preamble text.",
            "context": "Second context information.",
            "questions": [  # Test list form too
                {
                    "prompt": "Question 3: How does this work?",
                    "subprompts": [
                        {"question": "Subquestion 3.1: Describe the mechanism."},
                    ],
                }
            ],
        },
    ]

    # Create JSONL content for test.jsonl
    jsonl_content = "\n".join(json.dumps(r) for r in lingoly_records)

    # Create zip file with test.jsonl inside
    zip_path = input_dir / "lingoly.zip"
    create_zip(zip_path, {"test.jsonl": jsonl_content})

    config = ConvertLingolyToDolmaConfig(
        input_path=str(zip_path),
        output_path=str(output_dir),
        max_doc_length=200,  # Small length to test chunking
    )

    convert_lingoly_to_dolma(config)

    # Read output files
    output_files = list(output_dir.glob("*.jsonl"))
    assert len(output_files) > 0, "Expected at least one output file"

    all_records = []
    for output_file in output_files:
        all_records.extend(_read_jsonl(output_file))

    # Verify that text field exists and contains expected content
    assert len(all_records) > 0, "Expected at least one output record"

    for record in all_records:
        assert "text" in record, "Missing 'text' field in Dolma record"
        assert len(record["text"]) > 0, "Text field should not be empty"

    # Verify chunking occurred (should have multiple records due to small max_doc_length)
    assert len(all_records) >= 2, "Expected multiple chunks due to small max_doc_length"

    # Verify content includes preamble, context, and questions
    combined_text = "".join(r["text"] for r in all_records)
    assert "preamble about linguistics" in combined_text
    assert "Context: A study on language evolution" in combined_text
    assert "Question 1: What is the main topic?" in combined_text
    assert "Subquestion 1.1: Define linguistics." in combined_text


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


def test_eval_to_dolma(tmp_path: Path, write_jsonl_gz) -> None:
    """Test evaluation dataset to Dolma conversion."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    # Create sample eval records
    eval_records = [
        {
            "prompt": "What is the capital of France?",
            "response": "The capital of France is Paris.",
        },
        {
            "prompt": "Explain photosynthesis.",
            "response": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
        },
    ]

    # Write gzipped JSONL file
    input_file = input_dir / "eval_data.jsonl.gz"
    write_jsonl_gz(input_file, eval_records)

    config = ConvertEvalToDolmaConfig(
        input_path=str(input_dir),
        output_path=str(output_dir),
    )

    convert_eval_to_dolma(config)

    # Read output files
    output_files = list(output_dir.glob("*.jsonl.gz"))
    assert len(output_files) > 0, "Expected at least one output file"

    all_records = []
    for output_file in output_files:
        all_records.extend(_read_jsonl_gz(output_file))

    assert len(all_records) == 2, "Expected 2 output records"

    # Verify text field contains prompt + response
    for record in all_records:
        assert "text" in record, "Missing 'text' field"
        assert "prompt" in record, "Original 'prompt' field should be preserved"
        assert "response" in record, "Original 'response' field should be preserved"

    # Verify first record
    first = all_records[0]
    assert first["text"] == "What is the capital of France?\nThe capital of France is Paris."
    assert first["prompt"] == "What is the capital of France?"
    assert first["response"] == "The capital of France is Paris."

    # Verify second record
    second = all_records[1]
    expected_text = (
        "Explain photosynthesis.\n"
        "Photosynthesis is the process by which plants convert light energy into chemical energy."
    )
    assert second["text"] == expected_text
    assert second["prompt"] == "Explain photosynthesis."
    assert (
        second["response"] == "Photosynthesis is the process by which plants convert light energy into chemical energy."
    )
