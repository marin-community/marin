# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for legal dataset transforms."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from marin.transform.legal.transform_australianlegalcorpus import (
    Config as AustralianLegalCorpusConfig,
    main as transform_australianlegalcorpus,
)
from marin.transform.legal.transform_edgar import Config as EdgarConfig, main as transform_edgar
from marin.transform.legal.transform_hupd import Config as HupdConfig, main as transform_hupd
from marin.transform.legal.transform_multilegalpile import (
    Config as MultiLegalPileConfig,
    main as transform_multilegalpile,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record))
            f.write("\n")


def write_jsonl_xz(path: Path, records: list[dict]) -> None:
    """Write records to a JSONL.xz file."""
    import lzma

    path.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(path, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record))
            f.write("\n")


def write_parquet(path: Path, records: list[dict]) -> None:
    """Write records to a Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)


def test_transform_hupd(tmp_path: Path, create_tar_gz, read_all_jsonl_gz, validate_dolma_record) -> None:
    """Test HUPD transform converts tar.gz with JSON patent records to Dolma format."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    # Create sample HUPD records
    hupd_records = [
        {
            "application_number": "US-12345678-A",
            "title": "Improved Widget Design",
            "abstract": "An improved widget with enhanced functionality.",
            "claims": "1. A widget comprising...\n2. The widget of claim 1...",
            "full_description": "This invention relates to widgets and more particularly to improved widget designs.",
            "background": "Prior art widgets have limitations...",
            "summary": "The invention provides an improved widget...",
            "publication_number": "US-2020-0123456-A1",
            "decision": "ACCEPTED",
            "date_published": "2020-04-16",
            "date_produced": "2020-04-16",
            "main_cpc_label": "A01B",
            "cpc_labels": ["A01B", "A01C"],
            "main_ipcr_label": "A01B1/00",
            "ipcr_labels": ["A01B1/00", "A01C1/00"],
            "patent_number": "US-10987654-B2",
            "filing_date": "2018-05-01",
            "patent_issue_date": "2021-03-15",
            "abandon_date": None,
            "uspc_class": "171",
            "uspc_subclass": "1",
            "examiner_id": "12345",
            "examiner_name_last": "Smith",
            "examiner_name_first": "Jane",
            "examiner_name_middle": "A",
        },
        {
            "application_number": "US-87654321-A",
            "title": "Novel Process for Manufacturing",
            "abstract": "A novel manufacturing process with reduced costs.",
            "claims": "1. A method comprising the steps of...",
            "full_description": "The present invention relates to manufacturing processes.",
            "background": "Existing processes are inefficient...",
            "summary": "A more efficient process is provided...",
            "publication_number": "US-2019-0987654-A1",
            "decision": "REJECTED",
            "date_published": "2019-12-10",
            "date_produced": "2019-12-10",
            "main_cpc_label": "B01D",
            "cpc_labels": ["B01D", "B01F"],
            "main_ipcr_label": "B01D1/00",
            "ipcr_labels": ["B01D1/00"],
            "patent_number": None,
            "filing_date": "2018-08-20",
            "patent_issue_date": None,
            "abandon_date": "2020-05-15",
            "uspc_class": "210",
            "uspc_subclass": "2",
            "examiner_id": "67890",
            "examiner_name_last": "Johnson",
            "examiner_name_first": "Robert",
            "examiner_name_middle": "B",
        },
    ]

    # Create tar.gz with JSON files
    tar_path = input_dir / "2020.tar.gz"
    records_by_filename = {
        "2020/patent1.json": [hupd_records[0]],
        "2020/patent2.json": [hupd_records[1]],
    }
    create_tar_gz(tar_path, records_by_filename)

    # Run transform
    config = HupdConfig(
        input_path=str(input_dir),
        output_path=str(output_dir),
    )
    transform_hupd(config)

    # Verify output
    output_records = read_all_jsonl_gz(output_dir)
    assert len(output_records) == 2, f"Expected 2 records, got {len(output_records)}"

    # Validate Dolma schema
    for record in output_records:
        validate_dolma_record(record)

    # Check specific fields
    record_by_id = {r["id"]: r for r in output_records}
    assert "US-12345678-A" in record_by_id
    assert "US-87654321-A" in record_by_id

    # Validate first record
    r1 = record_by_id["US-12345678-A"]
    assert r1["source"] == "hupd"
    assert r1["created"] == "2020-04-16"
    assert "Title:\nImproved Widget Design" in r1["text"]
    assert "Abstract:\nAn improved widget" in r1["text"]
    assert "Claims:\n1. A widget comprising" in r1["text"]
    assert "Full Description:\nThis invention relates to widgets" in r1["text"]
    assert r1["metadata"]["publication_number"] == "US-2020-0123456-A1"
    assert r1["metadata"]["decision"] == "ACCEPTED"
    assert r1["metadata"]["main_cpc_label"] == "A01B"

    # Validate second record
    r2 = record_by_id["US-87654321-A"]
    assert r2["source"] == "hupd"
    assert r2["created"] == "2019-12-10"
    assert r2["metadata"]["decision"] == "REJECTED"
    assert r2["metadata"]["abandon_date"] == "2020-05-15"


def test_transform_edgar(tmp_path: Path, read_all_jsonl_gz, validate_dolma_record) -> None:
    """Test EDGAR transform converts parquet files to Dolma format."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    # Create sample EDGAR records with all required sections
    edgar_records = [
        {
            "cik": "0000012345",
            "year": 2020,
            "filename": "edgar_filing_2020_001.txt",
            "section_1": "Business Description: We manufacture widgets.",
            "section_1A": "Risk Factors: Market risks exist.",
            "section_1B": "Unresolved Staff Comments: None.",
            "section_2": "Properties: We own facilities in multiple states.",
            "section_3": "Legal Proceedings: No material proceedings.",
            "section_4": "Mine Safety Disclosures: Not applicable.",
            "section_5": "Market for Registrant's Common Equity.",
            "section_6": "Selected Financial Data.",
            "section_7": "Management's Discussion and Analysis.",
            "section_7A": "Quantitative and Qualitative Disclosures.",
            "section_8": "Financial Statements.",
            "section_9": "Changes in Accounting.",
            "section_9A": "Controls and Procedures.",
            "section_9B": "Other Information.",
            "section_10": "Directors and Officers.",
            "section_11": "Executive Compensation.",
            "section_12": "Security Ownership.",
            "section_13": "Certain Relationships.",
            "section_14": "Principal Accountant Fees.",
            "section_15": "Exhibits.",
        },
        {
            "cik": "0000098765",
            "year": 2021,
            "filename": "edgar_filing_2021_002.txt",
            "section_1": "Business: Technology company.",
            "section_1A": "Risks: Technological obsolescence.",
            "section_1B": "Comments: None.",
            "section_2": "Properties: Office space leased.",
            "section_3": "Legal: Patent litigation ongoing.",
            "section_4": "Mine Safety: N/A.",
            "section_5": "Market data.",
            "section_6": "Financial highlights.",
            "section_7": "MD&A discussion.",
            "section_7A": "Market risk disclosures.",
            "section_8": "Consolidated statements.",
            "section_9": "Accounting changes.",
            "section_9A": "Internal controls.",
            "section_9B": "Other matters.",
            "section_10": "Directors listed.",
            "section_11": "Compensation tables.",
            "section_12": "Ownership details.",
            "section_13": "Related transactions.",
            "section_14": "Audit fees.",
            "section_15": "Exhibit index.",
        },
    ]

    # Write parquet file
    parquet_path = input_dir / "edgar_2020.parquet"
    write_parquet(parquet_path, edgar_records)

    # Run transform
    config = EdgarConfig(
        input_path=str(input_dir),
        output_path=str(output_dir),
    )
    transform_edgar(config)

    # Verify output
    output_records = read_all_jsonl_gz(output_dir)
    assert len(output_records) == 2, f"Expected 2 records, got {len(output_records)}"

    # Validate Dolma schema
    for record in output_records:
        validate_dolma_record(record)

    # Check specific fields
    record_by_id = {r["id"]: r for r in output_records}
    assert "0000012345" in record_by_id
    assert "0000098765" in record_by_id

    # Validate first record
    r1 = record_by_id["0000012345"]
    assert r1["source"] == "edgar"
    assert r1["metadata"]["year"] == 2020
    assert r1["metadata"]["filename"] == "edgar_filing_2020_001.txt"
    # Check that sections are joined
    assert "Business Description: We manufacture widgets." in r1["text"]
    assert "Risk Factors: Market risks exist." in r1["text"]
    assert "Properties: We own facilities" in r1["text"]

    # Validate second record
    r2 = record_by_id["0000098765"]
    assert r2["source"] == "edgar"
    assert r2["metadata"]["year"] == 2021
    assert "Business: Technology company." in r2["text"]
    assert "Legal: Patent litigation ongoing." in r2["text"]


def test_transform_multilegalpile(tmp_path: Path, read_all_jsonl_gz, validate_dolma_record) -> None:
    """Test MultiLegalPile transform converts JSONL.xz files to Dolma format."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    # Create sample MultiLegalPile records
    multilegalpile_records = [
        {
            "text": "This is a legal statute from the European Union regarding data protection.",
            "type": "legislation",
            "jurisdiction": "EU",
        },
        {
            "text": "Court decision on intellectual property rights in Germany.",
            "type": "case_law",
            "jurisdiction": "Germany",
        },
        {
            "text": "French administrative regulation on environmental standards.",
            "type": "regulation",
            "jurisdiction": "France",
        },
    ]

    # Write JSONL.xz file
    jsonl_xz_path = input_dir / "legal_docs.jsonl.xz"
    write_jsonl_xz(jsonl_xz_path, multilegalpile_records)

    # Run transform
    config = MultiLegalPileConfig(
        input_path=str(input_dir),
        output_path=str(output_dir),
    )
    transform_multilegalpile(config)

    # Verify output
    output_records = read_all_jsonl_gz(output_dir)
    assert len(output_records) == 3, f"Expected 3 records, got {len(output_records)}"

    # Validate Dolma schema
    for record in output_records:
        validate_dolma_record(record)

    # Check that all records have proper fields
    for record in output_records:
        assert record["source"] == "multilegalpile"
        assert "id" in record
        assert len(record["id"]) == 64  # SHA256 hex digest
        assert "metadata" in record
        assert "type" in record["metadata"]
        assert "jurisdiction" in record["metadata"]

    # Check that texts are preserved
    output_texts = {r["text"] for r in output_records}
    input_texts = {r["text"] for r in multilegalpile_records}
    assert output_texts == input_texts

    # Verify metadata preservation
    for output_record in output_records:
        # Find matching input record by text
        matching_input = next(r for r in multilegalpile_records if r["text"] == output_record["text"])
        assert output_record["metadata"]["type"] == matching_input["type"]
        assert output_record["metadata"]["jurisdiction"] == matching_input["jurisdiction"]


def test_transform_australianlegalcorpus(tmp_path: Path, read_all_jsonl_gz, validate_dolma_record) -> None:
    """Test Australian Legal Corpus transform converts JSONL files to Dolma format."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    # Create sample Australian Legal Corpus records
    australian_records = [
        {
            "version_id": "auspol_12345_v1",
            "text": "High Court of Australia decision on constitutional matters.",
            "type": "decision",
            "jurisdiction": "Commonwealth",
            "source": "High Court of Australia",
            "citation": "[2020] HCA 15",
            "date": "2020-05-12",
            "when_scraped": "2021-03-01T10:30:00Z",
            "mime": "text/html",
            "url": "https://www.austlii.edu.au/cgi-bin/viewdoc/au/cases/cth/HCA/2020/15.html",
        },
        {
            "version_id": "nswleg_67890_v2",
            "text": "New South Wales legislation on planning and development.",
            "type": "legislation",
            "jurisdiction": "New South Wales",
            "source": "NSW Legislation",
            "citation": "Environmental Planning and Assessment Act 1979",
            "date": "1979-09-01",
            "when_scraped": "2021-03-05T14:20:00Z",
            "mime": "text/plain",
            "url": "https://legislation.nsw.gov.au/view/html/inforce/current/act-1979-203",
        },
        {
            "version_id": "viccase_54321_v1",
            "text": "Victorian Supreme Court ruling on contractual disputes.",
            "type": "case",
            "jurisdiction": "Victoria",
            "source": "Supreme Court of Victoria",
            "citation": "[2019] VSC 234",
            "date": "2019-11-20",
            "when_scraped": "2021-02-28T09:15:00Z",
            "mime": "application/pdf",
            "url": "https://www.austlii.edu.au/cgi-bin/viewdoc/au/cases/vic/VSC/2019/234.html",
        },
    ]

    # Write JSONL file
    jsonl_path = input_dir / "australian_legal.jsonl"
    write_jsonl(jsonl_path, australian_records)

    # Run transform
    config = AustralianLegalCorpusConfig(
        input_path=str(input_dir),
        output_path=str(output_dir),
    )
    transform_australianlegalcorpus(config)

    # Verify output
    output_records = read_all_jsonl_gz(output_dir)
    assert len(output_records) == 3, f"Expected 3 records, got {len(output_records)}"

    # Validate Dolma schema
    for record in output_records:
        validate_dolma_record(record)

    # Check specific fields
    record_by_id = {r["id"]: r for r in output_records}
    assert "auspol_12345_v1" in record_by_id
    assert "nswleg_67890_v2" in record_by_id
    assert "viccase_54321_v1" in record_by_id

    # Validate first record
    r1 = record_by_id["auspol_12345_v1"]
    assert r1["source"] == "australianlegalcorpus"
    assert r1["text"] == "High Court of Australia decision on constitutional matters."
    assert r1["created"] == "2020-05-12"
    assert r1["added"] == "2021-03-01T10:30:00Z"
    assert r1["metadata"]["type"] == "decision"
    assert r1["metadata"]["jurisdiction"] == "Commonwealth"
    assert r1["metadata"]["citation"] == "[2020] HCA 15"
    assert r1["metadata"]["source"] == "High Court of Australia"
    assert r1["metadata"]["mime"] == "text/html"
    assert "austlii.edu.au" in r1["metadata"]["url"]

    # Validate second record
    r2 = record_by_id["nswleg_67890_v2"]
    assert r2["source"] == "australianlegalcorpus"
    assert r2["metadata"]["type"] == "legislation"
    assert r2["metadata"]["jurisdiction"] == "New South Wales"
    assert r2["created"] == "1979-09-01"

    # Validate third record
    r3 = record_by_id["viccase_54321_v1"]
    assert r3["source"] == "australianlegalcorpus"
    assert r3["metadata"]["type"] == "case"
    assert r3["metadata"]["jurisdiction"] == "Victoria"


def test_transform_idempotency(tmp_path: Path, create_tar_gz, read_all_jsonl_gz) -> None:
    """Test that running transforms twice doesn't duplicate output files."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    # Create minimal HUPD test data
    hupd_record = {
        "application_number": "US-TEST-001",
        "title": "Test Patent",
        "abstract": "Test abstract",
        "claims": "Test claims",
        "full_description": "Test description",
        "background": "Test background",
        "summary": "Test summary",
        "publication_number": "US-TEST-PUB",
        "decision": "PENDING",
        "date_published": "2020-01-01",
        "date_produced": "2020-01-01",
        "main_cpc_label": "A01B",
        "cpc_labels": ["A01B"],
        "main_ipcr_label": "A01B1/00",
        "ipcr_labels": ["A01B1/00"],
        "patent_number": None,
        "filing_date": "2019-01-01",
        "patent_issue_date": None,
        "abandon_date": None,
        "uspc_class": "171",
        "uspc_subclass": "1",
        "examiner_id": "999",
        "examiner_name_last": "Tester",
        "examiner_name_first": "Test",
        "examiner_name_middle": "T",
    }

    tar_path = input_dir / "test.tar.gz"
    create_tar_gz(tar_path, {"test/patent.json": [hupd_record]})

    config = HupdConfig(
        input_path=str(input_dir),
        output_path=str(output_dir),
    )

    # Run transform first time
    transform_hupd(config)

    # Get output files and records after first run
    first_run_files = sorted(output_dir.glob("*.jsonl.gz"))
    first_run_records = read_all_jsonl_gz(output_dir)
    assert len(first_run_records) == 1

    # Run transform second time
    transform_hupd(config)

    # Get output files and records after second run
    second_run_files = sorted(output_dir.glob("*.jsonl.gz"))
    second_run_records = read_all_jsonl_gz(output_dir)

    # Verify no duplication - same number of files and records
    assert len(second_run_files) == len(first_run_files), "Files should not be duplicated"
    assert len(second_run_records) == len(first_run_records), "Records should not be duplicated"
    assert len(second_run_records) == 1, "Should still have exactly 1 record"
