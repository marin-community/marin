# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Observed PhiX read-structure and alignment-eligibility audit."""

import hashlib
import json

from experiments.post_training.bio_tasks.bam_artifacts import BamContract
from experiments.post_training.bio_tasks.contract import Column, Contract, TableContract
from experiments.post_training.bio_tasks.real_data import source_text
from experiments.post_training.bio_tasks.recipe_types import DataOrigin, Instance, OracleRuntime, Recipe, WorkflowScope

SUMMARY_FIELDS = {
    "input_pairs": ("integer", "pairs", "unchanged observed FASTQ pairs"),
    "native_alignment_records": ("integer", "SAM records", "all native alignment records"),
    "primary_read_records": ("integer", "reads", "primary read records, both mates counted separately"),
    "mate1_records": ("integer", "reads", "primary read-one records"),
    "mate2_records": ("integer", "reads", "primary read-two records"),
    "paired_layout": ("text", "layout", "paired or single-end input layout"),
    "modal_read_length": ("integer", "bases", "unique mode across both mates; null on a tie"),
    "modal_length_count": ("integer", "reads", "read records at the modal length"),
    "eligible_read_records": ("integer", "reads", "primary mapped proper-pair records with 30 <= MAPQ < 255"),
    "eligible_pairs_both_mates": ("integer", "pairs", "pairs whose two primary mates are both eligible"),
    "unmapped_primary_records": ("integer", "reads", "primary records with unmapped flag"),
    "mapq_255_primary_records": ("integer", "reads", "primary records with unavailable MAPQ 255"),
}

TABLE_FIELDS = {
    "primary_reads.tsv": {
        "read_id": "text",
        "mate": "integer",
        "read_length": "integer",
        "flag": "integer",
        "reference": "text",
        "position_1based": "integer",
        "mapq": "integer",
        "cigar": "text",
        "mate_reference": "text",
        "mate_position_1based": "integer",
        "template_length": "integer",
        "mapped": "integer",
        "proper_pair": "integer",
        "duplicate": "integer",
        "qc_fail": "integer",
        "mapq_255": "integer",
        "eligible": "integer",
        "sequence_sha256": "text",
        "quality_sha256": "text",
    },
    "read_lengths.tsv": {"mate": "integer", "read_length": "integer", "reads": "integer"},
    "flag_counts.tsv": {"records": "integer"},
    "eligible_reads.tsv": {"read_id": "text", "mate": "integer", "mapq": "integer", "flag": "integer"},
}


def bam_contract(reference: dict) -> Contract:
    """Bind complete BAM semantics, BAI retrieval, per-read tables and summary."""
    columns = {
        name: Column(kind=kind, unit=unit, description=description, nullable=name == "modal_read_length")
        for name, (kind, unit, description) in SUMMARY_FIELDS.items()
    }
    tables = {}
    for filename, fields in TABLE_FIELDS.items():
        table_columns = {
            name: Column(kind=kind, unit="native BAM read record", description=name.replace("_", " "))
            for name, kind in fields.items()
        }
        expected = {
            identifier: {name: int(value) if fields[name] == "integer" else value for name, value in row.items()}
            for identifier, row in reference["tables"][filename].items()
        }
        tables[filename] = TableContract(columns=table_columns, expected=expected, max_bytes=16 * 1024 * 1024)
    return Contract(
        columns=columns,
        expected=reference["summaries"],
        tables=tables,
        bams={"reads.bam": BamContract.model_validate(reference["bam"])},
    )


def generate_bam(_seed: int) -> Instance:
    reference = json.loads(source_text("ENA:ERR266411", "err266411-bam-reference.json.gz"))
    inputs = {
        f"reads_R{mate}.fastq": source_text("ENA:ERR266411", f"err266411-spread-r{mate}.fastq.gz") for mate in (1, 2)
    }
    inputs["reference.fa"] = source_text("RefSeq:NC_001422.1", "nc_001422-1.fa.gz")
    inputs["query.json"] = json.dumps(reference["query"], indent=2) + "\n"
    for name, content in inputs.items():
        if hashlib.sha256(content.encode()).hexdigest() != reference["source_sha256"][name]:
            raise ValueError(f"Changed BAM audit input: {name}")
    contract = bam_contract(reference)
    changed_eligible = [{**row, "eligible_read_records": row["eligible_read_records"] + 1} for row in contract.answer()]
    return Instance(
        "Map the supplied 12,000 unchanged, observed PhiX paired FASTQ records to the supplied NC_001422.1 "
        "reference. These are one technical subset from ENA ERR266411, not biological replicates. Input files "
        "are in /app/inputs. Use installed BWA-MEM 0.7.19: bwa index on a writable copy of reference.fa, then "
        "bwa mem -t 1 with reads_R1.fastq and reads_R2.fastq. Convert to BAM, coordinate-sort with SAMtools "
        "1.24 using one thread and build reads.bam.bai. Submit the complete native reads.bam and its index. "
        "Report one primary row per original mate in primary_reads.tsv, excluding only secondary (0x100) "
        "and supplementary (0x800) alignments from that table; retain all native records in the BAM. Primary "
        "records must preserve the complete read sequence and qualities. Use id=read_id:mate, where mate is "
        "1 or 2. read_length is the original FASTQ sequence length. sequence_sha256 and quality_sha256 hash "
        "the SAM-oriented SEQ and QUAL strings in the primary BAM record. Output read_lengths.tsv keyed R1:length "
        "or R2:length and flag_counts.tsv keyed by SAM flag name over ALL native alignment records. "
        "eligible_reads.tsv lists every eligible primary read id, with original read_id, mate, MAPQ and flag. "
        "Eligibility is mapped (!0x4), proper-pair bit (0x2), and 30 <= MAPQ < 255. This is mapping quality, "
        "not base quality; MAPQ 255 is unavailable. Count read records, so two eligible mates contribute two. "
        "Do not exclude duplicate (0x400) or QC-fail (0x200) reads. The reference genome is circular but this "
        "task reports native flags against its supplied linear representation; do not rescue crossing-origin "
        "pairs. Derive all answer.json fields from the same native BAM and input read-length distribution. "
        "The unique modal length is pooled across both mates and null if tied.",
        inputs,
        contract,
        {"changed_eligible_count": changed_eligible},
        data_origin=DataOrigin.REAL,
        source_ids=("ENA:ERR266411", "RefSeq:NC_001422.1"),
        derivation=(
            "12,000 systematically spaced original paired records from a checksum-verified observed run; "
            "native alignment to its deposited reference."
        ),
        workflow_scope=WorkflowScope.CONNECTED,
    )


RECIPES = (
    Recipe(
        id="real-phix-bam-read-structure",
        version="1",
        skills=(
            "paired-read identity",
            "native BAM/BAI",
            "SAM flag interpretation",
            "MAPQ filtering",
            "read-record accounting",
        ),
        formats=("paired FASTQ", "reference FASTA", "BAM", "BAI", "TSV", "JSON"),
        sources=("https://www.ebi.ac.uk/ena/browser/view/ERR266411", "https://www.ncbi.nlm.nih.gov/nuccore/NC_001422.1"),
        generate=generate_bam,
        oracle_timeout=900,
        oracle_runtime=OracleRuntime.NATIVE,
    ),
)
