# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Map observed PhiX read pairs and audit every primary BAM read record."""

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from experiments.post_training.bio_tasks.native.commands import execute
from experiments.post_training.bio_tasks.solvers.formats import fastq

FLAG_BITS = {
    "paired": 0x1,
    "proper_pair": 0x2,
    "unmapped": 0x4,
    "mate_unmapped": 0x8,
    "reverse": 0x10,
    "mate_reverse": 0x20,
    "read1": 0x40,
    "read2": 0x80,
    "secondary": 0x100,
    "qc_fail": 0x200,
    "duplicate": 0x400,
    "supplementary": 0x800,
}


def _write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["id"]))


def solve_bam(inputs: Path, output: Path) -> list[dict]:
    """Create native BAM/BAI and count read records under the declared policy."""
    inputs, output = inputs.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    query = json.loads((inputs / "query.json").read_text())
    if query != {"expected_pairs": 12000, "mapq_threshold": 30, "reference_accession": "NC_001422.1"}:
        raise ValueError("Unexpected BAM task query")
    reads: dict[str, dict[int, int]] = defaultdict(dict)
    mate_names = []
    for mate in (1, 2):
        names = []
        for name, bases, _ in fastq(inputs / f"reads_R{mate}.fastq"):
            if mate in reads[name]:
                raise ValueError("Duplicate FASTQ read identity")
            reads[name][mate] = len(bases)
            names.append(name)
        mate_names.append(names)
    if mate_names[0] != mate_names[1] or len(mate_names[0]) != query["expected_pairs"]:
        raise ValueError("FASTQ mates or declared pair count differ")
    reference = output / "reference.fa"
    if inputs != output:
        shutil.copyfile(inputs / "reference.fa", reference)
    execute(["bwa", "index", str(reference)], output, "bwa-index.stdout")
    sam = execute(
        [
            "bwa",
            "mem",
            "-t",
            "1",
            str(reference),
            str(inputs / "reads_R1.fastq"),
            str(inputs / "reads_R2.fastq"),
        ],
        output,
        "native.sam",
        timeout=600,
    )
    execute(["samtools", "view", "-b", "-o", str(output / "unsorted.bam"), str(sam)], output, "samtools-view.stdout")
    bam = output / "reads.bam"
    execute(
        ["samtools", "sort", "-@", "1", "-m", "256M", "-o", str(bam), str(output / "unsorted.bam")],
        output,
        "samtools-sort.stdout",
    )
    execute(["samtools", "index", str(bam)], output, "samtools-index.stdout")
    sam_text = execute(["samtools", "view", "-h", str(bam)], output, "sorted.sam")
    lengths: Counter[tuple[int, int]] = Counter()
    for pair in reads.values():
        for mate, length in pair.items():
            lengths[mate, length] += 1
    primary_rows = []
    eligible_rows = []
    flag_counts = Counter()
    primary_ids = set()
    all_records = 0
    for line in sam_text.read_text().splitlines():
        if line.startswith("@"):
            continue
        fields = line.split("\t")
        if len(fields) < 11:
            raise ValueError("Malformed native SAM record")
        (
            name,
            flag_text,
            reference,
            position,
            mapq_text,
            cigar,
            mate_reference,
            mate_position,
            template_length,
            sequence,
            quality,
        ) = fields[:11]
        flag, mapq = int(flag_text), int(mapq_text)
        mate = 1 if flag & 0x40 else 2 if flag & 0x80 else 0
        if mate == 0 or flag & 0xC0 == 0xC0 or name not in reads:
            raise ValueError("Native BAM mate identity is invalid")
        all_records += 1
        for bit_name, bit in FLAG_BITS.items():
            flag_counts[bit_name] += bool(flag & bit)
        if flag & (0x100 | 0x800):
            continue
        identifier = f"{name}:{mate}"
        if identifier in primary_ids or mate not in reads[name]:
            raise ValueError("Native BAM has duplicate or missing primary mate")
        primary_ids.add(identifier)
        length = reads[name][mate]
        if sequence == "*" or quality == "*" or len(sequence) != length or len(quality) != length:
            raise ValueError("Primary native alignment lost read sequence or quality")
        mapped = not bool(flag & 0x4)
        proper_pair = bool(flag & 0x2)
        eligible = mapped and proper_pair and 30 <= mapq < 255
        row = {
            "id": identifier,
            "read_id": name,
            "mate": mate,
            "read_length": length,
            "flag": flag,
            "reference": reference,
            "position_1based": int(position),
            "mapq": mapq,
            "cigar": cigar,
            "mate_reference": mate_reference,
            "mate_position_1based": int(mate_position),
            "template_length": int(template_length),
            "mapped": int(mapped),
            "proper_pair": int(proper_pair),
            "duplicate": int(bool(flag & 0x400)),
            "qc_fail": int(bool(flag & 0x200)),
            "mapq_255": int(mapq == 255),
            "eligible": int(eligible),
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "quality_sha256": hashlib.sha256(quality.encode()).hexdigest(),
        }
        primary_rows.append(row)
        if eligible:
            eligible_rows.append({"id": identifier, "read_id": name, "mate": mate, "mapq": mapq, "flag": flag})
    if len(primary_ids) != 2 * len(reads):
        raise ValueError("Native BAM is missing primary mate records")
    if not eligible_rows:
        raise ValueError("No eligible observed read records")
    _write_table(output / "primary_reads.tsv", primary_rows)
    _write_table(
        output / "read_lengths.tsv",
        [
            {"id": f"R{mate}:{length}", "mate": mate, "read_length": length, "reads": count}
            for (mate, length), count in lengths.items()
        ],
    )
    _write_table(
        output / "flag_counts.tsv",
        [{"id": name, "records": flag_counts[name]} for name in FLAG_BITS],
    )
    _write_table(output / "eligible_reads.tsv", eligible_rows)
    combined_lengths: Counter[int] = Counter()
    for (_, length), count in lengths.items():
        combined_lengths[length] += count
    modal_count = max(combined_lengths.values())
    modal_lengths = {length for length, count in combined_lengths.items() if count == modal_count}
    eligible_by_read: Counter[str] = Counter(row["read_id"] for row in eligible_rows)
    summary = {
        "id": "read_audit",
        "input_pairs": len(reads),
        "native_alignment_records": all_records,
        "primary_read_records": len(primary_rows),
        "mate1_records": sum(row["mate"] == 1 for row in primary_rows),
        "mate2_records": sum(row["mate"] == 2 for row in primary_rows),
        "paired_layout": "paired",
        "modal_read_length": next(iter(modal_lengths)) if len(modal_lengths) == 1 else None,
        "modal_length_count": modal_count,
        "eligible_read_records": len(eligible_rows),
        "eligible_pairs_both_mates": sum(count == 2 for count in eligible_by_read.values()),
        "unmapped_primary_records": sum(not row["mapped"] for row in primary_rows),
        "mapq_255_primary_records": sum(row["mapq_255"] for row in primary_rows),
    }
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "source_run": "ERR266411",
                "reference_accession": "NC_001422.1",
                "query": query,
                "commands": "commands.jsonl",
                "bam": "reads.bam",
                "index": "reads.bam.bai",
            },
            indent=2,
        )
        + "\n"
    )
    return [summary]
