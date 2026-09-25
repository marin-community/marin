# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare an independently measured private PhiX BAM audit reference."""

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pysam

from experiments.post_training.bio_tasks.bam_artifacts import summarize_bam
from experiments.post_training.bio_tasks.native.commands import execute
from experiments.post_training.bio_tasks.real_data import source_text

QUERY = {"expected_pairs": 12000, "mapq_threshold": 30, "reference_accession": "NC_001422.1"}
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
TABLES = ("primary_reads.tsv", "read_lengths.tsv", "flag_counts.tsv", "eligible_reads.tsv")


def source_lengths(path: Path) -> tuple[list[str], dict[str, int]]:
    """Read original identifiers and lengths without using the solver's parser."""
    names = []
    lengths = {}
    with path.open() as handle:
        while header := handle.readline():
            sequence, plus, quality = (handle.readline().rstrip("\r\n") for _ in range(3))
            name = header[1:].split()[0] if header.startswith("@") else ""
            if not name or not plus.startswith("+") or len(sequence) != len(quality) or name in lengths:
                raise ValueError("Malformed or repeated observed FASTQ record")
            names.append(name)
            lengths[name] = len(sequence)
    return names, lengths


def write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Empty observed BAM table: {path.name}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["id"]))


def native_bam(inputs: Path, output: Path) -> Path:
    """Align, sort and index the observed reads with the pinned native tools."""
    reference = output / "reference.fa"
    shutil.copyfile(inputs / "reference.fa", reference)
    execute(["bwa", "index", str(reference)], output, "bwa-index.stdout")
    sam = execute(
        ["bwa", "mem", "-t", "1", str(reference), str(inputs / "reads_R1.fastq"), str(inputs / "reads_R2.fastq")],
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
    return bam


def prepare(inputs: Path, output: Path) -> None:
    """Build native BAM, then measure all records through pysam rather than the oracle."""
    inputs, output = inputs.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if json.loads((inputs / "query.json").read_text()) != QUERY:
        raise ValueError("Unexpected observed BAM query")
    names1, lengths1 = source_lengths(inputs / "reads_R1.fastq")
    names2, lengths2 = source_lengths(inputs / "reads_R2.fastq")
    if names1 != names2 or len(names1) != QUERY["expected_pairs"]:
        raise ValueError("Observed FASTQ pair identities or count differ")
    bam_path = native_bam(inputs, output)
    primary, eligible = [], []
    flag_counts = Counter()
    primary_ids = set()
    native_records = 0
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for record in bam.fetch(until_eof=True):
            native_records += 1
            if native_records > 100_000:
                raise ValueError("Unexpected number of native BAM records")
            for name, bit in FLAG_BITS.items():
                flag_counts[name] += bool(record.flag & bit)
            if record.is_secondary or record.is_supplementary:
                continue
            mate = 1 if record.is_read1 else 2 if record.is_read2 else 0
            read_id = record.query_name
            if mate == 0 or (record.is_read1 and record.is_read2) or read_id not in lengths1:
                raise ValueError("Native BAM has invalid read/mate identity")
            identifier = f"{read_id}:{mate}"
            if identifier in primary_ids:
                raise ValueError("Native BAM has duplicate primary read")
            primary_ids.add(identifier)
            length = (lengths1 if mate == 1 else lengths2)[read_id]
            fields = record.to_string().split("\t")
            sequence, quality = fields[9:11]
            if sequence == "*" or quality == "*" or len(sequence) != length or len(quality) != length:
                raise ValueError("Native BAM lost observed read bases or qualities")
            passed = not record.is_unmapped and record.is_proper_pair and 30 <= record.mapping_quality < 255
            primary.append(
                {
                    "id": identifier,
                    "read_id": read_id,
                    "mate": mate,
                    "read_length": length,
                    "flag": record.flag,
                    "reference": fields[2],
                    "position_1based": record.reference_start + 1,
                    "mapq": record.mapping_quality,
                    "cigar": fields[5],
                    "mate_reference": fields[6],
                    "mate_position_1based": record.next_reference_start + 1,
                    "template_length": record.template_length,
                    "mapped": int(not record.is_unmapped),
                    "proper_pair": int(record.is_proper_pair),
                    "duplicate": int(record.is_duplicate),
                    "qc_fail": int(record.is_qcfail),
                    "mapq_255": int(record.mapping_quality == 255),
                    "eligible": int(passed),
                    "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                    "quality_sha256": hashlib.sha256(quality.encode()).hexdigest(),
                }
            )
            if passed:
                eligible.append(
                    {
                        "id": identifier,
                        "read_id": read_id,
                        "mate": mate,
                        "mapq": record.mapping_quality,
                        "flag": record.flag,
                    }
                )
    if len(primary_ids) != 2 * QUERY["expected_pairs"] or not eligible:
        raise ValueError("Observed BAM has missing primary reads or no eligible reads")
    write_table(output / "primary_reads.tsv", primary)
    counts = Counter((mate, length) for mate, lengths in ((1, lengths1), (2, lengths2)) for length in lengths.values())
    write_table(
        output / "read_lengths.tsv",
        [
            {"id": f"R{mate}:{length}", "mate": mate, "read_length": length, "reads": count}
            for (mate, length), count in counts.items()
        ],
    )
    write_table(output / "flag_counts.tsv", [{"id": name, "records": flag_counts[name]} for name in FLAG_BITS])
    write_table(output / "eligible_reads.tsv", eligible)
    pooled = Counter()
    for (_, length), count in counts.items():
        pooled[length] += count
    frequency = max(pooled.values())
    modes = [length for length, count in pooled.items() if count == frequency]
    by_read = Counter(row["read_id"] for row in eligible)
    summary = {
        "input_pairs": len(names1),
        "native_alignment_records": native_records,
        "primary_read_records": len(primary),
        "mate1_records": sum(row["mate"] == 1 for row in primary),
        "mate2_records": sum(row["mate"] == 2 for row in primary),
        "paired_layout": "paired",
        "modal_read_length": modes[0] if len(modes) == 1 else None,
        "modal_length_count": frequency,
        "eligible_read_records": len(eligible),
        "eligible_pairs_both_mates": sum(count == 2 for count in by_read.values()),
        "unmapped_primary_records": sum(row["mapped"] == 0 for row in primary),
        "mapq_255_primary_records": sum(row["mapq_255"] for row in primary),
    }
    references, record_digests, records = summarize_bam(bam_path, 128 * 1024 * 1024, 16 * 1024 * 1024, 100_000)
    if records != native_records:
        raise ValueError("Independent BAM record scans disagree")
    tables = {}
    for filename in TABLES:
        with (output / filename).open(newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        if len(rows) != len({row["id"] for row in rows}):
            raise ValueError(f"Duplicate table IDs: {filename}")
        tables[filename] = {row.pop("id"): row for row in rows}
    source_hashes = {
        name: hashlib.sha256((inputs / name).read_bytes()).hexdigest()
        for name in ("reads_R1.fastq", "reads_R2.fastq", "reference.fa", "query.json")
    }
    reference = {
        "query": QUERY,
        "source_sha256": source_hashes,
        "summaries": {"read_audit": summary},
        "tables": tables,
        "bam": {
            "references": references,
            "record_digests": dict(record_digests),
            "records": records,
            "max_bytes": min(128 * 1024 * 1024, max(8 * 1024 * 1024, 2 * bam_path.stat().st_size)),
            "index_max_bytes": min(16 * 1024 * 1024, max(64 * 1024, 2 * (output / "reads.bam.bai").stat().st_size)),
        },
    }
    content = json.dumps(reference, sort_keys=True, separators=(",", ":")).encode()
    compressed = output / "private-bam-reference.json.gz"
    with compressed.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
        handle.write(content)
    versions = {
        "bwa_stderr": (
            subprocess.run(["bwa"], capture_output=True, text=True, check=False, timeout=10).stderr.splitlines()[:2]
        ),
        "samtools": (
            subprocess.run(
                ["samtools", "--version"], capture_output=True, text=True, check=True, timeout=10
            ).stdout.splitlines()[0]
        ),
        "pysam": pysam.__version__,
    }
    (output / "reference-manifest.json").write_text(
        json.dumps(
            {
                "source_sha256": source_hashes,
                "reference_content_sha256": hashlib.sha256(content).hexdigest(),
                "reference_compressed_sha256": hashlib.sha256(compressed.read_bytes()).hexdigest(),
                "native_bam_bytes": bam_path.stat().st_size,
                "native_bai_bytes": (output / "reads.bam.bai").stat().st_size,
                "record_count": records,
                "tool_versions": versions,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--inputs", type=Path)
    source.add_argument("--from-catalog", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.from_catalog:
        inputs = args.output.parent / (args.output.name + "-inputs")
        inputs.mkdir(parents=True, exist_ok=False)
        for mate in (1, 2):
            (inputs / f"reads_R{mate}.fastq").write_text(
                source_text("ENA:ERR266411", f"err266411-spread-r{mate}.fastq.gz")
            )
        (inputs / "reference.fa").write_text(source_text("RefSeq:NC_001422.1", "nc_001422-1.fa.gz"))
        (inputs / "query.json").write_text(json.dumps(QUERY, indent=2) + "\n")
    else:
        inputs = args.inputs
    prepare(inputs, args.output)


if __name__ == "__main__":
    main()
