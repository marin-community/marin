# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble observed read pairs and measure complete contigs against a reference."""

import argparse
import gzip
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from Bio import SeqIO

from experiments.post_training.bio_tasks.native.commands import execute
from experiments.post_training.bio_tasks.real_data import tsv_text


def prepare(inputs: Path, output: Path) -> None:
    """Run SPAdes/minimap2 and independently check every aligned nucleotide."""
    inputs, output = inputs.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    query = json.loads((inputs / "query.json").read_text())
    read_qc = []
    names = []
    for mate in (1, 2):
        counts = Counter()
        identifiers = []
        for record in SeqIO.parse(inputs / f"reads_R{mate}.fastq", "fastq"):
            identifiers.append(record.id)
            qualities = record.letter_annotations["phred_quality"]
            counts.update(
                reads=1,
                bases=len(record),
                q20_bases=sum(value >= 20 for value in qualities),
                q30_bases=sum(value >= 30 for value in qualities),
                n_bases=str(record.seq).upper().count("N"),
            )
        names.append(identifiers)
        read_qc.append({"id": f"R{mate}", **counts})
    if names[0] != names[1] or len(set(names[0])) != len(names[0]):
        raise ValueError("Observed input pairs must have matching unique identifiers")
    reference_records = SeqIO.to_dict(SeqIO.parse(inputs / "reference.fa", "fasta"))
    if len(reference_records) != 1:
        raise ValueError("This assembly assessment requires one complete reference sequence")
    reference_id, reference_record = next(iter(reference_records.items()))
    reference_sequence = str(reference_record.seq).upper()
    execute(["spades.py", "--version"], output, "spades-version.txt")
    execute(["minimap2", "--version"], output, "minimap2-version.txt")
    assembly = output / "native-assembly"
    execute(
        [
            "spades.py",
            "--isolate",
            "--only-assembler",
            "-1",
            str(inputs / "reads_R1.fastq"),
            "-2",
            str(inputs / "reads_R2.fastq"),
            "-k",
            ",".join(map(str, query["kmers"])),
            "-t",
            "1",
            "-m",
            "4",
            "-o",
            str(assembly),
        ],
        output,
        "spades.stdout",
        timeout=900,
    )
    shutil.copyfile(assembly / "contigs.fasta", output / "contigs.fa")
    contigs = SeqIO.to_dict(SeqIO.parse(output / "contigs.fa", "fasta"))
    if not contigs:
        raise ValueError("SPAdes returned no contigs")
    paf = execute(
        [
            "minimap2",
            "-x",
            "asm5",
            "--secondary=no",
            "--eqx",
            "-c",
            "--cs=long",
            "-t",
            "1",
            str(inputs / "reference.fa"),
            str(output / "contigs.fa"),
        ],
        output,
        "native-alignments.paf",
    )
    depth = [0] * len(reference_sequence)
    aligned_positions = defaultdict(set)
    alignment_count = Counter()
    alignment_rows = []
    for line in paf.read_text().splitlines():
        fields = line.split("\t")
        name, strand, target = fields[0], fields[4], fields[5]
        qlength, qstart, qend = map(int, fields[1:4])
        tlength, tstart, tend, matches, block_length, mapq = map(int, fields[6:12])
        if target != reference_id or tlength != len(reference_sequence) or qlength != len(contigs[name]):
            raise ValueError("Native alignment references disagree with supplied sequences")
        tags = {tag.split(":", 2)[0]: tag.split(":", 2)[2] for tag in fields[12:]}
        cigar = tags["cg"]
        operations = re.findall(r"(\d+)([=XID])", cigar)
        if "".join(length + operation for length, operation in operations) != cigar:
            raise ValueError(f"Unexpected assembly-alignment CIGAR: {cigar}")
        sequence = str(contigs[name].seq if strand == "+" else contigs[name].seq.reverse_complement()).upper()
        qposition = qstart if strand == "+" else qlength - qend
        tposition = tstart
        counts = Counter({"matches": 0, "substitutions": 0, "inserted_bases": 0, "deleted_bases": 0})
        for raw_length, operation in operations:
            length = int(raw_length)
            if operation in "=X":
                left = sequence[qposition : qposition + length]
                right = reference_sequence[tposition : tposition + length]
                if len(left) != length or len(right) != length:
                    raise ValueError("Alignment exceeds sequence bounds")
                equal = sum(a == b for a, b in zip(left, right, strict=True))
                if equal != (length if operation == "=" else 0):
                    raise ValueError("CIGAR disagrees with observed contig/reference residues")
                counts["matches" if operation == "=" else "substitutions"] += length
                for offset in range(length):
                    depth[tposition + offset] += 1
                    position = qposition + offset if strand == "+" else qlength - 1 - qposition - offset
                    aligned_positions[name].add(position)
                qposition += length
                tposition += length
            elif operation == "I":
                counts["inserted_bases"] += length
                qposition += length
            else:
                counts["deleted_bases"] += length
                tposition += length
        expected_qend = qend if strand == "+" else qlength - qstart
        if (qposition, tposition) != (expected_qend, tend):
            raise ValueError("CIGAR endpoints disagree with PAF coordinates")
        if counts["matches"] != matches or sum(counts.values()) != block_length:
            raise ValueError("PAF alignment counts disagree with complete CIGAR/residue checks")
        alignment_count[name] += 1
        alignment_rows.append(
            {
                "id": f"{name}:{strand}:{qstart}:{qend}:{tstart}:{tend}",
                "contig": name,
                "reference": target,
                "strand": strand,
                "query_start": qstart,
                "query_end": qend,
                "reference_start": tstart,
                "reference_end": tend,
                "mapping_quality": mapq,
                "alignment_columns": block_length,
                **counts,
                "cigar": cigar,
            }
        )
    if not alignment_rows or len({row["id"] for row in alignment_rows}) != len(alignment_rows):
        raise ValueError("Expected distinct contig/reference alignments")
    contig_rows = [
        {
            "id": name,
            "length": len(record),
            "gc_bases": str(record.seq).upper().count("G") + str(record.seq).upper().count("C"),
            "ambiguous_bases": sum(base not in "ACGT" for base in str(record.seq).upper()),
            "alignments": alignment_count[name],
            "aligned_query_bases": len(aligned_positions[name]),
        }
        for name, record in sorted(contigs.items())
    ]
    coverage_rows = [{"id": f"{reference_id}:{position}", "contig_depth": count} for position, count in enumerate(depth)]
    lengths = sorted((len(record) for record in contigs.values()), reverse=True)
    cumulative = 0
    nx = {}
    for index, length in enumerate(lengths, 1):
        cumulative += length
        for label, denominator in (("n50", sum(lengths)), ("ng50", len(reference_sequence))):
            if label not in nx and 2 * cumulative >= denominator:
                nx[label] = length
                nx["l50" if label == "n50" else "lg50"] = index
    summary = {
        "read_pairs": len(names[0]),
        "input_bases": sum(row["bases"] for row in read_qc),
        "contigs": len(contigs),
        "assembled_bases": sum(lengths),
        "reference_length": len(reference_sequence),
        "n50": nx["n50"],
        "l50": nx["l50"],
        "ng50": nx.get("ng50"),
        "lg50": nx.get("lg50"),
        "aligned_contigs": sum(row["alignments"] > 0 for row in contig_rows),
        "reference_covered_bases": sum(value > 0 for value in depth),
        "reference_uncovered_bases": depth.count(0),
        "multiply_covered_reference_bases": sum(value > 1 for value in depth),
        "aligned_query_bases": sum(row["aligned_query_bases"] for row in contig_rows),
        **{
            name: sum(row[name] for row in alignment_rows)
            for name in ("matches", "substitutions", "inserted_bases", "deleted_bases")
        },
    }
    tables = {
        "read_qc.tsv": read_qc,
        "contig_metrics.tsv": contig_rows,
        "alignments.tsv": alignment_rows,
        "reference_coverage.tsv": coverage_rows,
    }
    for name, rows in tables.items():
        (output / name).write_text(tsv_text(sorted(rows, key=lambda row: row["id"])))
    (output / "answer.json").write_text(json.dumps([{"id": "assembly", **summary}], indent=2) + "\n")
    reference = {
        "query": query,
        "summaries": {"assembly": summary},
        "contigs": {name: str(record.seq).upper() for name, record in contigs.items()},
        "tables": {
            name: {row["id"]: {key: value for key, value in row.items() if key != "id"} for row in rows}
            for name, rows in tables.items()
        },
        "interpretation": (
            "Method-specific assembly of an observed archive prefix. Reference coverage is aligned contig breadth, "
            "not read depth or proof of sequence truth. A circular genome can produce terminal overlap; "
            "no circularization or reference-guided sequence correction is performed."
        ),
    }
    (output / "reference.json.gz").write_bytes(gzip.compress((json.dumps(reference, indent=2) + "\n").encode(), mtime=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.inputs, args.output)


if __name__ == "__main__":
    main()
