# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Input-reading assembly oracle with independent long-cs alignment checks."""

import csv
import json
import re
import shutil
from itertools import accumulate
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.post_training.bio_tasks.native.commands import execute
from experiments.post_training.bio_tasks.solvers.formats import fasta, fastq, reverse_complement


def solve_assembly(inputs: Path, output: Path) -> list[dict]:
    """Rebuild contigs and derive complete assembly measurements from observed inputs."""
    inputs, output = inputs.resolve(), output.resolve()
    query = json.loads((inputs / "query.json").read_text())
    read_rows = []
    paired_ids = []
    for mate in (1, 2):
        identifiers = []
        row = {"id": f"R{mate}", "reads": 0, "bases": 0, "q20_bases": 0, "q30_bases": 0, "n_bases": 0}
        for name, sequence, qualities in fastq(inputs / f"reads_R{mate}.fastq"):
            identifiers.append(name)
            row["reads"] += 1
            row["bases"] += len(sequence)
            row["q20_bases"] += sum(ord(value) >= 53 for value in qualities)
            row["q30_bases"] += sum(ord(value) >= 63 for value in qualities)
            row["n_bases"] += sequence.upper().count("N")
        paired_ids.append(identifiers)
        read_rows.append(row)
    if paired_ids[0] != paired_ids[1] or len(set(paired_ids[0])) != len(paired_ids[0]):
        raise ValueError("Paired FASTQ identifiers differ or repeat")
    reference = fasta(inputs / "reference.fa")
    if len(reference) != 1:
        raise ValueError("Expected one complete reference genome")
    reference_id, reference_sequence = next(iter(reference.items()))
    with TemporaryDirectory(prefix="assembly-oracle-", dir=output) as temporary:
        work = Path(temporary)
        assembly = work / "spades"
        execute(
            [
                "spades.py",
                "-o",
                str(assembly),
                "-1",
                str(inputs / "reads_R1.fastq"),
                "-2",
                str(inputs / "reads_R2.fastq"),
                "--only-assembler",
                "--isolate",
                "-k",
                ",".join(str(kmer) for kmer in query["kmers"]),
                "-t",
                "1",
                "-m",
                "4",
            ],
            output,
            "spades.stdout",
            timeout=900,
        )
        shutil.copyfile(assembly / "contigs.fasta", output / "contigs.fa")
    contigs = fasta(output / "contigs.fa")
    if not contigs:
        raise ValueError("Assembly has no contigs")
    paf = execute(
        [
            "minimap2",
            "-x",
            "asm5",
            "-t",
            "1",
            "-c",
            "--eqx",
            "--cs=long",
            "--secondary=no",
            str(inputs / "reference.fa"),
            str(output / "contigs.fa"),
        ],
        output,
        "native-alignments.paf",
    )
    coverage = [0] * len(reference_sequence)
    query_positions = {name: set() for name in contigs}
    alignments = []
    for line in paf.read_text().splitlines():
        fields = line.split("\t")
        name, strand = fields[0], fields[4]
        start, stop = int(fields[2]), int(fields[3])
        target_start, target_end = int(fields[7]), int(fields[8])
        if fields[5] != reference_id or int(fields[6]) != len(reference_sequence):
            raise ValueError("Alignment target differs from the supplied reference")
        sequence = contigs[name] if strand == "+" else reverse_complement(contigs[name])
        qpos = start if strand == "+" else len(sequence) - stop
        rpos = target_start
        cs = next(field[5:] for field in fields[12:] if field.startswith("cs:Z:"))
        tokens = re.findall(r"=[A-Za-z]+|\*[A-Za-z]{2}|[+-][A-Za-z]+", cs)
        if "".join(tokens) != cs:
            raise ValueError("Unsupported long-cs alignment operation")
        operations = []
        measures = dict.fromkeys(("matches", "substitutions", "inserted_bases", "deleted_bases"), 0)
        for token in tokens:
            code, bases = token[0], token[1:].upper()
            if code == "*":
                reference_bases, query_bases, operation = bases[0], bases[1], "X"
                measures["substitutions"] += 1
            elif code == "=":
                reference_bases, query_bases, operation = bases, bases, "="
                measures["matches"] += len(bases)
            elif code == "+":
                reference_bases, query_bases, operation = "", bases, "I"
                measures["inserted_bases"] += len(bases)
            else:
                reference_bases, query_bases, operation = bases, "", "D"
                measures["deleted_bases"] += len(bases)
            if reference_sequence[rpos : rpos + len(reference_bases)].upper() != reference_bases:
                raise ValueError("Long-cs reference residues disagree with FASTA")
            if sequence[qpos : qpos + len(query_bases)].upper() != query_bases:
                raise ValueError("Long-cs query residues disagree with assembled FASTA")
            length = max(len(reference_bases), len(query_bases))
            if operations and operations[-1][1] == operation:
                operations[-1] = (operations[-1][0] + length, operation)
            else:
                operations.append((length, operation))
            if code in "=*":
                for offset in range(length):
                    coverage[rpos + offset] += 1
                    query_positions[name].add(qpos + offset if strand == "+" else len(sequence) - qpos - offset - 1)
            qpos += len(query_bases)
            rpos += len(reference_bases)
        if qpos != (stop if strand == "+" else len(sequence) - start) or rpos != target_end:
            raise ValueError("Long-cs endpoint does not match the reported alignment span")
        cigar = "".join(f"{length}{operation}" for length, operation in operations)
        if f"cg:Z:{cigar}" not in fields[12:]:
            raise ValueError("Long-cs operations disagree with CIGAR")
        alignments.append(
            {
                "id": f"{name}:{strand}:{start}:{stop}:{target_start}:{target_end}",
                "contig": name,
                "reference": reference_id,
                "strand": strand,
                "query_start": start,
                "query_end": stop,
                "reference_start": target_start,
                "reference_end": target_end,
                "mapping_quality": int(fields[11]),
                "alignment_columns": sum(measures.values()),
                **measures,
                "cigar": cigar,
            }
        )
    if not alignments:
        raise ValueError("No assembly alignments were found")
    contig_rows = [
        {
            "id": name,
            "length": len(sequence),
            "gc_bases": sum(base in "GC" for base in sequence.upper()),
            "ambiguous_bases": sum(base not in "ACGT" for base in sequence.upper()),
            "alignments": sum(row["contig"] == name for row in alignments),
            "aligned_query_bases": len(query_positions[name]),
        }
        for name, sequence in sorted(contigs.items())
    ]
    tables = {
        "read_qc.tsv": read_rows,
        "contig_metrics.tsv": contig_rows,
        "alignments.tsv": alignments,
        "reference_coverage.tsv": [
            {"id": f"{reference_id}:{position}", "contig_depth": value} for position, value in enumerate(coverage)
        ],
    }
    for filename, rows in tables.items():
        with (output / filename).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: row["id"]))
    lengths = sorted(map(len, contigs.values()), reverse=True)
    cumulative = list(accumulate(lengths))
    n_index = next(i for i, count in enumerate(cumulative) if count * 2 >= cumulative[-1])
    ng_index = next((i for i, count in enumerate(cumulative) if count * 2 >= len(reference_sequence)), None)
    return [
        {
            "id": "assembly",
            "read_pairs": len(paired_ids[0]),
            "input_bases": sum(row["bases"] for row in read_rows),
            "contigs": len(contigs),
            "assembled_bases": cumulative[-1],
            "reference_length": len(reference_sequence),
            "n50": lengths[n_index],
            "l50": n_index + 1,
            "ng50": None if ng_index is None else lengths[ng_index],
            "lg50": None if ng_index is None else ng_index + 1,
            "aligned_contigs": sum(bool(positions) for positions in query_positions.values()),
            "reference_covered_bases": len(coverage) - coverage.count(0),
            "reference_uncovered_bases": coverage.count(0),
            "multiply_covered_reference_bases": sum(value >= 2 for value in coverage),
            "aligned_query_bases": sum(map(len, query_positions.values())),
            **{
                name: sum(row[name] for row in alignments)
                for name in ("matches", "substitutions", "inserted_bases", "deleted_bases")
            },
        }
    ]
