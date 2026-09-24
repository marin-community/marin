# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cluster an observed proteome and measure complete native representative artifacts."""

import argparse
import csv
import gzip
import json
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from Bio import SeqIO

from experiments.post_training.bio_tasks.native.commands import execute
from experiments.post_training.bio_tasks.real_data import tsv_text


def prepare(inputs: Path, output: Path) -> None:
    """Run pinned Linclust and independently reconcile membership, FASTA and metadata."""
    inputs, output = inputs.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    query = json.loads((inputs / "query.json").read_text())
    proteins = SeqIO.to_dict(
        SeqIO.parse(inputs / "proteins.fa", "fasta"), key_function=lambda record: record.id.split("|")[1]
    )
    with (inputs / "proteins.tsv").open() as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    metadata = {row["Entry"]: row for row in rows}
    if len(metadata) != len(rows) or set(metadata) != set(proteins):
        raise ValueError("Proteome and metadata identities disagree")
    if any(len(protein) != int(metadata[key]["Length"]) for key, protein in proteins.items()):
        raise ValueError("Proteome sequence and metadata lengths disagree")
    execute(["mmseqs", "version"], output, "mmseqs-version.txt")
    with TemporaryDirectory(prefix="linclust-", dir=inputs.parent) as temporary:
        work = Path(temporary)
        database, clusters, representatives = (str(work / name) for name in ("proteins", "clusters", "representatives"))
        execute(
            ["mmseqs", "createdb", str(inputs / "proteins.fa"), database, "--dbtype", "1", "--shuffle", "0"],
            output,
            "createdb.stdout",
        )
        execute(
            [
                "mmseqs",
                "linclust",
                database,
                clusters,
                str(work / "tmp"),
                "--min-seq-id",
                str(query["minimum_identity"]),
                "-c",
                str(query["coverage"]),
                "--cov-mode",
                "0",
                "--cluster-mode",
                "2",
                "--alignment-mode",
                "3",
                "--kmer-per-seq",
                str(query["kmer_per_sequence"]),
                "--threads",
                "1",
            ],
            output,
            "linclust.stdout",
            timeout=900,
        )
        execute(
            ["mmseqs", "createtsv", database, database, clusters, str(output / "native-clusters.tsv"), "--threads", "1"],
            output,
            "createtsv.stdout",
        )
        execute(["mmseqs", "createsubdb", clusters, database, representatives], output, "createsubdb.stdout")
        execute(
            ["mmseqs", "convert2fasta", representatives, str(output / "native-representatives.fa")],
            output,
            "convert2fasta.stdout",
        )
    groups = defaultdict(set)
    members = set()
    membership = []
    with (output / "native-clusters.tsv").open() as source:
        for representative, member in csv.reader(source, delimiter="\t"):
            if representative not in proteins or member not in proteins or member in members:
                raise ValueError("Native cluster output has unexpected or repeated protein identities")
            members.add(member)
            groups[representative].add(member)
            membership.append(
                {
                    "id": member,
                    "representative": representative,
                    "length": len(proteins[member]),
                    "sequence_version": int(metadata[member]["Sequence version"]),
                }
            )
    if members != set(proteins) or any(key not in group for key, group in groups.items()):
        raise ValueError("Native clusters must partition the whole proteome and contain their representative")
    native_representatives = SeqIO.to_dict(
        SeqIO.parse(output / "native-representatives.fa", "fasta"), key_function=lambda record: record.id.split("|")[1]
    )
    if set(native_representatives) != set(groups):
        raise ValueError("Extracted representative identities differ from the native cluster representatives")
    if any(record.seq != proteins[key].seq for key, record in native_representatives.items()):
        raise ValueError("Native representative extraction changed a sequence")
    cluster_rows = []
    for representative, group in sorted(groups.items()):
        lengths = [len(proteins[member]) for member in group]
        cluster_rows.append(
            {
                "id": representative,
                "members": len(group),
                "distinct_sequences": len({str(proteins[member].seq) for member in group}),
                "total_residues": sum(lengths),
                "minimum_length": min(lengths),
                "maximum_length": max(lengths),
                "representative_length": len(proteins[representative]),
            }
        )
    retained = {key: str(record.seq) for key, record in sorted(native_representatives.items())}
    with (output / "representatives.fa").open("w") as destination:
        for key, sequence in retained.items():
            destination.write(f">{key}\n{sequence}\n")
    summary = {
        "proteins": len(proteins),
        "clusters": len(groups),
        "singletons": sum(len(group) == 1 for group in groups.values()),
        "multimember_clusters": sum(len(group) > 1 for group in groups.values()),
        "nonrepresentative_proteins": len(proteins) - len(groups),
        "input_residues": sum(map(len, proteins.values())),
        "representative_residues": sum(map(len, retained.values())),
    }
    (output / "membership.tsv").write_text(tsv_text(sorted(membership, key=lambda row: row["id"])))
    (output / "clusters.tsv").write_text(tsv_text(cluster_rows))
    (output / "answer.json").write_text(json.dumps([{"id": "proteome", **summary}]) + "\n")
    reference = {
        "query": query,
        "summaries": {"proteome": summary},
        "membership": {row["id"]: {key: value for key, value in row.items() if key != "id"} for row in membership},
        "clusters": {row["id"]: {key: value for key, value in row.items() if key != "id"} for row in cluster_rows},
        "representatives": retained,
        "biopython_version": version("biopython"),
        "interpretation": (
            "Reproduction of the specified Linclust heuristic; similarity clusters do not establish "
            "orthology or function."
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
