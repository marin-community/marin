# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Input-reading MMseqs2 oracle with standard-library artifact measurements."""

import csv
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.post_training.bio_tasks.native.commands import execute
from experiments.post_training.bio_tasks.solvers.formats import fasta


def solve_clusters(inputs: Path, output: Path) -> list[dict]:
    """Recompute native clusters, extract representatives and reconcile every protein."""
    inputs, output = inputs.resolve(), output.resolve()
    query = json.loads((inputs / "query.json").read_text())
    sequences = {name.split("|")[1]: sequence for name, sequence in fasta(inputs / "proteins.fa").items()}
    with (inputs / "proteins.tsv").open() as handle:
        metadata = {row["Entry"]: row for row in csv.DictReader(handle, delimiter="\t")}
    if sequences.keys() != metadata.keys():
        raise ValueError("Protein metadata and FASTA identities differ")
    with TemporaryDirectory(prefix="mmseqs-oracle-", dir=output) as temporary:
        work = Path(temporary)
        database, clustered, representatives = (str(work / name) for name in ("db", "clusters", "representatives"))
        commands = [
            ["mmseqs", "createdb", str(inputs / "proteins.fa"), database, "--shuffle", "0", "--dbtype", "1"],
            [
                "mmseqs",
                "linclust",
                database,
                clustered,
                str(work / "tmp"),
                "--threads",
                "1",
                "--cluster-mode",
                "2",
                "--cov-mode",
                "0",
                "--alignment-mode",
                "3",
                "--min-seq-id",
                str(query["minimum_identity"]),
                "-c",
                str(query["coverage"]),
                "--kmer-per-seq",
                str(query["kmer_per_sequence"]),
            ],
            [
                "mmseqs",
                "createtsv",
                database,
                database,
                clustered,
                str(output / "native-clusters.tsv"),
                "--threads",
                "1",
            ],
            ["mmseqs", "createsubdb", clustered, database, representatives],
            ["mmseqs", "convert2fasta", representatives, str(output / "native-representatives.fa")],
        ]
        for index, command in enumerate(commands):
            execute(command, output, f"mmseqs-{index}.stdout", timeout=300)
    membership = {}
    with (output / "native-clusters.tsv").open() as handle:
        for line in handle:
            representative, member = line.rstrip("\n").split("\t")
            if member in membership or member not in sequences or representative not in sequences:
                raise ValueError("Cluster output must assign every input protein once")
            membership[member] = representative
    if membership.keys() != sequences.keys():
        raise ValueError("Clusters omit input proteins")
    native = {name.split("|")[1]: sequence for name, sequence in fasta(output / "native-representatives.fa").items()}
    if set(native) != set(membership.values()) or any(native[key] != sequences[key] for key in native):
        raise ValueError("Native representatives differ from assigned unchanged sequences")
    if any(membership[representative] != representative for representative in native):
        raise ValueError("Representative must belong to its own cluster")
    protein_rows = []
    cluster_rows = []
    sizes = Counter(membership.values())
    for member, representative in sorted(membership.items()):
        protein_rows.append(
            {
                "id": member,
                "representative": representative,
                "length": len(sequences[member]),
                "sequence_version": int(metadata[member]["Sequence version"]),
            }
        )
    for representative in sorted(native):
        group = [sequence for member, sequence in sequences.items() if membership[member] == representative]
        lengths = list(map(len, group))
        cluster_rows.append(
            {
                "id": representative,
                "members": sizes[representative],
                "distinct_sequences": len(set(group)),
                "total_residues": sum(lengths),
                "minimum_length": min(lengths),
                "maximum_length": max(lengths),
                "representative_length": len(native[representative]),
            }
        )
    for name, rows in [("membership.tsv", protein_rows), ("clusters.tsv", cluster_rows)]:
        with (output / name).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    with (output / "representatives.fa").open("w") as handle:
        for representative in sorted(native):
            handle.write(f">{representative}\n{native[representative]}\n")
    return [
        {
            "id": "proteome",
            "proteins": len(sequences),
            "clusters": len(sizes),
            "singletons": list(sizes.values()).count(1),
            "multimember_clusters": sum(size > 1 for size in sizes.values()),
            "nonrepresentative_proteins": len(sequences) - len(native),
            "input_residues": sum(map(len, sequences.values())),
            "representative_residues": sum(map(len, native.values())),
        }
    ]
