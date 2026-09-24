# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run pinned phylogenetic packages and measure their complete trees with Biopython."""

import argparse
import csv
import gzip
import hashlib
import json
import shutil
from itertools import combinations
from pathlib import Path

from Bio import Phylo, SeqIO

from experiments.post_training.bio_tasks.native.commands import execute

SEED = 20260924


def prepare(proteins: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    records = json.loads(gzip.decompress(proteins.read_bytes()))["proteins"]
    representatives = {}
    for key, record in sorted(records.items()):
        representatives.setdefault(record["sequence"], key)
    sequences = {key: sequence for sequence, key in representatives.items()}
    with (output / "accessions.tsv").open("w") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["accession", "representative", "taxon", "organism", "sequence_version"])
        for key, record in sorted(records.items()):
            writer.writerow(
                [
                    key,
                    representatives[record["sequence"]],
                    record["taxon"],
                    record["organism"],
                    record["sequence_version"],
                ]
            )
    raw = output / "proteins.fa"
    raw.write_text("".join(f">{key}\n{sequence}\n" for key, sequence in sequences.items()))
    execute(["mafft", "--auto", "--amino", "--thread", "1", str(raw)], output, "alignment.fa")
    alignment = output / "alignment.fa"
    aligned = {record.id: str(record.seq).upper() for record in SeqIO.parse(alignment, "fasta")}
    if {key: sequence.replace("-", "") for key, sequence in aligned.items()} != sequences:
        raise ValueError("Alignment changed observed proteins")
    if len({len(sequence) for sequence in aligned.values()}) != 1:
        raise ValueError("Ragged alignment")
    methods = {
        "iqtree": [
            "iqtree3",
            "-s",
            str(alignment),
            "-st",
            "AA",
            "-m",
            "LG+G4",
            "-nt",
            "1",
            "-seed",
            str(SEED),
            "-pre",
            str(output / "iqtree"),
        ],
        "fasttree": ["FastTree", "-lg", "-gamma", str(alignment)],
        "raxml": [
            "raxmlHPC-SSE3",
            "-s",
            str(alignment),
            "-m",
            "PROTGAMMALG",
            "-p",
            str(SEED),
            "-n",
            "cox1",
            "-w",
            str(output),
        ],
    }
    for method, command in methods.items():
        execute(
            command,
            output,
            "fasttree.nwk" if method == "fasttree" else method + ".stdout",
            timeout={"iqtree": 900, "fasttree": 60, "raxml": 600}[method],
        )
    shutil.copyfile(output / "iqtree.treefile", output / "iqtree.nwk")
    shutil.copyfile(output / "RAxML_bestTree.cox1", output / "raxml.nwk")
    write_reference(proteins, output)


def write_reference(proteins: Path, output: Path) -> None:
    """Measure complete native outputs independently of the task oracle."""
    records = json.loads(gzip.decompress(proteins.read_bytes()))["proteins"]
    representatives = {}
    for key, record in sorted(records.items()):
        representatives.setdefault(record["sequence"], key)
    aligned = {record.id: str(record.seq).upper() for record in SeqIO.parse(output / "alignment.fa", "fasta")}
    if {key: sequence.replace("-", "") for key, sequence in aligned.items()} != {
        key: sequence for sequence, key in representatives.items()
    }:
        raise ValueError("Reference alignment changed observed proteins")
    if len({len(sequence) for sequence in aligned.values()}) != 1:
        raise ValueError("Ragged reference alignment")
    taxa = sorted(aligned)
    trees = {method: Phylo.read(output / f"{method}.nwk", "newick") for method in ("iqtree", "fasttree", "raxml")}
    splits = {}
    summaries = {}
    for method, tree in trees.items():
        tips = [tip.name for tip in tree.get_terminals()]
        if sorted(tips) != taxa:
            raise ValueError(f"{method} changed leaf identities")
        edges = {}
        for node in tree.find_clades():
            if node is tree.root:
                continue
            descendants = {tip.name for tip in node.get_terminals()}
            sides = (tuple(sorted(descendants)), tuple(sorted(set(taxa) - descendants)))
            side = min(sides, key=lambda values: (len(values), values))
            edges[side] = edges.get(side, 0.0) + node.branch_length
        splits[method] = {side for side in edges if len(side) > 1}
        total = sum(edges.values())
        summaries[method] = {
            "tips": len(tips),
            "edges": len(edges),
            "tree_length": total,
            "treeness": sum(length for side, length in edges.items() if len(side) > 1) / total,
        }
    distances = {
        f"{left}:{right}": {method: tree.distance(left, right) for method, tree in trees.items()}
        for left, right in combinations(taxa, 2)
    }
    comparisons = {
        f"{left}:{right}": {
            "rf_distance": len(splits[left] ^ splits[right]),
            "normalized_rf": len(splits[left] ^ splits[right]) / (len(splits[left]) + len(splits[right])),
        }
        for left, right in combinations(sorted(trees), 2)
    }
    reference = {
        "sequences": len(taxa),
        "source_accessions": len(records),
        "identical_sequence_policy": "Keep lexicographically first accession per unchanged amino-acid sequence",
        "alignment_columns": len(next(iter(aligned.values()))),
        "trees": {method: (output / f"{method}.nwk").read_text() for method in trees},
        "summaries": summaries,
        "distances": distances,
        "comparisons": comparisons,
        "analysis": "Biopython 1.86 Phylo; unrooted weighted splits and patristic distances",
        "input_sha256": hashlib.sha256(proteins.read_bytes()).hexdigest(),
        "model_caveat": (
            "IQ-TREE LG+G4, RAxML PROTGAMMALG and FastTree LG+CAT followed by gamma rescaling "
            "have different optimization procedures. Do not compare raw likelihoods across these runs. "
            "These are single-gene estimates, without a validated species-tree interpretation."
        ),
        "query": {"seed": SEED},
    }
    (output / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proteins", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.proteins.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
