# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Infer COX1 gene trees from the supplied alignment and compare their weighted splits."""

import csv
import json
import shutil
import subprocess
from itertools import combinations
from pathlib import Path

from experiments.post_training.bio_tasks.solvers.newick import newick, weighted_splits


def solve_phylogeny(inputs: Path, output: Path) -> list[dict]:
    query = json.loads((inputs / "query.json").read_text())
    alignment = inputs / "alignment.fa"
    commands = {
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
            str(query["seed"]),
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
            str(query["seed"]),
            "-n",
            "cox1",
            "-w",
            str(output),
        ],
    }
    for method, command in commands.items():
        filename = "fasttree.nwk" if method == "fasttree" else method + ".stdout"
        with (output / filename).open("wb") as stdout, (output / (method + ".stderr")).open("wb") as stderr:
            subprocess.run(command, check=True, stdout=stdout, stderr=stderr, timeout=900 if method == "iqtree" else 300)
    shutil.copyfile(output / "iqtree.treefile", output / "iqtree.nwk")
    shutil.copyfile(output / "RAxML_bestTree.cox1", output / "raxml.nwk")
    trees = {method: weighted_splits(newick((output / f"{method}.nwk").read_text())) for method in commands}
    taxa = trees["iqtree"][0]
    if any(tips != taxa for tips, _ in trees.values()):
        raise ValueError("Inference methods returned different leaf identities")
    summaries = []
    splits = {}
    for method, (tips, edges) in trees.items():
        total = sum(edges.values())
        splits[method] = {key for key in edges if "," in key}
        summaries.append(
            {
                "id": method,
                "tips": len(tips),
                "edges": len(edges),
                "tree_length": total,
                "treeness": sum(length for key, length in edges.items() if "," in key) / total,
            }
        )
    edge_sets = {
        method: [(set(key.split(",")), length) for key, length in edges.items()] for method, (_, edges) in trees.items()
    }
    with (output / "distances.tsv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", *commands], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for left, right in combinations(taxa, 2):
            writer.writerow(
                {
                    "id": f"{left}:{right}",
                    **{
                        method: sum(length for side, length in edges if (left in side) != (right in side))
                        for method, edges in edge_sets.items()
                    },
                }
            )
    with (output / "comparisons.tsv").open("w") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "rf_distance", "normalized_rf"], delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for left, right in combinations(sorted(trees), 2):
            distance = len(splits[left] ^ splits[right])
            writer.writerow(
                {
                    "id": f"{left}:{right}",
                    "rf_distance": distance,
                    "normalized_rf": distance / (len(splits[left]) + len(splits[right])),
                }
            )
    return summaries
