# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assembly metrics, taxonomy ancestry, and ecological indices."""

import math
import re
from collections import defaultdict
from functools import partial
from itertools import combinations
from pathlib import Path

from experiments.post_training.bio_tasks.solvers.formats import fasta, tab_rows, table


def solve_assembly(inputs: Path, operation: str) -> list[dict]:
    answer = []
    if operation == "assembly-nx":
        lengths = sorted(map(len, fasta(inputs / "assembly.fa").values()), reverse=True)
        for prefix, total in [("N", sum(lengths)), ("NG", int((inputs / "genome_size.txt").read_text()))]:
            for percentile in [50, 90]:
                accumulated = 0
                result = {"id": f"{prefix}{percentile}", "length": None, "contigs": None}
                for i, length in enumerate(lengths, 1):
                    accumulated += length
                    if accumulated * 100 >= total * percentile:
                        result.update(length=length, contigs=i)
                        break
                answer.append(result)
    elif operation == "assembly-gap-runs":
        for name, sequence in fasta(inputs / "assembly.fa").items():
            for i, match in enumerate(re.finditer("N{10,}", sequence.upper()), 1):
                answer.append(
                    {
                        "id": f"{name}:{i}",
                        "start": match.start(),
                        "end": match.end(),
                        "length": match.end() - match.start(),
                    }
                )
    elif operation == "contig-depth-breadth":
        depths = {name: [0] * len(sequence) for name, sequence in fasta(inputs / "assembly.fa").items()}
        for name, start, end, depth in tab_rows(inputs / "depth.bedgraph"):
            depths[name][int(start) : int(end)] = [int(depth)] * (int(end) - int(start))
        for name, values in depths.items():
            answer.append(
                {
                    "id": name,
                    "mean_depth": sum(values) / len(values),
                    "breadth_ge_three": sum(value >= 3 for value in values) / len(values),
                }
            )
    elif operation == "taxonomic-lca":
        parents = {}
        for line in (inputs / "nodes.dmp").read_text().splitlines():
            fields = [part.strip() for part in line.split("|")]
            parents[int(fields[0])] = int(fields[1])

        def lineage(taxid: int) -> list[int]:
            result = [taxid]
            while parents[taxid] != taxid:
                taxid = parents[taxid]
                result.append(taxid)
            return result

        hits = defaultdict(list)
        for row in table(inputs / "hits.csv"):
            hits[row["query"]].append(int(row["taxid"]))
        for name, ids in hits.items():
            paths = [lineage(taxid) for taxid in ids if taxid]
            ancestor = next(node for node in paths[0] if all(node in path for path in paths)) if paths else 0
            answer.append({"id": name, "taxid": ancestor})
    else:
        rows = table(inputs / "counts.csv")
        samples = {name: [int(row[name]) for row in rows] for name in rows[0] if name != "taxon"}
        proportions = {
            name: [value / sum(values) if sum(values) else 0.0 for value in values] for name, values in samples.items()
        }
        if operation == "bray-curtis":
            for a, b in combinations(sorted(samples), 2):
                denominator = sum(proportions[a]) + sum(proportions[b])
                numerator = sum(abs(x - y) for x, y in zip(proportions[a], proportions[b], strict=True))
                answer.append({"id": a + ":" + b, "distance": numerator / denominator if denominator else 0.0})
        else:
            for name, values in proportions.items():
                positive = [p for p in values if p > 0]
                answer.append(
                    {
                        "id": name,
                        "richness": len(positive),
                        "shannon": -sum(p * math.log(p) for p in positive) if positive else None,
                        "inverse_simpson": 1 / sum(p * p for p in positive) if positive else None,
                    }
                )
    return answer


NAMES = ("assembly-nx", "assembly-gap-runs", "contig-depth-breadth", "taxonomic-lca", "bray-curtis", "alpha-diversity")
SOLVERS = {name: partial(solve_assembly, operation=name) for name in NAMES}
