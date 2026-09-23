# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keyed sample aggregation and identifier mapping references."""

from collections import Counter, defaultdict
from pathlib import Path

from experiments.post_training.bio_tasks.solvers.formats import tab_rows, table


def solve_lanes(inputs: Path) -> list[dict]:
    counts = Counter()
    lanes = Counter()
    for row in table(inputs / "samples.csv"):
        if row["include"] != "1":
            continue
        lanes[row["sample"]] += 1
        for count in table(inputs / row["counts"]):
            counts[row["sample"], count["gene"]] += int(count["count"])
    return [
        {"id": sample + ":" + gene, "count": value, "lanes": lanes[sample]} for (sample, gene), value in counts.items()
    ]


def solve_identifiers(inputs: Path) -> list[dict]:
    mapping = defaultdict(set)
    for row in table(inputs / "mapping.csv"):
        mapping[row["symbol"]].add(row["ensembl"])
    query = set((inputs / "query.txt").read_text().splitlines())
    universe = set((inputs / "universe.txt").read_text().splitlines())
    ambiguous = sum(len(mapping[symbol]) > 1 for symbol in query)
    unmapped = sum(not mapping[symbol] for symbol in query)
    selected = set().union(*(mapping[symbol] for symbol in query if len(mapping[symbol]) == 1)) & universe
    return [
        {
            "id": name,
            "overlap": len(selected & set(genes)),
            "mapped_query_size": len(selected),
            "ambiguous_symbols": ambiguous,
            "unmapped_symbols": unmapped,
        }
        for name, _, *genes in tab_rows(inputs / "terms.gmt")
    ]


SOLVERS = {"sample-sheet-lanes": solve_lanes, "enrichment-identifier-mapping": solve_identifiers}
