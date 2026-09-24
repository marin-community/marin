# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce native retained terms under the observed uniform relation weights."""

import argparse
import collections
import csv
import gzip
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the completed 406-term native simplification case.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    data = args.source
    parents = collections.defaultdict(set)
    with gzip.open(data / "gosemsim-bundled-graph.tsv.gz", "rt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["Ontology"] == "BP":
                assert r["relationship"] not in ["is_a", "part_of"]
                parents[r["go_id"]].add(r["parent"])
    dbparents = collections.defaultdict(set)
    with gzip.open(data / "go-db-bp-parents.tsv.gz", "rt") as f:
        rows = csv.reader(f, delimiter="\t")
        assert next(rows) == ["go_id", "go_id", "RelationshipType"]
        for child, parent, _relationship in rows:
            dbparents[child].add(parent)

    def ancestors(term, graph):
        distances = {term: 0}
        frontier = collections.deque([term])
        while frontier:
            child = frontier.popleft()
            for parent in graph.get(child, ()):
                if parent not in distances:
                    distances[parent] = distances[child] + 1
                    frontier.append(parent)
        return distances

    with gzip.open(data / "adjusted_tested-ora.tsv.gz", "rt") as f:
        significant = [r for r in csv.DictReader(f, delimiter="\t") if float(r["p.adjust"]) < 0.05]
    terms = sorted(r["ID"] for r in significant)
    p = {r["ID"]: float(r["p.adjust"]) for r in significant}
    sv = {
        term: {ancestor: 0.7**depth for ancestor, depth in ancestors(term, parents).items() if ancestor != "all"}
        for term in terms
    }
    sums = {term: sum(v.values()) for term, v in sv.items()}
    counts = {term: len(ancestors(term, dbparents)) - 1 for term in terms}
    removed = set()
    closest_to_rounding_boundary = 1.0
    started = time.monotonic()
    comparisons = 0
    for center in terms:
        neighbors = []
        for term in terms:
            common = sv[center].keys() & sv[term].keys()
            value = (
                1.0 if term == center else sum(sv[center][a] + sv[term][a] for a in common) / (sums[center] + sums[term])
            )
            closest_to_rounding_boundary = min(closest_to_rounding_boundary, abs(value - 0.7005))
            if round(value, 3) > 0.7:
                neighbors.append(term)
            comparisons += 1
        if len(neighbors) > 1:
            best = min(p[t] for t in neighbors)
            ties = [t for t in neighbors if p[t] == best]
            selected = min(ties, key=lambda t: (-counts[t], t))
            removed.update(t for t in neighbors if t != selected)
    expected = set(terms) - removed
    with gzip.open(data / "adjusted_tested-simplified.tsv.gz", "rt") as f:
        observed = {r["ID"] for r in csv.DictReader(f, delimiter="\t")}
    assert expected == observed, {"missing": sorted(observed - expected), "extra": sorted(expected - observed)}
    report = {
        "status": "native-adjusted-tested-retained-set-reproduced",
        "significant_terms": len(terms),
        "retained": len(expected),
        "ordered_pair_comparisons": comparisons,
        "seconds": time.monotonic() - started,
        "closest_similarity_to_rounding_boundary": closest_to_rounding_boundary,
        "method": (
            "The observed native BP bundle uses only labels that getSV maps to other (0.7). First "
            "duplicate ancestors then equal shortest-path depths; reconstruct contributions by "
            "breadth-first traversal, apply native three-decimal rounding and local-neighborhood "
            "min-p/ancestor-count tie-breaking."
        ),
        "limits": (
            "Matches the complete retained-term set for the 406-term native case. Does not compare "
            "every native similarity value and does not validate the unfinished 3786-term case."
        ),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
