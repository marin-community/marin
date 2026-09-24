# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct GO reduction, optionally checking every native contribution vector."""

import argparse
import collections
import csv
import gzip
import json
import math
import time
from pathlib import Path


def ancestors(term: str, graph: dict[str, set[str]]) -> dict[str, int]:
    distances = {term: 0}
    frontier = collections.deque([term])
    while frontier:
        child = frontier.popleft()
        for parent in graph.get(child, ()):
            if parent not in distances:
                distances[parent] = distances[child] + 1
                frontier.append(parent)
    return distances


def similarity(left: str, right: str, values: dict[str, dict[str, float]], sums: dict[str, float]) -> float:
    if left == right:
        return 1.0
    common = values[left].keys() & values[right].keys()
    return sum(values[left][a] + values[right][a] for a in common) / (sums[left] + sums[right])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--native-intermediates", type=Path)
    parser.add_argument("--retained-output", type=Path)
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
    view_names = ["adjusted_tested", "raw_genome"] if args.native_intermediates else ["adjusted_tested"]
    views = {}
    for view in view_names:
        with gzip.open(data / f"{view}-ora.tsv.gz", "rt") as f:
            views[view] = [r for r in csv.DictReader(f, delimiter="\t") if float(r["p.adjust"]) < 0.05]
    terms = sorted({r["ID"] for records in views.values() for r in records})
    values = {
        term: {ancestor: 0.7**depth for ancestor, depth in ancestors(term, parents).items() if ancestor != "all"}
        for term in terms
    }
    sums = {term: sum(v.values()) for term, v in values.items()}
    counts = {term: len(ancestors(term, dbparents)) - 1 for term in terms}
    native_checks = {}
    if args.native_intermediates:
        native = collections.defaultdict(dict)
        with gzip.open(args.native_intermediates / "native-s-values.tsv.gz", "rt") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                term, ancestor = row["term"], row["ancestor"]
                assert ancestor not in native[term]
                native[term][ancestor] = float(row["contribution"])
        assert set(native) == set(terms)
        max_error = 0.0
        for term in terms:
            expected = dict(values[term], all=0.0)
            assert native[term].keys() == expected.keys(), term
            for ancestor, value in expected.items():
                error = abs(native[term][ancestor] - value)
                assert error < 1e-14, (term, ancestor, value, native[term][ancestor])
                max_error = max(max_error, error)
        with gzip.open(args.native_intermediates / "native-ancestor-counts.tsv.gz", "rt") as f:
            native_counts = {r["term"]: int(r["ancestor_count"]) for r in csv.DictReader(f, delimiter="\t")}
        assert native_counts == counts
        pair_count = 0
        with gzip.open(args.native_intermediates / "native-pair-scores.tsv.gz", "rt") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                score = similarity(row["left"], row["right"], values, sums)
                assert math.isclose(score, float(row["score"]), rel_tol=0, abs_tol=1e-12), row
                assert round(score, 3) == float(row["rounded_score"]), row
                pair_count += 1
        native_checks = {
            "contribution_vectors": len(native),
            "contribution_entries": sum(len(v) for v in native.values()),
            "maximum_absolute_contribution_error": max_error,
            "ancestor_counts": len(native_counts),
            "native_pair_scores": pair_count,
        }
    reports = {}
    retained = {}
    for view, significant in views.items():
        view_terms = sorted(r["ID"] for r in significant)
        adjusted_p = {r["ID"]: float(r["p.adjust"]) for r in significant}
        removed = set()
        closest_to_rounding_boundary = 1.0
        started = time.monotonic()
        for center in view_terms:
            neighbors = []
            for term in view_terms:
                value = similarity(center, term, values, sums)
                closest_to_rounding_boundary = min(closest_to_rounding_boundary, abs(value - 0.7005))
                if round(value, 3) > 0.7:
                    neighbors.append(term)
            if len(neighbors) > 1:
                best = min(adjusted_p[t] for t in neighbors)
                ties = [t for t in neighbors if adjusted_p[t] == best]
                selected = min(ties, key=lambda t: (-counts[t], t))
                removed.update(t for t in neighbors if t != selected)
        # R and Python rounding at exact half-way values can differ; require separation from the decision boundary.
        assert closest_to_rounding_boundary > 1e-12
        retained[view] = sorted(set(view_terms) - removed)
        native_retained = data / f"{view}-simplified.tsv.gz"
        matched = False
        if native_retained.exists():
            with gzip.open(native_retained, "rt") as f:
                observed = {r["ID"] for r in csv.DictReader(f, delimiter="\t")}
            assert set(retained[view]) == observed, view
            matched = True
        reports[view] = {
            "significant_terms": len(view_terms),
            "retained": len(retained[view]),
            "ordered_pair_comparisons": len(view_terms) ** 2,
            "seconds": time.monotonic() - started,
            "closest_similarity_to_rounding_boundary": closest_to_rounding_boundary,
            "complete_native_retained_set_compared": matched,
        }
    report = {
        "status": "independent-reduction-audited",
        "native_intermediate_checks": native_checks,
        "views": reports,
        "method": (
            "The observed native BP bundle uses only labels that getSV maps to other (0.7). Reconstruct "
            "first-duplicate contributions by shortest-path traversal, apply native three-decimal rounding "
            "and local-neighborhood min-p/GO.db ancestor-count tie-breaking."
        ),
        "limits": (
            "Only completed native reductions are compared as full retained sets. Native intermediate "
            "comparisons, when supplied, cover every S-value and ancestor count but sample pair scores. "
            "The raw-genome full clusterProfiler call timed out. No Harbor or benchmark coverage validation."
        ),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.retained_output:
        args.retained_output.write_text(json.dumps(retained, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
