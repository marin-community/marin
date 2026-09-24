# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check native graph identity, relation semantics and eligible-term representation."""

import argparse
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the pinned native GO graph exports.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    data = args.source
    relation_counts = collections.Counter()
    bundled = set()
    bundled_terms = set()
    parents = collections.defaultdict(set)
    with gzip.open(data / "gosemsim-bundled-graph.tsv.gz", "rt") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["Ontology"] != "BP":
                continue
            relation_counts[row["relationship"]] += 1
            bundled.add((row["go_id"], row["parent"], row["relationship"]))
            bundled_terms.add(row["go_id"])
            bundled_terms.add(row["parent"])
            parents[row["go_id"]].add(row["parent"])
    database = set()
    database_terms = set()
    with gzip.open(data / "go-db-bp-parents.tsv.gz", "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        assert header == ["go_id", "go_id", "RelationshipType"]
        for child, parent, relationship in reader:
            database.add((child, parent, relationship))
            database_terms.add(child)
            database_terms.add(parent)
    visited = set()
    active = set()

    def visit(node):
        assert node not in active, "Bundled graph contains a cycle"
        if node in visited:
            return
        active.add(node)
        for parent in parents.get(node, ()):
            visit(parent)
        active.remove(node)
        visited.add(node)

    for node in list(parents):
        visit(node)
    views = {}
    for name in ["adjusted_tested", "raw_genome"]:
        with gzip.open(data / (name + "-ora.tsv.gz"), "rt") as f:
            significant = {r["ID"] for r in csv.DictReader(f, delimiter="\t") if float(r["p.adjust"]) < 0.05}
        views[name] = {
            "significant_terms": len(significant),
            "absent_from_bundled_graph": sorted(significant - bundled_terms),
            "absent_from_go_db_graph": sorted(significant - database_terms),
        }
    report = {
        "status": "native-bundled-graph-inspected",
        "bundled_bp_terms_including_parents": len(bundled_terms),
        "bundled_unique_bp_edges": len(bundled),
        "bundled_bp_relationship_counts": dict(relation_counts),
        "effective_native_weight_by_literal_label": {
            label: 0.8 if label == "is_a" else 0.6 if label == "part_of" else 0.7 for label in relation_counts
        },
        "go_db_bp_terms_including_parents": len(database_terms),
        "go_db_unique_bp_edges": len(database),
        "exact_edges_only_in_bundle": len(bundled - database),
        "exact_edges_only_in_go_db": len(database - bundled),
        "bundled_graph_acyclic": True,
        "views": views,
        "export_schema_note": (
            "GO.db toTable has duplicate go_id column names; consume child/parent by position or rename "
            "explicitly before dictionary parsing."
        ),
        "weight_note": (
            "The actual bundled labels are interpreted literally by pinned getSV. It treats every label "
            "except is_a and part_of as other (0.7). Do not silently normalize labels when reproducing "
            "native Wang scores."
        ),
        "file_sha256": {
            name: hashlib.sha256((data / name).read_bytes()).hexdigest()
            for name in ["gosemsim-bundled-graph.tsv.gz", "go-db-bp-parents.tsv.gz"]
        },
        "limits": (
            "No independent full similarity-matrix or retained-term recomputation yet; no benchmark mapping promoted."
        ),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
