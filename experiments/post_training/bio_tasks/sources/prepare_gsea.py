# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare native ranked GO enrichment and check each enrichment walk independently."""

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import time
from collections.abc import Iterator
from pathlib import Path

import gseapy
import numpy as np
import pandas as pd


def read_rows(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert importlib.metadata.version("gseapy") == "1.3.0"
    args.output.mkdir(parents=True, exist_ok=True)
    genes = {}
    seen_genes = set()
    excluded_genes = []
    for row in read_rows(args.source / "gene-results.tsv.gz"):
        gene = row["gene"]
        assert gene not in seen_genes and gene.isdecimal()
        seen_genes.add(gene)
        if row["stat"] == "NA" or not math.isfinite(float(row["stat"])):
            excluded_genes.append(gene)
        else:
            genes[gene] = float(row["stat"])
    ordered = sorted(genes.items(), key=lambda pair: (-pair[1], pair[0]))
    ranking = pd.DataFrame(ordered, columns=["gene", "statistic"])
    ranking.to_csv(args.output / "ranking.tsv", sep="\t", index=False, float_format="%.17g")
    sets = {}
    for row in read_rows(args.source / "full-bp-membership.tsv.gz"):
        sets.setdefault(row["term"], set()).add(row["gene"])
    sets = {term: sorted(members) for term, members in sorted(sets.items())}
    eligible = {}
    set_status = []
    for term, members in sets.items():
        observed = set(members) & genes.keys()
        status = "eligible" if 15 <= len(observed) <= 500 else "below_minimum" if len(observed) < 15 else "above_maximum"
        set_status.append(
            {"term": term, "annotated_genes": len(members), "ranked_genes": len(observed), "status": status}
        )
        if status == "eligible":
            eligible[term] = observed
    pd.DataFrame(set_status).to_csv(args.output / "set-status.tsv", sep="\t", index=False)
    with gzip.open(args.output / "go-bp.gmt.gz", "wt") as stream:
        for term, members in sets.items():
            stream.write("\t".join([term, "org.Mm.eg.db-3.22.0-BP", *members]) + "\n")
    started = time.monotonic()
    result = gseapy.prerank(
        rnk=ranking,
        gene_sets=sets,
        organism="mouse",
        outdir=None,
        min_size=15,
        max_size=500,
        permutation_num=1000,
        weight=1,
        ascending=None,
        threads=1,
        seed=42,
        no_plot=True,
        verbose=True,
        method="permutation",
    )
    native_seconds = time.monotonic() - started
    assert list(result.ranking.index) == [gene for gene, _ in ordered]
    np.testing.assert_array_equal(result.ranking.to_numpy(), ranking["statistic"].to_numpy())
    assert set(result.results) == set(eligible)
    ranked_ids = np.array([gene for gene, _ in ordered])
    statistics = np.array([value for _, value in ordered])
    positions = {gene: index for index, gene in enumerate(ranked_ids)}
    max_es_error = 0.0
    for term, members in eligible.items():
        native = result.results[term]
        hits = np.array(sorted(positions[gene] for gene in members))
        assert list(native["hits"]) == hits.tolist()
        weights = np.abs(statistics[hits])
        assert weights.sum() > 0 and len(hits) < len(statistics)
        walk = np.full(len(statistics), -1.0 / (len(statistics) - len(hits)))
        walk[hits] = weights / weights.sum()
        walk = np.cumsum(walk)
        high, low = int(walk.argmax()), int(walk.argmin())
        extreme = high if abs(walk[high]) > abs(walk[low]) else low
        expected_es = float(walk[extreme])
        error = abs(float(native["es"]) - expected_es)
        max_es_error = max(max_es_error, error)
        assert error <= 1e-8, (term, native["es"], expected_es)
        leading = hits[hits <= extreme] if expected_es >= 0 else hits[hits >= extreme][::-1]
        assert native["lead_genes"].split(";") == ranked_ids[leading].tolist(), term
        assert set(native["matched_genes"].split(";")) == members, term
        for field in ["pval", "fdr", "fwerp"]:
            assert math.isfinite(native[field]) and 0 <= native[field] <= 1, (term, field)
        assert math.isfinite(native["nes"]), term
    table = result.res2d.copy()
    assert table["Term"].is_unique and set(table["Term"]) == set(eligible)
    table.to_csv(args.output / "gsea-results.tsv", sep="\t", index=False, float_format="%.17g")
    table["abs_nes"] = table["NES"].abs()
    top5 = table.sort_values(["abs_nes", "Term"], ascending=[False, True]).head(5)
    top10 = (
        table[table["FDR q-val"] < 0.05]
        .sort_values(["FDR q-val", "abs_nes", "Term"], ascending=[True, False, True])
        .head(10)
    )
    for name, view in [("top5-absolute-nes", top5), ("top10-significant", top10)]:
        view.drop(columns=["abs_nes"]).to_csv(args.output / f"{name}.tsv", sep="\t", index=False, float_format="%.17g")
    metadata = {
        "status": "native-preparation-and-enrichment-walk-audit-only",
        "gseapy_version": importlib.metadata.version("gseapy"),
        "ranked_genes": len(genes),
        "nonfinite_statistic_genes": excluded_genes,
        "unannotated_ranked_genes": len(genes.keys() - set().union(*sets.values())),
        "genes_without_eligible_set": len(genes.keys() - set().union(*eligible.values())),
        "gene_sets": len(sets),
        "eligible_gene_sets": len(eligible),
        "native_seconds": native_seconds,
        "maximum_absolute_es_error": max_es_error,
        "top5_rows": len(top5),
        "top10_rows": len(top10),
        "limits": [
            "No Harbor task validation.",
            "Native NES/p/FDR have finite/range checks; null distributions are not independently recomputed.",
            "No benchmark coverage promoted.",
        ],
    }
    (args.output / "native-gsea-result.json").write_text(json.dumps(metadata, indent=2) + "\n")
    files = {}
    for path in sorted(args.output.iterdir()):
        if path.is_file():
            with path.open("rb") as stream:
                files[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    (args.output / "file-hashes.json").write_text(json.dumps(files, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
