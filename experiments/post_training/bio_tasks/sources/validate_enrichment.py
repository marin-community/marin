# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independently check native enrichment artifacts without loading the full membership matrix."""

import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
import math
import tarfile
from collections.abc import Iterator
from pathlib import Path

from scipy.stats import hypergeom, norm


def rows(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def identifiers(path: Path) -> set[str]:
    values = [row["gene"] for row in rows(path)]
    assert len(values) == len(set(values)), f"Duplicate gene identity: {path.name}"
    return set(values)


def finite(value: str) -> bool:
    return value != "NA" and math.isfinite(float(value))


def probability_matches(actual: float, expected: float) -> None:
    assert math.isfinite(actual) and 0 <= actual <= 1
    if expected == 0:
        assert actual < 1e-290
    elif expected < 1e-290:
        assert actual < 1e-280
    else:
        assert actual > 0 and abs(math.log(actual) - math.log(expected)) < 1e-8, (actual, expected)


def verify(source: Path, input_bundle: Path) -> dict:
    plan = json.loads((source / "launch-plan.json").read_text())
    assert (
        hashlib.sha256((source / "prepare_enrichment.R").read_bytes()).hexdigest()
        == plan["file_hashes"]["prepare_enrichment.R"]
    )
    assert json.loads((source / "namespace-preflight.json").read_text())["exit_code"] == 0
    assert json.loads((source / "analysis.json").read_text())["exit_code"] == 124
    data = source / "references"
    assert not (data / "raw_genome-simplified.tsv.gz").exists()
    assert not (source / "result.json").exists()
    genes = {}
    for row in rows(data / "gene-results.tsv.gz"):
        assert row["gene"] not in genes
        genes[row["gene"]] = row
    eligible = identifiers(data / "eligibility.tsv.gz")
    eligible_fit = {row["gene"] for row in rows(data / "eligibility.tsv.gz") if row["eligible"] == "TRUE"}
    assert len(eligible) == 27179 and set(genes) == eligible_fit and len(genes) == 16659
    selected_raw = {gene for gene, row in genes.items() if finite(row["pvalue"]) and float(row["pvalue"]) < 0.05}
    selected_adjusted = {gene for gene, row in genes.items() if finite(row["padj"]) and float(row["padj"]) < 0.05}
    finite_genes = {gene for gene, row in genes.items() if finite(row["pvalue"])}
    # Independently reconstruct eligibility and normalized means from the observed input bundle.
    factors = {r["sample"]: float(r["size_factor"]) for r in rows(data / "size-factors.tsv.gz")}
    native_eligibility = {r["gene"]: r for r in rows(data / "eligibility.tsv.gz")}
    seen_input = set()
    with tarfile.open(input_bundle, "r:gz") as bundle:
        samples = list(csv.DictReader(io.TextIOWrapper(bundle.extractfile("inputs/samples.tsv")), delimiter="\t"))
        selected_samples = sorted(r["sample"] for r in samples if r["population"] == "luminal")
        assert len(selected_samples) == 6 and set(factors) == set(selected_samples)
        reader = csv.DictReader(io.TextIOWrapper(bundle.extractfile("inputs/counts.tsv")), delimiter="\t")
        gene_column = reader.fieldnames[0]
        for row in reader:
            gene = row[gene_column]
            assert gene not in seen_input
            seen_input.add(gene)
            values = [int(row[sample]) for sample in selected_samples]
            total = sum(values)
            assert int(native_eligibility[gene]["count_sum"]) == total
            assert (native_eligibility[gene]["eligible"] == "TRUE") == (total >= 10)
            if total >= 10:
                mean = sum(value / factors[sample] for value, sample in zip(values, selected_samples, strict=True)) / 6
                assert math.isclose(float(genes[gene]["baseMean"]), mean, rel_tol=1e-10, abs_tol=1e-10)
    assert seen_input == eligible
    for row in genes.values():
        if finite(row["stat"]):
            assert math.isclose(
                float(row["stat"]), float(row["log2FoldChange"]) / float(row["lfcSE"]), rel_tol=1e-10, abs_tol=1e-10
            )
        if finite(row["pvalue"]):
            probability_matches(float(row["pvalue"]), 2 * float(norm.sf(abs(float(row["stat"])))))
    backgrounds = {name: identifiers(data / f"{name}-background.tsv.gz") for name in ["adjusted_tested", "raw_genome"]}
    selections = {name: identifiers(data / f"{name}-selected.tsv.gz") for name in backgrounds}
    annotation_genes = set()
    sizes = {name: collections.Counter() for name in backgrounds}
    overlaps = {name: collections.Counter() for name in backgrounds}
    memberships = 0
    for row in rows(data / "full-bp-membership.tsv.gz"):
        gene, term = row["gene"], row["term"]
        annotation_genes.add(gene)
        memberships += 1
        for name, background in backgrounds.items():
            if gene in background:
                sizes[name][term] += 1
            if gene in selections[name]:
                overlaps[name][term] += 1
    assert backgrounds["raw_genome"] == annotation_genes
    assert backgrounds["adjusted_tested"] == finite_genes & annotation_genes
    assert selections["raw_genome"] == selected_raw & annotation_genes
    assert selections["adjusted_tested"] == selected_adjusted & finite_genes & annotation_genes
    reports = {}
    for name, background in backgrounds.items():
        native = list(rows(data / f"{name}-ora.tsv.gz"))
        terms = [row["ID"] for row in native]
        expected_terms = {term for term, size in sizes[name].items() if 10 <= size <= 500 and overlaps[name][term] > 0}
        assert len(terms) == len(set(terms)) and set(terms) == expected_terms
        previous_p = -1
        for row in native:
            term = row["ID"]
            K, k = sizes[name][term], overlaps[name][term]
            N, n = len(background), len(selections[name])
            assert row["GeneRatio"] == f"{k}/{n}" and row["BgRatio"] == f"{K}/{N}"
            assert int(row["Count"]) == k
            actual = float(row["pvalue"])
            assert actual >= previous_p
            previous_p = actual
            probability_matches(actual, float(hypergeom.sf(k - 1, N, K, n)))
        # The native result is already ordered by p; this streams the BH recurrence in reverse.
        running_adjusted = 1.0
        for index in range(len(native) - 1, -1, -1):
            row = native[index]
            running_adjusted = min(running_adjusted, len(native) * float(row["pvalue"]) / (index + 1))
            probability_matches(float(row["p.adjust"]), running_adjusted)
        significant = {row["ID"] for row in native if float(row["p.adjust"]) < 0.05}
        if name == "raw_genome":
            reports[name] = {
                "background": len(background),
                "selected": len(selections[name]),
                "native_tests": len(terms),
                "significant": len(significant),
                "semantic_reduction": "not completed; analysis timed out",
            }
            continue
        retained = list(rows(data / f"{name}-retained.tsv.gz"))
        assert len(retained) == len(significant) and {row["term"] for row in retained} == significant
        simplified = list(rows(data / f"{name}-simplified.tsv.gz"))
        assert {row["ID"] for row in simplified} == {row["term"] for row in retained if row["retained"] == "TRUE"}
        source_rows = {row["ID"]: row for row in native}
        for row in simplified:
            assert row == source_rows[row["ID"]]
        reports[name] = {
            "background": len(background),
            "selected": len(selections[name]),
            "native_tests": len(terms),
            "zero_overlap_eligible": sum(
                10 <= size <= 500 and not overlaps[name][term] for term, size in sizes[name].items()
            ),
            "significant": len(significant),
            "retained": len(simplified),
        }
    return {
        "status": "completed-stages-audited-from-timed-out-run",
        "input_genes": len(eligible),
        "fitted_genes": len(genes),
        "full_annotation_genes": len(annotation_genes),
        "membership_rows": memberships,
        "views": reports,
        "probability_comparison": (
            "Absolute log difference <1e-8 above 1e-290; lower tails receive only a 1e-280 upper-bound "
            "check (or 1e-290 for reference underflow)."
        ),
        "limits": [
            "The native job timed out; raw-genome semantic reduction is incomplete.",
            "No Harbor task or solver run.",
            "Wang reduction recomputation and bundled graph reconciliation remain separate checks.",
            "GSEA not run by this preparation.",
            "No new validated benchmark mapping.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--input-bundle", required=True, type=Path)
    args = parser.parse_args()
    report = verify(args.source, args.input_bundle)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
