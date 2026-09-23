# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample wiring and identifier harmonization with downstream biological checks."""

import random
from functools import partial

from experiments.post_training.bio_tasks.contract import Column, Contract
from experiments.post_training.bio_tasks.recipe_types import Difficulty, Instance, Recipe, csv_text


def generate_workflow(seed: int, operation: str) -> Instance:
    rng = random.Random(seed)
    inputs, expected = {}, {}
    if operation == "sample-sheet-lanes":
        manifest = []
        for sample, lanes in [("patient_a", ["L1", "L2"]), ("patient_b", ["L2"])]:
            totals = {"g0": 0, "g1": 0}
            for lane in lanes:
                filename = f"{sample}_{lane}.csv"
                rows = []
                for gene in totals:
                    value = rng.randint(2, 15)
                    totals[gene] += value
                    rows.append({"gene": gene, "count": value})
                inputs[filename] = csv_text(rows)
                manifest.append({"sample": sample, "lane": lane, "counts": filename, "include": 1})
            for gene, value in totals.items():
                expected[sample + ":" + gene] = {"count": value, "lanes": len(lanes)}
        inputs["excluded.csv"] = csv_text([{"gene": "g0", "count": 999}, {"gene": "g1", "count": 999}])
        manifest.append({"sample": "patient_a", "lane": "L3", "counts": "excluded.csv", "include": 0})
        rng.shuffle(manifest)
        inputs["samples.csv"] = csv_text(manifest)
        columns = {
            "count": Column(
                kind="integer", unit="reads", description="sum over included lanes for this sample and gene"
            ),
            "lanes": Column(kind="integer", unit="lanes", description="number of included lane files for sample"),
        }
        prompt = (
            "Assemble biological-sample counts using samples.csv. Load only rows with include=1, and "
            "sum count files by the explicit sample key and gene ID. Lanes are technical splits, not "
            "additional biological samples. Do not form a Cartesian product of samples and lane names, "
            "and do not include excluded files. Return every included sample:gene with summed raw count"
            " and included lane count."
        )
        wrong = [
            {"id": k, **v, "count": v["count"] + 999 if k.startswith("patient_a:") else v["count"]}
            for k, v in expected.items()
        ]
        reason = "included_excluded_lane"
    else:
        assert operation == "enrichment-identifier-mapping"
        offset = rng.randint(100, 900)
        identifiers = [f"ENSG{offset+i}" for i in range(4)]
        mapping = [
            {"symbol": "ALPHA", "ensembl": identifiers[0]},
            {"symbol": "BETA", "ensembl": identifiers[1]},
            {"symbol": "AMBIG", "ensembl": identifiers[2]},
            {"symbol": "AMBIG", "ensembl": identifiers[3]},
            {"symbol": "ALIAS_ALPHA", "ensembl": identifiers[0]},
        ]
        include_beta = rng.choice([True, False])
        unknown_count = rng.randint(1, 3)
        universe = [identifiers[0], identifiers[2], *([identifiers[1]] if include_beta else [])]
        inputs = {
            "mapping.csv": csv_text(mapping),
            "query.txt": (
                "ALPHA\nALIAS_ALPHA\nBETA\nAMBIG\nALPHA\n" + "".join(f"UNKNOWN{i}\n" for i in range(unknown_count))
            ),
            "universe.txt": "\n".join(universe) + "\n",
            "terms.gmt": f"term1\tna\t{identifiers[0]}\t{identifiers[1]}\nterm2\tna\t{identifiers[2]}\n",
        }
        expected = {
            "term1": {
                "overlap": 1 + int(include_beta),
                "mapped_query_size": 1 + int(include_beta),
                "ambiguous_symbols": 1,
                "unmapped_symbols": unknown_count,
            },
            "term2": {
                "overlap": 0,
                "mapped_query_size": 1 + int(include_beta),
                "ambiguous_symbols": 1,
                "unmapped_symbols": unknown_count,
            },
        }
        columns = {
            name: Column(
                kind="integer",
                unit="identifiers" if name in {"overlap", "mapped_query_size"} else "distinct input symbols",
                description=description,
            )
            for name, description in [
                ("overlap", "unique mapped query IDs in term"),
                ("mapped_query_size", "unique unambiguous IDs after universe restriction"),
                ("ambiguous_symbols", "symbols with multiple distinct targets before universe filtering"),
                ("unmapped_symbols", "symbols with no mapping"),
            ]
        }
        prompt = (
            "Harmonize query.txt symbols to Ensembl IDs using mapping.csv before counting terms.gmt "
            "overlaps. Deduplicate input symbols; discard symbols mapping to multiple distinct IDs "
            "before restricting to universe.txt; discard unmapped symbols; then restrict IDs to the "
            "universe and deduplicate aliases. Report every term overlap and global mapped-query size, "
            "distinct ambiguous-symbol count and distinct unmapped-symbol count. Use term id. Do not "
            "resolve ambiguity merely because one target is outside the universe."
        )
        wrong = [{"id": k, **v, "mapped_query_size": v["mapped_query_size"] + 1} for k, v in expected.items()]
        reason = "resolved_ambiguity_after_universe_filtering"
    return Instance(
        prompt + " Inputs are in /app/inputs.", inputs, Contract(columns=columns, expected=expected), {reason: wrong}
    )


SKILLS = {
    "sample-sheet-lanes": ("sample-key-joins", "technical-lanes", "include-policy"),
    "enrichment-identifier-mapping": ("ambiguous-mapping", "aliases", "universe-order"),
}
RECIPES = tuple(
    Recipe(
        name,
        "2" if name in ["enrichment-identifier-mapping"] else "1",
        Difficulty.MEDIUM,
        skills,
        ("csv-header",) if name == "sample-sheet-lanes" else ("csv-header", "gmt", "gene-lists"),
        (
            ("https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html",)
            if name == "sample-sheet-lanes"
            else ("https://bioconductor.org/packages/release/bioc/html/AnnotationDbi.html",)
        ),
        partial(generate_workflow, operation=name),
    )
    for name, skills in SKILLS.items()
)
