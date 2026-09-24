# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assembly quality and community ecology task generators."""

import math
import random
from functools import partial
from itertools import combinations

from experiments.post_training.bio_tasks.contract import Column, Contract
from experiments.post_training.bio_tasks.recipe_types import Instance, Recipe, csv_text


def generate_assembly(seed: int, operation: str) -> Instance:
    rng = random.Random(seed)
    inputs, expected = {}, {}
    if operation == "assembly-nx":
        unit = rng.randint(10, 25)
        lengths = [8 * unit, 4 * unit, 2 * unit, unit]
        rng.shuffle(lengths)
        inputs = {
            "assembly.fa": "".join(
                f">c{i}\n" + ("ACGTN" * ((length + 4) // 5))[:length] + "\n" for i, length in enumerate(lengths)
            ),
            "genome_size.txt": str(20 * unit) + "\n",
        }
        expected = {
            "N50": {"length": 8 * unit, "contigs": 1},
            "N90": {"length": 2 * unit, "contigs": 3},
            "NG50": {"length": 4 * unit, "contigs": 2},
            "NG90": {"length": None, "contigs": None},
        }
        columns = {
            "length": Column(
                kind="integer",
                unit="bases",
                description="contig length at cumulative threshold, null if not reached",
                nullable=True,
            ),
            "contigs": Column(
                kind="integer",
                unit="contigs",
                description="number accumulated through threshold, null if not reached",
                nullable=True,
            ),
        }
        prompt = (
            "Calculate N50, N90, NG50, and NG90 for assembly.fa. Count all sequence characters "
            "including N toward contig length. Sort contigs by decreasing length; N uses total assembly"
            " length, NG uses genome_size.txt. Report the crossing contig length and number of contigs "
            "accumulated. Unreached thresholds give null for both fields. Use metric name as id."
        )
        wrong = [{"id": k, **v, "length": 4 * unit if k == "N50" else v["length"]} for k, v in expected.items()]
        reason = "used_median_contig_length"
    elif operation == "assembly-gap-runs":
        records = []
        for i in range(3):
            a, b, c = [rng.randint(3, 8) for _ in range(3)]
            gap = rng.randint(10, 15)
            sequence = "A" * a + "N" * gap + "C" * b + "n" * 5 + "G" * c
            records.append(f">c{i}\n{sequence}\n")
            expected[f"c{i}:1"] = {"start": a, "end": a + gap, "length": gap}
        inputs = {"assembly.fa": "".join(records)}
        columns = {
            name: Column(kind="integer", unit="0-based bases" if name != "length" else "bases", description=description)
            for name, description in [
                ("start", "inclusive gap start"),
                ("end", "exclusive gap end"),
                ("length", "length of maximal N run"),
            ]
        }
        prompt = (
            "Find maximal case-insensitive N runs of at least 10 bases in assembly.fa. Shorter runs are"
            " not assembly gaps for this task. Return 0-based half-open start/end and length for each "
            "qualifying run, using contig:1-based_qualifying_gap_index in coordinate order."
        )
        wrong = [{"id": k, **v, "end": v["end"] - 1} for k, v in expected.items()]
        reason = "closed_interval_gap_end"
    elif operation == "contig-depth-breadth":
        bedgraph = []
        fasta = []
        for i in range(3):
            length = rng.randint(30, 45)
            depth = rng.randint(3, 8)
            a, b = rng.randint(3, 8), rng.randint(5, 10)
            fasta.append(f">c{i}\n" + "A" * length + "\n")
            bedgraph.extend([f"c{i}\t0\t{a}\t2", f"c{i}\t{a}\t{a+b}\t{depth}"])
            expected[f"c{i}"] = {"mean_depth": (2 * a + depth * b) / length, "breadth_ge_three": b / length}
        inputs = {"assembly.fa": "".join(fasta), "depth.bedgraph": "\n".join(bedgraph) + "\n"}
        columns = {
            "mean_depth": Column(
                kind="number",
                unit="reads per base",
                description="mean across entire contig including absent zero intervals",
                atol=1e-10,
                rtol=1e-8,
            ),
            "breadth_ge_three": Column(
                kind="number",
                unit="fraction",
                description="fraction of full contig with depth >=3",
                atol=1e-10,
                rtol=1e-8,
            ),
        }
        prompt = (
            "Compute mean depth and fraction of bases at depth >=3 for each assembly.fa contig from "
            "nonoverlapping 0-based half-open depth.bedgraph intervals. Missing regions have depth "
            "zero. Weight by interval length and use full FASTA contig length as denominator. Use "
            "contig id."
        )
        wrong = [{"id": k, **v, "breadth_ge_three": 1.0} for k, v in expected.items()]
        reason = "ignored_uncovered_bases"
    elif operation == "taxonomic-lca":
        root = 1
        offset = rng.randint(10, 50) * 10
        genus_a, genus_b, s1, s2, s3 = [offset + i for i in range(5)]
        nodes = [
            (root, root, "no rank"),
            (genus_a, root, "genus"),
            (genus_b, root, "genus"),
            (s1, genus_a, "species"),
            (s2, genus_a, "species"),
            (s3, genus_b, "species"),
        ]
        inputs = {
            "nodes.dmp": "".join(f"{node}\t|\t{parent}\t|\t{rank}\t|\n" for node, parent, rank in nodes),
            "hits.csv": csv_text(
                [
                    {"query": query, "taxid": taxid}
                    for query, taxids in [("q0", [s1, s2]), ("q1", [s1, s3]), ("q2", [s3, s3, 0]), ("q3", [0])]
                    for taxid in taxids
                ]
            ),
        }
        expected = {"q0": {"taxid": genus_a}, "q1": {"taxid": root}, "q2": {"taxid": s3}, "q3": {"taxid": 0}}
        columns = {
            "taxid": Column(
                kind="integer",
                unit="taxonomy identifier",
                description="lowest common ancestor, or 0 if no classified hits",
            )
        }
        prompt = (
            "Assign each query in hits.csv to the lowest common ancestor of its nonzero taxids using "
            "nodes.dmp (NCBI-style pipe-delimited taxid, parent, rank fields). Ignore taxid 0 hits; a "
            "query with no classified hit gets 0. Repeated hits do not change the LCA. The root is "
            "self-parented. Use query id."
        )
        wrong = [{"id": k, "taxid": root} for k in expected]
        reason = "returned_root_for_every_query"
    else:
        a, b = rng.randint(3, 9), rng.randint(2, 8)
        ratio = rng.randint(2, 6)
        if operation == "bray-curtis":
            samples = {"a": [a, ratio * a, 0], "b": [0, b, ratio * b], "c": [0, 0, 0], "d": [0, 0, 0]}
            expected = {
                left
                + ":"
                + right: {
                    "distance": (
                        ratio / (ratio + 1)
                        if (left, right) == ("a", "b")
                        else 0.0 if (left, right) == ("c", "d") else 1.0
                    )
                }
                for left, right in combinations(samples, 2)
            }
            columns = {
                "distance": Column(
                    kind="number",
                    unit="dissimilarity",
                    description="Bray-Curtis on relative abundances",
                    atol=1e-10,
                    rtol=1e-8,
                )
            }
            prompt = (
                "Compute all unordered pairwise Bray-Curtis dissimilarities after normalizing each nonempty"
                " sample in counts.csv to relative abundances. Empty samples remain zero; two empty samples"
                " have distance 0, exactly one empty sample has distance 1. Use lexical sample1:sample2 "
                "IDs. Do not compare raw library sizes."
            )
            wrong = [{"id": k, "distance": 0.0} for k in expected]
            reason = "collapsed_distinct_communities"
        else:
            samples = {"a": [a, a, a], "b": [ratio * b, b, 0], "c": [0, 0, 0], "d": [a, 0, 0]}
            h = math.log(ratio + 1) - ratio / (ratio + 1) * math.log(ratio)
            expected = {
                "a": {"richness": 3, "shannon": math.log(3), "inverse_simpson": 3.0},
                "b": {"richness": 2, "shannon": h, "inverse_simpson": (ratio + 1) ** 2 / (ratio**2 + 1)},
                "c": {"richness": 0, "shannon": None, "inverse_simpson": None},
                "d": {"richness": 1, "shannon": 0.0, "inverse_simpson": 1.0},
            }
            columns = {
                "richness": Column(kind="integer", unit="taxa", description="taxa with positive counts"),
                "shannon": Column(
                    kind="number",
                    unit="nats",
                    description="-sum p ln(p); null for empty sample",
                    nullable=True,
                    atol=1e-10,
                    rtol=1e-8,
                ),
                "inverse_simpson": Column(
                    kind="number",
                    unit="effective taxa",
                    description="1/sum(p squared); null for empty sample",
                    nullable=True,
                    atol=1e-10,
                    rtol=1e-8,
                ),
            }
            prompt = (
                "Compute observed richness, Shannon entropy using natural logarithms, and inverse Simpson "
                "diversity from counts.csv. Use within-sample relative abundances and ignore zero taxa in "
                "log terms. Empty samples have richness 0 and null entropy/diversity. Use sample column "
                "names as IDs."
            )
            wrong = [
                {"id": k, **v, "shannon": v["shannon"] / math.log(2) if v["shannon"] is not None else None}
                for k, v in expected.items()
            ]
            reason = "used_log_base_two"
        inputs = {
            "counts.csv": csv_text(
                [{"taxon": f"t{i}", **{name: values[i] for name, values in samples.items()}} for i in range(3)]
            )
        }
    return Instance(
        prompt + " Inputs are in /app/inputs.", inputs, Contract(columns=columns, expected=expected), {reason: wrong}
    )


SKILLS = {
    "assembly-nx": ("length-weighted-contiguity", "genome-size", "unreached-thresholds"),
    "assembly-gap-runs": ("gap-runs", "case-insensitivity", "half-open-coordinates"),
    "contig-depth-breadth": ("depth", "coverage-breadth", "zero-regions"),
    "taxonomic-lca": ("taxonomy-tree", "common-ancestor", "unclassified-hits"),
    "bray-curtis": ("community-dissimilarity", "library-normalization", "empty-samples"),
    "alpha-diversity": ("richness", "entropy", "effective-species"),
}
FORMATS = {
    "assembly-nx": ("fasta",),
    "assembly-gap-runs": ("fasta",),
    "contig-depth-breadth": ("fasta", "bedgraph"),
    "taxonomic-lca": ("ncbi-taxdump-nodes-profile", "csv-header"),
    "bray-curtis": ("csv-header",),
    "alpha-diversity": ("csv-header",),
}
RECIPES = tuple(
    Recipe(
        name,
        "1",
        skills,
        FORMATS[name],
        (
            ("https://quast.sourceforge.net/docs/manual.html",)
            if name.startswith("assembly")
            else ("https://scikit.bio/docs/latest/diversity.html",)
        ),
        partial(generate_assembly, operation=name),
    )
    for name, skills in SKILLS.items()
)
