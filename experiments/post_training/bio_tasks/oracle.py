# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent input-reading solutions, packaged outside solver-visible task inputs."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote

from experiments.post_training.bio_tasks.solvers.assays import SOLVERS as ASSAYS_SOLVERS
from experiments.post_training.bio_tasks.solvers.assembly import SOLVERS as ASSEMBLY_SOLVERS
from experiments.post_training.bio_tasks.solvers.expression import SOLVERS as EXPRESSION_SOLVERS
from experiments.post_training.bio_tasks.solvers.formats import table
from experiments.post_training.bio_tasks.solvers.imaging import SOLVERS as IMAGING_SOLVERS
from experiments.post_training.bio_tasks.solvers.intervals import SOLVERS as INTERVAL_SOLVERS
from experiments.post_training.bio_tasks.solvers.networks import SOLVERS as NETWORKS_SOLVERS
from experiments.post_training.bio_tasks.solvers.phylogeny import SOLVERS as PHYLOGENY_SOLVERS
from experiments.post_training.bio_tasks.solvers.protein_alignment import OUTPUT_SOLVERS as ALIGNMENT_OUTPUT_SOLVERS
from experiments.post_training.bio_tasks.solvers.reads import SOLVERS as READ_SOLVERS
from experiments.post_training.bio_tasks.solvers.real_assembly import solve_assembly
from experiments.post_training.bio_tasks.solvers.real_clinical import SOLVERS as REAL_CLINICAL_SOLVERS
from experiments.post_training.bio_tasks.solvers.real_clusters import solve_clusters
from experiments.post_training.bio_tasks.solvers.real_domains import solve_domains
from experiments.post_training.bio_tasks.solvers.real_enrichment import solve_enrichment
from experiments.post_training.bio_tasks.solvers.real_expression import SOLVERS as REAL_EXPRESSION_SOLVERS
from experiments.post_training.bio_tasks.solvers.real_genomes import SOLVERS as REAL_GENOMES_SOLVERS
from experiments.post_training.bio_tasks.solvers.real_interaction import solve_interaction
from experiments.post_training.bio_tasks.solvers.real_phylogeny import solve_phylogeny as solve_cox1_phylogeny
from experiments.post_training.bio_tasks.solvers.real_reads import OUTPUT_SOLVERS as READ_OUTPUT_SOLVERS
from experiments.post_training.bio_tasks.solvers.real_rnaseq import OUTPUT_SOLVERS as RNASEQ_OUTPUT_SOLVERS
from experiments.post_training.bio_tasks.solvers.real_singlecell import solve_singlecell
from experiments.post_training.bio_tasks.solvers.real_structure import SOLVERS as REAL_STRUCTURE_SOLVERS
from experiments.post_training.bio_tasks.solvers.repo_formats import SOLVERS as REPO_FORMATS_SOLVERS
from experiments.post_training.bio_tasks.solvers.repo_sequences import SOLVERS as REPO_SEQUENCES_SOLVERS
from experiments.post_training.bio_tasks.solvers.sequence import SOLVERS as SEQUENCE_SOLVERS
from experiments.post_training.bio_tasks.solvers.statistics import SOLVERS as STATISTICS_SOLVERS
from experiments.post_training.bio_tasks.solvers.structure import SOLVERS as STRUCTURE_SOLVERS
from experiments.post_training.bio_tasks.solvers.variants import SOLVERS as VARIANTS_SOLVERS
from experiments.post_training.bio_tasks.solvers.workflow import SOLVERS as WORKFLOW_SOLVERS

OUTPUT_SOLVERS = {
    **READ_OUTPUT_SOLVERS,
    **ALIGNMENT_OUTPUT_SOLVERS,
    **RNASEQ_OUTPUT_SOLVERS,
    "real-cox1-tree-comparison": solve_cox1_phylogeny,
    "real-singlecell-read-qc": solve_singlecell,
    "real-proteome-domain-search": solve_domains,
    "real-proteome-clustering": solve_clusters,
    "real-rnaseq-population-interaction": solve_interaction,
    "real-phix-assembly": solve_assembly,
    "real-rnaseq-shrinkage-enrichment-audit": solve_enrichment,
}


def solve_sequence(inputs: Path) -> list[dict]:
    genome = {}
    for line in (inputs / "genome.fa").read_text().splitlines():
        if line.startswith(">"):
            record_id = line[1:].split()[0]
            genome[record_id] = ""
        else:
            genome[record_id] += line
    answer = []
    for line in (inputs / "annotations.gff3").read_text().splitlines():
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        if fields[2] != "exon":
            continue
        attributes = {key: unquote(value) for key, value in (item.split("=", 1) for item in fields[8].split(";"))}
        sequence = genome[unquote(fields[0])][int(fields[3]) - 1 : int(fields[4])]
        if fields[6] == "-":
            sequence = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        answer.append({"id": attributes["Parent"], "sequence": sequence, "length": len(sequence)})
    return answer


def solve_intervals(inputs: Path) -> list[dict]:
    minimum = int((inputs / "minimum.txt").read_text())
    peaks = [line.split("\t") for line in (inputs / "peaks.bed").read_text().splitlines()]
    features = [line.split("\t") for line in (inputs / "features.bed").read_text().splitlines()]
    return [
        {
            "id": feature[3],
            "peak_count": sum(
                peak[0] == feature[0]
                and min(int(peak[2]), int(feature[2])) - max(int(peak[1]), int(feature[1])) >= minimum
                for peak in peaks
            ),
        }
        for feature in features
    ]


def solve_counts(inputs: Path) -> list[dict]:
    cells = table(inputs / "cells.csv")
    counts = table(inputs / "raw_counts.csv")
    by_id = {row["cell_id"]: row for row in cells}
    genes = sorted(set(counts[0]) - {"cell_id"})
    donors = sorted({row["donor"] for row in cells})
    totals: dict[tuple[str, str], int] = defaultdict(int)
    sizes: dict[str, int] = defaultdict(int)
    for row in counts:
        cell = by_id[row["cell_id"]]
        if cell["cell_type"] != "T" or cell["qc_pass"] != "1":
            continue
        donor = cell["donor"]
        sizes[donor] += 1
        for gene in genes:
            totals[donor, gene] += int(row[gene])
    return [
        {"id": f"{donor}/{gene}", "count": totals[donor, gene], "n_cells": sizes[donor]}
        for donor in donors
        for gene in genes
    ]


def solve_fractions(inputs: Path) -> list[dict]:
    specimens = table(inputs / "specimens.csv")
    by_id = {row["specimen_id"]: row for row in specimens}
    totals = {row["patient_id"]: [0, 0] for row in specimens}
    for cell in table(inputs / "cells.csv"):
        specimen = by_id[cell["specimen_id"]]
        if specimen["visit"] != "baseline" or specimen["tissue"] != "blood" or cell["qc_pass"] != "1":
            continue
        patient = specimen["patient_id"]
        totals[patient][0] += cell["cell_type"] == "T"
        totals[patient][1] += 1
    return [
        {"id": patient, "numerator": n, "denominator": d, "fraction": n / d if d else None}
        for patient, (n, d) in totals.items()
    ]


def solve_read_qc(inputs: Path) -> list[dict]:
    config = json.loads((inputs / "thresholds.json").read_text())
    reads = defaultdict(list)
    for filename in ("reads_R1.fastq", "reads_R2.fastq"):
        lines = (inputs / filename).read_text().splitlines()
        for index in range(0, len(lines), 4):
            name, sequence, _, quality = lines[index : index + 4]
            fragment = name[1:].rsplit("/", 1)[0]
            mean_quality = sum(ord(q) - 33 for q in quality) / len(quality)
            passed = (
                len(sequence) >= config["minimum_length"]
                and mean_quality >= config["minimum_mean_quality"]
                and sequence.count("N") <= config["maximum_ns"]
            )
            reads[fragment].append((passed, len(sequence)))
    return [
        {"id": key, "keep": int(all(passed for passed, _ in mates)), "total_bases": sum(n for _, n in mates)}
        for key, mates in reads.items()
    ]


def solve_genotypes(inputs: Path) -> list[dict]:
    included = {row["sample_id"] for row in table(inputs / "samples.csv") if row["include"] == "1"}
    totals = {}
    samples = []
    for line in (inputs / "variants.vcf").read_text().splitlines():
        if line.startswith("##"):
            continue
        fields = line.split("\t")
        if line.startswith("#CHROM"):
            samples = fields[9:]
            continue
        counts = totals.setdefault(fields[2], [0, 0])
        gt_index = fields[8].split(":").index("GT")
        for sample, call in zip(samples, fields[9:], strict=True):
            if sample not in included:
                continue
            genotype = call.split(":")[gt_index]
            alleles = [value for value in genotype.replace("|", "/").split("/") if value != "."]
            counts[0] += len(alleles)
            counts[1] += alleles.count("1")
    return [
        {"id": variant, "called_alleles": n, "alt_alleles": a, "alt_frequency": a / n if n else None}
        for variant, (n, a) in totals.items()
    ]


def solve_tpm(inputs: Path) -> list[dict]:
    mapping = {row["transcript_id"]: row["gene_id"] for row in table(inputs / "mapping.csv")}
    quant = [row for row in table(inputs / "quant.csv") if row["is_decoy"] == "0"]
    rates = {row["transcript_id"]: float(row["count"]) / float(row["effective_length"]) for row in quant}
    normalizer = sum(rates.values())
    counts = defaultdict(int)
    abundance = defaultdict(float)
    for row in quant:
        gene = mapping[row["transcript_id"]]
        counts[gene] += int(row["count"])
        abundance[gene] += rates[row["transcript_id"]] / normalizer * 1_000_000
    return [{"id": gene, "count": counts[gene], "tpm": abundance[gene]} for gene in sorted(set(mapping.values()))]


def solve_sites(inputs: Path) -> list[dict]:
    answer = []
    for row in table(inputs / "alignments.csv"):
        sequences = []
        for line in (inputs / row["file"]).read_text().splitlines():
            if line.startswith(">"):
                sequences.append("")
            else:
                sequences[-1] += line
        eligible = variable = informative = 0
        for column in zip(*sequences, strict=True):
            states = Counter(base for base in column if base in "ACGT")
            if states.total() < 4:
                continue
            eligible += 1
            variable += len(states) >= 2
            informative += sum(count >= 2 for count in states.values()) >= 2
        answer.append(
            {
                "id": row["alignment_id"],
                "eligible_sites": eligible,
                "variable_sites": variable,
                "informative_sites": informative,
            }
        )
    return answer


def solve_tree_branches(inputs: Path) -> list[dict]:
    trees = defaultdict(list)
    for edge in table(inputs / "edges.csv"):
        trees[edge["tree_id"]].append(edge)
    answer = []
    for tree, edges in trees.items():
        parents = {edge["parent"] for edge in edges}
        total = sum(float(edge["length"]) for edge in edges)
        internal = sum(float(edge["length"]) for edge in edges if edge["child"] in parents)
        answer.append({"id": tree, "total_length": total, "internal_length": internal, "treeness": internal / total})
    return answer


def solve_busco(inputs: Path) -> list[dict]:
    complete = defaultdict(set)
    fragmented = set()
    for hit in table(inputs / "hits.csv"):
        if hit["status"] == "complete":
            complete[hit["ortholog_id"]].add(hit["hit_id"])
        else:
            fragmented.add(hit["ortholog_id"])
    counts = {"S": 0, "D": 0, "F": 0, "M": 0}
    for ortholog in table(inputs / "inventory.csv"):
        key = ortholog["ortholog_id"]
        size = len(complete[key])
        category = "D" if size > 1 else "S" if size == 1 else "F" if key in fragmented else "M"
        counts[category] += 1
    return [{"id": "assembly", **counts, "C": counts["S"] + counts["D"], "N": sum(counts.values())}]


def solve_taxonomy(inputs: Path) -> list[dict]:
    parents = {row["taxid"]: row["parent"] for row in table(inputs / "taxonomy.csv")}
    assignments = table(inputs / "assignments.csv")
    direct = Counter(row["taxid"] for row in assignments)
    clades = Counter()
    for taxid, count in direct.items():
        if taxid == "0":
            continue
        while taxid:
            clades[taxid] += count
            taxid = parents[taxid]
    return [
        {"id": taxid, "direct": direct[taxid], "clade": clades[taxid], "fraction": clades[taxid] / len(assignments)}
        for taxid in parents
    ]


def solve_images(inputs: Path) -> list[dict]:
    answer = []
    for image in json.loads((inputs / "images.json").read_text()):
        objects = defaultdict(list)
        for r, row in enumerate(image["mask"]):
            for c, label in enumerate(row):
                if label != 0:
                    objects[label].append((r, c, image["intensity"][r][c]))
        for label, pixels in objects.items():
            count = len(pixels)
            answer.append(
                {
                    "id": f"{image['id']}/{label}",
                    "pixels": count,
                    "area": count * image["spacing_x"] * image["spacing_y"],
                    "centroid_x": (sum(c for _, c, _ in pixels) / count + 0.5) * image["spacing_x"],
                    "centroid_y": (sum(r for r, _, _ in pixels) / count + 0.5) * image["spacing_y"],
                    "mean_intensity": sum(value for _, _, value in pixels) / count,
                }
            )
    return answer


SOLVERS = {
    **REAL_CLINICAL_SOLVERS,
    **REAL_EXPRESSION_SOLVERS,
    **REAL_GENOMES_SOLVERS,
    **REAL_STRUCTURE_SOLVERS,
    **REPO_FORMATS_SOLVERS,
    **REPO_SEQUENCES_SOLVERS,
    **WORKFLOW_SOLVERS,
    **ASSAYS_SOLVERS,
    **IMAGING_SOLVERS,
    **NETWORKS_SOLVERS,
    **STRUCTURE_SOLVERS,
    **ASSEMBLY_SOLVERS,
    **PHYLOGENY_SOLVERS,
    **EXPRESSION_SOLVERS,
    **VARIANTS_SOLVERS,
    **SEQUENCE_SOLVERS,
    **STATISTICS_SOLVERS,
    **READ_SOLVERS,
    **INTERVAL_SOLVERS,
    "strand-extraction": solve_sequence,
    "interval-overlap": solve_intervals,
    "donor-counts": solve_counts,
    "cell-fractions": solve_fractions,
    "paired-read-qc": solve_read_qc,
    "genotype-alleles": solve_genotypes,
    "transcript-tpm": solve_tpm,
    "alignment-sites": solve_sites,
    "tree-branches": solve_tree_branches,
    "busco-summary": solve_busco,
    "taxonomy-counts": solve_taxonomy,
    "image-measurements": solve_images,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", choices=[*SOLVERS, *OUTPUT_SOLVERS])
    parser.add_argument("--inputs", type=Path, default=Path("/app/inputs"))
    parser.add_argument("--answer", type=Path, default=Path("/app/answer.json"))
    args = parser.parse_args()
    if args.recipe in OUTPUT_SOLVERS:
        answer = OUTPUT_SOLVERS[args.recipe](args.inputs, args.answer.parent)
    else:
        answer = SOLVERS[args.recipe](args.inputs)
    args.answer.write_text(json.dumps(answer, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
