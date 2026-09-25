# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fresh construction-ledger references, independent of the packaged oracle solvers."""

import json
import random
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from experiments.post_training.bio_tasks.contract import Column, Contract
from experiments.post_training.bio_tasks.generators.assays import RECIPES as ASSAYS_RECIPES
from experiments.post_training.bio_tasks.generators.assembly import RECIPES as ASSEMBLY_RECIPES
from experiments.post_training.bio_tasks.generators.expression import RECIPES as EXPRESSION_RECIPES
from experiments.post_training.bio_tasks.generators.imaging import RECIPES as IMAGING_RECIPES
from experiments.post_training.bio_tasks.generators.intervals import RECIPES as INTERVAL_RECIPES
from experiments.post_training.bio_tasks.generators.networks import RECIPES as NETWORKS_RECIPES
from experiments.post_training.bio_tasks.generators.phylogeny import RECIPES as PHYLOGENY_RECIPES
from experiments.post_training.bio_tasks.generators.reads import RECIPES as READ_RECIPES
from experiments.post_training.bio_tasks.generators.real_assembly import RECIPES as REAL_ASSEMBLY_RECIPES
from experiments.post_training.bio_tasks.generators.real_bam import RECIPES as REAL_BAM_RECIPES
from experiments.post_training.bio_tasks.generators.real_clinical import RECIPES as REAL_CLINICAL_RECIPES
from experiments.post_training.bio_tasks.generators.real_clusters import RECIPES as REAL_CLUSTER_RECIPES
from experiments.post_training.bio_tasks.generators.real_domains import RECIPES as REAL_DOMAIN_RECIPES
from experiments.post_training.bio_tasks.generators.real_enrichment import RECIPES as REAL_ENRICHMENT_RECIPES
from experiments.post_training.bio_tasks.generators.real_expression import RECIPES as REAL_EXPRESSION_RECIPES
from experiments.post_training.bio_tasks.generators.real_genomes import RECIPES as REAL_GENOMES_RECIPES
from experiments.post_training.bio_tasks.generators.real_heme_pocket import RECIPES as REAL_HEME_POCKET_RECIPES
from experiments.post_training.bio_tasks.generators.real_interaction import RECIPES as REAL_INTERACTION_RECIPES
from experiments.post_training.bio_tasks.generators.real_phylogeny import RECIPES as REAL_PHYLOGENY_RECIPES
from experiments.post_training.bio_tasks.generators.real_proteins import RECIPES as REAL_PROTEINS_RECIPES
from experiments.post_training.bio_tasks.generators.real_reads import RECIPES as REAL_READS_RECIPES
from experiments.post_training.bio_tasks.generators.real_rnaseq import RECIPES as REAL_RNASEQ_RECIPES
from experiments.post_training.bio_tasks.generators.real_singlecell import RECIPES as REAL_SINGLECELL_RECIPES
from experiments.post_training.bio_tasks.generators.real_singlecell_representation import (
    RECIPES as REAL_SINGLECELL_REPRESENTATION_RECIPES,
)
from experiments.post_training.bio_tasks.generators.real_structure import RECIPES as REAL_STRUCTURE_RECIPES
from experiments.post_training.bio_tasks.generators.repo_formats import RECIPES as REPO_FORMATS_RECIPES
from experiments.post_training.bio_tasks.generators.repo_sequences import RECIPES as REPO_SEQUENCES_RECIPES
from experiments.post_training.bio_tasks.generators.sequence import RECIPES as SEQUENCE_RECIPES
from experiments.post_training.bio_tasks.generators.statistics import RECIPES as STATISTICS_RECIPES
from experiments.post_training.bio_tasks.generators.structure import RECIPES as STRUCTURE_RECIPES
from experiments.post_training.bio_tasks.generators.variants import RECIPES as VARIANTS_RECIPES
from experiments.post_training.bio_tasks.generators.workflow import RECIPES as WORKFLOW_RECIPES
from experiments.post_training.bio_tasks.recipe_types import Instance, Recipe, csv_text


def sequence_extraction(seed: int) -> Instance:
    rng = random.Random(seed)
    genome = "".join(rng.choices("ACGT", k=300))
    transcripts, expected, wrong_strand, wrong_coordinate = [], {}, [], []
    complement = {"A": "T", "C": "G", "G": "C", "T": "A"}
    for index in range(6):
        start = 1 + index * 45
        length = rng.randint(8, 19)
        # Extract by enumerating genomic positions; the solver uses Python slices.
        bases = [base for position, base in enumerate(genome, 1) if start <= position < start + length]
        strand = "+" if index % 2 == 0 else "-"
        sequence = "".join(bases if strand == "+" else [complement[base] for base in reversed(bases)])
        record_id = f"tx{index}"
        transcripts.append({"id": record_id, "start": start, "end": start + length - 1, "strand": strand})
        expected[record_id] = {"sequence": sequence, "length": length}
        wrong_strand.append({"id": record_id, "sequence": "".join(bases), "length": length})
        wrong_coordinate.append({"id": record_id, "sequence": sequence[1:], "length": length - 1})
    rng.shuffle(transcripts)
    annotations = ["##gff-version 3", "##sequence-region synthetic_chromosome 1 300"]
    for transcript in transcripts:
        record_id = transcript["id"]
        attributes = (
            ("gene", f"ID=gene_{record_id};Name=shared%3Blabel"),
            ("mRNA", f"Name=shared%3Blabel;Parent=gene_{record_id};ID={record_id}"),
            ("exon", f"Parent={record_id};ID=exon_{record_id}"),
        )
        for feature_type, feature_attributes in attributes:
            annotations.append(
                "\t".join(
                    [
                        "synthetic_chromosome",
                        "synthetic",
                        feature_type,
                        str(transcript["start"]),
                        str(transcript["end"]),
                        ".",
                        transcript["strand"],
                        ".",
                        feature_attributes,
                    ]
                )
            )
    fasta = ">unannotated_contig\n" + "N" * 300 + "\n>synthetic_chromosome\n"
    fasta += "\n".join(genome[offset : offset + 60] for offset in range(0, len(genome), 60)) + "\n"
    contract = Contract(
        columns={
            "sequence": Column(kind="text", unit="DNA bases", description="uppercase transcript-oriented sequence"),
            "length": Column(kind="integer", unit="bases", description="sequence length"),
        },
        expected=expected,
    )
    return Instance(
        "Extract every single-exon mRNA in annotations.gff3 from genome.fa. These GFF3 annotations contain "
        "gene, mRNA, and exon features; each mRNA has exactly one exon with a single Parent. Join the exon's "
        "Parent to the mRNA ID and match the GFF3 seqid to the FASTA record ID. Coordinates are 1-based, "
        "closed on both ends. For minus-strand transcripts return the reverse complement. "
        "Use the decoded mRNA ID as output id, not the gene/exon ID or nonunique Name. Ignore unannotated "
        "FASTA records. Inputs are in /app/inputs.",
        {"genome.fa": fasta, "annotations.gff3": "\n".join(annotations) + "\n"},
        contract,
        {
            "ignored_strand": wrong_strand,
            "off_by_one": wrong_coordinate,
            "used_exon_ids": [{**row, "id": f"exon_{row['id']}"} for row in contract.answer()],
            "ignored_seqid": [{**row, "sequence": "N" * row["length"]} for row in contract.answer()],
        },
    )


def interval_overlap(seed: int) -> Instance:
    rng = random.Random(seed)
    features = [
        {"id": f"gene{i}", "chrom": "chr1", "start": i * 30, "end": i * 30 + rng.randint(10, 20)} for i in range(6)
    ]
    peaks = []
    for feature in features:
        offsets = [rng.randrange(feature["end"] - feature["start"]) for _ in range(rng.randint(0, 5))]
        offsets.append(feature["end"] - feature["start"])
        for offset in offsets:
            peaks.append(
                {
                    "id": f"peak{len(peaks)}",
                    "chrom": "chr1",
                    "start": feature["start"] + offset,
                    "end": feature["start"] + offset + 8,
                }
            )
    peaks.append({"id": "decoy", "chrom": "chr2", "start": 0, "end": 200})
    minimum = rng.choice([1, 3, 5])
    expected, touching = {}, []
    for feature in features:
        positions = set(range(feature["start"], feature["end"]))
        overlaps = [
            p
            for p in peaks
            if p["chrom"] == feature["chrom"] and len(positions & set(range(p["start"], p["end"]))) >= minimum
        ]
        expected[feature["id"]] = {"peak_count": len(overlaps)}
        touching.append({"id": feature["id"], "peak_count": len(overlaps) + 1})
    rng.shuffle(peaks)
    rng.shuffle(features)
    contract = Contract(
        columns={"peak_count": Column(kind="integer", unit="distinct peaks", description="qualifying peak count")},
        expected=expected,
    )
    return Instance(
        f"For every feature in features.bed, count distinct peaks in peaks.bed overlapping by at least {minimum} "
        "bases on the same chromosome. These headerless BED4 files contain chromosome, start, end, name, "
        "separated by tabs; coordinates are 0-based half-open intervals [start, end). "
        "Boundary contact is not overlap. Do not merge peaks. Report every feature id, including zeros. "
        "Inputs are in /app/inputs.",
        {
            "features.bed": "".join(f"{f['chrom']}\t{f['start']}\t{f['end']}\t{f['id']}\n" for f in features),
            "peaks.bed": "".join(f"{p['chrom']}\t{p['start']}\t{p['end']}\t{p['id']}\n" for p in peaks),
            "minimum.txt": str(minimum),
        },
        contract,
        {"counted_boundary_contacts": touching},
    )


def donor_counts(seed: int) -> Instance:
    rng = random.Random(seed)
    metadata, counts, normalized, expected = [], [], [], {}
    genes = ["G0", "G1", "G2"]
    for donor in range(5):
        donor_id = f"donor{donor}"
        # donor4 has no eligible cells, which must produce explicit zero rows.
        eligible = rng.randint(2, 6) if donor != 4 else 0
        totals = {gene: 0 for gene in genes}
        for cell in range(eligible + 3):
            cell_id = f"{donor_id}_cell{cell}"
            included = cell < eligible
            metadata.append(
                {
                    "cell_id": cell_id,
                    "donor": donor_id,
                    "cell_type": "T" if included or cell == eligible + 2 else "B",
                    "qc_pass": 1 if cell != eligible + 2 else 0,
                }
            )
            values = {gene: rng.randint(0, 20) for gene in genes}
            if included:
                for gene, value in values.items():
                    totals[gene] += value
            counts.append({"cell_id": cell_id, **values})
            normalized.append({"cell_id": cell_id, **{g: round(v / 7, 4) for g, v in values.items()}})
        for gene in genes:
            expected[f"{donor_id}/{gene}"] = {"count": totals[gene], "n_cells": eligible}
    rng.shuffle(metadata)
    rng.shuffle(counts)
    rng.shuffle(normalized)
    contract = Contract(
        columns={
            "count": Column(kind="integer", unit="raw counts", description="sum over eligible donor cells"),
            "n_cells": Column(kind="integer", unit="cells", description="number of contributing cells"),
        },
        expected=expected,
    )
    pooled = [
        {
            "id": key,
            "count": sum(v["count"] for k, v in expected.items() if k.endswith(key[-2:])),
            "n_cells": value["n_cells"],
        }
        for key, value in expected.items()
    ]
    means = [
        {"id": key, "count": value["count"] // max(1, value["n_cells"]), "n_cells": value["n_cells"]}
        for key, value in expected.items()
    ]
    return Instance(
        "Construct donor-level pseudobulk raw counts for T cells with qc_pass=1. Join cells.csv and raw_counts.csv "
        "by cell_id; their row orders differ. normalized.csv is a transformed layer and must not be summed. "
        "Include every donor appearing in cells.csv and every gene column of raw_counts.csv. "
        "Use donor/gene as the output id. Donors with no eligible cells require zero counts and zero n_cells. "
        "Inputs are in /app/inputs.",
        {"cells.csv": csv_text(metadata), "raw_counts.csv": csv_text(counts), "normalized.csv": csv_text(normalized)},
        contract,
        {"pooled_donors": pooled, "averaged_counts": means},
    )


def cell_fractions(seed: int) -> Instance:
    rng = random.Random(seed)
    specimens, cells, expected = [], [], {}
    pooled_numerator = pooled_denominator = 0
    for patient in range(5):
        patient_id = f"patient{patient}"
        numerator = rng.randint(0, 5) if patient != 4 else 0
        denominator = numerator + rng.randint(2, 9) if patient != 4 else 0
        pooled_numerator += numerator
        pooled_denominator += denominator
        expected[patient_id] = {
            "numerator": numerator,
            "denominator": denominator,
            "fraction": numerator / denominator if denominator else None,
        }
        for visit, tissue in (("baseline", "blood"), ("followup", "blood"), ("baseline", "tumor")):
            specimen = f"{patient_id}_{visit}_{tissue}"
            specimens.append({"specimen_id": specimen, "patient_id": patient_id, "visit": visit, "tissue": tissue})
            selected = visit == "baseline" and tissue == "blood"
            size = denominator if selected else rng.randint(8, 16)
            for index in range(size + 2):
                # Barcodes repeat across specimens; the join key is specimen_id.
                cells.append(
                    {
                        "specimen_id": specimen,
                        "barcode": f"cell{index}",
                        "cell_type": "T" if index < (numerator if selected else 5) else "B",
                        "qc_pass": int(index < size),
                    }
                )
    rng.shuffle(specimens)
    rng.shuffle(cells)
    contract = Contract(
        columns={
            "numerator": Column(kind="integer", unit="cells", description="eligible T cells"),
            "denominator": Column(kind="integer", unit="cells", description="all eligible cells"),
            "fraction": Column(
                kind="number",
                unit="proportion",
                description="numerator/denominator; null if zero",
                nullable=True,
                atol=1e-8,
                rtol=1e-8,
            ),
        },
        expected=expected,
    )
    pooled = [{"id": key, **value, "fraction": pooled_numerator / pooled_denominator} for key, value in expected.items()]
    wrong_denominator = [
        {
            "id": key,
            **value,
            "denominator": value["denominator"] + 2,
            "fraction": value["numerator"] / (value["denominator"] + 2),
        }
        for key, value in expected.items()
    ]
    return Instance(
        "For every patient in specimens.csv, calculate the T-cell fraction among qc_pass=1 cells in baseline "
        "blood specimens only. Join cells.csv to specimens.csv using specimen_id. Cell identity is "
        "(specimen_id, barcode); barcodes are not globally unique. The denominator includes all eligible cell types. "
        "Report numerator, denominator, and fraction for each patient_id. A zero denominator requires fraction=null. "
        "Inputs are in /app/inputs.",
        {"specimens.csv": csv_text(specimens), "cells.csv": csv_text(cells)},
        contract,
        {"pooled_patient_fraction": pooled, "included_failed_qc": wrong_denominator},
    )


def read_qc(seed: int) -> Instance:
    rng = random.Random(seed)
    minimum = rng.choice([12, 16, 20])
    quality = rng.choice([20, 25, 30])
    mates: list[list[str]] = [[], []]
    expected = {}
    for pair in range(6):
        total = 0
        for mate in range(2):
            length = minimum + rng.randint(0, 6)
            q = quality + rng.randint(1, 5)
            ns = 0
            if pair == 1 and mate == 0:
                q = quality - 1
            if pair == 2 and mate == 1:
                length = minimum - 1
            if pair == 3 and mate == 0:
                ns = 1
            if pair == 4 and mate == 1:
                ns = 2
            if pair == 5:
                length, q = minimum, quality
            sequence = "N" * ns + "".join(rng.choices("ACGT", k=length - ns))
            mates[mate].append(f"@pair{pair}/{mate + 1}\n{sequence}\n+\n{chr(q + 33) * length}\n")
            total += length
        expected[f"pair{pair}"] = {"keep": int(pair in (0, 3, 5)), "total_bases": total}
    for reads in mates:
        rng.shuffle(reads)
    contract = Contract(
        columns={
            "keep": Column(kind="integer", unit="decision", description="1 if both mates pass, otherwise 0"),
            "total_bases": Column(kind="integer", unit="bases", description="sum of both original mate lengths"),
        },
        expected=expected,
    )
    wrong_pairing = contract.answer()
    wrong_pairing[2]["keep"] = 1
    strict_boundary = contract.answer()
    strict_boundary[5]["keep"] = 0
    return Instance(
        f"Filter paired FASTQ reads. Each mate must have length >= {minimum}, mean Phred quality >= {quality}, "
        "and at most one N base. Qualities use Phred+33. Keep a fragment only if both mates pass; do not trim. "
        "Mate files have different orders: pair the common read ID before /1 or /2. Report every fragment, "
        "including dropped fragments, using that common ID. total_bases includes both original mates even "
        "for dropped fragments. Inputs are in /app/inputs.",
        {
            "reads_R1.fastq": "".join(mates[0]),
            "reads_R2.fastq": "".join(mates[1]),
            "thresholds.json": json.dumps({"minimum_length": minimum, "minimum_mean_quality": quality, "maximum_ns": 1}),
        },
        contract,
        {"kept_pair_with_failed_mate": wrong_pairing, "rejected_inclusive_boundary": strict_boundary},
    )


def genotype_alleles(seed: int) -> Instance:
    rng = random.Random(seed)
    sample_order = list(range(6))
    rng.shuffle(sample_order)
    records, expected = [], {}
    choices = [[0, 0], [0, 1], [1, 1], [1], [0, None], [1, None], [None, None]]
    for variant in range(6):
        called, alternate = 0, 0
        genotypes = {}
        for sample in range(6):
            alleles = list(rng.choice(choices))
            if variant == 0:
                alleles = [[0, 1], [1], [1, None], [0, 0], [1, 1], [1, 1]][sample]
            if variant == 5:
                alleles = [None, None]
            if sample < 4:
                for allele in alleles:
                    called += allele is not None
                    alternate += allele == 1
            genotype = rng.choice(["/", "|"]).join("." if a is None else str(a) for a in alleles)
            genotypes[sample] = f"{genotype}:{rng.randint(5, 60)}"
        records.append(
            "\t".join(
                [
                    "chr1",
                    str(10 + variant * 20),
                    f"v{variant}",
                    "A",
                    "C",
                    ".",
                    "PASS",
                    ".",
                    "GT:DP",
                    *[genotypes[sample] for sample in sample_order],
                ]
            )
        )
        expected[f"v{variant}"] = {
            "called_alleles": called,
            "alt_alleles": alternate,
            "alt_frequency": alternate / called if called else None,
        }
    samples = [{"sample_id": f"s{i}", "include": int(i < 4)} for i in range(6)]
    rng.shuffle(samples)
    vcf = (
        "##fileformat=VCFv4.3\n##contig=<ID=chr1,length=1000>\n"
        '##FILTER=<ID=PASS,Description="All filters passed">\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">\n'
        + "\t".join(
            [
                "#CHROM",
                "POS",
                "ID",
                "REF",
                "ALT",
                "QUAL",
                "FILTER",
                "INFO",
                "FORMAT",
                *[f"s{sample}" for sample in sample_order],
            ]
        )
        + "\n"
        + "\n".join(records)
        + "\n"
    )
    contract = Contract(
        columns={
            "called_alleles": Column(kind="integer", unit="alleles", description="nonmissing included allele calls"),
            "alt_alleles": Column(kind="integer", unit="alleles", description="included allele-1 calls"),
            "alt_frequency": Column(
                kind="number",
                unit="proportion",
                description="alt/called; null when called=0",
                nullable=True,
                atol=1e-8,
                rtol=1e-8,
            ),
        },
        expected=expected,
    )
    diploid = [
        {"id": key, **row, "called_alleles": 8, "alt_frequency": row["alt_alleles"] / 8} for key, row in expected.items()
    ]
    dropped_partial = contract.answer()
    dropped_partial[0].update(called_alleles=5, alt_alleles=2, alt_frequency=2 / 5)
    return Instance(
        "Compute alternate-allele frequency for every record ID in variants.vcf (VCF 4.3), using only samples with "
        "include=1 in samples.csv. Match VCF sample columns by name and read GT from FORMAT; "
        "DP does not affect inclusion. "
        "These are biallelic sites: 0 is reference and 1 is alternate. GT may be "
        "haploid or diploid, and / versus | does not affect counting. Ignore each missing allele (.) individually: "
        "1/. contributes one alternate and one called allele. Divide by called alleles, not samples or assumed "
        "diploid ploidy. With no called alleles return frequency=null. Inputs are in /app/inputs.",
        {"variants.vcf": vcf, "samples.csv": csv_text(samples)},
        contract,
        {"assumed_diploid_denominator": diploid, "dropped_partial_genotype": dropped_partial},
    )


def transcript_tpm(seed: int) -> Instance:
    rng = random.Random(seed)
    rates = [rng.randint(1, 9), rng.randint(2, 12), rng.randint(1, 8), 0]
    lengths = [rng.randint(1, 2), rng.randint(3, 4), rng.randint(5, 7), 2]
    mapping = [{"transcript_id": f"tx{i}", "gene_id": gene} for i, gene in enumerate(["gA", "gA", "gB", "gC"])]
    quant = [
        {"transcript_id": f"tx{i}", "count": rate * lengths[i], "effective_length": lengths[i] * 1000, "is_decoy": 0}
        for i, rate in enumerate(rates)
    ]
    quant.append({"transcript_id": "decoy", "count": 1000, "effective_length": 1000, "is_decoy": 1})
    expected = {
        "gA": {
            "count": quant[0]["count"] + quant[1]["count"],
            "tpm": float(Fraction(rates[0] + rates[1], sum(rates)) * 1_000_000),
        },
        "gB": {"count": quant[2]["count"], "tpm": float(Fraction(rates[2], sum(rates)) * 1_000_000)},
        "gC": {"count": 0, "tpm": 0.0},
    }
    rng.shuffle(mapping)
    rng.shuffle(quant)
    contract = Contract(
        columns={
            "count": Column(kind="integer", unit="supplied counts", description="sum of nondecoy transcript counts"),
            "tpm": Column(
                kind="number",
                unit="TPM",
                description="sum of nondecoy transcript TPM for this gene",
                atol=1e-6,
                rtol=1e-8,
            ),
        },
        expected=expected,
    )
    cpm = [
        {"id": key, **row, "tpm": row["count"] / sum(v["count"] for v in expected.values()) * 1_000_000}
        for key, row in expected.items()
    ]
    decoy = [{"id": key, **row, "tpm": row["tpm"] * sum(rates) / (sum(rates) + 1000)} for key, row in expected.items()]
    return Instance(
        "Recompute transcript TPM from quant.csv, then sum TPM and supplied counts by the gene IDs in mapping.csv. "
        "First exclude is_decoy=1 rows. Transcript rate is count/effective_length in bases; TPM is that rate "
        "divided by the sum of nondecoy rates, times 1,000,000. Sum transcript TPM after normalization, and retain "
        "zero-count genes. These supplied counts are inputs to this arithmetic task, not inferred molecule truth. "
        "Use gene_id as output id. Inputs are in /app/inputs.",
        {"quant.csv": csv_text(quant), "mapping.csv": csv_text(mapping)},
        contract,
        {"used_cpm_instead_of_tpm": cpm, "included_decoy_in_normalizer": decoy},
    )


def alignment_sites(seed: int) -> Instance:
    rng = random.Random(seed)
    # Pattern labels establish the reference before base relabeling and sequence assembly.
    patterns = [
        ("AAAAAA", 1, 0, 0),
        ("AAAACC", 1, 1, 1),
        ("AAAAAC", 1, 1, 0),
        ("ACGTNN", 1, 1, 0),
        ("AA--CC", 1, 1, 1),
        ("ANN---", 0, 0, 0),
    ]
    expected, inputs, inventory = {}, {}, []
    for alignment in range(3):
        columns = []
        eligible = variable = informative = 0
        for pattern, e, v, p in patterns:
            repeats = rng.randint(1, 4)
            if alignment == 2 and e:
                continue
            columns.extend([pattern] * repeats)
            eligible, variable, informative = eligible + e * repeats, variable + v * repeats, informative + p * repeats
        rng.shuffle(columns)
        alphabet = rng.sample(list("ACGT"), 4)
        translation = str.maketrans("ACGT", "".join(alphabet))
        sequences = ["".join(column[row] for column in columns).translate(translation) for row in range(6)]
        rng.shuffle(sequences)
        name = f"alignment{alignment}"
        inputs[name + ".fa"] = "".join(f">taxon{i}\n{s}\n" for i, s in enumerate(sequences))
        inventory.append({"alignment_id": name, "file": name + ".fa"})
        expected[name] = {"eligible_sites": eligible, "variable_sites": variable, "informative_sites": informative}
    inputs["alignments.csv"] = csv_text(inventory)
    contract = Contract(
        columns={
            name: Column(kind="integer", unit="sites", description=description)
            for name, description in [
                ("eligible_sites", "columns with at least four A/C/G/T calls"),
                ("variable_sites", "eligible columns with at least two distinct A/C/G/T states"),
                ("informative_sites", "eligible columns with at least two states each occurring at least twice"),
            ]
        },
        expected=expected,
    )
    wrong = [{"id": key, **row, "informative_sites": row["variable_sites"]} for key, row in expected.items()]
    return Instance(
        "Summarize each FASTA alignment listed in alignments.csv. Ignore N and - when counting states. "
        "A column is eligible with at least four A/C/G/T calls. Among eligible columns, variable means at least "
        "two distinct states; parsimony-informative means at least two states each observed at least twice. "
        "Count only eligible columns in variable/informative totals. Report every alignment_id, including "
        "alignments with no eligible columns. Inputs are in /app/inputs.",
        inputs,
        contract,
        {"confused_variable_with_informative": wrong},
    )


def tree_branches(seed: int) -> Instance:
    rng = random.Random(seed)
    edges, expected = [], {}
    for tree in range(3):
        if tree == 2:
            topology = [("root", "a"), ("root", "b"), ("root", "c")]
            internal_indices = set()
        else:
            topology = [
                ("root", "u"),
                ("u", "a"),
                ("u", "b"),
                ("root", "v"),
                ("v", "c"),
                ("v", "d"),
                ("root", "outgroup"),
            ]
            internal_indices = {0, 3}
        lengths = [rng.randint(1, 20) for _ in topology]
        total, internal = sum(lengths), sum(lengths[i] for i in internal_indices)
        for (parent, child), length in zip(topology, lengths, strict=True):
            edges.append({"tree_id": f"tree{tree}", "parent": parent, "child": child, "length": length / 100})
        expected[f"tree{tree}"] = {
            "total_length": total / 100,
            "internal_length": internal / 100,
            "treeness": internal / total,
        }
    rng.shuffle(edges)
    contract = Contract(
        columns={
            "total_length": Column(
                kind="number", unit="substitutions/site", description="sum of all edge lengths", atol=1e-10, rtol=1e-8
            ),
            "internal_length": Column(
                kind="number",
                unit="substitutions/site",
                description="sum of edges whose child has children",
                atol=1e-10,
                rtol=1e-8,
            ),
            "treeness": Column(
                kind="number", unit="ratio", description="internal_length/total_length", atol=1e-10, rtol=1e-8
            ),
        },
        expected=expected,
    )
    terminal = [
        {
            "id": key,
            **row,
            "internal_length": row["total_length"] - row["internal_length"],
            "treeness": 1 - row["treeness"],
        }
        for key, row in expected.items()
    ]
    return Instance(
        "Compute branch summaries for each rooted tree in edges.csv. Every row is one parent-to-child edge. "
        "Internal edges have a child that itself has children; edges to leaves are terminal. Count root-to-internal "
        "edges, and do not invent a branch above the root. treeness=internal_length/total_length. All supplied "
        "trees have positive total length. Node names are local to tree_id. Use tree_id as output id. "
        "Inputs are in /app/inputs.",
        {"edges.csv": csv_text(edges)},
        contract,
        {"used_terminal_lengths": terminal},
    )


def busco_summary(seed: int) -> Instance:
    rng = random.Random(seed)
    inventory, hits = [], []
    counts = {category: rng.randint(1, 3) for category in ("S", "D", "F", "M")}
    for category, number in counts.items():
        for _ in range(number):
            ortholog = f"ortholog{len(inventory)}"
            inventory.append({"ortholog_id": ortholog})
            if category == "M":
                continue
            multiplicity = rng.randint(2, 4) if category == "D" else 1
            for index in range(multiplicity):
                hit = {
                    "ortholog_id": ortholog,
                    "hit_id": f"{ortholog}_hit{index}",
                    "status": "fragmented" if category == "F" else "complete",
                }
                hits.extend([hit, dict(hit)])  # Duplicate rows do not create distinct hits.
            if category in ("S", "D"):
                hits.append({"ortholog_id": ortholog, "hit_id": f"{ortholog}_fragment", "status": "fragmented"})
    rng.shuffle(hits)
    rng.shuffle(inventory)
    expected = {"assembly": {**counts, "C": counts["S"] + counts["D"], "N": sum(counts.values())}}
    contract = Contract(
        columns={
            name: Column(kind="integer", unit="orthologs", description=description)
            for name, description in [
                ("S", "complete single-copy IDs"),
                ("D", "complete duplicated IDs"),
                ("F", "fragmented-only IDs"),
                ("M", "IDs with no hits"),
                ("C", "all complete IDs, S+D"),
                ("N", "lineage inventory size"),
            ]
        },
        expected=expected,
    )
    duplicate_rows = contract.answer()
    duplicate_rows[0].update(S=0, D=counts["S"] + counts["D"])
    overshadowed = contract.answer()
    overshadowed[0]["F"] += counts["S"] + counts["D"]
    return Instance(
        "Reconcile BUSCO-style ortholog summaries from a frozen lineage inventory (inventory.csv) and hits.csv. "
        "For each inventory ortholog, count distinct complete hit_id values: one means S, two or more means D. "
        "With no complete hit, any fragmented hit means F; with no hit, M. Complete hits take precedence over "
        "fragmented hits. Duplicate rows do not create distinct hits. Report one record with id=assembly, "
        "C=S+D and N=S+D+F+M. This task summarizes supplied hits; it does not search sequences. "
        "Inputs are in /app/inputs.",
        {"inventory.csv": csv_text(inventory), "hits.csv": csv_text(hits)},
        contract,
        {"treated_duplicate_rows_as_hits": duplicate_rows, "counted_fragmented_beneath_complete": overshadowed},
    )


def taxonomy_counts(seed: int) -> Instance:
    rng = random.Random(seed)
    # Construction paths specify each lineage independently of the oracle's parent traversal.
    paths = [(1,), (1, 2), (1, 2, 3), (1, 2, 4), (1, 2, 3, 5), (1, 2, 3, 6), (1, 2, 4, 7)]
    nodes = [{"taxid": path[-1], "parent": path[-2] if len(path) > 1 else ""} for path in paths]
    assignments, direct, clade = [], {path[-1]: 0 for path in paths}, {path[-1]: 0 for path in paths}
    for path in paths:
        number = rng.randint(1, 5)
        direct[path[-1]] = number
        for ancestor in path:
            clade[ancestor] += number
        for _ in range(number):
            assignments.append({"fragment_id": f"fragment{len(assignments)}", "taxid": path[-1]})
    for _ in range(rng.randint(2, 6)):
        assignments.append({"fragment_id": f"fragment{len(assignments)}", "taxid": 0})
    expected = {
        str(taxid): {"direct": direct[taxid], "clade": clade[taxid], "fraction": clade[taxid] / len(assignments)}
        for taxid in direct
    }
    rng.shuffle(nodes)
    rng.shuffle(assignments)
    contract = Contract(
        columns={
            "direct": Column(kind="integer", unit="fragments", description="assignments directly to this taxid"),
            "clade": Column(kind="integer", unit="fragments", description="assignments to this taxid or descendants"),
            "fraction": Column(
                kind="number",
                unit="proportion",
                description="clade/all fragments, including unclassified",
                atol=1e-8,
                rtol=1e-8,
            ),
        },
        expected=expected,
    )
    wrong = [{"id": key, **row, "fraction": row["clade"] / sum(direct.values())} for key, row in expected.items()]
    no_ancestors = [
        {"id": key, **row, "clade": row["direct"], "fraction": row["direct"] / len(assignments)}
        for key, row in expected.items()
    ]
    return Instance(
        "Summarize fragment assignments using the frozen parent tree in taxonomy.csv. assignments.csv contains "
        "one row per fragment; taxid=0 means unclassified and is absent from the tree. For every taxonomy taxid, "
        "report direct assignments, assignments to its entire clade (including itself), and clade fraction of ALL "
        "fragments, including unclassified. Clade counts overlap across ancestors; do not sum them as disjoint "
        "abundances. The root has an empty parent. Use string taxid as output id. Inputs are in /app/inputs.",
        {"taxonomy.csv": csv_text(nodes), "assignments.csv": csv_text(assignments)},
        contract,
        {"excluded_unclassified_from_denominator": wrong, "ignored_descendant_assignments": no_ancestors},
    )


def image_measurements(seed: int) -> Instance:
    rng = random.Random(seed)
    images, expected = [], {}
    for image in range(2):
        dx, dy = rng.choice([0.25, 0.5]), rng.choice([1.0, 3.0])
        mask = [[0] * 7 for _ in range(5)]
        intensities = [[rng.randint(40, 80) for _ in range(7)] for _ in range(5)]
        objects = {1: [(r, c) for r in (0, 1) for c in (0, 1, 2)], 2: [(2, 3), (3, 4)], 3: [(4, 6)]}
        for label, coordinates in objects.items():
            values = [rng.randint(1, 30) for _ in coordinates]
            for (r, c), value in zip(coordinates, values, strict=True):
                mask[r][c], intensities[r][c] = label, value
            count = len(coordinates)
            expected[f"image{image}/{label}"] = {
                "pixels": count,
                "area": count * dx * dy,
                "centroid_x": float(sum(Fraction(2 * c + 1, 2) for _, c in coordinates) / count) * dx,
                "centroid_y": float(sum(Fraction(2 * r + 1, 2) for r, _ in coordinates) / count) * dy,
                "mean_intensity": sum(values) / count,
            }
        images.append({"id": f"image{image}", "spacing_x": dx, "spacing_y": dy, "mask": mask, "intensity": intensities})
    contract = Contract(
        columns={
            "pixels": Column(kind="integer", unit="pixels", description="number of pixels with this label"),
            "area": Column(
                kind="number",
                unit="micrometers squared",
                description="pixels times spacing_x times spacing_y",
                atol=1e-8,
                rtol=1e-8,
            ),
            "centroid_x": Column(
                kind="number",
                unit="micrometers",
                description="mean physical column-center coordinate",
                atol=1e-8,
                rtol=1e-8,
            ),
            "centroid_y": Column(
                kind="number",
                unit="micrometers",
                description="mean physical row-center coordinate",
                atol=1e-8,
                rtol=1e-8,
            ),
            "mean_intensity": Column(
                kind="number",
                unit="arbitrary intensity units",
                description="mean within this label",
                atol=1e-8,
                rtol=1e-8,
            ),
        },
        expected=expected,
    )
    unscaled = [{"id": key, **row, "area": row["pixels"]} for key, row in expected.items()]
    swapped = [
        {"id": key, **row, "centroid_x": row["centroid_y"], "centroid_y": row["centroid_x"]}
        for key, row in expected.items()
    ]
    return Instance(
        "Measure each nonzero object label in the supplied masks in images.json. Do not segment images or split "
        "disconnected pixels sharing a label. Label 0 is background; border objects are retained. "
        "Arrays are [row][column]. "
        "Pixel (r,c) occupies [c*spacing_x,(c+1)*spacing_x] by [r*spacing_y,(r+1)*spacing_y] micrometers; use its "
        "center for centroids. Report area, unweighted physical centroid, pixel count, and mean intensity per "
        "image/label id. Inputs are in /app/inputs.",
        {"images.json": json.dumps(images)},
        contract,
        {"ignored_pixel_scale": unscaled, "swapped_axes": swapped},
    )


RECIPES = (
    *(
        Recipe(
            "strand-extraction",
            "2",
            ("coordinates", "strand", "sequence-extraction", "feature-parent-joins", "sequence-identifiers"),
            ("fasta-dna", "gff3-single-exon"),
            (
                "https://github.com/biopython/biopython/blob/08fc09086afe0b57215d2515660e0c032b55c0dd/Tests/test_SeqFeature.py",
            ),
            sequence_extraction,
        ),
        Recipe(
            "interval-overlap",
            "2",
            ("coordinates", "overlap", "feature-identity"),
            ("bed4", "text-integer"),
            ("https://github.com/arq5x/bedtools2/tree/614e9a5c5935ab86e873dab9072fbbaf003c1b7e",),
            interval_overlap,
        ),
        Recipe(
            "donor-counts",
            "1",
            ("sample-joins", "raw-counts", "biological-replication"),
            ("csv-header",),
            ("https://github.com/scverse/scanpy/tree/0d5fd16234865619d2f5097d33fc4281900a2bc2",),
            donor_counts,
        ),
        Recipe(
            "cell-fractions",
            "1",
            ("sample-joins", "cohort-selection", "denominators"),
            ("csv-header",),
            ("https://github.com/scverse/scanpy/tree/0d5fd16234865619d2f5097d33fc4281900a2bc2",),
            cell_fractions,
        ),
        Recipe(
            "paired-read-qc",
            "1",
            ("read-quality", "mate-identity", "thresholds"),
            ("fastq-phred33", "json"),
            ("https://github.com/OpenGene/fastp/blob/8a2397b6628ae14127efdb7566f67fc05f9aea56/src/filter.cpp",),
            read_qc,
        ),
        Recipe(
            "genotype-alleles",
            "1",
            ("ploidy", "missing-calls", "allele-denominators"),
            ("vcf4.3", "csv-header"),
            ("https://github.com/samtools/bcftools/tree/edf7fd96c5da562ecfd99fb7f9e4b9eb597aeae8",),
            genotype_alleles,
        ),
        Recipe(
            "transcript-tpm",
            "1",
            ("transcript-joins", "abundance-units", "decoys"),
            ("csv-header",),
            ("https://github.com/COMBINE-lab/salmon/tree/5515b7f05a90341b6652adfdb807e7cf14295518",),
            transcript_tpm,
        ),
        Recipe(
            "alignment-sites",
            "1",
            ("site-states", "missing-bases", "parsimony-informative-sites"),
            ("fasta-alignment", "csv-header"),
            ("https://github.com/JLSteenwyk/PhyKIT/tree/3e59b123e6ffd1ba1f298603a4dc4e7b04c69b0e",),
            alignment_sites,
        ),
        Recipe(
            "tree-branches",
            "1",
            ("root-conventions", "branch-lengths", "treeness"),
            ("csv-header",),
            ("https://github.com/morgannprice/fasttree/tree/a5a2723ea1e64faf3da7ea514521cfa348891add",),
            tree_branches,
        ),
        Recipe(
            "busco-summary",
            "1",
            ("ortholog-identity", "deduplication", "completeness-categories"),
            ("csv-header",),
            ("https://gitlab.com/ezlab/busco/-/tree/cd071053c38c5060f75d0b370cb66c4edc8e59a1",),
            busco_summary,
        ),
        Recipe(
            "taxonomy-counts",
            "1",
            ("taxonomy", "hierarchical-counts", "unclassified-denominator"),
            ("csv-header",),
            ("https://github.com/DerrickWood/kraken2/tree/8c190b1b668825935dbf6dee5f969227dc8269bb",),
            taxonomy_counts,
        ),
        Recipe(
            "image-measurements",
            "1",
            ("object-identity", "pixel-spacing", "image-measurement"),
            ("json",),
            ("https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regionprops.html",),
            image_measurements,
        ),
    ),
    *REAL_EXPRESSION_RECIPES,
    *REAL_RNASEQ_RECIPES,
    *REAL_INTERACTION_RECIPES,
    *REAL_ENRICHMENT_RECIPES,
    *REAL_SINGLECELL_RECIPES,
    *REAL_SINGLECELL_REPRESENTATION_RECIPES,
    *REAL_CLINICAL_RECIPES,
    *REAL_GENOMES_RECIPES,
    *REAL_READS_RECIPES,
    *REAL_PROTEINS_RECIPES,
    *REAL_PHYLOGENY_RECIPES,
    *REAL_STRUCTURE_RECIPES,
    *REAL_HEME_POCKET_RECIPES,
    *REAL_DOMAIN_RECIPES,
    *REAL_CLUSTER_RECIPES,
    *REAL_ASSEMBLY_RECIPES,
    *REAL_BAM_RECIPES,
    *INTERVAL_RECIPES,
    *READ_RECIPES,
    *STATISTICS_RECIPES,
    *SEQUENCE_RECIPES,
    *VARIANTS_RECIPES,
    *EXPRESSION_RECIPES,
    *PHYLOGENY_RECIPES,
    *ASSEMBLY_RECIPES,
    *STRUCTURE_RECIPES,
    *NETWORKS_RECIPES,
    *IMAGING_RECIPES,
    *ASSAYS_RECIPES,
    *WORKFLOW_RECIPES,
    *REPO_SEQUENCES_RECIPES,
    *REPO_FORMATS_RECIPES,
)


DOMAIN_RECIPES = {
    "sequence": SEQUENCE_RECIPES + REPO_SEQUENCES_RECIPES + REAL_GENOMES_RECIPES,
    "genomic-intervals": INTERVAL_RECIPES,
    "sequencing-reads": READ_RECIPES + REAL_READS_RECIPES + REAL_BAM_RECIPES,
    "variants": VARIANTS_RECIPES,
    "expression": (
        EXPRESSION_RECIPES
        + REAL_EXPRESSION_RECIPES
        + REAL_RNASEQ_RECIPES
        + REAL_INTERACTION_RECIPES
        + REAL_ENRICHMENT_RECIPES
        + REAL_SINGLECELL_RECIPES
        + REAL_SINGLECELL_REPRESENTATION_RECIPES
    ),
    "statistics": STATISTICS_RECIPES + REAL_CLINICAL_RECIPES,
    "phylogeny": PHYLOGENY_RECIPES + REAL_PROTEINS_RECIPES + REAL_PHYLOGENY_RECIPES,
    "assembly-and-ecology": ASSEMBLY_RECIPES + REAL_ASSEMBLY_RECIPES,
    "structures-and-proteomics": (
        STRUCTURE_RECIPES
        + REAL_STRUCTURE_RECIPES
        + REAL_HEME_POCKET_RECIPES
        + REAL_DOMAIN_RECIPES
        + REAL_CLUSTER_RECIPES
    ),
    "networks": NETWORKS_RECIPES,
    "imaging-and-spatial": IMAGING_RECIPES,
    "assays-and-metabolomics": ASSAYS_RECIPES,
    "workflow-and-identifiers": WORKFLOW_RECIPES,
}
EXPLICIT_DOMAINS = {
    "strand-extraction": "sequence",
    "interval-overlap": "genomic-intervals",
    "donor-counts": "expression",
    "cell-fractions": "expression",
    "paired-read-qc": "sequencing-reads",
    "genotype-alleles": "variants",
    "transcript-tpm": "expression",
    "alignment-sites": "phylogeny",
    "tree-branches": "phylogeny",
    "busco-summary": "assembly-and-ecology",
    "taxonomy-counts": "assembly-and-ecology",
    "image-measurements": "imaging-and-spatial",
    "paf-query-coverage": "sequencing-reads",
    "sam-pair-concordance": "sequencing-reads",
    "vcf-sample-qc": "variants",
    "fastqc-report-reconciliation": "sequencing-reads",
    "alignment-partitions": "phylogeny",
    "newick-split-support": "phylogeny",
    "gtf-coverage-counts": "expression",
    "bedgraph-threshold-peaks": "genomic-intervals",
    "sra-spot-export": "sequencing-reads",
}


def annotated_recipes(recipes: tuple[Recipe, ...]) -> tuple[Recipe, ...]:
    coverage = json.loads(Path(__file__).with_name("repository_coverage.json").read_text())["repositories"]
    domains = {recipe.id: domain for domain, members in DOMAIN_RECIPES.items() for recipe in members}
    domains.update(EXPLICIT_DOMAINS)
    result = []
    for recipe in recipes:
        repositories = [row for row in coverage if recipe.id in row["recipes"]]
        result.append(
            replace(
                recipe,
                domain=domains[recipe.id],
                repositories=tuple(row["name"] for row in repositories),
                sources=tuple(dict.fromkeys([*recipe.sources, *(row["source_url"] for row in repositories)])),
            )
        )
    unknown = {identifier for row in coverage for identifier in row["recipes"]} - {recipe.id for recipe in recipes}
    if unknown:
        raise ValueError(f"Repository coverage references unknown recipes: {sorted(unknown)}")
    return tuple(result)


RECIPES = annotated_recipes(RECIPES)
