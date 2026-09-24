# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Observed phage genomes with deposited coding annotations and translations."""

import json
import random
import re
from collections import Counter
from functools import partial
from itertools import pairwise, product

import numpy as np

from experiments.post_training.bio_tasks.contract import Column, Contract
from experiments.post_training.bio_tasks.real_data import source_text
from experiments.post_training.bio_tasks.recipe_types import DataOrigin, Instance, Recipe

ACCESSIONS = ("NC_001422.1", "NC_001416.1", "NC_001604.1")


def generate_real_genome(seed: int, operation: str) -> Instance:
    rng = random.Random(seed)
    accession = rng.choice(ACCESSIONS)
    stem = accession.lower().replace(".", "-")
    source = f"RefSeq:{accession}"
    reference = json.loads(source_text(source, f"{stem}-reference.json.gz"))
    # Authoring reads the original GenBank sequence; the isolated solver reads FASTA/GFF3.
    origin = source_text(source, f"{stem}.gb.gz").split("ORIGIN", 1)[1].split("//", 1)[0]
    genome = re.sub("[^acgt]", "", origin).upper()
    genes = reference["genes"]
    query = {"operation": operation, "topology": reference["topology"]}
    inputs = {
        "genome.fa": source_text(source, f"{stem}.fa.gz"),
        "annotations.gff3": source_text(source, f"{stem}.gff3.gz"),
    }
    prompt = (
        f"Analyze the unchanged {accession} genome and deposited coding annotations in /app/inputs. "
        "Read operation parameters and genome topology from query.json. GFF3 coordinates are 1-based inclusive; "
        "repeated CDS IDs are parts of one coding sequence. "
        "The part=i/n attribute specifies biological 5'-to-3' concatenation order, including origin-crossing "
        "or overlapping parts. Orient each part by its strand before concatenating; do not discard phase bases "
        "from later parts. All supplied CDSs are complete and reproduce deposited proteins. "
    )
    expected = {}
    wrong = {}
    if operation == "real-genome-cds-extraction":
        expected = {
            gene["id"]: {"sequence": gene["coding_sequence"], "length": len(gene["coding_sequence"])} for gene in genes
        }
        columns = {
            "sequence": Column(kind="text", description="complete oriented CDS including stop codon", unit="DNA"),
            "length": Column(kind="integer", description="CDS length", unit="bases"),
        }
        wrong = {key: {**row, "sequence": row["sequence"][::-1]} for key, row in expected.items()}
        prompt += "Report each complete coding sequence and its length, keyed by the CDS ID."
        mutation = "reverse_sequence_without_complement"
    elif operation == "real-genome-translation":
        expected = {gene["id"]: {"protein": gene["protein"], "length": len(gene["protein"])} for gene in genes}
        columns = {
            "protein": Column(kind="text", description="translation without terminal stop", unit="amino acids"),
            "length": Column(kind="integer", description="protein length", unit="residues"),
        }
        wrong = {key: {"protein": row["protein"] + "*", "length": row["length"] + 1} for key, row in expected.items()}
        prompt += (
            "Extract and translate every complete CDS using its transl_table (1 or 11). "
            "Translate the initiating codon as methionine even for an alternative bacterial start; "
            "omit the terminal stop. Report protein sequence and residue count for each CDS ID."
        )
        mutation = "retained_terminal_stop_as_residue"
    elif operation == "real-genome-gc3":
        for gene in genes:
            sequence = gene["coding_sequence"][:-3]
            thirds = sequence[2::3]
            gc = sum(base in "GC" for base in thirds)
            expected[gene["id"]] = {"gc3": gc / len(thirds), "gc_codons": gc, "codons": len(thirds)}
            wrong[gene["id"]] = {**expected[gene["id"]], "gc3": sum(base in "GC" for base in sequence) / len(sequence)}
        columns = {
            "gc3": Column(
                kind="number", description="fraction of sense codons ending G or C", unit="fraction", atol=1e-12
            ),
            "gc_codons": Column(kind="integer", description="G/C third-position codons", unit="codons"),
            "codons": Column(kind="integer", description="all coding codons excluding terminal stop", unit="codons"),
        }
        prompt += "For each CDS ID, report GC3 and its numerator/denominator after excluding the terminal stop codon."
        mutation = "overall_gc_instead_of_third_position_gc"
    elif operation == "real-genome-codon-counts":
        counts = Counter(
            gene["coding_sequence"][i : i + 3] for gene in genes for i in range(0, len(gene["coding_sequence"]) - 3, 3)
        )
        expected = {"".join(codon): {"count": counts["".join(codon)]} for codon in product("ACGT", repeat=3)}
        wrong = {key: dict(row) for key, row in expected.items()}
        for gene in genes:
            wrong[gene["coding_sequence"][-3:]]["count"] += 1
        columns = {
            "count": Column(
                kind="integer", description="sense-position occurrences across annotated CDSs", unit="codons"
            )
        }
        prompt += (
            "Count coding triplets across all supplied CDSs, including initiator triplets and excluding each terminal "
            "stop. Keep overlapping genes as separate biological coding records. Report all 64 uppercase DNA codon IDs, "
            "including zero counts. Do not convert alternative initiator DNA triplets to ATG."
        )
        mutation = "counted_terminal_stops"
    elif operation == "real-genome-overlap":
        coverage = np.zeros((len(genes), len(genome)), dtype=bool)
        for index, gene in enumerate(genes):
            for part in gene["parts"]:
                coverage[index, part["start"] : part["end"]] = True
        totals = coverage.sum(axis=0)
        for index, gene in enumerate(genes):
            covered = int(coverage[index].sum())
            overlap = int(((totals > 1) & coverage[index]).sum())
            expected[gene["id"]] = {"bases": covered, "overlap": overlap}
            wrong[gene["id"]] = {"bases": covered, "overlap": covered}
        columns = {
            "bases": Column(kind="integer", description="distinct genomic positions occupied by this CDS", unit="bases"),
            "overlap": Column(
                kind="integer", description="positions also occupied by another CDS ID on either strand", unit="bases"
            ),
        }
        prompt += (
            "For every CDS ID, report distinct genomic bases it occupies and how many belong to another CDS ID. "
            "Union overlapping parts of the same CDS first; count each genomic base once and include either strand."
        )
        mutation = "counted_self_as_overlapping_gene"
    elif operation == "real-genome-promoters":
        width = rng.choice((50, 100, 200))
        query["upstream_bases"] = width
        for gene in genes:
            first = gene["parts"][0]
            start = first["start"] if gene["strand"] == 1 else first["end"] - 1
            positions = [start - gene["strand"] * distance for distance in range(width, 0, -1)]
            if reference["topology"] == "linear":
                positions = [position for position in positions if 0 <= position < len(genome)]
            bases = "".join(genome[position % len(genome)] for position in positions)
            if gene["strand"] == -1:
                bases = bases.translate(str.maketrans("ACGT", "TGCA"))
            expected[gene["id"]] = {"sequence": bases, "length": len(bases)}
            shifted = "".join(genome[(position + gene["strand"]) % len(genome)] for position in positions)
            if gene["strand"] == -1:
                shifted = shifted.translate(str.maketrans("ACGT", "TGCA"))
            wrong[gene["id"]] = {"sequence": shifted, "length": len(shifted)}
        columns = {
            "sequence": Column(
                kind="text", description="upstream genomic window in coding-strand orientation", unit="DNA"
            ),
            "length": Column(kind="integer", description="retained upstream window length", unit="bases"),
        }
        prompt += (
            "Extract the upstream_bases positions immediately before each CDS's first biological base, excluding "
            "that base. Report windows in coding-strand 5'-to-3' order. Wrap only circular genomes; clip linear "
            "genomes at their ends. This is a coordinate window, not a claim that every window is a promoter."
        )
        mutation = "shifted_window_into_cds"
    elif operation == "real-genome-restriction-digest":
        motifs = {
            "EcoRI": {"motif": "GAATTC", "cut_after": 1},
            "BamHI": {"motif": "GGATCC", "cut_after": 1},
            "HindIII": {"motif": "AAGCTT", "cut_after": 1},
            "HaeIII": {"motif": "GGCC", "cut_after": 2},
        }
        query["enzymes"] = motifs
        sequence = genome + genome[:5] if reference["topology"] == "circular" else genome
        for enzyme, profile in motifs.items():
            motif = profile["motif"]
            cuts = sorted(
                (match.start() + profile["cut_after"]) % len(genome)
                for match in re.finditer(f"(?={motif})", sequence)
                if match.start() < len(genome)
            )
            boundaries = cuts if reference["topology"] == "circular" and cuts else [0, *cuts, len(genome)]
            fragments = [b - a for a, b in pairwise(boundaries)]
            if reference["topology"] == "circular" and cuts:
                fragments.append(len(genome) - cuts[-1] + cuts[0])
            expected[enzyme] = {
                "cut_offsets": ",".join(map(str, cuts)),
                "fragment_lengths": ",".join(map(str, sorted(fragments))),
            }
            wrong[enzyme] = {**expected[enzyme], "cut_offsets": ",".join(str(cut + 1) for cut in cuts)}
        columns = {
            "cut_offsets": Column(
                kind="text", description="sorted comma-separated 0-based cut boundaries; empty if none", unit="bases"
            ),
            "fragment_lengths": Column(
                kind="text", description="sorted comma-separated lengths retaining multiplicities", unit="bases"
            ),
        }
        prompt += (
            "Digest each enzyme separately using the palindromic motifs in query.json. Each cuts the forward strand "
            "cut_after bases after the motif's start. Report 0-based cut boundaries and fragment lengths per enzyme. "
            "Allow a site spanning the origin only for circular genomes, reducing cut offsets modulo genome length. "
            "An uncut genome contributes one full-length fragment. Ignore methylation and partial digestion."
        )
        mutation = "one_based_instead_of_zero_based_cut_offsets"
    else:
        raise ValueError(operation)
    inputs["query.json"] = json.dumps(query) + "\n"
    return Instance(
        prompt,
        inputs,
        Contract(columns=columns, expected=expected),
        {mutation: [{"id": key, **row} for key, row in wrong.items()]},
        data_origin=DataOrigin.REAL,
        source_ids=(source,),
        derivation="Full reference genome unchanged; complete CDS coordinates converted from deposited GenBank to GFF3. "
        "Biopython 1.86 extraction/translation checked against deposited proteins; all retained parts and exclusions "
        "are recorded in the source preparation. Instance query selects the declared operation and window size.",
    )


RECIPES = tuple(
    Recipe(
        name,
        "1",
        skills,
        ("FASTA", "GFF3"),
        tuple(f"https://www.ncbi.nlm.nih.gov/nuccore/{accession}" for accession in ACCESSIONS),
        partial(generate_real_genome, operation=name),
    )
    for name, skills in (
        ("real-genome-cds-extraction", ("strand", "compound-cds", "circular-coordinates")),
        ("real-genome-translation", ("compound-cds", "genetic-code", "alternative-start-codons")),
        ("real-genome-gc3", ("reading-frame", "coding-composition", "denominators")),
        ("real-genome-codon-counts", ("reading-frame", "overlapping-genes", "stop-codon-exclusion")),
        ("real-genome-overlap", ("interval-union", "overlapping-genes", "coordinate-conventions")),
        ("real-genome-promoters", ("strand", "circular-coordinates", "boundary-clipping")),
        (
            "real-genome-restriction-digest",
            ("restriction-sites", "circular-coordinates", "fragment-lengths"),
        ),
    )
)
