# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A complete chromosome gene-model audit on observed Drosophila annotation."""

import json
from dataclasses import dataclass, field

from experiments.post_training.bio_tasks.contract import Column, Contract, FastaContract, TableContract
from experiments.post_training.bio_tasks.real_data import source_text
from experiments.post_training.bio_tasks.recipe_types import DataOrigin, Instance, OracleRuntime, Recipe, WorkflowScope

SOURCE_ID = "Ensembl:BDGP6.54:115:chr4"
GTF_ASSET = "dmel-bdgp6.54-r115-chr4.gtf.gz"
FASTA_ASSET = "dmel-bdgp6.54-r115-chr4.fa.gz"
FAI_ASSET = "dmel-bdgp6.54-r115-chr4.fa.fai.gz"
PROMOTER_BASES = 200
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


@dataclass
class Transcript:
    gene_id: str
    strand: str
    exons: list[tuple[int, int]] = field(default_factory=list)
    utr5: list[tuple[int, int]] = field(default_factory=list)
    utr3: list[tuple[int, int]] = field(default_factory=list)
    cds: list[tuple[int, int]] = field(default_factory=list)
    stop_codons: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class Gene:
    name: str
    start: int
    end: int
    strand: str
    transcripts: set[str] = field(default_factory=set)


def gtf_attributes(text: str) -> dict[str, str]:
    """Parse Ensembl's quoted GTF attributes without interpreting display names."""
    fields = {}
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        key, value = part.split(" ", 1)
        if not value.startswith('"') or not value.endswith('"'):
            raise ValueError("Malformed GTF attribute")
        fields[key] = value[1:-1]
    return fields


def annotation(gtf: str, chromosome_bases: int) -> tuple[dict[str, Gene], dict[str, Transcript]]:
    """Retain every protein-coding gene and its protein-coding transcript isoforms."""
    rows = []
    genes: dict[str, Gene] = {}
    transcripts: dict[str, Transcript] = {}
    for line in gtf.splitlines():
        chrom, _, feature, start_text, end_text, _, strand, _, attrs_text = line.split("\t")
        start, end = int(start_text), int(end_text)
        if chrom != "4" or not 1 <= start <= end <= chromosome_bases or strand not in {"+", "-"}:
            raise ValueError("Annotation does not match the complete chromosome 4 FASTA")
        attrs = gtf_attributes(attrs_text)
        rows.append((feature, start, end, strand, attrs))
        if feature == "gene" and attrs.get("gene_biotype") == "protein_coding":
            gene_id = attrs["gene_id"]
            if gene_id in genes:
                raise ValueError("Duplicate gene feature")
            genes[gene_id] = Gene(attrs["gene_name"], start, end, strand)
        if feature == "transcript" and attrs.get("transcript_biotype") == "protein_coding":
            transcript_id = attrs["transcript_id"]
            if transcript_id in transcripts:
                raise ValueError("Duplicate transcript feature")
            transcripts[transcript_id] = Transcript(attrs["gene_id"], strand)
    for feature, start, end, strand, attrs in rows:
        transcript_id = attrs.get("transcript_id")
        if transcript_id not in transcripts:
            continue
        transcript = transcripts[transcript_id]
        if transcript.gene_id not in genes or genes[transcript.gene_id].strand != strand:
            raise ValueError("Transcript does not match a protein-coding parent gene")
        genes[transcript.gene_id].transcripts.add(transcript_id)
        if feature == "exon":
            transcript.exons.append((start, end))
        elif feature == "five_prime_utr":
            transcript.utr5.append((start, end))
        elif feature == "three_prime_utr":
            transcript.utr3.append((start, end))
        elif feature == "CDS":
            transcript.cds.append((start, end))
        elif feature == "stop_codon":
            transcript.stop_codons.append((start, end))
    if any(not gene.transcripts for gene in genes.values()) or any(not tx.exons for tx in transcripts.values()):
        raise ValueError("Incomplete protein-coding gene model")
    return genes, transcripts


def distinct_bases(intervals: list[tuple[int, int]]) -> int:
    """Count the union of 1-based closed reference intervals independently of row order."""
    positions = set()
    for start, end in intervals:
        positions.update(range(start, end + 1))
    return len(positions)


def median(values: list[int | float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return float(ordered[middle]) if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def reference(gtf: str, fasta: str) -> dict:
    """Compute the private reference directly from deposited rows and bases."""
    lines = fasta.splitlines()
    if not lines[0].startswith(">4 "):
        raise ValueError("Wrong reference FASTA")
    sequence = "".join(lines[1:]).upper()
    if len(sequence) != 1_348_131:
        raise ValueError("Incomplete chromosome 4 sequence")
    genes, transcripts = annotation(gtf, len(sequence))
    tx_rows = {}
    for transcript_id, transcript in sorted(transcripts.items()):
        tx_rows[transcript_id] = {
            "gene_id": transcript.gene_id,
            "strand": transcript.strand,
            "exon_count": len(transcript.exons),
            "exon_union_bases": distinct_bases(transcript.exons),
            "utr5_bases": distinct_bases(transcript.utr5),
            "utr3_bases": distinct_bases(transcript.utr3),
            "cds_including_stop_bases": distinct_bases(transcript.cds + transcript.stop_codons),
        }
    gene_rows = {}
    promoters = {}
    for gene_id, gene in sorted(genes.items()):
        tss = gene.start if gene.strand == "+" else gene.end
        if gene.strand == "+":
            promoter_start = max(1, tss - PROMOTER_BASES)
            promoter_end = tss - 1
            promoter = sequence[promoter_start - 1 : promoter_end]
        else:
            promoter_start = tss + 1
            promoter_end = min(len(sequence), tss + PROMOTER_BASES)
            promoter = sequence[promoter_start - 1 : promoter_end].translate(COMPLEMENT)[::-1]
        if not promoter:
            raise ValueError("Empty promoter at chromosome boundary")
        acgt = sum(base in "ACGT" for base in promoter)
        if not acgt:
            raise ValueError("Promoter has no callable bases")
        gc = sum(base in "GC" for base in promoter)
        promoters[gene_id] = promoter
        exon_intervals = [interval for tx_id in gene.transcripts for interval in transcripts[tx_id].exons]
        gene_rows[gene_id] = {
            "gene_name": gene.name,
            "strand": gene.strand,
            "start_1based": gene.start,
            "end_1based": gene.end,
            "transcript_count": len(gene.transcripts),
            "exon_union_bases": distinct_bases(exon_intervals),
            "promoter_start_1based": promoter_start,
            "promoter_end_1based": promoter_end,
            "promoter_gc_bases": gc,
            "promoter_acgt_bases": acgt,
            "promoter_ambiguous_bases": len(promoter) - acgt,
            "promoter_gc_fraction": gc / acgt,
        }
    summary = {
        "chr4": {
            "gene_count": len(genes),
            "transcript_count": len(transcripts),
            "multi_transcript_gene_count": sum(len(gene.transcripts) > 1 for gene in genes.values()),
            "median_utr5_bases_floor": int(median([row["utr5_bases"] for row in tx_rows.values()])),
            "median_cds_including_stop_bases_floor": int(
                median([row["cds_including_stop_bases"] for row in tx_rows.values()])
            ),
            "median_utr3_bases_floor": int(median([row["utr3_bases"] for row in tx_rows.values()])),
            "median_gene_exon_union_bases": median([row["exon_union_bases"] for row in gene_rows.values()]),
            "median_promoter_gc_fraction": median([row["promoter_gc_fraction"] for row in gene_rows.values()]),
        }
    }
    return {"summaries": summary, "transcripts": tx_rows, "genes": gene_rows, "promoters": promoters}


def gene_model_contract(result: dict) -> Contract:
    """Grade complete transcript, gene, sequence and aggregate artifacts."""

    def integer(unit: str, description: str) -> Column:
        return Column(kind="integer", unit=unit, description=description)

    def number(unit: str, description: str) -> Column:
        return Column(kind="number", unit=unit, description=description, atol=1e-9)

    def text(description: str) -> Column:
        return Column(kind="text", unit="annotation", description=description)

    return Contract(
        columns={
            "gene_count": integer("genes", "all protein-coding chromosome 4 genes"),
            "transcript_count": integer("transcripts", "all protein-coding isoforms"),
            "multi_transcript_gene_count": integer("genes", "genes with at least two protein-coding isoforms"),
            "median_utr5_bases_floor": integer("bases", "floored median union 5-prime UTR bases per transcript"),
            "median_cds_including_stop_bases_floor": integer(
                "bases", "floored median CDS plus stop-codon union bases per transcript"
            ),
            "median_utr3_bases_floor": integer("bases", "floored median union 3-prime UTR bases per transcript"),
            "median_gene_exon_union_bases": number("bases", "median exon union bases per gene"),
            "median_promoter_gc_fraction": number("fraction", "median per-gene promoter GC fraction"),
        },
        expected=result["summaries"],
        tables={
            "transcripts.tsv": TableContract(
                columns={
                    "gene_id": text("parent FlyBase gene ID"),
                    "strand": text("genomic strand"),
                    "exon_count": integer("exon rows", "number of supplied exon features"),
                    "exon_union_bases": integer("bases", "union of this transcript's exon intervals"),
                    "utr5_bases": integer("bases", "union of explicit five_prime_utr intervals, including zero"),
                    "utr3_bases": integer("bases", "union of explicit three_prime_utr intervals, including zero"),
                    "cds_including_stop_bases": integer(
                        "bases", "union of CDS and stop_codon intervals without double-counting overlap"
                    ),
                },
                expected=result["transcripts"],
                max_bytes=1_000_000,
            ),
            "genes.tsv": TableContract(
                columns={
                    "gene_name": text("deposited FlyBase gene name"),
                    "strand": text("genomic strand"),
                    "start_1based": integer("1-based coordinate", "inclusive gene start"),
                    "end_1based": integer("1-based coordinate", "inclusive gene end"),
                    "transcript_count": integer("transcripts", "protein-coding isoforms of this gene"),
                    "exon_union_bases": integer("bases", "nonoverlapping union across all protein-coding isoforms"),
                    "promoter_start_1based": integer("1-based coordinate", "inclusive genomic window start"),
                    "promoter_end_1based": integer("1-based coordinate", "inclusive genomic window end"),
                    "promoter_gc_bases": integer("bases", "G or C bases in oriented promoter"),
                    "promoter_acgt_bases": integer("bases", "A, C, G or T promoter bases"),
                    "promoter_ambiguous_bases": integer("bases", "promoter bases excluded from GC denominator"),
                    "promoter_gc_fraction": number("fraction", "GC divided by ACGT bases"),
                },
                expected=result["genes"],
                max_bytes=1_000_000,
            ),
        },
        fasta={"promoters.fa": FastaContract(sequences=result["promoters"], max_bytes=1_000_000)},
    )


def generate_gene_model(_seed: int) -> Instance:
    gtf = source_text(SOURCE_ID, GTF_ASSET)
    fasta = source_text(SOURCE_ID, FASTA_ASSET)
    fai = source_text(SOURCE_ID, FAI_ASSET)
    result = reference(gtf, fasta)
    contract = gene_model_contract(result)
    query = {
        "assembly": "BDGP6.54",
        "annotation_release": "Ensembl 115",
        "chromosome": "4",
        "gene_biotype": "protein_coding",
        "transcript_biotype": "protein_coding",
        "upstream_bases": PROMOTER_BASES,
    }
    return Instance(
        "Audit the complete observed Drosophila melanogaster BDGP6.54 chromosome 4 reference and Ensembl 115 "
        "annotation in /app/inputs. Read genome.fa, genome.fa.fai, annotations.gtf and query.json. Retain "
        "every gene with gene_biotype protein_coding and every one of its transcripts with "
        "transcript_biotype protein_coding; preserve all alternative isoforms, including zero-length UTRs. "
        "Use gene_id and transcript_id as stable IDs and gene_name only as a display label. GTF coordinates "
        "are 1-based inclusive; merge overlapping or touching exons and explicit five_prime_utr and "
        "three_prime_utr features before counting distinct bases. Do not infer UTRs from CDS or stop codons; "
        "CDS length includes stop_codon bases, unioning overlaps once. "
        "For each gene, union exons across its retained isoforms, then take the gene-level 5-prime TSS "
        "at start on + or end on -. Extract up to upstream_bases immediately upstream without the TSS "
        "base, clipping at chromosome ends. Report inclusive genomic promoter bounds in genes.tsv, "
        "but write promoters.fa in transcriptional 5-prime-to-3-prime orientation, reverse complementing "
        "minus-strand windows. Count G+C over only A/C/G/T; report ambiguous bases separately. "
        "Summaries are medians over the complete retained transcript or gene universes named by each "
        "field, with an even-sized median equal to the mean of the two middle values; floor the three "
        "transcript-region medians once after aggregation. These coordinate "
        "windows are operational promoters, not measured regulatory activity. All outputs derive from "
        "the same matched annotation and reference.",
        {
            "genome.fa": fasta,
            "genome.fa.fai": fai,
            "annotations.gtf": gtf,
            "query.json": json.dumps(query, sort_keys=True, indent=2) + "\n",
        },
        contract,
        {
            "median_over_gene_utr_instead_of_transcripts": [
                {**row, "median_utr5_bases_floor": row["median_utr5_bases_floor"] + 1} for row in contract.answer()
            ],
            "promoter_gc_over_all_bases": [
                {**row, "median_promoter_gc_fraction": row["median_promoter_gc_fraction"] + 0.01}
                for row in contract.answer()
            ],
        },
        data_origin=DataOrigin.REAL,
        source_ids=(SOURCE_ID,),
        derivation=(
            "All 9,582 chromosome 4 feature rows from Ensembl 115 BDGP6.54 chromosome GTF and the complete "
            "1,348,131-base chromosome 4 FASTA, with no gene or isoform subsampling. Query retains the "
            "declared protein-coding universe. Private reference uses direct positional unions; native "
            "oracle uses Biopython sequence parsing and interval sweeps."
        ),
        workflow_scope=WorkflowScope.CONNECTED,
    )


RECIPES = (
    Recipe(
        id="real-dmel-gene-model-audit",
        version="1",
        skills=(
            "gene-transcript identity",
            "strand-aware promoter extraction",
            "UTR lengths",
            "exon union",
            "GC composition",
        ),
        formats=("FASTA", "FAI", "GTF", "TSV", "JSON"),
        sources=("https://ftp.ensembl.org/pub/release-115/gtf/drosophila_melanogaster/",),
        generate=generate_gene_model,
        oracle_timeout=1800,
        oracle_runtime=OracleRuntime.NATIVE,
    ),
)
