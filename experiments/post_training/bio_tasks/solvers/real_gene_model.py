# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Biopython oracle for the complete Drosophila chromosome 4 gene-model audit."""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def merged_bases(intervals: list[tuple[int, int]]) -> int:
    """Count covered bases from sorted half-open intervals."""
    total = 0
    right = 0
    for start, end in sorted(intervals):
        if start >= right:
            total += end - start
            right = end
        elif end > right:
            total += end - right
            right = end
    return total


def middle(values: list[int | float]) -> float:
    values = sorted(values)
    midpoint = len(values) // 2
    if len(values) % 2:
        return float(values[midpoint])
    return (values[midpoint - 1] + values[midpoint]) / 2


def write_table(path: Path, rows: dict[str, dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", *next(iter(rows.values()))], delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows({"id": key, **row} for key, row in sorted(rows.items()))


def solve_gene_model(inputs: Path, output: Path) -> list[dict]:
    """Read all eligible GTF models and write the complete sequence and audit artifacts."""
    from Bio import SeqIO  # noqa: PLC0415 - installed only in this recipe's native environment

    output.mkdir(parents=True, exist_ok=True)
    query = json.loads((inputs / "query.json").read_text())
    if query != {
        "assembly": "BDGP6.54",
        "annotation_release": "Ensembl 115",
        "chromosome": "4",
        "gene_biotype": "protein_coding",
        "transcript_biotype": "protein_coding",
        "upstream_bases": 200,
    }:
        raise ValueError("Unexpected gene-model audit query")
    records = list(SeqIO.parse(inputs / "genome.fa", "fasta"))
    if len(records) != 1 or records[0].id != "4":
        raise ValueError("Expected complete chromosome 4 FASTA")
    chromosome = records[0].seq.upper()
    fai = (inputs / "genome.fa.fai").read_text().strip().split("\t")
    if len(fai) != 5 or fai[0] != "4" or int(fai[1]) != len(chromosome):
        raise ValueError("FAI and FASTA disagree")

    rows = []
    genes = {}
    transcripts = {}
    for line in (inputs / "annotations.gtf").read_text().splitlines():
        fields = line.split("\t")
        if len(fields) != 9 or fields[0] != "4":
            raise ValueError("Expected complete chromosome 4 GTF")
        feature = fields[2]
        attributes = dict(re.findall(r'(\w+) "([^"]*)";', fields[8]))
        start = int(fields[3]) - 1
        end = int(fields[4])
        strand = fields[6]
        if not 0 <= start < end <= len(chromosome) or strand not in {"+", "-"}:
            raise ValueError("Invalid GTF coordinate")
        rows.append((feature, start, end, strand, attributes))
        if feature == "gene" and attributes.get("gene_biotype") == query["gene_biotype"]:
            genes[attributes["gene_id"]] = {
                "gene_name": attributes["gene_name"],
                "strand": strand,
                "start_1based": start + 1,
                "end_1based": end,
            }
        if feature == "transcript" and attributes.get("transcript_biotype") == query["transcript_biotype"]:
            transcripts[attributes["transcript_id"]] = {
                "gene_id": attributes["gene_id"],
                "strand": strand,
                "exons": [],
                "five_prime_utr": [],
                "three_prime_utr": [],
                "CDS": [],
                "stop_codon": [],
            }
    for feature, start, end, _, attributes in rows:
        transcript = transcripts.get(attributes.get("transcript_id"))
        if transcript is not None and feature in {"exon", "five_prime_utr", "three_prime_utr", "CDS", "stop_codon"}:
            transcript[{"exon": "exons"}.get(feature, feature)].append((start, end))

    transcript_rows = {}
    by_gene = defaultdict(list)
    for transcript_id, transcript in transcripts.items():
        gene_id = transcript["gene_id"]
        if gene_id not in genes or transcript["strand"] != genes[gene_id]["strand"] or not transcript["exons"]:
            raise ValueError("Incomplete protein-coding transcript")
        by_gene[gene_id].append(transcript_id)
        transcript_rows[transcript_id] = {
            "gene_id": gene_id,
            "strand": transcript["strand"],
            "exon_count": len(transcript["exons"]),
            "exon_union_bases": merged_bases(transcript["exons"]),
            "utr5_bases": merged_bases(transcript["five_prime_utr"]),
            "utr3_bases": merged_bases(transcript["three_prime_utr"]),
            "cds_including_stop_bases": merged_bases(transcript["CDS"] + transcript["stop_codon"]),
        }

    gene_rows = {}
    promoter_sequences = {}
    for gene_id, gene in genes.items():
        if not by_gene[gene_id]:
            raise ValueError("Protein-coding gene has no protein-coding transcript")
        tss = gene["start_1based"] if gene["strand"] == "+" else gene["end_1based"]
        if gene["strand"] == "+":
            start = max(0, tss - 1 - query["upstream_bases"])
            end = tss - 1
            promoter = chromosome[start:end]
        else:
            start = tss
            end = min(len(chromosome), tss + query["upstream_bases"])
            promoter = chromosome[start:end].reverse_complement()
        sequence = str(promoter)
        callable_bases = sum(base in "ACGT" for base in sequence)
        gc_bases = sum(base in "GC" for base in sequence)
        if not sequence or not callable_bases:
            raise ValueError("Promoter cannot be scored")
        promoter_sequences[gene_id] = sequence
        gene_rows[gene_id] = {
            **gene,
            "transcript_count": len(by_gene[gene_id]),
            "exon_union_bases": merged_bases(
                [interval for tx_id in by_gene[gene_id] for interval in transcripts[tx_id]["exons"]]
            ),
            "promoter_start_1based": start + 1,
            "promoter_end_1based": end,
            "promoter_gc_bases": gc_bases,
            "promoter_acgt_bases": callable_bases,
            "promoter_ambiguous_bases": len(sequence) - callable_bases,
            "promoter_gc_fraction": gc_bases / callable_bases,
        }

    write_table(output / "transcripts.tsv", transcript_rows)
    write_table(output / "genes.tsv", gene_rows)
    with (output / "promoters.fa").open("w") as handle:
        for gene_id, sequence in sorted(promoter_sequences.items()):
            handle.write(f">{gene_id}\n{sequence}\n")
    return [
        {
            "id": "chr4",
            "gene_count": len(genes),
            "transcript_count": len(transcripts),
            "multi_transcript_gene_count": sum(len(members) > 1 for members in by_gene.values()),
            "median_utr5_bases_floor": int(middle([row["utr5_bases"] for row in transcript_rows.values()])),
            "median_cds_including_stop_bases_floor": int(
                middle([row["cds_including_stop_bases"] for row in transcript_rows.values()])
            ),
            "median_utr3_bases_floor": int(middle([row["utr3_bases"] for row in transcript_rows.values()])),
            "median_gene_exon_union_bases": middle([row["exon_union_bases"] for row in gene_rows.values()]),
            "median_promoter_gc_fraction": middle([row["promoter_gc_fraction"] for row in gene_rows.values()]),
        }
    ]
