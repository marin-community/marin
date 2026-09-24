# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Observed paired-read assembly with complete contig and reference measurements."""

import json

from experiments.post_training.bio_tasks.contract import Column, Contract, FastaContract, TableContract
from experiments.post_training.bio_tasks.real_data import source_text
from experiments.post_training.bio_tasks.recipe_types import DataOrigin, Instance, OracleRuntime, Recipe, WorkflowScope


def assembly_contract(reference: dict) -> Contract:
    """Verify every assembled residue, alignment measurement and reference position."""
    summary_columns = {
        name: Column(kind="integer", unit=unit, description=description, nullable=name in {"ng50", "lg50"})
        for name, unit, description in (
            ("read_pairs", "pairs", "all observed input read pairs"),
            ("input_bases", "bases", "sum of both mates' sequence lengths"),
            ("contigs", "contigs", "all native contigs, including unaligned contigs"),
            ("assembled_bases", "bases", "sum of complete contig lengths"),
            ("reference_length", "bases", "complete supplied reference length"),
            ("n50", "bases", "contig length crossing half the assembled length"),
            ("l50", "contigs", "number of descending-length contigs needed to cross half the assembled length"),
            ("ng50", "bases", "contig length crossing half the reference length; null if never reached"),
            ("lg50", "contigs", "number needed to cross half the reference length; null if never reached"),
            ("aligned_contigs", "contigs", "contigs with at least one reported reference alignment"),
            ("reference_covered_bases", "bases", "reference positions paired to at least one query base"),
            ("reference_uncovered_bases", "bases", "reference positions with zero aligned contig depth"),
            ("multiply_covered_reference_bases", "bases", "reference positions paired to at least two query bases"),
            ("aligned_query_bases", "bases", "union of matched or substituted query positions within each contig"),
            ("matches", "aligned bases", "sum of exact matches over all reported alignments"),
            ("substitutions", "aligned bases", "sum of nucleotide substitutions over all reported alignments"),
            ("inserted_bases", "bases", "query insertions over all reported alignments"),
            ("deleted_bases", "bases", "reference deletions over all reported alignments"),
        )
    }
    table_fields = {
        "read_qc.tsv": {
            "reads": ("reads", "records for this mate"),
            "bases": ("bases", "complete input bases"),
            "q20_bases": ("bases", "Phred+33 quality at least 20"),
            "q30_bases": ("bases", "Phred+33 quality at least 30"),
            "n_bases": ("bases", "N residues in input reads"),
        },
        "contig_metrics.tsv": {
            "length": ("bases", "complete native contig length"),
            "gc_bases": ("bases", "G and C residues"),
            "ambiguous_bases": ("bases", "residues other than A, C, G or T"),
            "alignments": ("alignments", "all reported alignments of this contig"),
            "aligned_query_bases": ("bases", "union of matched or substituted original query positions"),
        },
        "alignments.tsv": {
            "query_start": ("0-based coordinate", "inclusive query start on the original contig"),
            "query_end": ("0-based coordinate", "exclusive query end on the original contig"),
            "reference_start": ("0-based coordinate", "inclusive reference start"),
            "reference_end": ("0-based coordinate", "exclusive reference end"),
            "mapping_quality": ("MAPQ", "native PAF mapping quality"),
            "alignment_columns": ("columns", "matches, substitutions, inserted bases and deleted bases"),
            **{
                name: (summary_columns[name].unit, summary_columns[name].description)
                for name in ("matches", "substitutions", "inserted_bases", "deleted_bases")
            },
        },
        "reference_coverage.tsv": {"contig_depth": ("aligned bases", "number of query bases paired to this position")},
    }
    tables = {}
    for filename, fields in table_fields.items():
        columns = {name: Column(kind="integer", unit=unit, description=text) for name, (unit, text) in fields.items()}
        if filename == "alignments.tsv":
            columns.update(
                {
                    name: Column(kind="text", unit=unit, description=text)
                    for name, unit, text in (
                        ("contig", "FASTA identifier", "native contig identifier"),
                        ("reference", "FASTA identifier", "supplied reference identifier"),
                        ("strand", "+ or -", "query alignment orientation"),
                        ("cigar", "extended CIGAR", "complete native =, X, I and D operations"),
                    )
                }
            )
        tables[filename] = TableContract(columns=columns, expected=reference["tables"][filename], max_bytes=1024 * 1024)
    return Contract(
        columns=summary_columns,
        expected=reference["summaries"],
        fasta={"contigs.fa": FastaContract(sequences=reference["contigs"], max_bytes=1024 * 1024)},
        tables=tables,
    )


def generate_assembly(_seed: int) -> Instance:
    reference = json.loads(source_text("ENA:ERR266411", "err266411-assembly-reference.json.gz"))
    contract = assembly_contract(reference)
    inputs = {
        f"reads_R{mate}.fastq": source_text("ENA:ERR266411", f"err266411-spread-r{mate}.fastq.gz") for mate in (1, 2)
    }
    inputs["reference.fa"] = source_text("RefSeq:NC_001422.1", "nc_001422-1.fa.gz")
    inputs["query.json"] = json.dumps(reference["query"], indent=2) + "\n"
    return Instance(
        "Assemble the supplied observed PhiX paired reads and assess contiguity and agreement with the deposited "
        "reference genome. Inputs are in /app/inputs. The FASTQs are systematically spaced records across the full "
        "ENA ERR266411 run, with original sequences, qualities and matching mate identifiers. This technical "
        "subset is not biological replication. First report complete per-mate read and quality counts in read_qc.tsv, "
        "with ids R1 and R2. Use all input reads unchanged. Run the installed SPAdes with --isolate --only-assembler, "
        "the k-mer lengths in query.json, one thread and a 4 GB memory limit. Retain every native contig and its "
        "identifier in contigs.fa; use contigs.fasta rather than scaffolds. Align these contigs to reference.fa "
        "with minimap2 -x asm5 --secondary=no --eqx -c --cs=long -t 1. Include every reported alignment, including "
        "split alignments, in alignments.tsv. Use id=contig:strand:query_start:query_end:reference_start:reference_end. "
        "Coordinates are zero-based and half-open; query coordinates refer to the original forward contig. "
        "Write contig_metrics.tsv keyed by native contig ID and reference_coverage.tsv keyed by "
        "reference_id:zero_based_position for every reference position, including zeros. Count contig depth only "
        "where a query base is paired to a reference base (= or X); insertions and deletions do not contribute. "
        "Within each contig, count aligned query positions once when alignments overlap. Sum matches and errors "
        "over alignments without that deduplication. Write answer.json for id=assembly with complete input, "
        "assembly and reference accounting. For N50/L50 and NG50/LG50, sort all contig lengths descending and "
        "include the first contig reaching at least half the relevant length; do not filter short contigs. "
        "NG50/LG50 use the reference length and are null if the threshold is never reached. The reference is "
        "circular, but this task measures the native linear contigs and alignments without circularization or "
        "reference-guided correction. Contig-alignment breadth describes reference agreement; it is not read "
        "depth or a guarantee of assembly correctness.",
        inputs,
        contract,
        {
            "count_assembled_length_as_covered": [
                {**row, "reference_covered_bases": row["assembled_bases"]} for row in contract.answer()
            ]
        },
        data_origin=DataOrigin.REAL,
        source_ids=("ENA:ERR266411", "RefSeq:NC_001422.1"),
        derivation=(
            "12,000 observed read pairs systematically spaced across the checksum-verified run; "
            "native assembly and alignment."
        ),
        workflow_scope=WorkflowScope.CONNECTED,
    )


RECIPES = (
    Recipe(
        id="real-phix-assembly",
        version="1",
        skills=(
            "paired-read QC",
            "de novo assembly",
            "contiguity assessment",
            "alignment interpretation",
            "coverage union",
        ),
        formats=("paired FASTQ", "FASTA", "PAF", "CIGAR", "minimap2 cs", "TSV", "JSON"),
        sources=("https://www.ebi.ac.uk/ena/browser/view/ERR266411", "https://github.com/ablab/spades"),
        generate=generate_assembly,
        oracle_timeout=1800,
        oracle_runtime=OracleRuntime.NATIVE,
    ),
)
