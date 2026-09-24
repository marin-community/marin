# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete protein-domain and proteome-coverage artifact contracts."""

import json

from experiments.post_training.bio_tasks.contract import Column, Contract, TableContract
from experiments.post_training.bio_tasks.real_data import source_text
from experiments.post_training.bio_tasks.recipe_types import DataOrigin, Instance, OracleRuntime, Recipe, WorkflowScope

PROTEOME_SOURCE = "UniProt:UP000000625-20260924"
MODEL_SOURCE = "Pfam:selected-models-20260924"


def generate_domains(_seed: int) -> Instance:
    reference = json.loads(source_text(PROTEOME_SOURCE, "ecoli-k12-domain-reference.json.gz"))
    contract = domain_contract(reference)
    return Instance(
        "Search the complete supplied E. coli K-12 reference proteome against the three supplied Pfam "
        "profiles and produce an auditable domain-coverage report. All inputs are in /app/inputs. "
        "proteins.fa contains unchanged UniProt sequences; proteins.tsv supplies accessions, sequence "
        "versions and lengths. models.hmm contains the versioned profiles listed in query.json. "
        "Run HMMER hmmsearch with --cpu 1, --seed from query.json, --cut_ga, --noali and --domtblout. "
        "Use the supplied model gathering thresholds for both sequence and domain reporting; retain "
        "every reported domain. Write domains.tsv with one row per model/accession/native-domain-index "
        "identity, using id=MODEL:ACCESSION:INDEX. Report the native scores, conditional and independent "
        "domain E-values, HMM positions, and protein alignment and envelope boundaries. Public table "
        "coordinates are 1-based and inclusive. Extract the unchanged protein sequence inside the "
        "alignment boundaries, not the envelope. Write proteins.tsv for every supplied protein, including "
        "proteins with zero hits, keyed by UniProt accession. Preserve its length and sequence version, "
        "count domains and distinct models, and calculate the union of aligned protein positions across "
        "all models. Overlapping domains must not double-count residues. Report answer.json for every "
        "supplied model, including zero-hit models: searched proteins, distinct matched proteins, domain "
        "instances and summed per-protein aligned-position unions for that model. This search covers "
        "only the supplied profiles; absence of a hit does not establish absence of other domains or "
        "biological function.",
        {
            "query.json": json.dumps(reference["query"], indent=2) + "\n",
            "proteins.fa": source_text(PROTEOME_SOURCE, "ecoli-k12-proteins.fasta.gz"),
            "proteins.tsv": source_text(PROTEOME_SOURCE, "ecoli-k12-metadata.tsv.gz"),
            "models.hmm": "".join(
                source_text(MODEL_SOURCE, name) for name in ("pf00042.hmm.gz", "pf00109.hmm.gz", "pf00115.hmm.gz")
            ),
        },
        contract,
        {"searched_only_matches": [{**row, "searched_proteins": row["matched_proteins"]} for row in contract.answer()]},
        data_origin=DataOrigin.REAL,
        source_ids=(PROTEOME_SOURCE, MODEL_SOURCE),
        derivation=(
            "All 4,403 reviewed proteins from observed reference proteome UP000000625, unchanged sequences "
            "and metadata; three unchanged versioned Pfam profiles. Native HMMER search with Biopython "
            "reference measurements and an independent standard-library artifact parser."
        ),
        workflow_scope=WorkflowScope.CONNECTED,
    )


def domain_contract(reference: dict) -> Contract:
    """Check every domain sequence, boundary, score and searched protein identity."""
    return Contract(
        columns={
            name: Column(kind="integer", unit=unit, description=description)
            for name, unit, description in (
                ("searched_proteins", "proteins", "all supplied target proteins"),
                ("matched_proteins", "proteins", "distinct proteins with a passing domain for this model"),
                ("domains", "domains", "passing domain instances for this model"),
                ("covered_residues", "amino acids", "sum of per-protein aligned-position unions for this model"),
            )
        },
        expected=reference["summaries"],
        tables={
            "domains.tsv": TableContract(
                columns={
                    **{
                        name: Column(kind="text", unit=unit, description=description)
                        for name, unit, description in (
                            ("protein", "accession", "UniProt target accession"),
                            ("model", "accession.version", "complete Pfam model identity"),
                            ("sequence", "amino acids", "unchanged protein residues inside alignment boundaries"),
                        )
                    },
                    **{
                        name: Column(kind="integer", unit="1-based index", description=description)
                        for name, description in (
                            ("domain_index", "native domain number within the protein/model pair"),
                            ("alignment_start", "first aligned protein residue"),
                            ("alignment_end", "last aligned protein residue, included"),
                            ("envelope_start", "first envelope protein residue"),
                            ("envelope_end", "last envelope protein residue, included"),
                            ("model_start", "first aligned HMM position"),
                            ("model_end", "last aligned HMM position, included"),
                        )
                    },
                    **{
                        name: Column(kind="number", unit="bits", description=description, atol=0.05, rtol=0)
                        for name, description in (
                            ("protein_score", "native per-sequence bit score"),
                            ("domain_score", "native domain bit score"),
                        )
                    },
                    **{
                        name: Column(
                            kind="number", unit="expected false positives", description=description, atol=0, rtol=0.05
                        )
                        for name, description in (
                            ("independent_evalue", "native independent domain E-value"),
                            ("conditional_evalue", "native conditional domain E-value"),
                        )
                    },
                },
                expected=reference["domains"],
                max_bytes=2 * 1024 * 1024,
            ),
            "proteins.tsv": TableContract(
                columns={
                    name: Column(kind="integer", unit=unit, description=description)
                    for name, unit, description in (
                        ("length", "amino acids", "complete supplied protein length"),
                        ("sequence_version", "version", "supplied UniProt sequence version"),
                        ("domains", "domains", "all passing domain instances across supplied models"),
                        ("distinct_models", "models", "distinct matching models"),
                        ("covered_residues", "amino acids", "union of aligned protein positions across models"),
                    )
                },
                expected=reference["proteins"],
                max_bytes=2 * 1024 * 1024,
            ),
        },
    )


RECIPES = (
    Recipe(
        id="real-proteome-domain-search",
        version="1",
        skills=("profile search", "domain coordinates", "sequence extraction", "overlap-aware proteome coverage"),
        formats=("FASTA", "Pfam HMM", "HMMER domtblout", "TSV protein metadata", "JSON query"),
        sources=("https://www.uniprot.org/proteomes/UP000000625", "https://www.ebi.ac.uk/interpro/entry/pfam/"),
        generate=generate_domains,
        oracle_timeout=900,
        oracle_runtime=OracleRuntime.NATIVE,
    ),
)
