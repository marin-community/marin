# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned release metadata for datasets outside training registries."""

from dataclasses import dataclass

HF_ORIGIN = "Hugging Face"


@dataclass(frozen=True)
class HuggingFaceSource:
    dataset_id: str
    revision: str
    revised_at: str
    version: str
    splits: tuple[tuple[str, int], ...]
    family: str
    environment: str
    type: str
    turns: str
    classification_basis: str
    benchmark_basis: str
    verification: str
    license: str
    notes: str
    usage_url: str
    validation_url: str


PDBTHINK_DATASET = "open-athena/pdbthink-coordinate-tasks"
PDBTHINK_REVISION = "3734406cb97b1702844319f9a5d860cbbf8fe660"
PDBTHINK_RELEASE = f"https://huggingface.co/datasets/{PDBTHINK_DATASET}/blob/{PDBTHINK_REVISION}"

HF_SOURCES = (
    HuggingFaceSource(
        dataset_id=PDBTHINK_DATASET,
        revision=PDBTHINK_REVISION,
        revised_at="2026-10-02T18:39:51+00:00",
        version="1.3.0",
        splits=(("train", 91154), ("validation", 4411), ("test", 4435)),
        family="protein-coordinate-reasoning",
        environment="Harbor",
        type="RLVR",
        turns="Single-turn",
        classification_basis=(
            "The release requires one tool-free response to displayed protein coordinates; "
            "Harbor packages the environment and deterministic verifier."
        ),
        benchmark_basis="Training/evaluation tasks generated separately from the frozen PDBThink benchmark",
        verification="Bundled pdbthink-coordinate verifier 1.1.0; exact correctness with per-task numeric tolerances",
        license="Apache-2.0; coordinate provenance identifies the public Protein Data Bank sources",
        notes=(
            "19 coordinate families, from 2,671 PDB entries in 1,867 source groups. "
            "The frozen benchmark's entries, exact protein sequences and RCSB 30% clusters were excluded. "
            "No sequence-to-structure prediction or retired MECH tasks. "
            "Use the release's CoordinateNoToolsAgent with tools disabled; a terminal agent changes the protocol. "
            "28,045 training tasks leave 8,192 output tokens in Snowball's 32,768-token context; "
            "count exact native tokens for other models. "
            "v1.3.0 fixes inclusive numeric boundaries and clarifies G04 sulfur-pair exclusions. "
            "Publisher validation is linked; Atlas quality and difficulty require separate assessment. "
            "The GLM teacher v1.0.0 release was generated and scored against task v1.2.0."
        ),
        usage_url=f"{PDBTHINK_RELEASE}/USAGE.md",
        validation_url=f"{PDBTHINK_RELEASE}/audits/contract_revision/regressions.json",
    ),
)
