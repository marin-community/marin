# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checked-in release registrations, independent of training integrations."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

REGISTERED_ORIGIN = "Registered releases"


class ReleaseHost(StrEnum):
    HUGGING_FACE = "Hugging Face"
    HARBOR_HUB = "Harbor Hub"
    GIT = "Git"


@dataclass(frozen=True)
class ReleaseReference:
    host: ReleaseHost
    repository: str
    revision: str


@dataclass(frozen=True)
class ExecutionContract:
    environment: str
    type: Literal["RLVR", "Alignment", "Agentic"]
    turns: Literal["Single-turn", "Multi-turn"]
    tools: Literal["Disabled", "Allowed"]
    agent: str
    scoring: str


@dataclass(frozen=True)
class RegisteredSource:
    name: str
    release: ReleaseReference
    version: str
    revised_at: str
    url: str
    counts: tuple[tuple[str, int], ...]
    count_basis: str
    count_url: str
    execution: ExecutionContract
    classification_basis: str
    verifier_revision: str
    verifier_url: str
    family: str
    family_url: str
    is_benchmark: bool
    benchmark_basis: str
    license: str
    notes: str
    usage_url: str = ""
    validation_url: str = ""
    upstream_url: str = ""
    license_url: str = ""
    paper_url: str = ""


PDBTHINK_RELEASE = ReleaseReference(
    ReleaseHost.HUGGING_FACE, "open-athena/pdbthink-coordinate-tasks", "3734406cb97b1702844319f9a5d860cbbf8fe660"
)
PDBTHINK_URL = f"https://huggingface.co/datasets/{PDBTHINK_RELEASE.repository}/blob/{PDBTHINK_RELEASE.revision}"

REGISTERED_SOURCES = (
    RegisteredSource(
        name="pdbthink-coordinate-tasks",
        release=PDBTHINK_RELEASE,
        version="1.3.0",
        revised_at="2026-10-02T18:39:51+00:00",
        url=f"https://huggingface.co/datasets/{PDBTHINK_RELEASE.repository}/tree/{PDBTHINK_RELEASE.revision}",
        counts=(("train", 91154), ("validation", 4411), ("test", 4435)),
        count_basis="Pinned release manifest: sum of registered split counts",
        count_url=f"{PDBTHINK_URL}/manifest.json",
        execution=ExecutionContract(
            environment="Harbor",
            type="RLVR",
            turns="Single-turn",
            tools="Disabled",
            agent="CoordinateNoToolsAgent; supply only prompt.json to the solver",
            scoring="Bundled pdbthink-coordinate verifier 1.1.0; exact correctness with per-task numeric tolerances",
        ),
        classification_basis=(
            "The release requires one tool-free response to displayed protein coordinates; "
            "Harbor packages the environment and deterministic verifier."
        ),
        verifier_revision=PDBTHINK_RELEASE.revision,
        verifier_url=f"{PDBTHINK_URL}/generator_source.tar.gz",
        family="protein-coordinate-reasoning",
        family_url=f"{PDBTHINK_URL}/manifest.json",
        is_benchmark=False,
        benchmark_basis="Training/evaluation tasks generated separately from the frozen PDBThink benchmark",
        license="Apache-2.0; coordinate provenance identifies the public Protein Data Bank sources",
        notes=(
            "19 coordinate families, from 2,671 PDB entries in 1,867 source groups. "
            "The frozen benchmark's entries, exact protein sequences and RCSB 30% clusters were excluded. "
            "No sequence-to-structure prediction or retired MECH tasks. "
            "A terminal agent changes the protocol; the Atlas runner needs a tool-free adapter. "
            "28,045 training tasks leave 8,192 output tokens in Snowball's 32,768-token context; "
            "count exact native tokens for other models. "
            "v1.3.0 fixes inclusive numeric boundaries and clarifies G04 sulfur-pair exclusions. "
            "Publisher validation is linked; Atlas quality and difficulty require separate assessment. "
            "The GLM teacher v1.0.0 release was generated and scored against task v1.2.0."
        ),
        usage_url=f"{PDBTHINK_URL}/USAGE.md",
        validation_url=f"{PDBTHINK_URL}/audits/contract_revision/regressions.json",
    ),
)
