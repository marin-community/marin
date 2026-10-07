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
SKILL2ENV_RELEASE = ReleaseReference(
    ReleaseHost.HARBOR_HUB,
    "skill2env/skill2env",
    "sha256:bef4f739bde8d865af04c04a2cb1c84ecad83b52dd558f4d582dd46c7933666c",
)
SKILL2ENV_URL = f"https://hub.harborframework.com/datasets/{SKILL2ENV_RELEASE.repository}"
SKILL2ENV_REPOSITORY = "https://github.com/NVlabs/Skill2Env"
SKILL2ENV_CODE_REVISION = "3fe416cecfeb1ac1d62bd1b5e799f8f248c5ba91"

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
    RegisteredSource(
        name="skill2env",
        release=SKILL2ENV_RELEASE,
        version="1.0.1",
        revised_at="2026-09-18T10:42:07.429943+00:00",
        url=SKILL2ENV_URL,
        counts=(("all", 7496),),
        count_basis=(
            "Exact Harbor Hub dataset_version_task membership count for release 1.0.1 (revision 2), "
            "dataset version 102391e2-5a05-4ff5-a656-1cf7c473014e; checked 2026-10-01"
        ),
        count_url=SKILL2ENV_URL,
        execution=ExecutionContract(
            environment="Harbor",
            type="Agentic",
            turns="Multi-turn",
            tools="Allowed",
            agent="Terminal agent; preserve task network restrictions and resource limits",
            scoring=(
                "Native tests/test.sh component rewards; Atlas reports their arithmetic mean and requires "
                "all components to equal one within 1e-9 tolerance for a full pass. "
                "The additional LLM reward from tests/rubric.md is separate."
            ),
        ),
        classification_basis="Native Harbor terminal tasks in the official release",
        verifier_revision=SKILL2ENV_RELEASE.revision,
        verifier_url=SKILL2ENV_URL,
        family="terminal-agent",
        family_url=f"{SKILL2ENV_REPOSITORY}/tree/{SKILL2ENV_CODE_REVISION}",
        is_benchmark=False,
        benchmark_basis="Released RL training corpus; the paper's private S2EBench is a separate evaluation set",
        license=(
            "Project-owned source: Apache-2.0. Third-party skills and assets retain their own terms; "
            "task-level reuse terms require review."
        ),
        notes=(
            "Official release with reference solutions, programmatic verifiers, and behavioral rubrics. "
            "Harbor revision 2 has version 1.0.1 and tags latest and v1.0; this registration pins its digest. "
            "Issue #9630 records a three-task Terminus-2 assessment and a confirmed grader defect; "
            "the prepared Some issues rating has not been published. "
            "Two attempts exhausted 50 turns, and the paper's additional LLM rubric reward was not run. "
            "The shared Atlas runner still needs the component-reward and offline-tooling changes from #9632. "
            "Registration establishes neither corpus-wide difficulty nor training readiness."
        ),
        usage_url="https://github.com/marin-community/marin/issues/9630",
        upstream_url=f"{SKILL2ENV_REPOSITORY}/tree/{SKILL2ENV_CODE_REVISION}",
        license_url=f"{SKILL2ENV_REPOSITORY}/blob/{SKILL2ENV_CODE_REVISION}/README.md#license",
        paper_url="https://www.alphaxiv.org/abs/2609.reinforcing-agents-collective-skills",
    ),
)
