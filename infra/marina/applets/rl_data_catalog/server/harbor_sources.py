# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned native Harbor releases registered for Atlas quality and difficulty review."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HarborSource:
    package: str
    digest: str
    published_at: str
    task_count: int
    count_basis: str
    metadata_checked_at: str
    repository_url: str
    paper_url: str
    license: str
    license_url: str
    family: str
    benchmark_basis: str
    notes: str
    verification: str


HARBOR_SOURCES = (
    HarborSource(
        package="skill2env/skill2env",
        digest="sha256:bef4f739bde8d865af04c04a2cb1c84ecad83b52dd558f4d582dd46c7933666c",
        published_at="2026-09-18T10:42:07.429943+00:00",
        task_count=7496,
        count_basis=(
            "Exact Harbor Hub dataset_version_task membership count for release 1.0.1 (revision 2), "
            "dataset version 102391e2-5a05-4ff5-a656-1cf7c473014e; checked 2026-10-01"
        ),
        metadata_checked_at="2026-10-01",
        repository_url="https://github.com/NVlabs/Skill2Env/tree/3fe416cecfeb1ac1d62bd1b5e799f8f248c5ba91",
        paper_url="https://www.alphaxiv.org/abs/2609.reinforcing-agents-collective-skills",
        license=(
            "Project-owned source: Apache-2.0. Third-party skills and assets retain their own terms; "
            "task-level reuse terms require review."
        ),
        license_url="https://github.com/NVlabs/Skill2Env/blob/3fe416cecfeb1ac1d62bd1b5e799f8f248c5ba91/README.md#license",
        family="terminal-agent",
        benchmark_basis="Released RL training corpus; the paper's private S2EBench is a separate evaluation set",
        notes=(
            "Official Skill2Env release with reference solutions, programmatic verifiers, and behavioral rubrics. "
            "Metadata registration does not establish task quality or training readiness. "
            "This catalog pins the release; it does not follow the latest tag."
        ),
        verification="Native tests/test.sh; tests/rubric.md supplies additional behavioral criteria",
    ),
)
