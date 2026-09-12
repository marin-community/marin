# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read the pinned cleaned TaskTrove archive boundary without extracting files."""

import dataclasses
import io
import re
import tarfile
from dataclasses import dataclass

from tasktrove_verify.spec import Spec, mode_of, parse_spec

from taskcompendium.models import VERIFIER_REVISION, Source, VerifierSpec, relative_path

RELEASE = "2026.09.10.8"
RELEASE_ROOT = f"s3://marin-us-east-02a/marin/tasktrove/clean/{RELEASE}"
INSPECTED_CONVERTER_REVISION = "bef70bb8584d7e0c1391d88c9eacb1605b551a90"
RELEASE_PRODUCER_REVISION = "ccc5ff24cd16112a1e68c5dc6ca9f5ee4f6a52c2"
IMPORTER_REVISION = "taskcompendium-tasktrove-v0.2"
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
_VERIFIER_INSTALL = re.compile(
    rb'https://github\.com/marin-community/marin(?:\.git)?@([^#"\s]+)#subdirectory=lib/tasktrove-verify'
)


@dataclass(frozen=True)
class TaskArchive:
    source: Source
    family: str
    files: dict[str, bytes]

    @property
    def instructions(self) -> str:
        return self.files["instruction.md"].decode()

    @property
    def verifier(self) -> Spec:
        dockerfile = self.files.get("environment/Dockerfile")
        if dockerfile is not None and set(_VERIFIER_INSTALL.findall(dockerfile)) != {VERIFIER_REVISION.encode()}:
            raise ValueError(f"Task archive must declare the pinned verifier revision {VERIFIER_REVISION}")
        return parse_spec(self.files["tests/verifier.toml"].decode())


def read_archive(data: bytes, row: str, family: str) -> TaskArchive:
    """Read regular archive members with bounded expansion and no path traversal."""
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError("Task archive exceeds input limit")
    files: dict[str, bytes] = {}
    size = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if member.isdir():
                continue
            relative_path(name)
            if not member.isfile() or name in files:
                raise ValueError(f"Unsupported or duplicate archive member: {member.name}")
            size += member.size
            if size > MAX_ARCHIVE_BYTES:
                raise ValueError("Task archive exceeds expanded size limit")
            stream = archive.extractfile(member)
            assert stream is not None
            files[name] = stream.read()
    source = Source(RELEASE_ROOT, RELEASE, row, IMPORTER_REVISION)
    return TaskArchive(source, family, files)


def semantic_verifier(contract: Spec) -> VerifierSpec:
    """Remove execution/submission locations from a deterministic source verifier."""
    parameters = dataclasses.asdict(contract)
    parameters.pop("output", None)
    parameters.pop("workspace", None)
    return VerifierSpec(mode_of(contract), parameters)


def puzzle_instructions(instructions: str) -> str:
    """Remove the known all-puzzles file-delivery wrapper, preserving precision rules."""
    problem = "## Problem Statement"
    ending = "\n## Task\n"
    if (
        "## Deliverable" not in instructions
        or "## Puzzle Type" not in instructions
        or instructions.count(problem) != 1
        or instructions.count(ending) != 1
    ):
        raise ValueError("Unsupported all-puzzles instruction template")
    body = instructions.split(problem, 1)[1].split(ending, 1)[0].strip()
    if not body:
        raise ValueError("Empty puzzle statement")
    precision = "Coordinates should be rounded to 3 decimals where applicable."
    return f"{problem}\n\n{body}\n\n{precision}"
