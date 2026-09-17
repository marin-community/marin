# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import one supported TaskTrove Clean MCQA archive into a direct-chat TaskSpec."""

import io
import tarfile
import tomllib
from dataclasses import dataclass

from tasktrove_verify.spec import McqSpec, parse_spec

from taskcompendium.models import MultipleChoiceAnswer, Source, TaskRequirements, TaskSpec

RELEASE = "2026.09.10.9"
RELEASE_ROOT = f"s3://marin-us-east-02a/marin/tasktrove/clean/{RELEASE}"
IMPORTER_REVISION = "taskcompendium-tasktrove-mcqa-v0.1"
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
FAMILY = "qa-short-answer"
CONVERTER = "nemotron_mcqa"
_PREFIX = (
    "You are answering a multiple-choice question. Read the question below and write your final "
    "answer to `/app/answer.txt`.\n\n"
    "The verifier extracts a single letter (A/B/C/...) from your answer file using a regex pattern; "
    "the simplest valid output is a file containing exactly\n`Answer: X` (where X is your chosen letter).\n\n"
    "---\n\n"
)
_FORMAT_PREFIX = "Answer the following multiple choice question. The last line of your response should be"


@dataclass(frozen=True)
class TaskArchive:
    """A bounded, release-pinned TaskTrove archive."""

    row: str
    files: dict[str, bytes]

    @property
    def source(self) -> Source:
        return Source(RELEASE_ROOT, RELEASE, self.row, IMPORTER_REVISION)


def read_archive(data: bytes, row: str) -> TaskArchive:
    """Read regular archive members without extracting them to the host filesystem."""
    if not row or len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError("Task archive has an invalid row or exceeds the input limit")
    files: dict[str, bytes] = {}
    size = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if member.isdir():
                continue
            if not member.isfile() or not name or name.startswith("/") or ".." in name.split("/") or name in files:
                raise ValueError(f"Unsupported archive member: {member.name}")
            size += member.size
            if size > MAX_ARCHIVE_BYTES:
                raise ValueError("Task archive exceeds expanded size limit")
            stream = archive.extractfile(member)
            assert stream is not None
            files[name] = stream.read()
    return TaskArchive(row, files)


def _clean_instructions(instructions: str) -> str:
    if not instructions.startswith(_PREFIX):
        raise ValueError("Unsupported MCQA instruction template")
    prompt = instructions.removeprefix(_PREFIX)
    first_line, separator, question = prompt.partition("\n\n")
    if not separator or not first_line.startswith(_FORMAT_PREFIX) or not question.strip():
        raise ValueError("MCQA instruction template has no question")
    return question.strip()


def import_task(archive: TaskArchive) -> TaskSpec:
    """Import a cleaned Nemotron MCQA archive with its original verifier contract."""
    try:
        metadata = tomllib.loads(archive.files["task.toml"].decode())["metadata"]
        if metadata.get("family") != FAMILY or metadata.get("converter") != CONVERTER:
            raise ValueError("Unsupported TaskTrove MCQA source")
        contract = parse_spec(archive.files["tests/verifier.toml"].decode())
        if not isinstance(contract, McqSpec):
            raise ValueError("TaskTrove MCQA archive must declare an MCQ verifier")
        instructions = _clean_instructions(archive.files["instruction.md"].decode())
    except (KeyError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as error:
        raise ValueError(f"Invalid TaskTrove MCQA archive: {error}") from error
    return TaskSpec(
        id=f"tasktrove-{archive.row}",
        instructions=instructions,
        verifier=MultipleChoiceAnswer(contract.expected, contract.options),
        source=archive.source,
        requirements=TaskRequirements(),
    )
