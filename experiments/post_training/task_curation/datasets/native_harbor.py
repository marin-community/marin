# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve native Harbor packages for QUICK inspection without lowering their runtime."""

import base64
import hashlib
import io
import json
import tarfile
import tomllib
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from functools import partial
from itertools import batched
from pathlib import PurePosixPath

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    DockerBuildContext,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    ResourceGroups,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, RawRow
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import unpack_task_binary

VERSION = "2026.11.12.1"
HUB_READ_CONCURRENCY = 16


def unpack_native(row: dict, context: ConversionContext) -> dict:
    """Join separately shipped solution files without exposing them to the actor."""
    data = unpack_task_binary(row, context)
    solution = row.get("solution_binary")
    if not solution:
        return data
    with tarfile.open(fileobj=io.BytesIO(solution), mode="r:*") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if member.isdir():
                continue
            if not name.startswith("solution/"):
                raise ValueError(f"Separate solution archive contains a non-solution member: {name}")
            if member.issym() or member.islnk():
                data["archive_links"][name] = {"target": member.linkname}
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported solution archive member: {name}")
            content = archive.extractfile(member).read()
            encoded = base64.b64encode(content).decode()
            if name in data["files"] and data["files"][name] != encoded:
                raise ValueError(f"Conflicting native solution member: {name}")
            data["files"][name] = encoded
            data["file_metadata"][name] = {
                "mode": format(member.mode & 0o7777, "04o"),
                "mtime_ns": int(Decimal(member.pax_headers.get("mtime", str(member.mtime))) * 1_000_000_000),
            }
    data["solution_archive_sha256"] = hashlib.sha256(solution).hexdigest()
    return data


def read_hub_archive(root: StoragePath, context: ConversionContext, reference: str) -> dict:
    identity = hashlib.sha256(reference.encode()).hexdigest()
    archive = root / "tasks" / f"{identity}.tar.gz"
    return unpack_task_binary({"path": reference, "task_binary": archive.read_bytes()}, context)


def hub_rows(path: StoragePath, context: ConversionContext) -> Iterator[dict]:
    """Read precisely the immutable task references selected by Harbor acquisition."""
    manifest = json.loads(path.read_text())
    if set(manifest["selected"]) - set(manifest["task_refs"]):
        raise ValueError("Harbor release task count differs from its selected references")
    read = partial(read_hub_archive, path.parent, context)
    with ThreadPoolExecutor(max_workers=HUB_READ_CONCURRENCY) as executor:
        for references in batched(manifest["selected"], HUB_READ_CONCURRENCY):
            yield from executor.map(read, references)


def native_task(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    """Keep the native instruction, environment and verifier bytes behind Harbor's trial boundary."""
    data = row.data
    if data["archive_links"]:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="native_archive_links",
            detail="TaskCompendium inline regular files cannot preserve the archived links",
        )
    files = {name: base64.b64decode(value, validate=True) for name, value in data["files"].items()}
    missing = {"task.toml", "instruction.md", "tests/test.sh"} - files.keys()
    if missing:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="incomplete_native_package",
            detail=f"Missing native package files: {sorted(missing)}",
        )
    config = tomllib.loads(files["task.toml"].decode())
    if not config.get("environment", {}).get("docker_image") and "environment/Dockerfile" not in files:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="missing_native_environment",
            detail="The native package declares neither an image nor an environment Dockerfile",
        )
    resources = {}
    for name, content in sorted(files.items()):
        # Native extraction uses package-relative names, including tests/ and setup_files/.
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
            raise ValueError(f"Unsafe native package path: {name}")
        metadata = data["file_metadata"][name]
        resources[name] = inline_resource(name, content).model_copy(
            update={"mode": metadata["mode"], "mtime_ns": metadata["mtime_ns"]}
        )
    environment = config.get("environment", {})
    image = environment.get("docker_image")
    build = (
        None
        if image
        else DockerBuildContext(
            files=tuple(
                resource.model_copy(update={"path": name.removeprefix("environment/")})
                for name, resource in resources.items()
                if name.startswith("environment/")
            )
        )
    )
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=files["instruction.md"].decode()),)),
        environment_requirements=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            docker_image=image,
            docker_build=build,
            working_directory=environment.get("workdir"),
            environment_variables=environment.get("env", {}),
            setup_commands=(environment["setup"]["command"],) if environment.get("setup") else (),
        ),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=NoGrader(reason="The native Harbor Trial executes this package's declared verifier"),
        resources=ResourceGroups(
            verifier=tuple(resource for name, resource in resources.items() if not name.startswith("solution/")),
            oracle=tuple(resource for name, resource in resources.items() if name.startswith("solution/")),
        ),
        tags=("harbor",),
        labels={"native_archive_sha256": data["archive_sha256"], "native_import_version": VERSION},
    )
