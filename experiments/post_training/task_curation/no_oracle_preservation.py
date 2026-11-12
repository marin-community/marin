# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scan a native task's Dockerfile recipe and check a rebuild preserved its copies.

The native recipe's COPY steps define the workspace contract: which files the
rebuilt setup payload must stage, where, and with which bytes and modes. The
scanners read the recipe as written rather than through ``parse_instructions``:
the preservation check must describe the literal archive even when the rebuild
would defer on it, so a malformed recipe still yields a report instead of a crash.
"""

import hashlib
import io
import re
import shlex
import tarfile
from dataclasses import dataclass, replace

from experiments.post_training.task_curation.calibforge_fixtures import FIXTURE_ARCHIVE, SETUP_ROOT
from experiments.post_training.task_curation.environment_fidelity import (
    COPY_GLOB_CHARACTERS,
    DEFAULT_WORKDIR,
    ENVIRONMENT_DOCKERFILE,
    ENVIRONMENT_PREFIX,
)
from experiments.post_training.task_curation.task_tars import open_task_tar

COPY_SOURCE = re.compile(r"^\s*COPY\s+(?:--\S+\s+)*(?P<sources>.+)$", re.IGNORECASE | re.MULTILINE)
ADD_SOURCE = re.compile(r"^\s*ADD\s", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class FileExpectation:
    """One file the native recipe copies, with the bytes it must keep."""

    path: str
    sha256: str
    mode: int


@dataclass(frozen=True)
class PreservationResult:
    """Whether the rebuilt payload preserved every copied file."""

    task_id: str
    expected: tuple[FileExpectation, ...]
    missing: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    remoded: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not (self.missing or self.changed or self.remoded)


def _native_dockerfile(native_payload: bytes) -> str | None:
    """The native recipe's Dockerfile text, or None when the tar carries none."""
    with open_task_tar(native_payload) as archive:
        member = next((item for item in archive.getmembers() if item.name == ENVIRONMENT_DOCKERFILE), None)
        if member is None:
            return None
        return archive.extractfile(member).read().decode(errors="replace")


def native_workdir(dockerfile: str) -> str:
    """The workdir a native recipe declares, or the default when it sets none."""
    match = re.search(r"^\s*WORKDIR\s+(\S+)", dockerfile, re.IGNORECASE | re.MULTILINE)
    return match.group(1) if match else DEFAULT_WORKDIR


def _copy_steps(dockerfile: str) -> list[tuple[list[str], str]]:
    """Each COPY's flag-free sources and destination, in recipe order."""
    steps: list[tuple[list[str], str]] = []
    for match in COPY_SOURCE.finditer(dockerfile):
        tokens = [token for token in match.group("sources").split() if not token.startswith("--")]
        if len(tokens) < 2:
            continue
        steps.append((tokens[:-1], tokens[-1]))
    return steps


def copied_paths(dockerfile: str) -> list[str]:
    """The context-relative sources a native recipe copies into the workspace.

    These scans read the recipe as written rather than through ``parse_instructions``:
    the preservation check must describe the literal archive even when the rebuild
    would defer on it, so a malformed recipe still yields a report instead of a crash.
    """
    if ADD_SOURCE.search(dockerfile):
        raise ValueError("ADD recipes have no preservation contract")
    return [
        source.rstrip("/")
        for sources, _destination in _copy_steps(dockerfile)
        for source in sources
        if not any(character in source for character in COPY_GLOB_CHARACTERS)
    ]


def copied_destinations(dockerfile: str, workdir: str) -> tuple[str, ...]:
    """Absolute destinations the native recipe writes, resolved against its workdir."""
    return tuple(dict.fromkeys(mapping.destination for mapping in copy_mappings(dockerfile, workdir)))


def preservation_expectations(native_payload: bytes) -> tuple[FileExpectation, ...]:
    """Files and modes the native recipe copies, read from its own tar."""
    dockerfile = _native_dockerfile(native_payload)
    if dockerfile is None:
        raise ValueError("native task has no environment/Dockerfile")
    with open_task_tar(native_payload) as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        expectations: list[FileExpectation] = []
        for source in copied_paths(dockerfile):
            source_path = f"{ENVIRONMENT_PREFIX}{source}"
            prefix = source_path + "/"
            matches = [name for name in members if name == source_path or name.startswith(prefix)]
            for name in sorted(matches):
                content = archive.extractfile(members[name]).read()
                expectations.append(
                    FileExpectation(
                        path=name,
                        sha256=hashlib.sha256(content).hexdigest(),
                        mode=members[name].mode & 0o7777,
                    )
                )
    return tuple(expectations)


def removed_workspace_paths(dockerfile: str, workdir: str) -> tuple[str, ...]:
    """Absolute paths the native recipe deletes from the workspace.

    A recipe that generates data with a copied script and then removes the script
    leaves no such file behind, so a faithful rebuild must not be required to keep it.
    """
    removed: list[str] = []
    for line in dockerfile.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("RUN "):
            continue
        for segment in re.split(r"&&|\|\||;", stripped[4:]):
            try:
                tokens = shlex.split(segment, comments=False)
            except ValueError as error:
                raise ValueError(f"cannot parse native RUN segment: {segment!r}") from error
            while tokens and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=\S*", tokens[0]):
                tokens = tokens[1:]
            if not tokens or tokens[0] != "rm":
                continue
            for token in tokens[1:]:
                if token.startswith("-") or any(character in token for character in COPY_GLOB_CHARACTERS):
                    continue
                removed.append(token if token.startswith("/") else f"{workdir.rstrip('/')}/{token}")
    return tuple(dict.fromkeys(removed))


@dataclass(frozen=True)
class CopyMapping:
    source: str
    destination: str
    directory: bool


def copy_mappings(dockerfile: str, workdir: str) -> list[CopyMapping]:
    """Each copied source with its absolute destination and whether that is a directory."""
    mappings: list[CopyMapping] = []
    for sources, destination in _copy_steps(dockerfile):
        if any(character in destination for character in COPY_GLOB_CHARACTERS):
            continue
        is_directory = len(sources) > 1 or destination.endswith("/")
        absolute = destination if destination.startswith("/") else f"{workdir.rstrip('/')}/{destination}"
        absolute = absolute.rstrip("/") or "/"
        for source in sources:
            if any(character in source for character in COPY_GLOB_CHARACTERS):
                continue
            mappings.append(CopyMapping(source.rstrip("/"), absolute, is_directory))
    return mappings


def workspace_expectations(native_payload: bytes) -> tuple[FileExpectation, ...]:
    """The copied files the native recipe leaves in the workspace, read from its own tar."""
    expectations = preservation_expectations(native_payload)
    dockerfile = _native_dockerfile(native_payload)
    assert dockerfile is not None  # preservation_expectations raises when the archive has no Dockerfile
    workdir = native_workdir(dockerfile)
    removed = removed_workspace_paths(dockerfile, workdir)
    mappings = copy_mappings(dockerfile, workdir)
    kept: dict[str, FileExpectation] = {}
    for item in expectations:
        relative = item.path[len(ENVIRONMENT_PREFIX) :]
        for mapping in mappings:
            if relative == mapping.source:
                destination = (
                    f"{mapping.destination}/{relative.rsplit('/', 1)[-1]}" if mapping.directory else mapping.destination
                )
            elif relative.startswith(mapping.source + "/"):
                destination = f"{mapping.destination}/{relative[len(mapping.source) + 1:]}"
            else:
                continue
            if any(destination == path or destination.startswith(path.rstrip("/") + "/") for path in removed):
                continue
            kept[destination] = replace(item, path=destination)
    return tuple(kept.values())


def check_preservation(task_id: str, native_payload: bytes, rebuilt_payload: bytes) -> PreservationResult:
    """Compare the native recipe's copied files with the rebuilt setup payload."""
    expected = preservation_expectations(native_payload)
    payload_member = f"{SETUP_ROOT.lstrip(chr(47))}/{FIXTURE_ARCHIVE}"
    with open_task_tar(rebuilt_payload) as archive:
        member = next((item for item in archive.getmembers() if item.name == payload_member), None)
        if member is None:
            return PreservationResult(task_id=task_id, expected=expected, missing=tuple(item.path for item in expected))
        payload = archive.extractfile(member).read()
    staged = {}
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        for item in archive.getmembers():
            if item.isfile():
                staged[f"{ENVIRONMENT_PREFIX}{item.name}"] = (archive.extractfile(item).read(), item.mode & 0o7777)
    missing, changed, remoded = [], [], []
    for item in expected:
        found = staged.get(item.path)
        if found is None:
            missing.append(item.path)
            continue
        if hashlib.sha256(found[0]).hexdigest() != item.sha256:
            changed.append(item.path)
        if found[1] != item.mode:
            remoded.append(item.path)
    return PreservationResult(
        task_id=task_id, expected=expected, missing=tuple(missing), changed=tuple(changed), remoded=tuple(remoded)
    )


def _normalized_path(path: str) -> str:
    """A reported path with one relative prefix removed; absolute paths keep their root."""
    return path[2:] if path.startswith("./") else path


def _manifest(stdout: str) -> list[FileExpectation]:
    """Parse paired absolute-path mode and SHA-256 records from a container scan."""
    entries: dict[str, int] = {}
    digests: dict[str, str] = {}
    for line in stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        first, second = parts
        path = _normalized_path(second.strip())
        # stat prints a mode without leading zeros ("0" for a file a task
        # deliberately made unreadable), so accept one to four octal digits.
        if re.fullmatch(r"[0-7]{1,4}", first):
            entries[path] = int(first, 8)
        elif re.fullmatch(r"[0-9a-f]{64}", first):
            digests[path] = first
    return [FileExpectation(path, digest, entries[path]) for path, digest in digests.items() if path in entries]


@dataclass(frozen=True)
class WorkspaceComparison:
    """A workspace scan against the files the native recipe should have written."""

    written: int
    missing: tuple[str, ...]
    """Expected paths the scan found nowhere."""
    altered: tuple[str, ...]
    """Expected paths present with different bytes."""
    extra: tuple[str, ...]
    remoded: tuple[str, ...]
    """Paths present with a mode other than the native archive member's, as "digest:path:from->to"."""


def _compare_workspace(expectations: tuple[FileExpectation, ...], stdout: str) -> WorkspaceComparison:
    """Require the native bytes at each native COPY destination, including duplicate contents."""
    observed = {item.path: item for item in _manifest(stdout)}
    expected = {item.path: item for item in expectations}
    missing = tuple(path for path, item in sorted(expected.items()) if path not in observed)
    altered = tuple(
        path for path, item in sorted(expected.items()) if path in observed and observed[path].sha256 != item.sha256
    )
    extra = tuple(sorted(observed.keys() - expected.keys()))
    remoded = tuple(
        f"{item.sha256[:12]}:{path}:{oct(observed[path].mode)}->{oct(item.mode)}"
        for path, item in sorted(expected.items())
        if path in observed and observed[path].mode != item.mode
    )
    return WorkspaceComparison(len(observed), missing, altered, extra, remoded)
