# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify a source that ships no oracle solutions.

A source whose tasks carry no oracle solution (``has_solution`` false, or an
empty ``solution_binary``) cannot be qualified by executing its oracle. This
module checks payload preservation and setup execution for cohorts rebuilt onto
shared environments. Empty-submission grading runs separately through
``no_oracle_liveness``:

1. **Payload preservation** (offline). Every file the native recipe copies into
   the workspace must appear in the rebuilt task's setup payload with identical
   bytes and mode. This catches a rebuild that drops, rewrites or re-modes
   fixture data.
2. **Setup execution** (local Docker). Run the rebuilt setup script in the task's
   own base image and compare the files it writes against the same expectation
   set. Preservation proves the bytes travel; this proves the script lands them.
   Package installation runs by default. ``--stub-apt`` skips installs for a
   payload-only diagnostic and does not establish dependency provisioning.
3. **Empty-submission grading** (separate command). ``no_oracle_liveness``
   samples the grader on a fresh machine. Qualification requires a successful
   verifier exit, a reward below 1.0, and a positive structured executed-test
   count.

The offline report records preservation and setup pass rates by base and
overall. Verifier results require an explicit ``--liveness`` receipt file. A reward-only
sample is insufficient to establish ``oracle_validation=verifier_executed``.

Run the offline phase with::

    uv run python -m experiments.post_training.task_curation.no_oracle_qualification \
        --cohort /path/to/native-tasks.parquet \
        --rebuild-out /path/to/rebuild-output \
        --output /path/to/qc-report.json

Attach the separate liveness JSONL with ``--liveness`` to combine receipts.
Use ``no_oracle_liveness`` through the procedure in
``docs/references/task-curation.md`` for the separate reward sample.
"""

import argparse
import base64
import gzip
import hashlib
import io
import json
import re
import shlex
import subprocess
import tarfile
import tempfile
import tomllib
from collections import Counter, defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import pyarrow.parquet as pq

from experiments.post_training.task_curation.calibforge_fixtures import FIXTURE_ARCHIVE, SETUP_ROOT
from experiments.post_training.task_curation.environment_fidelity import (
    DEFAULT_WORKDIR,
    ENVIRONMENT_DOCKERFILE,
    ENVIRONMENT_PREFIX,
)
from experiments.post_training.task_curation.no_oracle_liveness import load_specs, manifest_bases

COPY_SOURCE = re.compile(r"^\s*COPY\s+(?:--\S+\s+)*(?P<sources>.+)$", re.IGNORECASE | re.MULTILINE)
ADD_SOURCE = re.compile(r"^\s*ADD\s", re.IGNORECASE | re.MULTILINE)

PASSING_VERIFIER_OUTCOMES = ("negative_verdict",)


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


@dataclass(frozen=True)
class SetupExecution:
    """One rebuilt setup script executed in its base image."""

    task_id: str
    outcome: str
    written: int = 0
    missing: tuple[str, ...] = ()
    """Expected paths the workspace scan found nowhere."""
    altered: tuple[str, ...] = ()
    """Expected paths present with different bytes."""
    extra: tuple[str, ...] = ()
    remoded: tuple[str, ...] = ()
    workspace_stdout: str | None = None
    """The raw workspace scan output, when the script ran to completion."""
    detail: str = ""

    @property
    def passed(self) -> bool:
        """Whether every expected file is present with the native bytes.

        A mode difference is reported in ``remoded`` and does not fail the check,
        because the native recipe's own ``chmod`` steps are replayed verbatim by the
        rebuild, so a faithful rebuild legitimately differs from the source byte mode.
        """
        return self.outcome == "executed" and not (self.missing or self.altered)


@dataclass(frozen=True)
class VerifierResult:
    """One verifier execution against the untouched workspace."""

    task_id: str
    outcome: str
    exit_code: int | None
    detail: str = ""
    tests_executed: int = 0

    @property
    def passed(self) -> bool:
        return self.outcome in PASSING_VERIFIER_OUTCOMES and self.exit_code == 0 and self.tests_executed > 0


@dataclass
class QualificationReport:
    """Per-task checks plus the pass rates admission reads."""

    preservation: list[PreservationResult] = field(default_factory=list)
    execution: list[SetupExecution] = field(default_factory=list)
    verifier: list[VerifierResult] = field(default_factory=list)
    bases: dict[str, str] = field(default_factory=dict)

    def _rate(self, results) -> float | None:
        if not results:
            return None
        return sum(1 for result in results if result.passed) / len(results)

    @property
    def preservation_rate(self) -> float | None:
        return self._rate(self.preservation)

    @property
    def execution_rate(self) -> float | None:
        return self._rate(self.execution)

    @property
    def verifier_rate(self) -> float | None:
        return self._rate(self.verifier)

    def by_base(self) -> dict[str, dict[str, float | int | None]]:
        grouped: dict[str, list[PreservationResult]] = defaultdict(list)
        for result in self.preservation:
            grouped[self.bases.get(result.task_id, "<unknown>")].append(result)
        return {
            base: {"tasks": len(results), "preservation_rate": self._rate(results)}
            for base, results in sorted(grouped.items())
        }

    def verdict(self) -> str:
        """``verifier_executed`` only when every check ran and every sample passed."""
        rates = [self.preservation_rate, self.execution_rate, self.verifier_rate]
        if any(rate is None for rate in rates):
            return "skipped"
        if any(rate != 1.0 for rate in rates):
            return "skipped"
        return "verifier_executed"

    def to_dict(self) -> dict:
        return {
            "preservation_rate": self.preservation_rate,
            "execution_rate": self.execution_rate,
            "verifier_rate": self.verifier_rate,
            "by_base": self.by_base(),
            "verdict": self.verdict(),
            "preservation": [
                {
                    "task_id": result.task_id,
                    "passed": result.passed,
                    "missing": list(result.missing),
                    "changed": list(result.changed),
                    "remoded": list(result.remoded),
                }
                for result in self.preservation
            ],
            "execution": [
                {
                    "task_id": result.task_id,
                    "outcome": result.outcome,
                    "written": result.written,
                    "missing": list(result.missing),
                    "altered": list(result.altered),
                    "extra": list(result.extra),
                    "remoded": list(result.remoded),
                    "passed": result.passed,
                    "detail": result.detail,
                }
                for result in self.execution
            ],
            "verifier": [
                {
                    "task_id": result.task_id,
                    "outcome": result.outcome,
                    "exit_code": result.exit_code,
                    "tests_executed": result.tests_executed,
                    "passed": result.passed,
                    "detail": result.detail,
                }
                for result in self.verifier
            ],
        }


def _read_tar(payload: bytes) -> tarfile.TarFile:
    raw = gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
    return tarfile.open(fileobj=io.BytesIO(raw))


def _native_dockerfile(native_payload: bytes) -> str | None:
    """The native recipe's Dockerfile text, or None when the tar carries none."""
    with _read_tar(native_payload) as archive:
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
        if not any(character in source for character in "$*?")
    ]


def copied_destinations(dockerfile: str, workdir: str) -> tuple[str, ...]:
    """Absolute destinations the native recipe writes, resolved against its workdir."""
    return tuple(dict.fromkeys(mapping.destination for mapping in copy_mappings(dockerfile, workdir)))


def preservation_expectations(native_payload: bytes) -> tuple[FileExpectation, ...]:
    """Files and modes the native recipe copies, read from its own tar."""
    dockerfile = _native_dockerfile(native_payload)
    if dockerfile is None:
        raise ValueError("native task has no environment/Dockerfile")
    with _read_tar(native_payload) as archive:
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
                if token.startswith("-") or any(character in token for character in "$*?"):
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
        if any(character in destination for character in "$*?"):
            continue
        is_directory = len(sources) > 1 or destination.endswith("/")
        absolute = destination if destination.startswith("/") else f"{workdir.rstrip('/')}/{destination}"
        absolute = absolute.rstrip("/") or "/"
        for source in sources:
            if any(character in source for character in "$*?"):
                continue
            mappings.append(CopyMapping(source.rstrip("/"), absolute, is_directory))
    return mappings


def workspace_expectations(native_payload: bytes) -> tuple[FileExpectation, ...]:
    """The copied files the native recipe leaves in the workspace, read from its own tar."""
    expectations = preservation_expectations(native_payload)
    dockerfile = _native_dockerfile(native_payload)
    if dockerfile is None:
        return expectations
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
    with _read_tar(rebuilt_payload) as archive:
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


DOCKER_SETUP_TIMEOUT = 300
"""Seconds one rebuilt setup script may run in its base image."""

APT_SHIM = """#!/bin/sh
# Skip package installation during payload-only diagnostics.
exit 0
"""


def _docker(*arguments: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *arguments], capture_output=True, text=True, **kwargs)


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


def _execute_container_setup(
    task_id: str,
    image: str,
    command: str,
    copies: tuple[tuple[Path, str], ...],
    dns: tuple[str, ...],
    timeout: int,
) -> SetupExecution:
    resolvers = tuple(part for server in dns for part in ("--dns", server))
    created = _docker("create", *resolvers, image, "sh", "-c", command)
    if created.returncode:
        return SetupExecution(task_id=task_id, outcome="docker_error", detail=created.stderr[-500:])
    container = created.stdout.strip()
    try:
        for source, destination in copies:
            copied = _docker("cp", str(source), f"{container}:{destination}")
            if copied.returncode:
                return SetupExecution(task_id=task_id, outcome="docker_error", detail=copied.stderr[-500:])
        started = _docker("start", "-a", container, timeout=timeout)
    except subprocess.TimeoutExpired:
        return SetupExecution(task_id=task_id, outcome="timeout", detail=f"{timeout}s")
    finally:
        removed = _docker("rm", "-f", container)
        if removed.returncode:
            raise RuntimeError(f"cannot remove Docker setup container {container}: {removed.stderr[-500:]}")
    if started.returncode:
        return SetupExecution(task_id=task_id, outcome="script_error", detail=started.stderr[-500:])
    return SetupExecution(task_id=task_id, outcome="executed", workspace_stdout=started.stdout)


def _workspace_scan(roots: tuple[str, ...]) -> str:
    paths = " ".join(shlex.quote(path) for path in dict.fromkeys(roots))
    return (
        f'for root in {paths}; do if [ -d "$root" ]; then '
        '(find "$root" -type f -printf \'%m %p\\n\' && find "$root" -type f -exec sha256sum {} +); '
        'elif [ -e "$root" ]; then (stat -c \'%a %n\' "$root" && sha256sum "$root"); fi; done'
    )


def _setup_comparison(task_id: str, expectations: tuple[FileExpectation, ...], stdout: str) -> SetupExecution:
    comparison = _compare_workspace(expectations, stdout)
    return SetupExecution(
        task_id=task_id,
        outcome="executed",
        written=comparison.written,
        missing=comparison.missing,
        altered=comparison.altered,
        extra=comparison.extra,
        remoded=comparison.remoded,
        workspace_stdout=stdout,
    )


def execute_setup_in_docker(
    task_id: str,
    native_payload: bytes,
    rebuilt_payload: bytes,
    *,
    image: str,
    workdir: str,
    setup_script: str,
    destinations: tuple[str, ...] = (),
    stub_apt: bool = False,
    dns: tuple[str, ...] = (),
    timeout: int = DOCKER_SETUP_TIMEOUT,
) -> SetupExecution:
    """Run one rebuilt setup script in its base image and compare the written files."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with _read_tar(rebuilt_payload) as archive:
            archive.extractall(root, filter="data")
        setup = root / SETUP_ROOT.lstrip("/")
        script = setup / setup_script
        if not script.exists():
            return SetupExecution(task_id=task_id, outcome="missing_setup_script")
        shim = root / "shim"
        shim.mkdir()
        for tool in ("apt-get", "apt"):
            (shim / tool).write_text(APT_SHIM)
            (shim / tool).chmod(0o755)
        command = (
            ("PATH=/shim:$PATH; " if stub_apt else "")
            + f"sh {SETUP_ROOT}/{setup_script} && "
            + _workspace_scan((workdir, *destinations))
        )
        result = _execute_container_setup(task_id, image, command, ((setup, SETUP_ROOT), (shim, "/shim")), dns, timeout)
    if result.workspace_stdout is None:
        return result
    return _setup_comparison(task_id, workspace_expectations(native_payload), result.workspace_stdout)


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
    """Present paths whose mode differs from the native recipe's own chmod."""


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


def execute_lowered_setup(
    spec: Mapping,
    expectations: tuple[FileExpectation, ...],
    *,
    destinations: tuple[str, ...] = (),
    timeout: int = DOCKER_SETUP_TIMEOUT,
    dns: tuple[str, ...] = (),
) -> SetupExecution:
    """Run a lowered spec's setup commands in its image and compare the written files.

    The spec is the artifact a rollout consumes: resources carry the fixtures and
    ``environment_requirements.setup_commands`` stage them. This executes exactly
    those commands, so a spec that declares a hook nothing runs cannot pass.
    """
    task = spec["task"]
    requirements = task["environment_requirements"]
    commands = tuple(requirements.get("setup_commands") or ())
    if not commands:
        return SetupExecution(task_id=task["id"], outcome="no_setup_commands")
    workdir = requirements.get("working_directory") or DEFAULT_WORKDIR
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for resource in task["resources"].get("all") or ():
            target = root / "setup_files" / Path(resource["path"]).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(base64.b64decode(resource["source"]["content_base64"], validate=True))
            mode = resource.get("mode")
            if mode is not None:
                target.chmod(int(mode, 8))
        command = " && ".join(commands) + " && " + _workspace_scan((workdir, *destinations))
        result = _execute_container_setup(
            task["id"], requirements["docker_image"], command, ((root / "setup_files", SETUP_ROOT),), dns, timeout
        )
    if result.workspace_stdout is None:
        return result
    return _setup_comparison(task["id"], expectations, result.workspace_stdout)


def summarize(report: QualificationReport) -> dict:
    """Admission summary with the pass rates and the recorded verdict."""
    return {
        "checked": len(report.preservation),
        "verified": len(report.verifier),
        "preservation_rate": report.preservation_rate,
        "verifier_rate": report.verifier_rate,
        "minimum_rate": 1.0,
        "executed": len(report.execution),
        "execution_rate": report.execution_rate,
        "setup_failures": dict(Counter(result.outcome for result in report.execution if not result.passed)),
        "verifier_failures": dict(Counter(result.outcome for result in report.verifier if not result.passed)),
        "by_base": report.by_base(),
        "oracle_validation": report.verdict(),
    }


def _native_tasks(path: Path):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=64, columns=["path", "task_binary"]):
        columns = batch.to_pydict()
        for task_id, blob in zip(columns["path"], columns["task_binary"], strict=True):
            yield task_id, bytes(blob)


@dataclass(frozen=True)
class RebuiltMetadata:
    image: str
    workdir: str
    setup_script: str


def _rebuilt_metadata(rebuilt_payload: bytes) -> RebuiltMetadata:
    """The image, workdir and setup script a rebuilt task declares."""
    with _read_tar(rebuilt_payload) as archive:
        member = next((item for item in archive.getmembers() if item.name == "task.toml"), None)
        if member is None:
            raise ValueError("rebuilt task has no task.toml")
        document = tomllib.loads(archive.extractfile(member).read().decode(errors="replace"))
    environment = document.get("environment") or {}
    setup = environment.get("setup") or {}
    command = str(setup.get("command") or "")
    script = command.split()[-1].rsplit("/", 1)[-1]
    workdir = environment.get("workdir") or DEFAULT_WORKDIR
    return RebuiltMetadata(environment["docker_image"], workdir, script)


def _native_destinations(native_payload: bytes, workdir: str) -> tuple[str, ...]:
    """Absolute destinations the native recipe writes, resolved against its own workdir."""
    dockerfile = _native_dockerfile(native_payload)
    if dockerfile is None:
        return (workdir,)
    return copied_destinations(dockerfile, native_workdir(dockerfile))


def _execute_setups(
    native: dict[str, bytes],
    rebuilt: dict[str, bytes],
    workers: int,
    *,
    stub_apt: bool = False,
    dns: tuple[str, ...] = (),
) -> list[SetupExecution]:
    def run(task_id: str) -> SetupExecution:
        try:
            metadata = _rebuilt_metadata(rebuilt[task_id])
        except (KeyError, ValueError) as error:
            return SetupExecution(task_id=task_id, outcome="invalid_task", detail=str(error))
        return execute_setup_in_docker(
            task_id,
            native[task_id],
            rebuilt[task_id],
            image=metadata.image,
            workdir=metadata.workdir,
            setup_script=metadata.setup_script,
            destinations=_native_destinations(native[task_id], metadata.workdir),
            stub_apt=stub_apt,
            dns=dns,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, sorted(native)))


def _execute_lowered(
    lowered: dict[str, Mapping], native: dict[str, bytes], workers: int, *, dns: tuple[str, ...] = ()
) -> list[SetupExecution]:
    def run(task_id: str) -> SetupExecution:
        expectations = workspace_expectations(native[task_id])
        workdir = lowered[task_id]["task"]["environment_requirements"].get("working_directory") or DEFAULT_WORKDIR
        return execute_lowered_setup(
            lowered[task_id],
            expectations,
            destinations=_native_destinations(native[task_id], workdir),
            dns=dns,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, sorted(lowered)))


def _wanted_task_ids(args: argparse.Namespace, rebuilt: dict[str, bytes]) -> set[str]:
    """The task ids this run should cover: explicit ids plus a previous report's failures."""
    wanted = set(args.task)
    if args.retry_failed is not None:
        previous = json.loads(args.retry_failed.read_text())
        wanted |= {row["task_id"] for row in previous["execution"] if not row["passed"]}
        wanted |= {row["task_id"] for row in previous["preservation"] if not row["passed"]}
    return wanted & set(rebuilt)


def _merge_liveness_receipts(report: QualificationReport, path: Path, native: dict[str, bytes]) -> None:
    """Append each liveness receipt as a verifier result; every task must be known once."""
    seen: set[str] = set()
    for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip()):
        if row["task_id"] not in native or row["task_id"] in seen:
            raise ValueError(f"unknown or duplicate liveness task: {row['task_id']}")
        seen.add(row["task_id"])
        report.verifier.append(
            VerifierResult(
                row["task_id"], row["outcome"], row["exit_code"], row.get("detail", ""), row.get("tests_executed", 0)
            )
        )


def main() -> None:
    """Check preservation, execute setup in Docker or the lowered runtime, and merge liveness receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True, help="native cohort parquet (path, task_binary)")
    parser.add_argument("--rebuild-out", type=Path, required=True, help="rebuild output directory")
    parser.add_argument("--output", type=Path, required=True, help="qualification report JSON")
    parser.add_argument("--docker", action="store_true", help="also execute every rebuilt setup script in Docker")
    parser.add_argument("--stub-apt", action="store_true", help="skip package installation in the setup script")
    parser.add_argument("--workers", type=int, default=4, help="concurrent setup executions")
    parser.add_argument("--docker-dns", action="append", default=[], help="resolver for container DNS")
    parser.add_argument("--retry-failed", type=Path, help="re-run the tasks a previous report failed")
    parser.add_argument("--lowered", action="store_true", help="execute the lowered specs instead of the tar setups")
    parser.add_argument("--task", action="append", default=[], help="limit to these task ids")
    parser.add_argument("--liveness", type=Path, help="JSONL receipts from no_oracle_liveness")
    args = parser.parse_args()

    bases = manifest_bases(args.rebuild_out / "manifest.json")
    rebuilt = dict(_native_tasks(args.rebuild_out / "pending-tasks.parquet"))

    report = QualificationReport(bases=bases)
    wanted = _wanted_task_ids(args, rebuilt)
    native = {
        task_id: payload
        for task_id, payload in _native_tasks(args.cohort)
        if task_id in rebuilt and (not wanted or task_id in wanted)
    }
    for task_id, native_payload in native.items():
        report.preservation.append(check_preservation(task_id, native_payload, rebuilt[task_id]))

    if args.docker and args.lowered:
        lowered_path = args.rebuild_out / "lowered-tasks.parquet"
        lowered = {task_id: spec for task_id, spec in load_specs(lowered_path).items() if task_id in native}
        report.execution = _execute_lowered(lowered, native, args.workers, dns=tuple(args.docker_dns))
    elif args.docker:
        report.execution = _execute_setups(
            native, rebuilt, args.workers, stub_apt=args.stub_apt, dns=tuple(args.docker_dns)
        )

    if args.liveness is not None:
        _merge_liveness_receipts(report, args.liveness, native)

    summary = summarize(report)
    args.output.write_text(json.dumps({"summary": summary, **report.to_dict()}, indent=1) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "by_base"}))


if __name__ == "__main__":
    main()
