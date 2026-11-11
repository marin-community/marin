# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify a source that ships no oracle solutions.

A source whose tasks carry no oracle solution (``has_solution`` false, or an
empty ``solution_binary``) cannot be qualified by executing its oracle. This
module runs the two checks that replace it when a cohort has been rebuilt onto
shared environments:

1. **Payload preservation** (offline). Every file the native recipe copies into
   the workspace must appear in the rebuilt task's setup payload with identical
   bytes and mode. This catches a rebuild that drops, rewrites or re-modes
   fixture data.
2. **Setup execution** (local Docker). Run the rebuilt setup script in the task's
   own base image and compare the files it writes against the same expectation
   set. Preservation proves the bytes travel; this proves the script lands them.
   Package installation is stubbed because the rebuilt setup no longer installs
   task dependencies, and the base image owns provisioning.
3. **Verifier liveness** (sandbox). The task's own verifier must execute and
   return a well-formed negative verdict on the untouched workspace. A crash, a
   timeout or unparsable output is an infrastructure error, not a pass.

The report records the pass rate per base reference and overall, and its verdict
becomes ``oracle_validation=verifier_executed`` on the declaration. Until the
report exists the declaration must say ``skipped``: absence of oracle solutions
is not evidence that execution works.

Run the offline phase with::

    uv run python -m experiments.post_training.task_curation.no_oracle_qc \
        --cohort /path/to/native-tasks.parquet \
        --rebuild-out /path/to/rebuild-output \
        --output /path/to/qc-report.json

Add ``--sandbox`` to include the verifier-liveness phase; it needs a Daytona
credential and creates one sandbox per sampled task. Snapshot deletion has no
code path in this repository (see ``.agents/ops/daytona.md``); hand the created
snapshot names to an operator for cleanup.
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
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

ENVIRONMENT_PREFIX = "environment/"
SETUP_ROOT = "/setup_files"
FIXTURE_ARCHIVE = "calibforge-fixtures.tar.gz"
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
    extra: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    detail: str = ""

    @property
    def passed(self) -> bool:
        """Whether every expected file is present as written by the native recipe.

        The comparison is on content. A mode difference is reported in ``changed``
        because the native recipe's own ``chmod`` steps are replayed verbatim by the
        rebuild, so a faithful rebuild legitimately differs from the source byte mode.
        """
        return self.outcome == "executed" and not self.missing


@dataclass(frozen=True)
class VerifierResult:
    """One verifier execution against the untouched workspace."""

    task_id: str
    outcome: str
    exit_code: int | None
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.outcome in PASSING_VERIFIER_OUTCOMES


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

    def verdict(self, *, minimum_rate: float = 1.0) -> str:
        """``verifier_executed`` only when every check ran and every sample passed."""
        rates = [self.preservation_rate, self.execution_rate, self.verifier_rate]
        if any(rate is None for rate in rates):
            return "skipped"
        if any(rate < minimum_rate for rate in rates):
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
                    "extra": list(result.extra),
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
                    "passed": result.passed,
                    "detail": result.detail,
                }
                for result in self.verifier
            ],
        }


def _read_tar(payload: bytes) -> tarfile.TarFile:
    raw = gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
    return tarfile.open(fileobj=io.BytesIO(raw))


def copied_paths(dockerfile: str) -> list[str]:
    """The context-relative sources a native recipe copies into the workspace."""
    if ADD_SOURCE.search(dockerfile):
        raise ValueError("ADD recipes have no preservation contract")
    paths: list[str] = []
    for match in COPY_SOURCE.finditer(dockerfile):
        *sources, _destination = match.group("sources").split()
        for source in sources:
            if source.startswith("--") or "$" in source or "*" in source or "?" in source:
                continue
            paths.append(source.rstrip("/"))
    return paths


def copied_destinations(dockerfile: str, workdir: str) -> tuple[str, ...]:
    """Absolute destinations the native recipe writes, resolved against its workdir."""
    roots: list[str] = []
    for match in COPY_SOURCE.finditer(dockerfile):
        destination = match.group("sources").split()[-1]
        if "$" in destination or "*" in destination or "?" in destination:
            continue
        absolute = destination if destination.startswith("/") else f"{workdir.rstrip('/')}/{destination}"
        roots.append(absolute.rstrip("/") or "/")
    return tuple(dict.fromkeys(roots))


def preservation_expectations(native_payload: bytes) -> tuple[FileExpectation, ...]:
    """Files and modes the native recipe copies, read from its own tar."""
    with _read_tar(native_payload) as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        dockerfile_member = members.get(f"{ENVIRONMENT_PREFIX}Dockerfile")
        if dockerfile_member is None:
            raise ValueError("native task has no environment/Dockerfile")
        dockerfile = archive.extractfile(dockerfile_member).read().decode(errors="replace")
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
            except ValueError:
                continue
            while tokens and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=\S*", tokens[0]):
                tokens = tokens[1:]
            if not tokens or tokens[0] != "rm":
                continue
            for token in tokens[1:]:
                if token.startswith("-") or any(character in token for character in "$*?"):
                    continue
                removed.append(token if token.startswith("/") else f"{workdir.rstrip('/')}/{token}")
    return tuple(dict.fromkeys(removed))


def copy_mappings(dockerfile: str, workdir: str) -> list[tuple[str, str, bool]]:
    """Each copied source with its absolute destination and whether that is a directory."""
    mappings: list[tuple[str, str, bool]] = []
    for match in COPY_SOURCE.finditer(dockerfile):
        tokens = [token for token in match.group("sources").split() if not token.startswith("--")]
        if len(tokens) < 2:
            continue
        *sources, destination = tokens
        is_directory = len(sources) > 1 or destination.endswith("/")
        absolute = destination if destination.startswith("/") else f"{workdir.rstrip('/')}/{destination}"
        absolute = absolute.rstrip("/") or "/"
        for source in sources:
            if any(character in source for character in "$*?"):
                continue
            mappings.append((source.rstrip("/"), absolute, is_directory))
    return mappings


def workspace_expectations(native_payload: bytes) -> tuple[FileExpectation, ...]:
    """The copied files the native recipe leaves in the workspace, read from its own tar."""
    expectations = preservation_expectations(native_payload)
    with _read_tar(native_payload) as archive:
        member = next((item for item in archive.getmembers() if item.name == f"{ENVIRONMENT_PREFIX}Dockerfile"), None)
        if member is None:
            return expectations
        dockerfile = archive.extractfile(member).read().decode(errors="replace")
    workdir_match = re.search(r"^\s*WORKDIR\s+(\S+)", dockerfile, re.IGNORECASE | re.MULTILINE)
    workdir = workdir_match.group(1) if workdir_match else "/app"
    removed = removed_workspace_paths(dockerfile, workdir)
    if not removed:
        return expectations
    mappings = copy_mappings(dockerfile, workdir)
    kept: list[FileExpectation] = []
    for item in expectations:
        relative = item.path[len(ENVIRONMENT_PREFIX) :]
        destination = relative
        for source, absolute, is_directory in mappings:
            if relative == source:
                destination = f"{absolute}/{relative.rsplit('/', 1)[-1]}" if is_directory else absolute
                break
            if relative.startswith(source + "/"):
                destination = f"{absolute}/{relative[len(source) + 1:]}"
                break
        if any(destination == path or destination.startswith(path.rstrip("/") + "/") for path in removed):
            continue
        kept.append(item)
    return tuple(kept)


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
# The rebuilt setup no longer installs task dependencies; the base image owns
# provisioning. This keeps the script's apt steps from reaching the network.
exit 0
"""


def _docker(*arguments: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *arguments], capture_output=True, text=True, **kwargs)


def _normalized_path(path: str) -> str:
    """A reported path with one relative prefix removed; absolute paths keep their root."""
    return path[2:] if path.startswith("./") else path


def _manifest(stdout: str) -> list[tuple[str, int, str]]:
    r"""Parse ``mode path`` and ``sha256  path`` lines from one container run.

    ``find`` reports paths relative to the root it walks and ``stat`` reports the
    absolute path it was given, so both forms appear. Normalizing a single leading
    ``./`` keeps them consistent without touching a leading ``/``.
    """
    entries: dict[str, int] = {}
    digests: dict[str, str] = {}
    for line in stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        first, second = parts
        path = _normalized_path(second.strip())
        if re.fullmatch(r"[0-7]{3,4}", first):
            entries[path] = int(first, 8)
        elif re.fullmatch(r"[0-9a-f]{64}", first):
            digests[path] = first
    return [(digest, entries[path], path) for path, digest in digests.items() if path in entries]


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
        scan = " ".join(
            f"$( [ -e {shlex.quote(item)} ] && printf '%s ' {shlex.quote(item)} )" for item in (workdir, *destinations)
        )
        command = (
            ("PATH=/shim:$PATH; " if stub_apt else "")
            + f"sh {SETUP_ROOT}/{setup_script} && "
            + f'for root in {scan}; do if [ -d "$root" ]; then (cd "$root" && find . -type f'
            + " -printf '%m %p\\n' && find . -type f -exec sha256sum {} +); elif [ -e \"$root\" ]; then"
            + ' (stat -c \'%a %n\' "$root" && sha256sum "$root"); fi; done'
        )
        resolvers = tuple(part for server in dns for part in ("--dns", server))
        created = _docker("create", *resolvers, image, "sh", "-c", command)
        if created.returncode:
            return SetupExecution(task_id=task_id, outcome="docker_error", detail=created.stderr[-500:])
        container = created.stdout.strip()
        try:
            copied = _docker("cp", str(setup), f"{container}:{SETUP_ROOT}")
            if copied.returncode:
                return SetupExecution(task_id=task_id, outcome="docker_error", detail=copied.stderr[-500:])
            _docker("cp", str(shim), f"{container}:/shim")
            started = _docker("start", "-a", container, timeout=timeout)
        except subprocess.TimeoutExpired:
            return SetupExecution(task_id=task_id, outcome="timeout", detail=f"{timeout}s")
        finally:
            _docker("rm", "-f", container)
    if started.returncode:
        return SetupExecution(task_id=task_id, outcome="script_error", detail=started.stderr[-500:])
    written, missing, extra, changed = _compare_workspace(workspace_expectations(native_payload), started.stdout)
    return SetupExecution(
        task_id=task_id, outcome="executed", written=written, missing=missing, extra=extra, changed=changed
    )


def _compare_workspace(
    expectations: tuple[FileExpectation, ...], stdout: str
) -> tuple[int, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Compare a container's workspace against the expected files.

    Content decides the verdict. Modes are reported separately because the native
    recipe's own ``chmod`` steps are replayed verbatim by the rebuild, so a faithful
    rebuild can legitimately end with a different mode than the archived source file.
    """
    manifest = _manifest(stdout)
    observed: dict[str, set[int]] = {}
    for digest, mode, _path in manifest:
        observed.setdefault(digest, set()).add(mode)
    expected = {item.sha256: item for item in expectations}
    missing = tuple(f"{digest[:12]}:{item.path}" for digest, item in sorted(expected.items()) if digest not in observed)
    extra = tuple(sorted({f"{digest[:12]}:{path}" for digest, _mode, path in manifest if digest not in expected}))
    changed = tuple(
        f"{digest[:12]}:{'/'.join(oct(mode) for mode in sorted(observed[digest]))}->{oct(item.mode)}"
        for digest, item in sorted(expected.items())
        if digest in observed and item.mode not in observed[digest]
    )
    return len(manifest), missing, extra, changed


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
    workdir = requirements.get("working_directory") or "/app"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for resource in task["resources"].get("all") or ():
            target = root / "setup_files" / Path(resource["path"]).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(base64.b64decode(resource["source"]["content_base64"], validate=True))
            mode = resource.get("mode")
            if mode is not None:
                target.chmod(int(mode, 8))
        roots = (workdir, *destinations)
        scan = " ".join(f"$( [ -e {shlex.quote(item)} ] && printf '%s ' {shlex.quote(item)} )" for item in roots)
        command = (
            " && ".join(commands)
            + f' && for root in {scan}; do if [ -d "$root" ]; then (cd "$root" && find . -type f'
            + " -printf '%m %p\\n' && find . -type f -exec sha256sum {} +); elif [ -e \"$root\" ]; then"
            + ' (stat -c \'%a %n\' "$root" && sha256sum "$root"); fi; done'
        )
        resolvers = tuple(part for server in dns for part in ("--dns", server))
        created = _docker("create", *resolvers, requirements["docker_image"], "sh", "-c", command)
        if created.returncode:
            return SetupExecution(task_id=task["id"], outcome="docker_error", detail=created.stderr[-500:])
        container = created.stdout.strip()
        try:
            copied = _docker("cp", str(root / "setup_files"), f"{container}:{SETUP_ROOT}")
            if copied.returncode:
                return SetupExecution(task_id=task["id"], outcome="docker_error", detail=copied.stderr[-500:])
            started = _docker("start", "-a", container, timeout=timeout)
        except subprocess.TimeoutExpired:
            return SetupExecution(task_id=task["id"], outcome="timeout", detail=f"{timeout}s")
        finally:
            _docker("rm", "-f", container)
    if started.returncode:
        return SetupExecution(task_id=task["id"], outcome="script_error", detail=started.stderr[-500:])
    written, missing, extra, changed = _compare_workspace(expectations, started.stdout)
    return SetupExecution(
        task_id=task["id"],
        outcome="executed",
        written=written,
        missing=missing,
        extra=extra,
        changed=changed,
    )


def classify_verifier_run(*, exit_code: int | None, timed_out: bool, stdout: str) -> str:
    """Classify one verifier execution against an untouched workspace.

    A verifier that ran returns a negative verdict: it executed and rejected the
    unsolved submission. A timeout, crash or unparsable report is an
    infrastructure error and never counts as a pass.
    """
    if timed_out:
        return "timeout"
    if exit_code is None:
        return "infra_error"
    text = stdout.strip()
    if not text:
        return "infra_error"
    reward_line = next((line for line in reversed(text.splitlines()) if line.strip()), "")
    try:
        reward = float(reward_line.split()[-1])
    except ValueError:
        return "malformed"
    if exit_code != 0:
        return "infra_error"
    return "negative_verdict" if reward < 1.0 else "positive_reward"


def summarize(report: QualificationReport, *, minimum_rate: float = 1.0) -> dict:
    """Admission summary with the pass rates and the recorded verdict."""
    return {
        "checked": len(report.preservation),
        "verified": len(report.verifier),
        "preservation_rate": report.preservation_rate,
        "verifier_rate": report.verifier_rate,
        "minimum_rate": minimum_rate,
        "executed": len(report.execution),
        "execution_rate": report.execution_rate,
        "setup_failures": dict(Counter(result.outcome for result in report.execution if not result.passed)),
        "verifier_failures": dict(Counter(result.outcome for result in report.verifier if not result.passed)),
        "by_base": report.by_base(),
        "oracle_validation": report.verdict(minimum_rate=minimum_rate),
    }


def _native_tasks(path: Path):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=64, columns=["path", "task_binary"]):
        columns = batch.to_pydict()
        for task_id, blob in zip(columns["path"], columns["task_binary"], strict=True):
            yield task_id, bytes(blob.as_py() if hasattr(blob, "as_py") else blob)


def _rebuilt_metadata(rebuilt_payload: bytes) -> tuple[str, str, str]:
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
    workdir = environment.get("workdir") or "/app"
    return environment["docker_image"], workdir, script


def _native_destinations(native_payload: bytes, workdir: str) -> tuple[str, ...]:
    """Absolute destinations the native recipe writes, resolved against its own workdir."""
    with _read_tar(native_payload) as archive:
        member = next((item for item in archive.getmembers() if item.name == "environment/Dockerfile"), None)
        if member is None:
            return (workdir,)
        dockerfile = archive.extractfile(member).read().decode(errors="replace")
    native_workdir = "/app"
    match = re.search(r"^\s*WORKDIR\s+(\S+)", dockerfile, re.IGNORECASE | re.MULTILINE)
    if match:
        native_workdir = match.group(1)
    return copied_destinations(dockerfile, native_workdir)


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
            image, workdir, script = _rebuilt_metadata(rebuilt[task_id])
        except (KeyError, ValueError) as error:
            return SetupExecution(task_id=task_id, outcome="invalid_task", detail=str(error))
        return execute_setup_in_docker(
            task_id,
            native[task_id],
            rebuilt[task_id],
            image=image,
            workdir=workdir,
            setup_script=script,
            destinations=_native_destinations(native[task_id], workdir),
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
        workdir = lowered[task_id]["task"]["environment_requirements"].get("working_directory") or "/app"
        return execute_lowered_setup(
            lowered[task_id],
            expectations,
            destinations=_native_destinations(native[task_id], workdir),
            dns=dns,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, sorted(lowered)))


def main() -> None:
    """Run the offline preservation phase and write the qualification report."""
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
    parser.add_argument("--sandbox", action="store_true", help="also run verifier liveness (needs Daytona)")
    parser.add_argument("--sample-per-base", type=int, default=20, help="verifier samples per base reference")
    args = parser.parse_args()

    rebuilt_path = args.rebuild_out / "pending-tasks.parquet"
    manifest = json.loads((args.rebuild_out / "manifest.json").read_text())
    bases = {row["task_id"]: row["base_ref"] for row in manifest["tasks"] if row.get("status", "").startswith("rebuilt")}

    rebuilt = {}
    for batch in pq.ParquetFile(rebuilt_path).iter_batches(batch_size=64, columns=["path", "task_binary"]):
        columns = batch.to_pydict()
        for task_id, blob in zip(columns["path"], columns["task_binary"], strict=True):
            rebuilt[task_id] = bytes(blob.as_py() if hasattr(blob, "as_py") else blob)

    report = QualificationReport(bases=bases)
    wanted = set(args.task)
    if args.retry_failed is not None:
        previous = json.loads(args.retry_failed.read_text())
        wanted |= {row["task_id"] for row in previous["execution"] if not row["passed"]}
        wanted |= {row["task_id"] for row in previous["preservation"] if not row["passed"]}
    native = {
        task_id: payload
        for task_id, payload in _native_tasks(args.cohort)
        if task_id in rebuilt and (not wanted or task_id in wanted)
    }
    for task_id, native_payload in native.items():
        report.preservation.append(check_preservation(task_id, native_payload, rebuilt[task_id]))

    if args.docker and args.lowered:
        lowered = {}
        lowered_path = args.rebuild_out / "lowered-tasks.parquet"
        for batch in pq.ParquetFile(lowered_path).iter_batches(batch_size=32, columns=["path", "lowered_task_spec"]):
            columns = batch.to_pydict()
            for task_id, spec in zip(columns["path"], columns["lowered_task_spec"], strict=True):
                if task_id in native:
                    lowered[task_id] = json.loads(spec)
        report.execution = _execute_lowered(lowered, native, args.workers, dns=tuple(args.docker_dns))
    elif args.docker:
        report.execution = _execute_setups(
            native, rebuilt, args.workers, stub_apt=args.stub_apt, dns=tuple(args.docker_dns)
        )

    if args.sandbox:
        raise SystemExit(
            "Verifier liveness needs a Daytona credential and creates one sandbox per sampled task; "
            "run it through the reviewed procedure in docs/references/task-curation.md"
        )

    summary = summarize(report)
    args.output.write_text(json.dumps({"summary": summary, **report.to_dict()}, indent=1) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "by_base"}))


if __name__ == "__main__":
    main()
