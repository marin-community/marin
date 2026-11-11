# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild CalibForge V1 as a snapshot-safe cohort.

The native cohort bakes per-task data and dependencies into 2,181 distinct
build contexts. The rebuild lowers every decontaminated task onto a small
fixed set of base images, moving the task's files and dependency installs
into a setup script that runs at sandbox start — so a whole cohort needs a
handful of snapshots instead of one per task. Anything the fixed bases
cannot serve is deferred with an explicit reason, never silently altered.
"""

import argparse
import gzip
import hashlib
import io
import json
import re
import shlex
import tarfile
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq
import tomlkit
from tomlkit.items import Table

FIXTURE_ARCHIVE = "calibforge-fixtures.tar.gz"
SETUP_SCRIPT = "calibforge-initialize.sh"
SETUP_ROOT = "/setup_files"
ENVIRONMENT_PREFIX = "environment/"
REBUILT = "rebuilt_pending_oracle_and_admission"

NORMALIZED_CPUS = 2
NORMALIZED_MEMORY_MB = 4096
NORMALIZED_STORAGE_MB = 10240
"""One machine shape for the rebuilt cohort.

Snapshot identity includes the resource shape, so leaving each task's declared
shape in place multiplies the cohort's snapshots by every distinct size. Tasks
that need more than this shape are deferred rather than silently shrunk."""

# Fixed base set: one image per runtime family in the V1 FROM census. Digests
# are resolved and recorded at packaging time; the rebuild stages references
# and refuses unmapped families.
FIXED_BASES = {
    "ubuntu:24.04": "ubuntu:24.04",
    "aweaiteam/calibforge:tb2-verifier-base-v1": "aweaiteam/calibforge:tb2-verifier-base-v1",
    "aweaiteam/calibforge:fasttext-base-v1": "aweaiteam/calibforge:fasttext-base-v1",
    "python:3.13-slim-bookworm": "python:3.13-slim-bookworm",
    "python:3.12-slim-bookworm": "python:3.12-slim-bookworm",
    "python:3.11-slim": "python:3.11-slim-bookworm",
    "python:3.11-slim-bookworm": "python:3.11-slim-bookworm",
    "python:3.10-slim-bookworm": "python:3.10-slim-bookworm",
    "debian:12-slim": "debian:12-slim",
    "debian:13.0-slim": "debian:13.0-slim",
}

_META_COMMANDS = {"from", "cmd", "entrypoint", "label", "expose", "stopsignal", "healthcheck", "shell"}
_INNOCUOUS_RUN = {"mkdir", "chmod", "touch", "ln", "rm"}
_METACHARACTERS = "$`;&|<>()*?[~"


class DeferredRebuild(ValueError):
    """A native recipe needs a contract the fixed bases do not serve."""


@dataclass(frozen=True)
class Instruction:
    """One parsed Dockerfile instruction."""

    command: str
    args: list[str]
    flags: list[str]
    json_form: bool
    raw: str = ""
    """The unsplit remainder after the command word, as written in the Dockerfile."""


@dataclass(frozen=True)
class RebuiltTask:
    """A lowered task plus the evidence admission needs."""

    task_binary: bytes
    base_ref: str
    workdir: str
    payload_sha256: str
    setup_sha256: str


def parse_instructions(dockerfile: str) -> list[Instruction]:
    """Parse the instruction subset the fixed bases serve; defer the rest.

    Handles line continuations, comments, JSON-form RUN arrays and heredoc
    detection. Anything outside FROM/WORKDIR/COPY/RUN defers.
    """
    instructions: list[Instruction] = []
    buffer = ""
    for raw_line in dockerfile.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        candidate = f"{buffer} {stripped}".strip() if buffer else stripped
        if candidate.endswith("\\"):
            # Continuation: the next line joins after removing the backslash and
            # any trailing whitespace it hides.
            buffer = candidate[:-1].rstrip()
            continue
        line, buffer = candidate, ""
        if "<<" in line:
            raise DeferredRebuild("requires_heredoc_support")
        words = line.split(None, 1)
        if not words:
            continue
        command = words[0].lower()
        raw_args = words[1] if len(words) > 1 else ""
        json_form = False
        # Dockerfile instruction flags lead the line; anything after content begins
        # (e.g. apt-get's --no-install-recommends) is a shell argument, not a flag.
        flags = []
        while raw_args.startswith("--"):
            flag = raw_args.split(None, 1)[0]
            flags.append(flag)
            raw_args = raw_args[len(flag) :].lstrip()
        if command == "run" and raw_args.lstrip().startswith("["):
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError as error:
                raise DeferredRebuild("requires_run_json_support") from error
            json_form = True
            args = [str(item) for item in parsed]
        else:
            try:
                parts = shlex.split(raw_args, comments=True, posix=True)
            except ValueError as error:
                raise DeferredRebuild("requires_balanced_quotation") from error
            if not parts:
                continue
            args = parts
        instructions.append(Instruction(command=command, args=args, flags=flags, json_form=json_form, raw=raw_args))
    if buffer:
        raise DeferredRebuild("requires_continuation_terminator")
    return instructions


def _dependency_run(arguments: list[str]) -> bool:
    """True for package installs and index refreshes the setup script may run at start."""
    if arguments[0] in {"apt-get", "apt"}:
        return "install" in arguments or "update" in arguments
    if arguments[0] in {"pip", "pip3"}:
        return len(arguments) > 1 and arguments[1] == "install"
    if arguments[0] == "python3" and len(arguments) > 3:
        return arguments[1:3] == ["-m", "pip"] and "install" in arguments[3:]
    return False


def _tainted(arguments: list[str]) -> bool:
    return any(character in argument for argument in arguments for character in _METACHARACTERS)


def compile_recipe(
    instructions: list[Instruction], files: dict[str, bytes], modes: dict[str, int]
) -> tuple[str, str, str, bytes]:
    """Compile a native recipe into (base_ref, workdir, setup script, payload tar).

    Dependency installs move into the setup script verbatim; file copies are
    staged from the fixture payload; the FROM line maps onto the fixed base
    set. Everything else defers with the reason admission records.
    """
    bases = [item for item in instructions if item.command == "from"]
    if len(bases) != 1 or len(bases[0].args) != 1:
        raise DeferredRebuild("requires_multistage_support")
    original_from = bases[0].args[0]
    base_key, _, original_digest = original_from.partition("@")
    if base_key not in FIXED_BASES:
        raise DeferredRebuild("requires_unmapped_base:" + original_from)
    # A digest-pinned original keeps its digest on the normalized reference; floating
    # tags stay references until packaging resolves their digests.
    base_ref = f"{FIXED_BASES[base_key]}@{original_digest}" if original_digest else FIXED_BASES[base_key]
    workdir = "/"
    commands = [
        "#!/bin/sh",
        "set -eu",
        "stage=$(mktemp -d /tmp/calibforge-setup.XXXXXXXX)",
        "trap 'rm -rf \"$stage\"' EXIT",
        f'tar -xzf {SETUP_ROOT}/{FIXTURE_ARCHIVE} -C "$stage"',
    ]
    env_exports: list[str] = []
    resources: dict[str, bytes] = {}
    copied_files: set[str] = set()
    for item in instructions:
        if item.command in _META_COMMANDS:
            continue
        if item.command == "env":
            if item.json_form or item.flags:
                raise DeferredRebuild("requires_env_flags_support")
            if not item.args or any("=" not in argument for argument in item.args) or _tainted(item.args):
                raise DeferredRebuild("requires_env_support")
            env_exports.extend("export " + shlex.quote(argument) for argument in item.args)
            continue
        if item.command == "arg":
            raise DeferredRebuild("requires_arg_support")
        if item.command == "workdir":
            value = item.args[0]
            if "$" in value or ".." in PurePosixPath(value).parts:
                raise DeferredRebuild("requires_dynamic_workdir_support")
            workdir = str(PurePosixPath(workdir) / value)
            commands.append("mkdir -p " + shlex.quote(workdir))
            continue
        if item.command == "run":
            if item.flags:
                raise DeferredRebuild("requires_run_flags_support")
            if item.json_form:
                segments = [item.args]
            else:
                segments: list[list[str]] = [[]]
                for token in item.raw.split():
                    if token in {"&&", ";"}:
                        segments.append([])
                    elif token in {"|", "<", ">", "&"}:
                        raise DeferredRebuild("requires_pipeline_support")
                    else:
                        segments[-1].append(token)
            for raw_segment in segments:
                if not raw_segment:
                    raise DeferredRebuild("requires_empty_run")
                inline_env = ""
                while raw_segment and re.match(r"^[A-Za-z_][A-Za-z_0-9]*=", raw_segment[0]):
                    inline_env += raw_segment[0] + " "
                    raw_segment = raw_segment[1:]
                arguments = raw_segment
                fixture_generator = (
                    len(arguments) == 2
                    and arguments[0] == "python3"
                    and arguments[1].startswith("/app/")
                    and arguments[1].endswith(".py")
                    and arguments[1] in copied_files
                )
                if _dependency_run(arguments) or arguments[0] in _INNOCUOUS_RUN or fixture_generator:
                    if "`" in " ".join(raw_segment):
                        raise DeferredRebuild("requires_command_substitution_support")
                    prefix = inline_env
                    if arguments[0] in {"apt-get", "apt"} and "DEBIAN_FRONTEND" not in inline_env:
                        prefix = "DEBIAN_FRONTEND=noninteractive " + prefix
                    # Emit the segment verbatim: globs and variables keep their shell
                    # semantics at sandbox start (the recipe is the task's own content).
                    commands.append(f"(cd {shlex.quote(workdir)} && {prefix}{' '.join(raw_segment)})")
                    continue
                raise DeferredRebuild("requires_unsupported_run:" + arguments[0])
            continue
        if item.command != "copy":
            raise DeferredRebuild("requires_instruction_support:" + item.command)
        if item.flags:
            raise DeferredRebuild("requires_copy_flags_support")
        *sources, destination = item.args
        if any(character in destination for character in "$*?[") or ".." in PurePosixPath(destination).parts:
            raise DeferredRebuild("requires_dynamic_destination_support")
        target = str(PurePosixPath(workdir) / destination)
        directory_target = destination.endswith("/") or len(sources) > 1
        for source_name in sources:
            if (
                any(character in source_name for character in "$*?[")
                or PurePosixPath(source_name).is_absolute()
                or ".." in PurePosixPath(source_name).parts
            ):
                raise DeferredRebuild("requires_dynamic_source_support")
            prefix = source_name.rstrip("/") + "/"
            matches = [name for name in files if name == source_name or name.startswith(prefix)]
            if not matches:
                raise DeferredRebuild("missing_source:" + source_name)
            is_directory = any(name != source_name and name.startswith(prefix) for name in matches)
            if is_directory:
                directory_source = source_name.rstrip("/")
                for name in matches:
                    # Every file under a directory source must reach the setup payload;
                    # the script copies the staged directory, not the build context.
                    resources[name] = files[name]
                    copied_files.add(str(PurePosixPath(target) / name[len(prefix) :]))
                commands.append("mkdir -p " + shlex.quote(target))
                commands.append(
                    "cp -a " + f'"$stage"/{shlex.quote(directory_source)}' + "/. " + shlex.quote(target + "/")
                )
                continue
            name = matches[0]
            resources[name] = files[name]
            file_target = str(PurePosixPath(target) / PurePosixPath(name).name) if directory_target else target
            copied_files.add(file_target)
            commands.append("mkdir -p " + shlex.quote(str(PurePosixPath(file_target).parent)))
            commands.append("cp -a " + f'"$stage"/{shlex.quote(name)}' + " " + shlex.quote(file_target))
    commands.append(f"rm -f {SETUP_ROOT}/{FIXTURE_ARCHIVE} {SETUP_ROOT}/{SETUP_SCRIPT}")
    commands.append(f"rmdir {SETUP_ROOT}")
    if env_exports:
        commands[2:2] = env_exports
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        for name in sorted(resources):
            content = resources[name]
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = modes.get(name, 0o644)
            archive.addfile(info, io.BytesIO(content))
    return base_ref, workdir, "\n".join(commands) + "\n", gzip.compress(payload.getvalue(), mtime=0)


def rebuild_task(task_binary: bytes) -> RebuiltTask:
    """Lower one native task tar onto the fixed base set."""
    with tarfile.open(
        fileobj=io.BytesIO(gzip.decompress(task_binary) if task_binary[:2] == b"\x1f\x8b" else task_binary)
    ) as archive:
        files: dict[str, bytes] = {}
        modes: dict[str, int] = {}
        environment_modes: dict[str, int] = {}
        for member in archive.getmembers():
            if member.isdir() or member.issym():
                if member.issym():
                    raise DeferredRebuild("requires_symlink_support")
                continue
            files[member.name] = archive.extractfile(member).read()
            modes[member.name] = member.mode
            if member.name.startswith(ENVIRONMENT_PREFIX):
                environment_modes[member.name[len(ENVIRONMENT_PREFIX) :]] = member.mode
    dockerfile = files.pop("environment/Dockerfile", None)
    if dockerfile is None:
        raise DeferredRebuild("missing_environment_dockerfile")
    instructions = parse_instructions(dockerfile.decode(errors="replace"))
    context_files = {
        name[len(ENVIRONMENT_PREFIX) :]: content
        for name, content in files.items()
        if name.startswith(ENVIRONMENT_PREFIX)
    }
    base_ref, workdir, setup_script, payload = compile_recipe(instructions, context_files, environment_modes)
    output_files = {name: content for name, content in files.items() if not name.startswith(ENVIRONMENT_PREFIX)}
    output_files["task.toml"] = _rewrite_task_toml(files.get("task.toml"), base_ref, workdir)
    output_files["environment/Dockerfile"] = f"FROM {base_ref}\nWORKDIR {workdir}\n".encode()
    output_files[f"setup_files/{FIXTURE_ARCHIVE}"] = payload
    output_files[f"setup_files/{SETUP_SCRIPT}"] = setup_script.encode()
    packed = io.BytesIO()
    with tarfile.open(fileobj=packed, mode="w") as archive:
        for name in sorted(output_files):
            content = output_files[name]
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = modes.get(name, 0o644)
            archive.addfile(info, io.BytesIO(content))
    return RebuiltTask(
        task_binary=gzip.compress(packed.getvalue(), mtime=0),
        base_ref=base_ref,
        workdir=workdir,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        setup_sha256=hashlib.sha256(setup_script.encode()).hexdigest(),
    )


def _size_mb(value: object) -> int | None:
    """Parse a task.toml size such as 4096 or "8G" into MiB."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().upper()
    if text.endswith("G"):
        return int(float(text[:-1]) * 1024)
    if text.endswith("M"):
        return int(float(text[:-1]))
    return int(float(text))


def _rewrite_task_toml(original: bytes | None, base_ref: str, workdir: str) -> bytes:
    """Rewrite task.toml onto the fixed base with a setup-at-start hook.

    The machine shape is normalized so the cohort's snapshots collapse to one
    per base image; a task that declares more than the normalized shape defers.
    """
    if original is None:
        raise DeferredRebuild("missing_task_toml")
    config = tomllib.loads(original.decode(errors="replace"))
    environment = config.get("environment")
    if not isinstance(environment, dict):
        raise DeferredRebuild("missing_environment_section")
    if "setup" in environment:
        raise DeferredRebuild("existing_setup_hook")
    if environment.get("gpus") or environment.get("tpu"):
        raise DeferredRebuild("requires_accelerator_machine")
    if (environment.get("cpus") or 0) > NORMALIZED_CPUS:
        raise DeferredRebuild("requires_larger_machine_shape")
    if (_size_mb(environment.get("memory")) or 0) > NORMALIZED_MEMORY_MB:
        raise DeferredRebuild("requires_larger_machine_shape")
    if (_size_mb(environment.get("storage")) or 0) > NORMALIZED_STORAGE_MB:
        raise DeferredRebuild("requires_larger_machine_shape")
    document = tomlkit.parse(original.decode(errors="replace"))
    table = document["environment"]
    assert isinstance(table, Table)
    table["docker_image"] = base_ref
    table["workdir"] = workdir
    table["cpus"] = NORMALIZED_CPUS
    table["memory"] = f"{NORMALIZED_MEMORY_MB // 1024}G"
    table["storage"] = f"{NORMALIZED_STORAGE_MB // 1024}G"
    table["setup"] = {"command": f"sh {SETUP_ROOT}/{SETUP_SCRIPT}"}
    return tomlkit.dumps(document).encode()


def main() -> None:
    """Rebuild a native cohort parquet: staged tasks + evidence manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True, help="native cohort parquet (path, task_binary)")
    parser.add_argument("--excluded", type=Path, required=True, help="decontamination exclusion list (json array)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    excluded = set(json.loads(args.excluded.read_text()))
    records = []
    evidence = []
    base_refs: list[str] = []
    for batch in pq.ParquetFile(args.cohort).iter_batches(batch_size=64, columns=["path", "task_binary"]):
        for path, blob in zip(batch.to_pydict()["path"], batch.to_pydict()["task_binary"], strict=True):
            raw = bytes(blob.as_py() if hasattr(blob, "as_py") else blob)
            if path in excluded:
                evidence.append({"task_id": path, "status": "excluded_decontamination"})
                continue
            record: dict = {"task_id": path, "source_archive_sha256": hashlib.sha256(raw).hexdigest()}
            try:
                rebuilt = rebuild_task(raw)
            except DeferredRebuild as error:
                record.update(status="deferred", reason=str(error))
                evidence.append(record)
                continue
            base_refs.append(rebuilt.base_ref)
            record.update(
                status=REBUILT,
                base_ref=rebuilt.base_ref,
                workdir=rebuilt.workdir,
                payload_sha256=rebuilt.payload_sha256,
                setup_sha256=rebuilt.setup_sha256,
                archive_sha256=hashlib.sha256(rebuilt.task_binary).hexdigest(),
            )
            records.append({"path": path, "source": path, "task_binary": rebuilt.task_binary})
            evidence.append(record)
    schema = pa.schema([("path", pa.string()), ("source", pa.string()), ("task_binary", pa.binary())])
    pq.write_table(pa.Table.from_pylist(records, schema=schema), args.output / "pending-tasks.parquet")
    manifest = {
        "cohort_sha256": hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
        "excluded_sha256": hashlib.sha256(args.excluded.read_bytes()).hexdigest(),
        "cohort_count": len(evidence),
        "excluded_count": sum(1 for row in evidence if row["status"] == "excluded_decontamination"),
        "rebuilt_count": len(records),
        "deferred_count": sum(1 for row in evidence if row["status"] == "deferred"),
        # The Atlas counter measures the snapshot identities from these refs; the
        # rebuild itself stays free of the Harbor measurement environment.
        "snapshot_identity_count": None,
        "base_refs": sorted(set(base_refs)),
        "digest_pinned": False,
        "admission": "not_run",
        "tasks": evidence,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "tasks"}))


if __name__ == "__main__":
    main()
