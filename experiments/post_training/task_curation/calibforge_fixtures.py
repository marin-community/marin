# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lower CalibForge's dependency-free fixture recipes before dependency qualification."""

import argparse
import gzip
import hashlib
import io
import json
import shlex
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

FIXTURE_ARCHIVE = "calibforge-fixtures.tar.gz"
SETUP_SCRIPT = "calibforge-initialize.sh"
COMPILED_PENDING = "compiled_pending_equivalence_and_admission"


@dataclass(frozen=True)
class FixtureBundle:
    base_image: str
    workdir: str
    setup_script: str
    payload: bytes


class DeferredEnvironment(ValueError):
    """A native recipe needs a lowering contract not yet qualified."""


def fixture_bundle(context: Path, instructions: list[dict]) -> FixtureBundle:
    """Compile ordered public copies without executing source commands or using private files."""
    if any("/setup_files" in argument for item in instructions for argument in item["args"]):
        raise DeferredEnvironment("requires_reserved_setup_path_support")
    unsupported = {item["command"] for item in instructions} - {"from", "workdir", "copy", "run"}
    if unsupported:
        raise DeferredEnvironment("requires_recipe_support:" + ",".join(sorted(unsupported)))
    bases = [item for item in instructions if item["command"] == "from"]
    if len(bases) != 1 or len(bases[0]["args"]) != 1:
        raise DeferredEnvironment("requires_multistage_support")
    ignore = context / ".dockerignore"
    if ignore.exists() and ignore.read_text().strip():
        raise DeferredEnvironment("requires_dockerignore_support")
    workdir = "/"
    commands = [
        "#!/bin/sh",
        "set -eu",
        "stage=$(mktemp -d /tmp/calibforge-setup.XXXXXXXX)",
        "trap 'rm -rf \"$stage\"' EXIT",
        f'tar -xzf /setup_files/{FIXTURE_ARCHIVE} -C "$stage"',
    ]
    resources: dict[str, Path] = {}
    copied_files: set[str] = set()
    executed_scripts: set[str] = set()
    for item in instructions:
        if item["flags"] or item["heredocs"]:
            raise DeferredEnvironment("requires_copy_flags_or_heredoc_support")
        if item["command"] == "from":
            continue
        if item["command"] == "workdir":
            value = item["args"][0]
            if "$" in value or ".." in PurePosixPath(value).parts:
                raise DeferredEnvironment("requires_dynamic_workdir_support")
            workdir = str(PurePosixPath(workdir) / value)
            commands.append("mkdir -p " + shlex.quote(workdir))
            continue
        if item["command"] == "run":
            json_form = (item["attributes"] or {}).get("json")
            if not json_form and "\n" in item["args"][0]:
                raise DeferredEnvironment("requires_multiline_run_support")
            arguments = item["args"] if json_form else shlex.split(item["args"][0], comments=True)
            fixture_generator = (
                len(arguments) == 2
                and arguments[0] == "python3"
                and arguments[1].startswith("/app/")
                and arguments[1].endswith(".py")
                and arguments[1] in copied_files
            )
            remove_generator = len(arguments) == 3 and arguments[:2] == ["rm", "-f"] and arguments[2] in executed_scripts
            if (
                not arguments
                or not (arguments[0] in {"mkdir", "chmod", "touch"} or fixture_generator or remove_generator)
                or any(any(character in argument for character in "$`;&|<>()*?[~") for argument in arguments)
            ):
                raise DeferredEnvironment("requires_dependency_or_complex_run_support")
            if fixture_generator:
                executed_scripts.add(arguments[1])
            if remove_generator:
                executed_scripts.remove(arguments[2])
            commands.append("(cd " + shlex.quote(workdir) + " && " + shlex.join(arguments) + ")")
            continue
        *sources, destination = item["args"]
        if any(character in destination for character in "$*?[") or ".." in PurePosixPath(destination).parts:
            raise DeferredEnvironment("requires_dynamic_destination_support")
        target = str(PurePosixPath(workdir) / destination)
        directory_target = destination.endswith("/") or len(sources) > 1
        for source in sources:
            if (
                any(character in source for character in "$*?[")
                or PurePosixPath(source).is_absolute()
                or ".." in PurePosixPath(source).parts
            ):
                raise DeferredEnvironment("requires_dynamic_source_support")
            path = context / source
            if not path.exists():
                raise DeferredEnvironment("missing_source:" + source)
            members = [path, *path.rglob("*")] if path.is_dir() else [path]
            if any(member.is_symlink() for member in members):
                raise DeferredEnvironment("requires_symlink_support")
            for member in members:
                resources[str(member.relative_to(context))] = member
            staged = '"$stage"/' + shlex.quote(str(PurePosixPath(source)))
            if path.is_dir():
                copied_files.update(
                    str(PurePosixPath(target) / member.relative_to(path).as_posix())
                    for member in members
                    if member.is_file()
                )
                commands.extend(
                    ["mkdir -p " + shlex.quote(target), "cp -a " + staged + "/. " + shlex.quote(target + "/")]
                )
            else:
                copied_files.add(str(PurePosixPath(target) / path.name) if directory_target else target)
                parent = target if directory_target else str(PurePosixPath(target).parent)
                commands.append("mkdir -p " + shlex.quote(parent))
                commands.append("cp -a " + staged + " " + shlex.quote(target + ("/" if directory_target else "")))
    commands.append(f"rm -f /setup_files/{FIXTURE_ARCHIVE} /setup_files/{SETUP_SCRIPT}")
    commands.append("rmdir /setup_files")
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, path in sorted(resources.items()):
            info = archive.gettarinfo(str(path), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            if path.is_file():
                with path.open("rb") as content:
                    archive.addfile(info, content)
            else:
                archive.addfile(info)
    return FixtureBundle(
        bases[0]["args"][0], workdir, "\n".join(commands) + "\n", gzip.compress(output.getvalue(), mtime=0)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = json.loads(args.inventory.read_text())
    records = []
    for task in inventory["tasks"]:
        identity = task["task_id"]
        record = {"task_id": identity, "source_dockerfile_sha256": task["dockerfile_sha256"]}
        try:
            bundle = fixture_bundle(args.source / identity / "environment", task["instructions"])
        except DeferredEnvironment as error:
            record.update(status="deferred", reason=str(error))
            records.append(record)
            continue
        target = args.output / "tasks" / identity
        target.mkdir(parents=True)
        (target / FIXTURE_ARCHIVE).write_bytes(bundle.payload)
        (target / SETUP_SCRIPT).write_text(bundle.setup_script)
        record.update(
            status=COMPILED_PENDING,
            base_image=bundle.base_image,
            workdir=bundle.workdir,
            payload_sha256=hashlib.sha256(bundle.payload).hexdigest(),
            setup_sha256=hashlib.sha256(bundle.setup_script.encode()).hexdigest(),
        )
        records.append(record)
    result = {
        "inventory_sha256": hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
        "task_count": len(records),
        "compiled_count": sum(row["status"] == COMPILED_PENDING for row in records),
        "admission": "not_run",
        "tasks": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "tasks"}))


if __name__ == "__main__":
    main()
