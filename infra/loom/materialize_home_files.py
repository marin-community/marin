# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize Secret Manager values into Loom's shared session home."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SESSION_HOME = Path("/home/app")
STAGING_MOUNT = Path("/run/loom-home-files")
LEDGER_NAME = ".loom-managed-home-files.json"
PATH_PATTERN = re.compile(r"(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+")
PROJECT_PATTERN = re.compile(r"[a-z0-9-]+")
SECRET_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
VERSION_PATTERN = re.compile(r"[0-9]+")
ALLOWED_MODES = frozenset({"0400", "0600"})
VOLUME_NAME = "loom_loom_home"


@dataclass(frozen=True)
class SecretHomeFile:
    path: str
    project: str
    secret: str
    version: str
    mode: str


@dataclass(frozen=True)
class StagedHomeFile:
    path: str
    source: str
    mode: int


def _validated_path(value: object) -> str:
    if not isinstance(value, str) or not PATH_PATTERN.fullmatch(value):
        raise ValueError(f"invalid managed home path {value!r}")
    if value == LEDGER_NAME or any(part in {".", ".."} for part in value.split("/")):
        raise ValueError(f"invalid managed home path {value!r}")
    return value


def _load_secret_manifest(path: Path) -> tuple[SecretHomeFile, ...]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("home-file manifest must be a list")
    result: list[SecretHomeFile] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("home-file manifest entries must be objects")
        target = _validated_path(item.get("path"))
        project = item.get("project")
        secret = item.get("secret")
        version = item.get("version")
        mode = item.get("mode")
        if not isinstance(project, str) or not PROJECT_PATTERN.fullmatch(project):
            raise ValueError(f"invalid Secret Manager project for {target!r}")
        if not isinstance(secret, str) or not SECRET_PATTERN.fullmatch(secret):
            raise ValueError(f"invalid Secret Manager secret for {target!r}")
        if not isinstance(version, str) or not VERSION_PATTERN.fullmatch(version):
            raise ValueError(f"invalid Secret Manager version for {target!r}")
        if not isinstance(mode, str) or mode not in ALLOWED_MODES:
            raise ValueError(f"invalid mode for {target!r}")
        if target in seen:
            raise ValueError(f"duplicate managed home path {target!r}")
        seen.add(target)
        result.append(SecretHomeFile(target, project, secret, version, mode))
    return tuple(result)


def _load_staged_plan(path: Path) -> tuple[StagedHomeFile, ...]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("staged home-file plan must be a list")
    result: list[StagedHomeFile] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("staged home-file entries must be objects")
        target = _validated_path(item.get("path"))
        source = item.get("source")
        mode = item.get("mode")
        if not isinstance(source, str) or not VERSION_PATTERN.fullmatch(source):
            raise ValueError(f"invalid staged source for {target!r}")
        if not isinstance(mode, str) or mode not in ALLOWED_MODES:
            raise ValueError(f"invalid mode for {target!r}")
        if target in seen:
            raise ValueError(f"duplicate managed home path {target!r}")
        seen.add(target)
        result.append(StagedHomeFile(target, source, int(mode, 8)))
    return tuple(result)


def prepare_home_files(manifest_path: Path, image: str) -> None:
    """Fetch every declared secret, then apply the complete plan in the session image."""
    entries = _load_secret_manifest(manifest_path)
    with tempfile.TemporaryDirectory(prefix="loom-home-files-", dir="/run") as staging_value:
        staging = Path(staging_value)
        plan: list[dict[str, str]] = []
        for index, entry in enumerate(entries):
            source = str(index)
            payload = staging / source
            with payload.open("wb") as output:
                subprocess.run(
                    [
                        "gcloud",
                        "secrets",
                        "versions",
                        "access",
                        entry.version,
                        f"--project={entry.project}",
                        f"--secret={entry.secret}",
                    ],
                    check=True,
                    stdout=output,
                )
            payload.chmod(0o600)
            plan.append({"path": entry.path, "source": source, "mode": entry.mode})
        plan_path = staging / "plan.json"
        plan_path.write_text(json.dumps(plan, sort_keys=True, separators=(",", ":")))
        plan_path.chmod(0o600)
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network=none",
                "--user=0:0",
                "--entrypoint=python3",
                "--mount",
                f"type=volume,source={VOLUME_NAME},target={SESSION_HOME}",
                "--mount",
                f"type=bind,source={staging},target={STAGING_MOUNT},readonly",
                "--mount",
                f"type=bind,source={Path(__file__).resolve()},target=/run/materialize-home-files.py,readonly",
                image,
                "/run/materialize-home-files.py",
                "apply",
                f"--plan={STAGING_MOUNT / 'plan.json'}",
            ],
            check=True,
        )


def _open_directory(parent_fd: int, name: str, owner: tuple[int, int]) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
        os.chown(name, *owner, dir_fd=parent_fd, follow_symlinks=False)
        return os.open(name, flags, dir_fd=parent_fd)


def _target_directory(home_fd: int, path: str, owner: tuple[int, int]) -> tuple[int, str]:
    parts = path.split("/")
    current_fd = os.dup(home_fd)
    try:
        for part in parts[:-1]:
            next_fd = _open_directory(current_fd, part, owner)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd, parts[-1]
    except BaseException:
        os.close(current_fd)
        raise


def _regular_target_or_missing(parent_fd: int, name: str) -> None:
    try:
        status = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"managed home target {name!r} is not a regular file")


def _temporary_file(parent_fd: int, name: str, mode: int, owner: tuple[int, int]) -> tuple[str, int]:
    _regular_target_or_missing(parent_fd, name)
    temporary = f".{name}.loom-{secrets.token_hex(8)}"
    output_fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=parent_fd,
    )
    os.fchmod(output_fd, mode)
    os.fchown(output_fd, *owner)
    return temporary, output_fd


def _stage_file(
    parent_fd: int,
    name: str,
    source: Path,
    mode: int,
    owner: tuple[int, int],
) -> str:
    temporary, output_fd = _temporary_file(parent_fd, name, mode, owner)
    try:
        with source.open("rb") as input_file, os.fdopen(output_fd, "wb", closefd=False) as output_file:
            shutil.copyfileobj(input_file, output_file)
            output_file.flush()
            os.fsync(output_fd)
    except BaseException:
        os.unlink(temporary, dir_fd=parent_fd)
        raise
    finally:
        os.close(output_fd)
    return temporary


def _managed_paths(home_fd: int) -> set[str]:
    try:
        ledger_fd = os.open(LEDGER_NAME, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=home_fd)
    except FileNotFoundError:
        return set()
    try:
        status = os.fstat(ledger_fd)
        if not stat.S_ISREG(status.st_mode):
            raise ValueError("managed home-file ledger is not a regular file")
        with os.fdopen(ledger_fd, closefd=False) as ledger:
            raw = json.load(ledger)
    finally:
        os.close(ledger_fd)
    if not isinstance(raw, list):
        raise ValueError("managed home-file ledger must be a list")
    return {_validated_path(item) for item in raw}


def _write_ledger(home_fd: int, paths: set[str], owner: tuple[int, int]) -> None:
    payload = (json.dumps(sorted(paths), separators=(",", ":")) + "\n").encode()
    temporary, ledger_fd = _temporary_file(home_fd, LEDGER_NAME, 0o600, owner)
    try:
        with os.fdopen(ledger_fd, "wb", closefd=False) as ledger:
            ledger.write(payload)
            ledger.flush()
            os.fsync(ledger_fd)
    except BaseException:
        os.unlink(temporary, dir_fd=home_fd)
        raise
    finally:
        os.close(ledger_fd)
    os.replace(temporary, LEDGER_NAME, src_dir_fd=home_fd, dst_dir_fd=home_fd)


def apply_home_files(plan_path: Path, home: Path = SESSION_HOME, staging: Path = STAGING_MOUNT) -> None:
    """Atomically replace declared files and remove only previously managed paths."""
    entries = _load_staged_plan(plan_path)
    home_status = home.stat(follow_symlinks=False)
    if not stat.S_ISDIR(home_status.st_mode):
        raise ValueError(f"session home {home} is not a directory")
    owner = home_status.st_uid, home_status.st_gid
    home_fd = os.open(home, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    staged: list[tuple[int, str, str]] = []
    try:
        previous_paths = _managed_paths(home_fd)
        next_paths = {entry.path for entry in entries}
        for entry in entries:
            source = staging / entry.source
            source_status = source.stat(follow_symlinks=False)
            if not stat.S_ISREG(source_status.st_mode):
                raise ValueError(f"staged source for {entry.path!r} is not a regular file")
            parent_fd, name = _target_directory(home_fd, entry.path, owner)
            try:
                temporary = _stage_file(parent_fd, name, source, entry.mode, owner)
            except BaseException:
                os.close(parent_fd)
                raise
            staged.append((parent_fd, temporary, name))

        for parent_fd, temporary, name in staged:
            os.replace(temporary, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        for removed_path in previous_paths - next_paths:
            parent_fd, name = _target_directory(home_fd, removed_path, owner)
            try:
                _regular_target_or_missing(parent_fd, name)
                try:
                    os.unlink(name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
            finally:
                os.close(parent_fd)
        _write_ledger(home_fd, next_paths, owner)
    finally:
        for parent_fd, temporary, _ in staged:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            os.close(parent_fd)
        os.close(home_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--image", required=True)
    apply = subparsers.add_parser("apply")
    apply.add_argument("--plan", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare_home_files(args.manifest, args.image)
    else:
        apply_home_files(args.plan)


if __name__ == "__main__":
    main()
