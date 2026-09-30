#!/usr/bin/env python3
"""Restore a manifest-bound continuation seed without trusting transport layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import NoReturn

PROTECTED_PREFIXES = (".env", "secret", "token")


def fail(message: str) -> NoReturn:
    raise SystemExit(message)


def safe_member(relative: str, digest: str) -> PurePosixPath:
    if not isinstance(relative, str):
        fail("restore seed contains an unsafe member")
    parsed = PurePosixPath(relative)
    if (
        parsed.is_absolute()
        or ".." in parsed.parts
        or not parsed.parts
        or any(
            part == ".tmp" or part.lower().startswith(PROTECTED_PREFIXES)
            for part in parsed.parts
        )
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        fail("restore seed contains an unsafe member")
    return parsed


def display(relative: str) -> str:
    parts = relative.split("/")
    if any(part == ".tmp" or part.lower().startswith(PROTECTED_PREFIXES) for part in parts):
        return "[protected path]"
    return relative


def member_map(root: Path) -> dict[str, str]:
    actual: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            fail("restore seed contains a symlink")
        if path.is_file():
            actual[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return actual


def assert_member_map(expected: dict[str, str], actual: dict[str, str]) -> None:
    if actual == expected:
        return
    groups = {
        "missing": sorted(set(expected) - set(actual)),
        "extra": sorted(set(actual) - set(expected)),
        "changed": sorted(name for name in set(expected) & set(actual) if expected[name] != actual[name]),
    }
    detail = "; ".join(
        f"{kind}={len(values)}" + (f" ({', '.join(display(value) for value in values[:8])})" if values else "")
        for kind, values in groups.items()
    )
    fail(f"restore seed members are missing, extra, or changed: {detail}")


def extract_archive(archive: Path, expected_sha256: str, destination: Path) -> None:
    if not archive.is_file():
        fail("restore seed archive is missing")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected_sha256:
        fail("restore seed archive fingerprint mismatch")
    try:
        with tarfile.open(archive, mode="r:gz") as source:
            seen: set[str] = set()
            for member in source.getmembers():
                name = member.name
                parsed = PurePosixPath(name)
                if (
                    not name
                    or parsed.is_absolute()
                    or ".." in parsed.parts
                    or member.issym()
                    or member.islnk()
                    or member.isdev()
                    or member.isfifo()
                ):
                    fail("restore seed archive contains an unsafe member")
                if member.isdir():
                    continue
                if not member.isfile() or name in seen:
                    fail("restore seed archive contains an unsafe member")
                seen.add(name)
                payload = source.extractfile(member)
                if payload is None:
                    fail("restore seed archive member is unreadable")
                output = destination / parsed
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("wb") as handle:
                    shutil.copyfileobj(payload, handle)
                output.chmod(member.mode & 0o777)
    except (OSError, tarfile.TarError):
        fail("restore seed archive is unreadable")


def load_seed(manifest_path: Path, seed: Path, results: Path | None) -> None:
    try:
        document = json.loads(manifest_path.read_text())
        expected = document["seed_members"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit("restore seed manifest is invalid") from error
    if not isinstance(expected, dict) or not expected:
        fail("restore seed has no declared members")
    for relative, digest in expected.items():
        safe_member(relative, digest)

    archive = document.get("seed_archive")
    with tempfile.TemporaryDirectory(prefix="capability-restore-seed-") as temporary:
        root = seed
        if archive is not None:
            if not isinstance(archive, dict):
                fail("restore seed archive declaration is invalid")
            name, digest = archive.get("name"), archive.get("sha256")
            if not isinstance(name, str) or PurePosixPath(name).parts != (name,):
                fail("restore seed archive declaration is invalid")
            if not isinstance(digest, str) or len(digest) != 64:
                fail("restore seed archive declaration is invalid")
            root = Path(temporary) / "seed"
            root.mkdir()
            extract_archive(seed.parent / name, digest, root)
        elif not root.is_dir():
            fail("restore seed directory is missing")
        actual = member_map(root)
        assert_member_map(expected, actual)
        if results is None:
            return
        collisions = [
            relative
            for relative in sorted(expected)
            if (results / safe_member(relative, expected[relative])).exists()
            or (results / safe_member(relative, expected[relative])).is_symlink()
        ]
        if collisions:
            fail(
                "new result prefix already contains restore members: "
                + ", ".join(display(relative) for relative in collisions[:8])
            )
        for relative, digest in sorted(expected.items()):
            parsed = safe_member(relative, digest)
            source = root / parsed
            destination = results / parsed
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary_copy = destination.with_name(destination.name + ".seed-part")
            shutil.copyfile(source, temporary_copy)
            if hashlib.sha256(temporary_copy.read_bytes()).hexdigest() != digest:
                fail("restore seed copy checksum mismatch")
            temporary_copy.chmod(source.stat().st_mode & 0o777)
            temporary_copy.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("seed", type=Path)
    parser.add_argument("results", type=Path, nargs="?")
    args = parser.parse_args()
    if not args.check_only and args.results is None:
        parser.error("results is required unless --check-only is used")
    load_seed(args.manifest, args.seed, None if args.check_only else args.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
