#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare and optionally merge exact marin-style consumer updates."""

import argparse
import json
import re
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath

from scripts.ci.dependency_update import (
    changed_worktree_files,
    merge_when_green,
    prepare_update_branch,
    publish_update,
    validate_changed_files,
)
from scripts.ci.dependency_update_policy import PullRequestPolicy
from scripts.ci.marin_style_consumers import (
    LEGACY_MANAGED_FILES,
    LockMode,
    MarinStyleConsumer,
    marin_style_consumer,
    marin_style_consumer_matrix,
)

MARIN_STYLE_REPOSITORY = "https://github.com/marin-community/marin-style"
MARIN_STYLE_MANIFEST = ".agents/marin-style/manifest.json"
MARIN_STYLE_BRANCH = "automation/marin-style"
MARIN_STYLE_TITLE = "[dependencies] Advance marin-style"
MARIN_STYLE_PIN = re.compile(
    r"https://github\.com/marin-community/marin-style(?:\.git)?@([0-9a-f]{40})"
)
REVISION = re.compile(r"[0-9a-f]{40}")
CONTENT_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
MANAGED_PREFIXES = (".agents/marin-style/", ".agents/skills/")
PR_BODY = """Advance marin-style to `{revision}` and regenerate its shared agent guidance.

Changed paths are restricted to registered revision pins, the generated lockfile where applicable, and files owned by the old and new marin-style manifests.
"""


@dataclass(frozen=True)
class GeneratedManifest:
    revision: str
    files: tuple[tuple[str, str], ...]

    @classmethod
    def from_text(cls, text: str, *, expected_revision: str) -> "GeneratedManifest":
        payload = json.loads(text)
        if not isinstance(payload, dict) or set(payload) != {"files", "format", "revision"}:
            raise ValueError("invalid marin-style manifest shape")
        if payload["format"] != 1 or payload["revision"] != expected_revision:
            raise ValueError(f"manifest does not describe marin-style revision {expected_revision}")
        files = payload["files"]
        if not isinstance(files, dict) or not files:
            raise ValueError("marin-style manifest has no files")
        for path, digest in files.items():
            if not isinstance(path, str) or not isinstance(digest, str):
                raise ValueError("marin-style manifest files must map paths to digests")
            relative = PurePosixPath(path)
            if (
                relative.is_absolute()
                or relative.as_posix() != path
                or ".." in relative.parts
                or not path.startswith(MANAGED_PREFIXES)
                or path == MARIN_STYLE_MANIFEST
            ):
                raise ValueError(f"invalid marin-style managed path: {path!r}")
            if CONTENT_DIGEST.fullmatch(digest) is None:
                raise ValueError(f"invalid digest for marin-style managed path: {path!r}")
        return cls(revision=expected_revision, files=tuple(sorted(files.items())))


@dataclass(frozen=True)
class GeneratedMarinStyleUpdate:
    old_revision: str
    new_revision: str
    changed_files: tuple[str, ...]
    policy: PullRequestPolicy


class ManifestMode(StrEnum):
    VALIDATE = "validate"
    BOOTSTRAP = "bootstrap"


class MergeMode(StrEnum):
    PUBLISH = "publish"
    MERGE = "merge"


def _run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        list(args),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _manifest_for_revision(revision: str) -> GeneratedManifest:
    output = _run(
        "uvx",
        "--from",
        f"git+{MARIN_STYLE_REPOSITORY}@{revision}",
        "marin-style",
        "managed-files",
    )
    return GeneratedManifest.from_text(output, expected_revision=revision)


def _checked_in_manifest(repo_root: Path, *, expected_revision: str) -> GeneratedManifest | None:
    path = repo_root / MARIN_STYLE_MANIFEST
    if path.is_symlink():
        raise ValueError(f"marin-style manifest must be a regular file: {path}")
    if not path.exists():
        return None
    return GeneratedManifest.from_text(path.read_text(), expected_revision=expected_revision)


def _old_revision(repo_root: Path, consumer: MarinStyleConsumer) -> str:
    canonical = repo_root / "infra/pre-commit.py"
    matches = MARIN_STYLE_PIN.findall(canonical.read_text())
    if len(matches) != 1:
        raise ValueError(f"expected one marin-style revision in {canonical}")
    old_revision = matches[0]
    for relative in consumer.pin_files:
        path = repo_root / relative
        text = path.read_text()
        if old_revision not in text:
            raise ValueError(f"registered pin file does not contain {old_revision}: {path}")
    return old_revision


def _replace_pin_files(
    repo_root: Path,
    consumer: MarinStyleConsumer,
    *,
    old_revision: str,
    new_revision: str,
) -> None:
    for relative in consumer.pin_files:
        path = repo_root / relative
        text = path.read_text()
        updated = text.replace(old_revision, new_revision)
        if updated == text or old_revision in updated:
            raise ValueError(f"failed to replace marin-style revision in {path}")
        path.write_text(updated)


def _update_uv_lock(repo_root: Path, *, old_revision: str, new_revision: str) -> None:
    subprocess.run(
        ["uv", "lock", "--upgrade-package", "marin-style"],
        cwd=repo_root,
        check=True,
    )
    lock_path = repo_root / "uv.lock"
    with lock_path.open("rb") as lock_file:
        payload = tomllib.load(lock_file)
    packages = [package for package in payload.get("package", []) if package.get("name") == "marin-style"]
    if len(packages) != 1:
        raise ValueError("uv.lock must contain exactly one marin-style package")
    package = packages[0]
    source = package.get("source")
    if not isinstance(source, dict) or new_revision not in str(source.get("git", "")):
        raise ValueError("uv.lock does not contain the target marin-style revision")
    if not isinstance(package.get("version"), str) or old_revision in lock_path.read_text():
        raise ValueError("uv.lock contains an invalid marin-style package record")


def generate_marin_style_update(
    *,
    repo_root: Path,
    consumer: MarinStyleConsumer,
    revision: str,
    manifest_mode: ManifestMode,
) -> GeneratedMarinStyleUpdate:
    """Generate and validate one consumer update in the current checkout."""
    if REVISION.fullmatch(revision) is None:
        raise ValueError("marin-style revision must be a full lowercase commit SHA")
    old_revision = _old_revision(repo_root, consumer)
    if old_revision == revision:
        raise ValueError("consumer already pins the target marin-style revision")

    new_manifest = _manifest_for_revision(revision)
    old_manifest = _checked_in_manifest(repo_root, expected_revision=old_revision)
    if old_manifest is None:
        if manifest_mode is not ManifestMode.BOOTSTRAP:
            raise ValueError("consumer has no marin-style manifest; run the reviewed bootstrap first")
        new_paths = frozenset(path for path, _ in new_manifest.files)
        missing_legacy_paths = sorted(LEGACY_MANAGED_FILES - new_paths)
        if missing_legacy_paths:
            raise ValueError(f"bootstrap revision removes legacy managed files: {missing_legacy_paths}")
        old_paths = LEGACY_MANAGED_FILES
    else:
        if manifest_mode is ManifestMode.BOOTSTRAP:
            raise ValueError("consumer already has a marin-style manifest")
        installed_old_manifest = _manifest_for_revision(old_revision)
        if installed_old_manifest != old_manifest:
            raise ValueError("checked-in manifest does not match the pinned marin-style revision")
        old_paths = frozenset(path for path, _ in old_manifest.files)

    _replace_pin_files(
        repo_root,
        consumer,
        old_revision=old_revision,
        new_revision=revision,
    )
    subprocess.run(
        [
            "uvx",
            "--from",
            f"git+{MARIN_STYLE_REPOSITORY}@{revision}",
            "marin-style",
            "sync",
            "--repo-root",
            str(repo_root),
        ],
        check=True,
    )
    written_manifest = _checked_in_manifest(repo_root, expected_revision=revision)
    if written_manifest != new_manifest:
        raise ValueError("marin-style sync wrote a manifest that differs from the installed package")

    if consumer.lock_mode is LockMode.UV:
        _update_uv_lock(repo_root, old_revision=old_revision, new_revision=revision)

    new_paths = frozenset(path for path, _ in new_manifest.files)
    allowed_files = frozenset(
        {
            *consumer.pin_files,
            *consumer.lock_files,
            *old_paths,
            *new_paths,
            MARIN_STYLE_MANIFEST,
        }
    )
    policy = PullRequestPolicy(
        base_branch=consumer.base_branch,
        head_branch=MARIN_STYLE_BRANCH,
        title=MARIN_STYLE_TITLE,
        allowed_files=allowed_files,
        required_checks=consumer.required_checks,
    )
    changed_files = validate_changed_files(changed_worktree_files(repo_root), policy=policy)
    if not changed_files:
        raise ValueError("marin-style update produced no changed files")
    return GeneratedMarinStyleUpdate(
        old_revision=old_revision,
        new_revision=revision,
        changed_files=changed_files,
        policy=policy,
    )


def _preflight(consumer: MarinStyleConsumer) -> None:
    installed = _run(
        "gh",
        "api",
        "installation/repositories",
        "--paginate",
        "--jq",
        ".repositories[].full_name",
    ).splitlines()
    if consumer.repository not in installed:
        raise ValueError(f"updater App is not installed on {consumer.repository}")
    default_branch = _run(
        "gh",
        "repo",
        "view",
        consumer.repository,
        "--json",
        "defaultBranchRef",
        "--jq",
        ".defaultBranchRef.name",
    ).strip()
    if default_branch != consumer.base_branch:
        raise ValueError(f"unexpected default branch for {consumer.repository}: {default_branch!r}")
    for label in ("agent-generated", "dependencies"):
        _run("gh", "api", f"repos/{consumer.repository}/labels/{label}")


def update_consumer(
    *,
    consumer: MarinStyleConsumer,
    revision: str,
    merge_mode: MergeMode,
    app_slug: str,
    manifest_mode: ManifestMode,
) -> str | None:
    """Prepare, publish, and optionally merge one consumer update."""
    if manifest_mode is ManifestMode.BOOTSTRAP and merge_mode is MergeMode.MERGE:
        raise ValueError("the manifest bootstrap requires human review")
    _preflight(consumer)
    repo_root = Path.cwd()
    old_revision = _old_revision(repo_root, consumer)
    initial_policy = PullRequestPolicy(
        base_branch=consumer.base_branch,
        head_branch=MARIN_STYLE_BRANCH,
        title=MARIN_STYLE_TITLE,
        allowed_files=frozenset(),
        required_checks=consumer.required_checks,
    )
    branch = prepare_update_branch(policy=initial_policy, repository=consumer.repository)
    if old_revision == revision:
        if branch.pull_request_url:
            raise ValueError(f"consumer is current but has an open automation PR: {branch.pull_request_url}")
        return None

    update = generate_marin_style_update(
        repo_root=repo_root,
        consumer=consumer,
        revision=revision,
        manifest_mode=manifest_mode,
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        body_file = Path(temp_dir) / "marin-style-update.md"
        body_file.write_text(PR_BODY.format(revision=revision))
        pull_request = publish_update(
            policy=update.policy,
            repository=consumer.repository,
            body_file=body_file,
            expected_remote_sha=branch.expected_remote_sha,
            pull_request_url=branch.pull_request_url,
            push_mode=branch.push_mode,
        )
    if merge_mode is MergeMode.MERGE:
        merge_when_green(
            pr=pull_request.url,
            repository=consumer.repository,
            app_slug=app_slug,
            policy=update.policy,
            expected_head_sha=pull_request.head_sha,
            timeout=3600,
            poll_interval=30,
        )
    return pull_request.url


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    matrix = subparsers.add_parser("matrix", help="print the consumer matrix")
    matrix.add_argument("--consumer", default="")
    run = subparsers.add_parser("run", help="update one consumer checkout")
    run.add_argument("--consumer", required=True)
    run.add_argument("--revision", required=True)
    run.add_argument("--app-slug", required=True)
    run.add_argument("--auto-merge", action="store_true")
    run.add_argument("--bootstrap", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "matrix":
        print(marin_style_consumer_matrix(args.consumer))
        return
    url = update_consumer(
        consumer=marin_style_consumer(args.consumer),
        revision=args.revision,
        merge_mode=MergeMode.MERGE if args.auto_merge else MergeMode.PUBLISH,
        app_slug=args.app_slug,
        manifest_mode=ManifestMode.BOOTSTRAP if args.bootstrap else ManifestMode.VALIDATE,
    )
    if url is None:
        print(f"{args.consumer} already pins {args.revision}")
    else:
        print(f"Updated {url}")


if __name__ == "__main__":
    main()
