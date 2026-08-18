#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Discover, prepare, and optionally merge marin-style consumer updates."""

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
    DEPENDENCY_UPDATE_LABELS,
    changed_worktree_files,
    merge_when_protected_checks_green,
    prepare_update_branch,
    publish_update,
    validate_changed_files,
)
from scripts.ci.dependency_update_policy import PullRequestPolicy

ORGANIZATION = "marin-community"
CONTROL_REPOSITORY = f"{ORGANIZATION}/marin"
MARIN_STYLE_REPOSITORY = f"https://github.com/{ORGANIZATION}/marin-style"
MARIN_STYLE_PACKAGE = "marin-style"
MARIN_STYLE_MANIFEST = ".agents/marin-style/manifest.json"
MARIN_STYLE_BRANCH = "automation/marin-style"
MARIN_STYLE_TITLE = "[dependencies] Advance marin-style"
MARIN_STYLE_PIN = re.compile(rf"{re.escape(MARIN_STYLE_REPOSITORY)}(?:\.git)?@([0-9a-f]{{40}})")
REVISION = re.compile(r"[0-9a-f]{40}")
CONTENT_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
MANAGED_PREFIXES = (".agents/marin-style/", ".agents/skills/")
CANONICAL_PIN_FILE = "infra/pre-commit.py"
UV_LOCK_FILE = "uv.lock"
PR_BODY = (
    "Advance marin-style to `{revision}` and regenerate its shared agent guidance.\n\n"
    "Changed paths are restricted to discovered marin-style pins, its generated lockfile, "
    "and files owned by the old and new manifests.\n"
)

# Remove this bootstrap boundary after every installed consumer has a manifest.
LEGACY_MANAGED_FILES = frozenset(
    {
        ".agents/marin-style/AGENTS-core.md",
        ".agents/marin-style/TESTING-core.md",
        ".agents/skills/commit/SKILL.md",
        ".agents/skills/consult-echo/SKILL.md",
        ".agents/skills/consult-echo/scripts/echo.py",
        ".agents/skills/debug/SKILL.md",
        ".agents/skills/file-issue/SKILL.md",
        ".agents/skills/task-logbook/SKILL.md",
        ".agents/skills/write-design-doc/SKILL.md",
        ".agents/skills/write-ops-log/SKILL.md",
        ".agents/skills/write-tests/SKILL.md",
        ".agents/skills/writing-style/SKILL.md",
        ".agents/skills/writing-style/ai-writing-donts.md",
        ".agents/skills/writing-style/blog-posts.md",
        ".agents/skills/writing-style/discord.md",
        ".agents/skills/writing-style/issues.md",
        ".agents/skills/writing-style/pull-requests.md",
        ".agents/skills/writing-style/reference-docs.md",
        ".agents/skills/writing-style/reports.md",
        ".agents/skills/writing-style/tutorials.md",
    }
)


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


@dataclass(frozen=True)
class LockedMarinStyle:
    version: str
    source: str


class ManifestMode(StrEnum):
    VALIDATE = "validate"
    BOOTSTRAP = "bootstrap"


class MergeMode(StrEnum):
    PUBLISH = "publish"
    MERGE = "merge"


class ConsumerUpdateStatus(StrEnum):
    CURRENT = "current"
    PUBLISHED = "published"
    MERGED = "merged"


@dataclass(frozen=True)
class ConsumerUpdateResult:
    status: ConsumerUpdateStatus
    pull_request_url: str


def _run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        list(args),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def installed_consumer_matrix_json(selected: str = "") -> str:
    """Return installed repositories as a GitHub Actions matrix."""
    rows = _run(
        "gh",
        "api",
        "installation/repositories",
        "--paginate",
        "--jq",
        ".repositories[] | [.full_name, .name, .default_branch] | @tsv",
    ).splitlines()
    consumers: list[dict[str, str]] = []
    for row in rows:
        repository, name, base_branch = row.split("\t")
        if repository == CONTROL_REPOSITORY:
            continue
        if not repository.startswith(f"{ORGANIZATION}/") or not name or not base_branch:
            raise ValueError(f"invalid updater installation repository: {row!r}")
        consumers.append(
            {
                "name": name,
                "repository": repository,
                "repository_name": name,
                "base_branch": base_branch,
            }
        )
    if selected:
        consumers = [row for row in consumers if selected in {row["name"], row["repository"]}]
        if not consumers:
            raise ValueError(f"updater App is not installed on requested consumer {selected!r}")
    if not consumers:
        raise ValueError("updater App has no installed consumer repositories")
    return json.dumps({"include": sorted(consumers, key=lambda row: row["repository"])}, separators=(",", ":"))


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


def _old_revision(repo_root: Path) -> str:
    canonical = repo_root / CANONICAL_PIN_FILE
    matches = MARIN_STYLE_PIN.findall(canonical.read_text())
    if len(matches) != 1:
        raise ValueError(f"expected one marin-style revision in {canonical}")
    return matches[0]


def _tracked_revision_paths(repo_root: Path, revision: str) -> frozenset[str]:
    result = subprocess.run(
        ["git", "grep", "--fixed-strings", "--name-only", revision, "--"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        result.check_returncode()
    return frozenset(result.stdout.splitlines())


def _is_direct_pin_path(path: str) -> bool:
    if path in {CANONICAL_PIN_FILE, "pyproject.toml"}:
        return True
    relative = PurePosixPath(path)
    return relative.suffix in {".yaml", ".yml"} and relative.parts[:2] == (".github", "workflows")


def _direct_pin_files(
    repo_root: Path,
    *,
    revision: str,
    managed_paths: frozenset[str],
    uses_uv_lock: bool,
) -> tuple[str, ...]:
    ignored = {*managed_paths, MARIN_STYLE_MANIFEST}
    if uses_uv_lock:
        ignored.add(UV_LOCK_FILE)
    candidates = _tracked_revision_paths(repo_root, revision) - ignored
    unexpected: list[str] = []
    direct: list[str] = []
    for relative in sorted(candidates):
        if not _is_direct_pin_path(relative):
            unexpected.append(relative)
            continue
        lines = [line for line in (repo_root / relative).read_text().splitlines() if revision in line]
        if not lines or any(
            "marin-style" not in line.lower() and "marin_style_rev" not in line.lower() for line in lines
        ):
            unexpected.append(relative)
            continue
        direct.append(relative)
    if unexpected:
        raise ValueError(f"unexpected files reference the pinned marin-style revision: {unexpected}")
    if CANONICAL_PIN_FILE not in direct:
        raise ValueError(f"{CANONICAL_PIN_FILE} is not a direct marin-style pin")
    return tuple(direct)


def _locked_marin_style(repo_root: Path) -> LockedMarinStyle | None:
    lock_path = repo_root / UV_LOCK_FILE
    if not lock_path.exists():
        return None
    with lock_path.open("rb") as lock_file:
        payload = tomllib.load(lock_file)
    packages = [package for package in payload.get("package", []) if package.get("name") == MARIN_STYLE_PACKAGE]
    if not packages:
        return None
    if len(packages) != 1:
        raise ValueError("uv.lock must contain at most one marin-style package")
    package = packages[0]
    source = package.get("source")
    version = package.get("version")
    if not isinstance(source, dict) or not isinstance(source.get("git"), str) or not isinstance(version, str):
        raise ValueError("uv.lock contains an invalid marin-style package record")
    return LockedMarinStyle(version=version, source=source["git"])


def _replace_pin_files(repo_root: Path, pin_files: tuple[str, ...], *, old_revision: str, new_revision: str) -> None:
    for relative in pin_files:
        path = repo_root / relative
        text = path.read_text()
        updated = text.replace(old_revision, new_revision)
        if updated == text or old_revision in updated:
            raise ValueError(f"failed to replace marin-style revision in {path}")
        path.write_text(updated)


def _update_uv_lock(repo_root: Path, *, old_revision: str, new_revision: str) -> None:
    subprocess.run(["uv", "lock", "--upgrade-package", MARIN_STYLE_PACKAGE], cwd=repo_root, check=True)
    package = _locked_marin_style(repo_root)
    if package is None:
        raise ValueError("uv.lock must contain exactly one marin-style package")
    if new_revision not in package.source:
        raise ValueError("uv.lock does not contain the target marin-style revision")
    if old_revision in (repo_root / UV_LOCK_FILE).read_text():
        raise ValueError("uv.lock contains an invalid marin-style package record")


def generate_marin_style_update(
    *,
    repo_root: Path,
    base_branch: str,
    revision: str,
    manifest_mode: ManifestMode,
) -> GeneratedMarinStyleUpdate:
    """Generate and validate one consumer update in the current checkout."""
    if REVISION.fullmatch(revision) is None:
        raise ValueError("marin-style revision must be a full lowercase commit SHA")
    old_revision = _old_revision(repo_root)
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
        if _manifest_for_revision(old_revision) != old_manifest:
            raise ValueError("checked-in manifest does not match the pinned marin-style revision")
        old_paths = frozenset(path for path, _ in old_manifest.files)

    locked_marin_style = _locked_marin_style(repo_root)
    if locked_marin_style is not None and old_revision not in locked_marin_style.source:
        raise ValueError("uv.lock does not contain the pinned marin-style revision")
    uses_uv_lock = locked_marin_style is not None
    pin_files = _direct_pin_files(
        repo_root,
        revision=old_revision,
        managed_paths=old_paths,
        uses_uv_lock=uses_uv_lock,
    )
    if old_revision != revision:
        _replace_pin_files(repo_root, pin_files, old_revision=old_revision, new_revision=revision)
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
    if uses_uv_lock and old_revision != revision:
        _update_uv_lock(repo_root, old_revision=old_revision, new_revision=revision)

    new_paths = frozenset(path for path, _ in new_manifest.files)
    lock_files = {UV_LOCK_FILE} if uses_uv_lock else set()
    policy = PullRequestPolicy(
        base_branch=base_branch,
        head_branch=MARIN_STYLE_BRANCH,
        title=MARIN_STYLE_TITLE,
        allowed_files=frozenset({*pin_files, *lock_files, *old_paths, *new_paths, MARIN_STYLE_MANIFEST}),
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


def _preflight(repository: str, base_branch: str) -> None:
    owner, separator, name = repository.partition("/")
    if owner != ORGANIZATION or separator != "/" or not name or "/" in name:
        raise ValueError(f"invalid consumer repository: {repository!r}")
    actual_branch = _run(
        "gh",
        "repo",
        "view",
        repository,
        "--json",
        "defaultBranchRef",
        "--jq",
        ".defaultBranchRef.name",
    ).strip()
    if actual_branch != base_branch:
        raise ValueError(f"unexpected default branch for {repository}: {actual_branch!r}")
    for label in DEPENDENCY_UPDATE_LABELS:
        _run("gh", "api", f"repos/{repository}/labels/{label}")


def update_consumer(
    *,
    repository: str,
    base_branch: str,
    revision: str,
    merge_mode: MergeMode,
    app_slug: str,
    manifest_mode: ManifestMode,
) -> ConsumerUpdateResult:
    """Return whether the consumer is current, published, or merged."""
    if manifest_mode is ManifestMode.BOOTSTRAP and merge_mode is MergeMode.MERGE:
        raise ValueError("the manifest bootstrap requires human review")
    _preflight(repository, base_branch)
    repo_root = Path.cwd()
    old_revision = _old_revision(repo_root)
    initial_policy = PullRequestPolicy(
        base_branch=base_branch,
        head_branch=MARIN_STYLE_BRANCH,
        title=MARIN_STYLE_TITLE,
        allowed_files=frozenset(),
    )
    branch = prepare_update_branch(policy=initial_policy, repository=repository)
    if old_revision == revision and manifest_mode is ManifestMode.VALIDATE:
        if _checked_in_manifest(repo_root, expected_revision=revision) is None:
            raise ValueError("consumer has no marin-style manifest; run the reviewed bootstrap first")
        if branch.pull_request_url:
            raise ValueError(f"consumer is current but has an open automation PR: {branch.pull_request_url}")
        return ConsumerUpdateResult(status=ConsumerUpdateStatus.CURRENT, pull_request_url="")

    update = generate_marin_style_update(
        repo_root=repo_root,
        base_branch=base_branch,
        revision=revision,
        manifest_mode=manifest_mode,
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        body_file = Path(temp_dir) / "marin-style-update.md"
        body_file.write_text(PR_BODY.format(revision=revision))
        pull_request = publish_update(
            policy=update.policy,
            repository=repository,
            body_file=body_file,
            expected_remote_sha=branch.expected_remote_sha,
            pull_request_url=branch.pull_request_url,
            push_mode=branch.push_mode,
        )
    if merge_mode is MergeMode.MERGE:
        merge_when_protected_checks_green(
            pr=pull_request.url,
            repository=repository,
            app_slug=app_slug,
            policy=update.policy,
            expected_head_sha=pull_request.head_sha,
            timeout=2400,
            poll_interval=30,
        )
    status = ConsumerUpdateStatus.MERGED if merge_mode is MergeMode.MERGE else ConsumerUpdateStatus.PUBLISHED
    return ConsumerUpdateResult(status=status, pull_request_url=pull_request.url)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    matrix = subparsers.add_parser("matrix", help="print installed consumer repositories")
    matrix.add_argument("--consumer", default="")
    run = subparsers.add_parser("run", help="update one consumer checkout")
    run.add_argument("--repository", required=True)
    run.add_argument("--base-branch", required=True)
    run.add_argument("--revision", required=True)
    run.add_argument("--app-slug", required=True)
    run.add_argument("--merge-mode", type=MergeMode, choices=MergeMode, required=True)
    run.add_argument("--manifest-mode", type=ManifestMode, choices=ManifestMode, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "matrix":
        print(installed_consumer_matrix_json(args.consumer))
        return
    result = update_consumer(
        repository=args.repository,
        base_branch=args.base_branch,
        revision=args.revision,
        merge_mode=MergeMode(args.merge_mode),
        app_slug=args.app_slug,
        manifest_mode=ManifestMode(args.manifest_mode),
    )
    if result.status is ConsumerUpdateStatus.CURRENT:
        print(f"{args.repository} already pins {args.revision}")
    else:
        print(f"{result.status}: {result.pull_request_url}")


if __name__ == "__main__":
    main()
