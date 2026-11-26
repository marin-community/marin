# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure a train publication's stored grader against current main's recipes.

A publication sealed at ``converter_revision`` stores the grader and verifier
files its recipe produced. When main later fixes those recipe files, the stored
grader predates the fix and nothing on the release branch can see it: the
``pipeline_reference == original_pipeline`` checks compare a release only with
its own branch's declaration, and the branch has not merged main. This module
measures both sides with git alone — the recipe files' blob hashes at the
sealed converter revision and at origin/main — and compares them into one typed
per-source finding, ``grader_stale_vs_main``, carrying the main fix commits and
PRs that touch the changed files.

The measurement is advisory by ruling (2026-11-26 6:45 AM EST): grader fixes
must not block training, so a stale source queues for re-seal instead of
refusing the release.
"""

import subprocess
from typing import Literal

from finestore.rl.curation import SourceUse
from finestore.rl.status_taxonomy import SourceStatus
from pydantic import BaseModel, ConfigDict, Field

DRIFT_FINDING = "grader_stale_vs_main"
"""The typed finding a stale publication carries."""

DRIFT_STATUSES = frozenset({SourceStatus.AVAILABLE.value, SourceStatus.PENDING.value})
"""The catalog statuses the drift measurement covers."""

DRIFT_USE = SourceUse.TRAIN.value
"""Only train publications queue for re-seal; eval graders refit at serving time."""


class FixCommit(BaseModel):
    """One main commit that touched a changed recipe file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sha: str
    subject: str
    pr: str | None = None
    """The PR number parsed from the subject's ``(#123)`` trailer, when present."""


class ChangedFile(BaseModel):
    """One recipe file whose sealed bytes differ from main's."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    sealed_blob: str
    main_blob: str | None = None
    """None when main no longer carries the path at all."""


class SealedRecipeReference(BaseModel):
    """One source's stored recipe side: the converter revision and its file blobs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    converter_revision: str | None = None
    """The git revision whose converter produced the stored publication; None when
    the source has no sealed census binding yet (nothing to compare)."""
    converter_version: str | None = None
    files: dict[str, str] = Field(default_factory=dict)
    """Repo path -> git blob hash at ``converter_revision``, for the declared recipe files."""


class MainRecipeReference(BaseModel):
    """Main's recipe side at the measured revision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    revision: str
    """origin/main's commit at measurement time."""
    merge_base: str
    """The merge base with the release branch: main's commits since this are the
    fixes the branch cannot see."""
    files: dict[str, str] = Field(default_factory=dict)
    """Repo path -> git blob hash at ``revision``; a path main no longer carries is absent."""
    commits: dict[str, list[FixCommit]] = Field(default_factory=dict)
    """Repo path -> the commits on main since ``merge_base`` that touch the path."""


class GraderDriftReference(BaseModel):
    """The measured comparison basis one release derives against.

    The plan derivation measures both sides at the freeze and pins this
    reference; the graph stage verifies it binds the sealed declarations and
    compares it per source, so the release process itself needs no git and no
    network.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    main: MainRecipeReference
    sources: dict[str, SealedRecipeReference]
    """Every declared source's sealed recipe side, keyed by canonical source id."""


class GraderDriftFinding(BaseModel):
    """One stale publication and the main fixes it predates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    finding: Literal["grader_stale_vs_main"]
    converter_revision: str
    converter_version: str | None
    changed_files: list[ChangedFile]
    fix_commits: list[FixCommit]
    """The main commits touching the changed files since the merge base, with PRs."""


def is_drift_population(declaration: dict) -> bool:
    """Whether the declaration is an Available or Pending train source with a sealed publication."""
    return (
        declaration.get("status") in DRIFT_STATUSES
        and declaration.get("intended_use") == DRIFT_USE
        and declaration.get("actor_census_input") is not None
    )


def compare_recipe(
    source_id: str,
    sealed: SealedRecipeReference,
    main: MainRecipeReference,
) -> GraderDriftFinding | None:
    """The drift finding for one sealed recipe, or None when main agrees with it.

    The converter version rides the finding for context; the blob comparison is
    the drift test, because recipe version constants live in the recipe files
    themselves — a version bump without a file change cannot change the stored
    grader.
    """
    if sealed.converter_revision is None:
        return None
    changed = [
        ChangedFile(path=path, sealed_blob=sealed_blob, main_blob=main.files.get(path))
        for path, sealed_blob in sorted(sealed.files.items())
        if main.files.get(path) != sealed_blob
    ]
    if not changed:
        return None
    fix_commits: list[FixCommit] = []
    seen: set[str] = set()
    for row in changed:
        for commit in main.commits.get(row.path, ()):
            if commit.sha not in seen:
                seen.add(commit.sha)
                fix_commits.append(commit)
    return GraderDriftFinding(
        source_id=source_id,
        finding=DRIFT_FINDING,
        converter_revision=sealed.converter_revision,
        converter_version=sealed.converter_version,
        changed_files=changed,
        fix_commits=fix_commits,
    )


def _git(repo_root, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"git {' '.join(args)} failed at {repo_root}: {result.stderr.strip()[:300]}")
    return result.stdout.strip()


def head_revision(repo_root) -> str:
    """The checkout's own HEAD revision (the release branch side of the merge base)."""
    return _git(repo_root, "rev-parse", "HEAD")


def _blob_at(repo_root, revision: str, path: str) -> str | None:
    """The file's git blob hash at a revision; None when the revision lacks the path."""
    check = subprocess.run(["git", "cat-file", "-e", f"{revision}:{path}"], cwd=repo_root, capture_output=True)
    if check.returncode != 0:
        return None
    return _git(repo_root, "rev-parse", f"{revision}:{path}")


def measure_main_recipe_reference(
    repo_root,
    *,
    source_revision: str,
    paths,
    main_ref: str = "origin/main",
    fetch: bool = True,
) -> MainRecipeReference:
    """Measure main's recipe side: its revision, file blobs and the unseen fix commits.

    ``fetch`` updates the remote ref first so the measurement is current; the
    derivation fetches fail-fast because a release cannot derive its drift
    reference against a stale main. Tests point ``main_ref`` at a local branch
    with ``fetch=False``.
    """
    if fetch:
        _git(repo_root, "fetch", "origin", "main")
        main_ref = "FETCH_HEAD"
    revision = _git(repo_root, "rev-parse", main_ref)
    merge_base = _git(repo_root, "merge-base", source_revision, revision)
    files = {path: _blob_at(repo_root, revision, path) for path in paths}
    files = {path: blob for path, blob in files.items() if blob is not None}
    commits: dict[str, list[FixCommit]] = {}
    for path in sorted(files):
        rows = _git(repo_root, "log", f"{merge_base}..{revision}", "--format=%H%x00%s", "--", path).splitlines()
        commits[path] = [
            FixCommit(sha=sha, subject=subject, pr=pr)
            for sha, subject, pr in (_parse_commit_row(row) for row in rows if row.strip())
        ]
    return MainRecipeReference(revision=revision, merge_base=merge_base, files=files, commits=commits)


def _parse_commit_row(row: str) -> tuple[str, str, str | None]:
    sha, _, subject = row.partition("\x00")
    pr = None
    if subject.endswith(")") and "(#" in subject:
        candidate = subject[subject.rindex("(#") + 2 : -1]
        if candidate.isdigit():
            pr = candidate
    return sha, subject, pr


def measure_sealed_recipe(source_id: str, declaration: dict, repo_root) -> SealedRecipeReference:
    """Measure one source's stored recipe side from its sealed declaration document."""
    binding = declaration.get("actor_census_input")
    pipeline = declaration.get("pipeline")
    revision = binding.get("converter_revision") if binding else None
    version = binding.get("converter_version") if binding else None
    paths = list((pipeline or {}).get("files") or [])
    if revision is None:
        return SealedRecipeReference(source_id=source_id)
    files = {path: _blob_at(repo_root, revision, path) for path in paths}
    files = {path: blob for path, blob in files.items() if blob is not None}
    return SealedRecipeReference(
        source_id=source_id, converter_revision=revision, converter_version=version, files=files
    )


def measure_grader_drift_reference(
    repo_root,
    *,
    source_revision: str,
    declared: dict[str, dict],
    main_ref: str = "origin/main",
    fetch: bool = True,
) -> GraderDriftReference:
    """Measure the whole comparison basis: main's side plus every declared source's sealed side."""
    paths = sorted(
        {path for declaration in declared.values() for path in (declaration.get("pipeline") or {}).get("files") or []}
    )
    main = measure_main_recipe_reference(
        repo_root, source_revision=source_revision, paths=paths, main_ref=main_ref, fetch=fetch
    )
    sources = {
        source_id: measure_sealed_recipe(source_id, declaration, repo_root)
        for source_id, declaration in sorted(declared.items())
    }
    return GraderDriftReference(main=main, sources=sources)
