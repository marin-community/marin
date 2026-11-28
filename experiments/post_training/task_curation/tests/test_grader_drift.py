# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the main-grader-drift stage (R41, ruling 2026-11-26 6:45 AM EST).

A train publication sealed at ``converter_revision`` stores the grader and
verifier files its recipe produced. The tests run the real measurement over a
local git fixture — one fixed recipe and one unfixed — and pin the stage's
typed findings, its advisory character (drift never blocks), and its fail-closed
binding to the sealed declarations.
"""

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation import grader_drift, source_status
from experiments.post_training.task_curation.grader_drift import (
    FixCommit,
    GraderDriftReference,
    MainRecipeReference,
    compare_recipe,
)
from experiments.post_training.task_curation.release_stages import main_grader_drift as drift_stage
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef

RECIPE = "datasets/recipe/grader.py"
VERIFIER = "datasets/recipe/verifier.py"
STALE = "Test:stale"
FIXED = "Test:fixed"
EVAL = "Test:eval-source"
UNBOUND = "Test:unbound-source"
FIX_SUBJECT = "Fix the reasoning gym grader (#12329)"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return result.stdout.strip()


@dataclass(frozen=True)
class RecipeRepo:
    """The git fixture: a sealed recipe commit and main one grader fix ahead."""

    root: Path
    sealed_revision: str
    main_revision: str

    def blob(self, revision: str, path: str) -> str | None:
        return grader_drift._blob_at(self.root, revision, path)


@pytest.fixture
def recipe_repo(tmp_path: Path) -> RecipeRepo:
    repo = tmp_path / "repo"
    repo.mkdir()

    def write(path: str, content: str) -> None:
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "fixture@example.org")
    _git(repo, "config", "user.name", "fixture")
    write(RECIPE, "SEALED\n")
    write(VERIFIER, "VERIFY\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "seal the publication")
    sealed = _git(repo, "rev-parse", "HEAD")
    write(RECIPE, "FIXED\n")
    _git(repo, "rm", "-q", VERIFIER)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", FIX_SUBJECT)
    return RecipeRepo(root=repo, sealed_revision=sealed, main_revision=_git(repo, "rev-parse", "HEAD"))


def _declaration(
    source_id: str,
    *,
    status: str = "Available",
    use: str = "train",
    revision: str | None,
    version: str | None,
    files: tuple[str, ...],
) -> dict:
    declaration = {"id": source_id, "status": status, "intended_use": use, "pipeline": {"files": list(files)}}
    if revision is not None:
        declaration["actor_census_input"] = {"converter_revision": revision, "converter_version": version}
    return declaration


def _declared(repo: RecipeRepo) -> dict[str, dict]:
    return {
        STALE: _declaration(STALE, revision=repo.sealed_revision, version="2026.11.1.0", files=(RECIPE, VERIFIER)),
        FIXED: _declaration(FIXED, revision=repo.main_revision, version="2026.11.2.0", files=(RECIPE,)),
        EVAL: _declaration(EVAL, use="eval", revision=repo.sealed_revision, version="2026.11.1.0", files=(RECIPE,)),
        UNBOUND: _declaration(UNBOUND, revision=None, version=None, files=(RECIPE,)),
    }


def _reference(repo: RecipeRepo, declared: dict[str, dict] | None = None) -> GraderDriftReference:
    return grader_drift.measure_grader_drift_reference(
        repo.root,
        source_revision=repo.sealed_revision,
        declared=declared or _declared(repo),
        main_ref="main",
        fetch=False,
    )


def test_measurement_reads_both_sides_from_git(recipe_repo: RecipeRepo):
    """The measurement reads the sealed blobs at the converter revision and main's blobs."""
    reference = _reference(recipe_repo)
    assert reference.main.revision == recipe_repo.main_revision
    assert reference.main.merge_base == recipe_repo.sealed_revision
    assert reference.main.files[RECIPE] == recipe_repo.blob(recipe_repo.main_revision, RECIPE)
    assert VERIFIER not in reference.main.files
    stale = reference.sources[STALE]
    assert stale.converter_revision == recipe_repo.sealed_revision
    assert stale.files[RECIPE] == recipe_repo.blob(recipe_repo.sealed_revision, RECIPE)
    assert reference.sources[UNBOUND].converter_revision is None
    assert reference.main.commits[RECIPE] == [FixCommit(sha=recipe_repo.main_revision, subject=FIX_SUBJECT, pr="12329")]


def test_comparison_flags_the_unfixed_source_with_its_fix_commits(recipe_repo: RecipeRepo):
    """The unfixed publication carries grader_stale_vs_main with the main fix commits; the fixed one is clean."""
    reference = _reference(recipe_repo)
    stale = compare_recipe(STALE, reference.sources[STALE], reference.main)
    assert stale is not None and stale.finding == "grader_stale_vs_main"
    assert [(row.path, row.main_blob) for row in stale.changed_files] == [
        (RECIPE, reference.main.files[RECIPE]),
        (VERIFIER, None),
    ]
    assert stale.fix_commits == reference.main.commits[RECIPE]
    assert stale.converter_version == "2026.11.1.0"
    assert compare_recipe(FIXED, reference.sources[FIXED], reference.main) is None
    assert compare_recipe(UNBOUND, reference.sources[UNBOUND], reference.main) is None


def _seal_declarations(tmp_path: Path, declared: dict[str, dict]) -> ArtifactRef:
    path = tmp_path / "declarations.json"
    path.write_text(json.dumps(declared, indent=2, sort_keys=True) + "\n")
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _inputs(
    tmp_path: Path, repo: RecipeRepo, declared: dict[str, dict], reference: GraderDriftReference
) -> drift_stage.GraderDriftInputs:
    return drift_stage.GraderDriftInputs(
        release_id="testrelease",
        source_revision=repo.sealed_revision,
        declarations=_seal_declarations(tmp_path, declared),
        reference=reference,
        output_root=str(tmp_path / "release" / "grader-drift"),
    )


def test_stage_seals_the_advisory_queue(recipe_repo: RecipeRepo, tmp_path: Path):
    """One fixed and one unfixed fixture: the queue holds exactly the stale source."""
    declared = _declared(recipe_repo)
    inputs = _inputs(tmp_path, recipe_repo, declared, _reference(recipe_repo))
    proof = drift_stage.run_main_grader_drift(inputs, dry_run=True)
    assert proof["stale_sources"] == 1 and proof["clean_sources"] == 1 and proof["skipped_sources"] == 2
    finding = proof["grader_drift"][0]
    assert finding["source_id"] == STALE and finding["finding"] == "grader_stale_vs_main"
    assert finding["fix_commits"][0]["pr"] == "12329"

    proof = drift_stage.run_main_grader_drift(inputs, dry_run=False)
    document = json.loads(StoragePath(proof["document"]["uri"]).read_text())
    assert document["advisory"] is True and document["finding_type"] == "grader_stale_vs_main"
    assert [row["source_id"] for row in document["findings"]] == [STALE]
    assert document["skipped_sources"] == [EVAL, UNBOUND]
    with pytest.raises(ValueError, match="already exists"):
        drift_stage.run_main_grader_drift(inputs, dry_run=False)


def test_stage_is_advisory_when_every_train_source_is_stale(recipe_repo: RecipeRepo, tmp_path: Path):
    """Drift never blocks: a fully stale estate still seals the queue (ruling: fixes must not block training)."""
    declared = _declared(recipe_repo)
    declared[FIXED] = _declaration(
        FIXED, revision=recipe_repo.sealed_revision, version="2026.11.1.0", files=(RECIPE, VERIFIER)
    )
    inputs = _inputs(tmp_path, recipe_repo, declared, _reference(recipe_repo, declared))
    proof = drift_stage.run_main_grader_drift(inputs, dry_run=True)
    assert proof["stale_sources"] == 2
    assert sorted(row["source_id"] for row in proof["grader_drift"]) == [FIXED, STALE]


def test_stage_fails_closed_on_a_reference_that_does_not_bind(recipe_repo: RecipeRepo, tmp_path: Path):
    """A pin that disagrees with the sealed declarations refuses the stage."""
    declared = _declared(recipe_repo)
    reference = _reference(recipe_repo)
    drifted_pin = reference.sources[STALE].model_copy(update={"converter_revision": "f" * 40})
    reference = reference.model_copy(update={"sources": {**reference.sources, STALE: drifted_pin}})
    with pytest.raises(ValueError, match="measures converter revision"):
        drift_stage.run_main_grader_drift(_inputs(tmp_path, recipe_repo, declared, reference), dry_run=True)


def test_stage_fails_closed_on_a_source_set_mismatch(recipe_repo: RecipeRepo, tmp_path: Path):
    declared = _declared(recipe_repo)
    reference = _reference(recipe_repo)
    partial = reference.model_copy(update={"sources": {STALE: reference.sources[STALE]}})
    with pytest.raises(ValueError, match="disagree on the source set"):
        drift_stage.run_main_grader_drift(_inputs(tmp_path, recipe_repo, declared, partial), dry_run=True)


def test_source_status_queues_the_stale_sources(recipe_repo: RecipeRepo, monkeypatch):
    """The extras re-seal queue carries the stale source with its display name and fix PRs."""
    declared = _declared(recipe_repo)
    reference = _reference(recipe_repo)

    def fake_source(source_id: str):
        declaration = declared[source_id]
        return SimpleNamespace(
            info=SimpleNamespace(id=source_id),
            files=tuple((declaration.get("pipeline") or {}).get("files") or ()),
            declaration=SimpleNamespace(model_dump=lambda mode=None, _declaration=declaration: _declaration),
        )

    monkeypatch.setattr(
        source_status, "all_sources", lambda: {sid: fake_source(sid) for sid in (STALE, FIXED, EVAL, UNBOUND)}
    )
    rows = source_status.grader_drift_rows(reference.main, repo_root=recipe_repo.root)
    assert [row["id"] for row in rows] == [STALE]
    assert rows[0]["display_name"] == "Test · stale"
    assert [file["path"] for file in rows[0]["changed_files"]] == [RECIPE, VERIFIER]
    assert rows[0]["fix_commits"][0]["pr"] == "12329"
    assert rows[0]["main_revision"] == recipe_repo.main_revision


def test_source_status_never_queues_without_a_sealed_binding(monkeypatch):
    """A source with no census binding has no stored grader to compare and never queues."""
    declaration = _declaration(UNBOUND, revision=None, version=None, files=(RECIPE,))
    source = SimpleNamespace(
        info=SimpleNamespace(id=UNBOUND),
        files=(RECIPE,),
        declaration=SimpleNamespace(model_dump=lambda mode=None: declaration),
    )
    monkeypatch.setattr(source_status, "all_sources", lambda: {UNBOUND: source})
    assert source_status.grader_drift_rows(MainRecipeReference(revision="e" * 40, merge_base="a" * 40)) == []
