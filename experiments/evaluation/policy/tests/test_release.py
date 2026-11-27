# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A release pins one compliant run per baseline and evaluation, is written once, and a patch keeps unchanged runs."""

import hashlib
import subprocess
from dataclasses import replace

import pytest
from evalrecords.policy_release import PolicyRelease, ReleasedRun, Verdict, release_tag
from rigging.config_discovery import find_project_root

from experiments.evaluation.policy import release
from experiments.evaluation.policy.launch import model_evaluations
from experiments.evaluation.policy.runs import RunCheck
from experiments.evaluation.policy.tests.test_runs import POLICY, QWEN_NAME, _archive, _runs, _write

TWO = replace(
    POLICY,
    baselines=(QWEN_NAME,),
    evaluations=tuple(evaluation for evaluation in POLICY.evaluations if evaluation.name in {"aime24", "gsm8k-0shot"}),
)


@pytest.fixture
def archived(monkeypatch):
    """Treat every finished run with an archive as compliant, as the archive checks would for a good run."""

    def compliant(runs, expected):
        finished = [run for run in runs if run.record is not None]
        if not finished:
            return None
        run = finished[-1]
        return run, RunCheck(record=run.record, state=Verdict.VALID, problems=(), warnings=())

    monkeypatch.setattr(release, "compliant_run", compliant)


def _released(run, record, run_id: str | None = None) -> ReleasedRun:
    return ReleasedRun(
        model=QWEN_NAME,
        evaluation=run.name,
        run_id=run.run_id if run_id is None else run_id,
        records_prefix=run.batch.records_prefix,
        commit=record.provenance.git_sha,
        record_sha256="0" * 64,
        archive_version="1",
        scores={},
    )


def test_a_release_pins_each_compliant_run_and_lists_pairs_without_one(qwen, records, archived):
    run = _runs(qwen, {"aime24"})["aime24"]
    record = _write(run)
    _archive(record.results_path, [(1000, 10)])
    candidate = release.release_candidate(TWO, find_project_root())
    (pinned,) = candidate.release.runs
    with open(run.record_path, "rb") as handle:
        assert pinned.record_sha256 == hashlib.sha256(handle.read()).hexdigest()
    assert (pinned.evaluation, pinned.run_id) == ("aime24", run.run_id)
    assert candidate.missing == (f"{QWEN_NAME} gsm8k-0shot: no compliant full run in HEAD's history",)


def test_a_patch_keeps_the_base_runs_whose_inputs_changed_only_in_version(qwen, records, monkeypatch):
    base = replace(TWO, version="marin-base")
    plan = model_evaluations(TWO, qwen)
    runs = _runs(qwen, {"aime24", "gsm8k-0shot"}, policy=base)
    record = _write(runs["aime24"])
    # gsm8k-0shot's released run has other inputs, as after its benchmark's VERSION was bumped.
    changed = runs["gsm8k-0shot"].run_id.replace(runs["gsm8k-0shot"].base, f"{runs['gsm8k-0shot'].base[:-1]}0")
    released = PolicyRelease(
        version="marin-base",
        commit=record.provenance.git_sha,
        base=None,
        runs=(_released(runs["aime24"], record), _released(runs["gsm8k-0shot"], record, run_id=changed)),
    )
    monkeypatch.setattr(release, "read_release", lambda version, _root: released)
    patch = replace(TWO, version="marin-patch", base="marin-base")
    carried = release.carried_runs(patch, qwen, plan, find_project_root())
    assert set(carried) == {"aime24"} and carried["aime24"].carried_from == "marin-base"


def _git(*args, cwd):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A checkout with an ``origin`` remote, and manifests under a temporary root."""
    origin, checkout = tmp_path / "origin.git", tmp_path / "checkout"
    _git("init", "-q", "--bare", str(origin), cwd=tmp_path)
    _git("init", "-q", str(checkout), cwd=tmp_path)
    (checkout / "a.txt").write_text("a\n")
    _git("add", ".", cwd=checkout)
    _git("commit", "-qm", "a", cwd=checkout)
    _git("remote", "add", "origin", str(origin), cwd=checkout)
    monkeypatch.setattr(release, "release_path", lambda version: str(tmp_path / "releases" / f"{version}.json"))
    (tmp_path / "releases").mkdir()
    return checkout


def test_a_release_is_written_once_and_tagged_and_a_retry_finishes_it(repo):
    head = _git("rev-parse", "HEAD", cwd=repo)
    manifest = PolicyRelease(version="marin-x", commit=head, base=None, runs=())
    digest = release.publish(manifest, repo)
    assert release.tagged_commit("marin-x", repo) == head
    assert release.read_release("marin-x", repo) == manifest
    assert release.publish(manifest, repo) == digest

    other = manifest.model_copy(update={"base": "marin-w"})
    with pytest.raises(ValueError, match="another manifest"):
        release.publish(other, repo)
    assert _git("ls-remote", "origin", f"refs/tags/{release_tag('marin-x')}^{{}}", cwd=repo).startswith(head)


def test_an_unreleased_version_has_no_release(repo):
    assert release.tagged_commit("marin-y", repo) is None
    with pytest.raises(ValueError, match="not released"):
        release.read_release("marin-y", repo)
