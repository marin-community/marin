# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the maintained snapshot-publication stage (R40 gate slice 3)."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from experiments.post_training.task_curation.release_stages import guard
from experiments.post_training.task_curation.release_stages import snapshot_publication as sp
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef


def _write_json(tmp_path, name, document) -> ArtifactRef:
    path = tmp_path / name
    path.write_text(json.dumps(document))
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _inputs(tmp_path, **overrides) -> sp.SnapshotPublicationInputs:
    refs = overrides.pop("refs", {})
    default_documents = {
        "census_receipt": ("census-default.json", {"results": []}),
        "identity_receipt": ("identity-default.json", {"fingerprints": {}}),
        "backend_receipt": ("backend-default.json", {"sources": []}),
    }
    defaults = {
        name: refs.get(name, _write_json(tmp_path, filename, document))
        for name, (filename, document) in default_documents.items()
    }
    base = dict(
        release_id="rtest",
        snapshot_id="assembly-test-v1",
        prior_snapshot_id="assembly-prior-v1",
        source_revision="f" * 40,
        catalog_root="s3://test/catalog",
        release_root=str(tmp_path / "releases/rtest"),
        targets=("Test:One", "Test:Two"),
        expected_source_count=2,
        expected_contract_count=2,
        expected_contract_tasks=10,
        **defaults,
    )
    return sp.SnapshotPublicationInputs(**{**base, **overrides})


def _init_repo(root: Path) -> str:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

    git("init", "--quiet")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    (root / "file.txt").write_text("one\n")
    git("add", "file.txt")
    git("commit", "--quiet", "-m", "one")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_verify_repo_state_refuses_dirty_checkout_and_wrong_revision(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    head = _init_repo(repo)
    inputs = _inputs(tmp_path, source_revision="a" * 40)
    with pytest.raises(ValueError, match="does not contain the plan's pinned revision"):
        sp.verify_repo_state(repo, inputs)
    inputs = _inputs(tmp_path, source_revision=head)
    (repo / "file.txt").write_text("dirty\n")
    with pytest.raises(ValueError, match="dirty"):
        sp.verify_repo_state(repo, inputs)


def test_verify_repo_state_attests_in_the_bundle_mode(tmp_path, monkeypatch):
    """The git-less Iris bundle returns the submitter-attested revision."""
    (tmp_path / "not-a-git-repo").mkdir()
    monkeypatch.setenv(guard.ATTESTED_REVISION_ENV, "b" * 40)
    inputs = _inputs(tmp_path, source_revision="b" * 40)
    assert sp.verify_repo_state(tmp_path / "not-a-git-repo", inputs) == "b" * 40


def test_verify_survey_alignment_refuses_mismatch(tmp_path):
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps({"fingerprints": {"Test:One": {}, "Test:Two": {}, "Test:Three": {}}}))
    identity = ArtifactRef(uri=str(identity_path), sha256=hashlib.sha256(identity_path.read_bytes()).hexdigest())
    inputs = _inputs(tmp_path, refs={"identity_receipt": identity})
    with pytest.raises(ValueError, match="fingerprints"):
        sp.verify_survey_alignment(inputs, {"Test:One": {}, "Test:Two": {}})


def test_seal_contracts_refuses_target_disagreement(tmp_path, monkeypatch):
    census_path = tmp_path / "census.json"
    census_path.write_text(json.dumps({"results": [{"declaration": {"id": "Test:One"}}]}))
    census = ArtifactRef(uri=str(census_path), sha256=hashlib.sha256(census_path.read_bytes()).hexdigest())
    inputs = _inputs(tmp_path, refs={"census_receipt": census})
    monkeypatch.setattr(sp, "derived_contract_receipts", lambda: {})
    with pytest.raises(ValueError, match="targets disagree"):
        sp.seal_contracts(inputs)


def test_seal_contracts_verifies_actor_binding_and_seals(tmp_path, monkeypatch):
    census = _write_json(
        tmp_path,
        "census.json",
        {
            "results": [
                {"declaration": {"id": "Test:One", "actor_census_input": {"x": 1, "measured_tasks": 5}}},
                {"declaration": {"id": "Test:Two", "actor_census_input": {"x": 2, "measured_tasks": 5}}},
            ]
        },
    )
    inputs = _inputs(tmp_path, refs={"census_receipt": census})

    def fake_receipt(binding: dict):
        return type(
            "Receipt",
            (),
            {
                "input": type("Input", (), {"model_dump": staticmethod(lambda mode=None: binding)})(),
                "model_dump": staticmethod(lambda mode=None: {"input": binding}),
            },
        )()

    monkeypatch.setattr(
        sp,
        "derived_contract_receipts",
        lambda: {
            "Test:One": fake_receipt({"x": 1, "measured_tasks": 5}),
            "Test:Two": fake_receipt({"x": 2, "measured_tasks": 5}),
        },
    )
    binding_ref = sp.seal_contracts(inputs)
    sealed = json.loads((tmp_path / "releases/rtest/fresh-contracts.json").read_text())
    assert [row["input"]["x"] for row in sealed["contracts"]] == [1, 2]
    assert (
        binding_ref.sha256 == hashlib.sha256((tmp_path / "releases/rtest/fresh-contracts.json").read_bytes()).hexdigest()
    )


def test_check_only_emits_alerts_and_zero_writes(tmp_path, monkeypatch):
    feed = tmp_path / "alerts.jsonl"
    repo = tmp_path / "repo"
    repo.mkdir()
    head = _init_repo(repo)
    inputs = _inputs(tmp_path, source_revision=head)
    monkeypatch.setattr(sp, "verify_current_declarations", lambda inputs: {"Test:One": {}, "Test:Two": {}})
    monkeypatch.setattr(sp, "verify_no_prior_drift", lambda inputs, current, prior: None)
    monkeypatch.setattr(sp, "verify_survey_alignment", lambda inputs, current: ({}, {}))
    proof = sp.run_snapshot_publication(
        inputs, check_only=True, repo_root=repo, prior_declarations_uri=str(tmp_path / "prior.json"), feeds=(feed,)
    )
    assert proof["sql_writes"] == 0
    assert proof["source_count"] == 2
    assert not (tmp_path / "releases/rtest").exists() or not list((tmp_path / "releases/rtest").iterdir())
    rows = [json.loads(line) for line in feed.read_text().splitlines()]
    assert [row["kind"] for row in rows] == ["STAGE_START", "STAGE_FINISH"]
