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
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, sealed_contract_rows


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
        "contracts": ("contracts-default.json", {"contracts": []}),
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
        removed_sources=("Test:Gone",),
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


def test_verify_no_prior_drift_allows_only_planned_removals(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "fingerprint_input", lambda row: json.dumps(row, sort_keys=True))
    prior = {"Test:Kept": {"id": "Test:Kept"}, "Test:Gone": {"id": "Test:Gone"}, "Test:One": {"id": "old"}}
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(json.dumps(prior))
    current = {"Test:Kept": {"id": "Test:Kept"}, "Test:One": {"id": "new"}, "Test:Two": {"id": "Test:Two"}}
    sp.verify_no_prior_drift(_inputs(tmp_path), current, str(prior_path))
    with pytest.raises(ValueError, match="non-target declaration drift: \\['Test:Gone'\\]"):
        sp.verify_no_prior_drift(_inputs(tmp_path, removed_sources=()), current, str(prior_path))
    with pytest.raises(ValueError, match="planned removals are still declared"):
        sp.verify_no_prior_drift(_inputs(tmp_path), {**current, "Test:Gone": {"id": "Test:Gone"}}, str(prior_path))


def test_verify_survey_alignment_refuses_mismatch(tmp_path):
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps({"fingerprints": {"Test:One": {}, "Test:Two": {}, "Test:Three": {}}}))
    identity = ArtifactRef(uri=str(identity_path), sha256=hashlib.sha256(identity_path.read_bytes()).hexdigest())
    inputs = _inputs(tmp_path, refs={"identity_receipt": identity})
    with pytest.raises(ValueError, match="fingerprints"):
        sp.verify_survey_alignment(inputs, {"Test:One": {}, "Test:Two": {}})


@pytest.mark.parametrize(
    ("status", "population", "accepted"),
    [("Excluded", None, True), ("Available", None, False), ("Excluded", {"measured_tasks": 3}, False)],
)
def test_identity_coverage_allows_only_unpopulated_excluded_inventory(tmp_path, status, population, accepted):
    current = {
        "Test:One": {"status": "Available", "actor_census_input": {"measured_tasks": 2}},
        "Test:Unfingerprinted": {"status": status, "actor_census_input": population},
    }
    binding = {"catalog_root": "s3://test/catalog", "snapshot_id": "assembly-test-v1"}
    documents = {
        "identity_receipt": {**binding, "fingerprints": {"Test:One": {"task_count": 2}}},
        "backend_receipt": {**binding, "sources": [{"source_id": source_id} for source_id in current]},
    }
    refs = {name: _write_json(tmp_path, name + ".json", document) for name, document in documents.items()}
    inputs = _inputs(tmp_path, refs=refs)
    if not accepted:
        with pytest.raises(ValueError, match=r"missing.*Test:Unfingerprinted"):
            sp.verify_survey_alignment(inputs, current)
        return
    assert sp.verify_survey_alignment(inputs, current) == (documents["identity_receipt"], documents["backend_receipt"])


@pytest.mark.parametrize(
    ("survey", "field"),
    [
        (None, None),
        ("identity_receipt", "snapshot_id"),
        ("backend_receipt", "snapshot_id"),
        ("identity_receipt", "catalog_root"),
        ("backend_receipt", "catalog_root"),
    ],
)
def test_survey_publication_binding_matches_the_destination(tmp_path, survey, field):
    current = {"Test:One": {}, "Test:Two": {}}
    binding = {"catalog_root": "s3://test/catalog", "snapshot_id": "assembly-test-v1"}
    documents = {
        "identity_receipt": {**binding, "fingerprints": current},
        "backend_receipt": {**binding, "sources": [{"source_id": sid} for sid in current]},
    }
    if survey is not None:
        documents[survey][field] = "another-publication"
    refs = {name: _write_json(tmp_path, name + ".json", document) for name, document in documents.items()}
    inputs = _inputs(tmp_path, refs=refs)
    if survey is not None:
        with pytest.raises(ValueError, match="publication binding"):
            sp.verify_survey_alignment(inputs, current)
        return
    assert sp.verify_survey_alignment(inputs, current) == (documents["identity_receipt"], documents["backend_receipt"])


def test_seal_contracts_refuses_uncensused_or_unregistered_targets(tmp_path, monkeypatch):
    """The census covers the whole population; every target needs a census row and a receipt."""
    one_only = _write_json(tmp_path, "census-one.json", {"results": [{"declaration": {"id": "Test:One"}}]})
    with pytest.raises(ValueError, match=r"targets missing from the census: \['Test:Two'\]"):
        sp.seal_contracts(_inputs(tmp_path, refs={"census_receipt": one_only}))

    population = _write_json(
        tmp_path,
        "census-population.json",
        {"results": [{"declaration": {"id": source}} for source in ("Test:One", "Test:Two", "Test:Other")]},
    )
    with pytest.raises(ValueError, match=r"targets without a derived-contract receipt: \['Test:One', 'Test:Two'\]"):
        sp.seal_contracts(_inputs(tmp_path, refs={"census_receipt": population}))


def test_seal_contracts_verifies_actor_binding_and_seals(tmp_path, monkeypatch):
    census = _write_json(
        tmp_path,
        "census.json",
        {
            "results": [
                {"declaration": {"id": "Test:One", "actor_census_input": {"x": 1, "measured_tasks": 5}}},
                {"declaration": {"id": "Test:Two", "actor_census_input": {"x": 2, "measured_tasks": 5}}},
                {"declaration": {"id": "Test:Other", "actor_census_input": {"x": 3, "measured_tasks": 7}}},
            ]
        },
    )
    contracts = _write_json(
        tmp_path,
        "contracts.json",
        {
            "contracts": [
                {"source_id": "Test:One", "input": {"x": 1, "measured_tasks": 5}},
                {"source_id": "Test:Two", "input": {"x": 2, "measured_tasks": 5}},
            ]
        },
    )
    inputs = _inputs(tmp_path, refs={"census_receipt": census, "contracts": contracts})
    binding_ref = sp.seal_contracts(inputs)
    sealed = json.loads((tmp_path / "releases/rtest/fresh-contracts.json").read_text())
    # The catalog stages read the sealed document through the same helper.
    assert [row["input"]["x"] for row in sealed_contract_rows(sealed)] == [1, 2]
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


def test_deliberate_exclusions_carry_the_served_declaration(tmp_path, monkeypatch):
    """A deliberately excluded source's served declaration substitutes into the
    publication's checked current (and would publish verbatim) — the frozen layer's
    re-pin is not adopted, and the carried document's digest is what the proof seals."""
    repo = tmp_path / "repo"
    repo.mkdir()
    head = _init_repo(repo)
    served_row = {"dataset_id": "served", "dataset_revision": "r1", "pipeline": None}
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(json.dumps({"Test:One": served_row}))
    inputs = _inputs(tmp_path, source_revision=head, deliberate_exclusions=("Test:One",))
    frozen_current = {
        "Test:One": {"dataset_id": "repinned", "dataset_revision": "r2", "pipeline": None},
        "Test:Two": {"dataset_id": "two", "dataset_revision": "r", "pipeline": None},
    }
    monkeypatch.setattr(sp, "verify_current_declarations", lambda inputs: dict(frozen_current))
    monkeypatch.setattr(sp, "verify_no_prior_drift", lambda inputs, current, prior: None)
    monkeypatch.setattr(sp, "verify_survey_alignment", lambda inputs, current: ({}, {}))
    proof = sp.run_snapshot_publication(inputs, check_only=True, repo_root=repo, prior_declarations_uri=str(prior_path))
    carried = {**frozen_current, "Test:One": served_row}
    expected = hashlib.sha256((json.dumps(carried, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
    assert proof["declarations_sha256"] == expected
