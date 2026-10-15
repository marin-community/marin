# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def review_modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    return importlib.import_module("make_review"), importlib.import_module("publish_review")


def _publication_files(root: Path, mode: str, verifyit_commit: str | None = None, harbor_commit: str | None = None):
    collection = {
        "schema_version": "0.3.0",
        "created_at": "2026-10-02T00:00:00Z",
        "subjects": [
            {
                "id": "source:test",
                "level": "source",
                "repository": "org/data",
                "dataset_revision": "data1",
                "source_id": "MarinSkyRL:test",
                "task_id": None,
                "task_path": None,
                "row_index": None,
            }
        ],
        "reviews": [],
        "tag_assignments": [],
        "source_mappings": [],
        "execution_provenance": {
            "marinskyrl_commit": "a" * 40,
            "marinskyrl_dirty": False,
            "marinskyrl_python_tree_sha256": "a" * 64,
            "harbor_commit": harbor_commit,
            "verifyit": (
                {
                    "version": "1",
                    "source_url": "https://github.com/marin-community/verifyit",
                    "source_commit": verifyit_commit,
                }
                if verifyit_commit is not None
                else None
            ),
        },
    }
    (root / "quality-reviews.json").write_text(json.dumps(collection))
    (root / "quality-review.schema.json").write_bytes(
        Path(__file__).with_name("quality-review.schema.json").read_bytes()
    )
    return {"origin": "MarinSkyRL", "revision": "a" * 40, "dataset_revision": "data1", "verifier_mode": mode}


def _add_runtime_reviews(root: Path, make_review, modes: list[bool | None]) -> None:
    evidence_path = root / "runtime-evidence.json"
    evidence_path.write_text("{}")
    evidence = make_review.evidence(evidence_path, root)
    path = root / "quality-reviews.json"
    collection = json.loads(path.read_text())
    for index, mode in enumerate(modes):
        review = make_review.review_record(
            f"runtime:{index}",
            "source:test",
            {"summary": "Native verification completed", "verdict": "unrated", "metrics": [], "findings": []},
            "runtime_execution",
            {
                "scope": "single_task",
                "sample_count": 1,
                "population_count": 1,
                "sampling_method": None,
                "seed": None,
                "samples": [],
            },
            True,
            [evidence],
            {"name": "test"},
        )
        review["attributes"] = {"verification": {"status": "verified"}}
        if mode is not None:
            review["attributes"]["verifyit_enabled"] = mode
        collection["reviews"].append(review)
    path.write_text(json.dumps(collection))


@pytest.mark.parametrize("executed", [None, "b" * 40, "c" * 40])
def test_publication_rejects_stale_verifyit_evidence_before_writing(tmp_path, monkeypatch, executed, review_modules):
    make_review, publish_review = review_modules
    payload = _publication_files(tmp_path, "verifyit", verifyit_commit=executed)
    _add_runtime_reviews(tmp_path, make_review, [True])
    payload["verifyit_revision"] = "c" * 40

    def atlas_sql(statement, _parameters):
        assert statement.startswith("SELECT"), "Stale evidence must never be published"
        return {"rows": [{"payload": payload}]}

    monkeypatch.setattr(publish_review, "sql", atlas_sql)
    if executed == payload["verifyit_revision"]:
        assert publish_review.validated_publication(tmp_path, "MarinSkyRL:test").payload == payload
    else:
        with pytest.raises(ValueError, match="Executed verifyit revision differs"):
            publish_review.publish_review(tmp_path, "MarinSkyRL:test")
    assert not (tmp_path / "atlas-publication.json").exists()


@pytest.mark.parametrize("modes", [[False], [None], [True, False], []])
def test_publication_rejects_verifyit_runtime_mode_before_writing(tmp_path, monkeypatch, modes, review_modules):
    make_review, publish_review = review_modules
    payload = _publication_files(tmp_path, "verifyit", verifyit_commit="c" * 40)
    _add_runtime_reviews(tmp_path, make_review, modes)
    payload["verifyit_revision"] = "c" * 40

    def atlas_sql(statement, _parameters):
        assert statement.startswith("SELECT"), "Invalid runtime evidence must never be published"
        return {"rows": [{"payload": payload}]}

    monkeypatch.setattr(publish_review, "sql", atlas_sql)
    with pytest.raises(ValueError, match="did not all enable verifyit"):
        publish_review.publish_review(tmp_path, "MarinSkyRL:test")
    assert not (tmp_path / "atlas-publication.json").exists()


def test_review_rejects_verifyit_environment_that_differs_from_selected_checkout(tmp_path, monkeypatch, review_modules):
    make_review, _ = review_modules
    skyrl = tmp_path / "skyrl"
    (skyrl / "skyrl-gym").mkdir(parents=True)
    (skyrl / "skyrl-gym/pyproject.toml").write_text(
        f'verifyit = "verifyit @ git+https://github.com/marin-community/verifyit@{"a" * 40}"\n'
    )
    task = make_review.Task("one", make_review.Route.GYM, "org/data", "data1", "source", "one", 0, [], {}, "aime", None)
    sample = make_review.TaskSample([task], 1, {("org/data", "data1", "source"): 1})
    execution = tmp_path / "tasks/0000/execution-0"
    execution.mkdir(parents=True)
    (execution / "native-code-index.json").write_text(
        json.dumps(
            [
                {
                    "package": {
                        "version": "1",
                        "source_url": "https://github.com/marin-community/verifyit",
                        "source_commit": "b" * 40,
                    }
                }
            ]
        )
    )
    monkeypatch.setattr(
        make_review, "attempt", lambda *_args: {"execution_path": "tasks/0000/execution-0", "verifyit_enabled": True}
    )
    identity = {"marinskyrl": {"commit": "c" * 40, "dirty": False, "python_tree_sha256": "c" * 64}}
    config = {
        "model": {},
        "runtime": {"marinskyrl_checkout": str(skyrl), "gym_config": {"aime": {"verifyit_enabled": True}}},
    }
    with pytest.raises(ValueError, match="Executed verifyit revision differs from the selected"):
        make_review.independent_reviews(sample, config, tmp_path, "snapshot", 1, 1, identity, {}, None)


def _git_commit(checkout: Path) -> str:
    subprocess.run(["git", "add", "."], cwd=checkout, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "Update fixture"],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()


def test_harbor_publication_accepts_unchanged_verifier_tree_and_rejects_changed_tree(
    tmp_path, monkeypatch, review_modules
):
    _, publish_review = review_modules
    checkout = tmp_path / "harbor"
    (checkout / "src/harbor/verifier").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=checkout, check=True, capture_output=True)
    verifier = checkout / "src/harbor/verifier/check.py"
    verifier.write_text("def check(): return True\n")
    original = _git_commit(checkout)
    (checkout / "README.md").write_text("Documentation change\n")
    executed = _git_commit(checkout)
    payload = _publication_files(tmp_path, "harbor", harbor_commit=executed)
    payload["harbor_verifier_revision"] = original
    (tmp_path / "run.json").write_text(
        json.dumps(
            {"config": {"runtime": {"harbor_checkout": str(checkout)}}, "harbor": {"commit": executed, "dirty": False}}
        )
    )

    def atlas_sql(statement, _parameters):
        assert statement.startswith("SELECT"), "Stale evidence must never be published"
        return {"rows": [{"payload": payload}]}

    monkeypatch.setattr(publish_review, "sql", atlas_sql)
    assert publish_review.validated_publication(tmp_path, "MarinSkyRL:test").payload == payload
    verifier.write_text("def check(): return False\n")
    payload["harbor_verifier_revision"] = _git_commit(checkout)
    with pytest.raises(ValueError, match="Executed Harbor verifier differs"):
        publish_review.publish_review(tmp_path, "MarinSkyRL:test")
    assert not (tmp_path / "atlas-publication.json").exists()
