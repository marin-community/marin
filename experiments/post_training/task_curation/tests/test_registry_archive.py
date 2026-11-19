# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The registry archive refuses publishes that would blank sibling shard rows."""

import json
from pathlib import Path

import pytest

import experiments.post_training.task_curation.registry as registry
from experiments.post_training.task_curation.registry import (
    _require_complete_shard_family,
    read_registry_json,
)


@pytest.fixture
def family(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A references root with one pointer file whose pinned document holds two rows."""
    references_root = tmp_path / "references"
    directory = references_root / "datasets" / "census"
    directory.mkdir(parents=True)
    pointer = directory / "part-000.json"
    document = {"alpha": {"count": 1}, "beta": {"count": 2}}
    served = tmp_path / "served" / "datasets" / "census" / "part-000.json"
    served.parent.mkdir(parents=True)
    served.write_text(json.dumps(document, indent=2) + "\n")

    def fake_download(**kwargs):
        assert kwargs["filename"] == served.relative_to(tmp_path).as_posix().removeprefix("served/")
        return str(served)

    monkeypatch.setattr(registry, "hf_hub_download", fake_download)
    pointer.write_text(
        json.dumps(
            {
                "repo": "open-athena/example-registry",
                "revision": "a" * 40,
                "path": served.relative_to(tmp_path).as_posix().removeprefix("served/"),
                "sha256": __import__("hashlib").sha256(served.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    return references_root


def test_refuses_a_publish_that_would_drop_sibling_rows(family: Path):
    documents = {"datasets/census/part-001.json": json.dumps({"gamma": {"count": 3}}, indent=2).encode()}
    with pytest.raises(ValueError, match="would drop 2 rows held by the unstaged shard"):
        _require_complete_shard_family(
            documents, family, Path("datasets/census"), "open-athena/example-registry", "a" * 40
        )
    # the refusal must happen before any publish-side effects: nothing was staged
    assert "datasets/census/part-000.json" not in documents


def test_allows_consolidation_when_every_sibling_row_is_staged(family: Path):
    documents = {
        "datasets/census/part-001.json": (
            json.dumps({"alpha": {"count": 1}, "beta": {"count": 2}, "gamma": {"count": 3}}, indent=2).encode()
        )
    }
    _require_complete_shard_family(documents, family, Path("datasets/census"), "open-athena/example-registry", "a" * 40)
    # the superseded sibling is blanked in the refreshed collection, as designed
    assert json.loads(documents["datasets/census/part-000.json"]) == {}


def test_staging_the_sibling_itself_skips_the_blank(family: Path):
    documents = {
        "datasets/census/part-000.json": json.dumps({"alpha": {"count": 1}, "beta": {"count": 2}}, indent=2).encode(),
        "datasets/census/part-001.json": json.dumps({"gamma": {"count": 3}}, indent=2).encode(),
    }
    _require_complete_shard_family(documents, family, Path("datasets/census"), "open-athena/example-registry", "a" * 40)
    assert json.loads(documents["datasets/census/part-000.json"]) == {"alpha": {"count": 1}, "beta": {"count": 2}}


def test_read_registry_json_still_resolves_the_fixture_pointer(family: Path):
    pointer = family / "datasets" / "census" / "part-000.json"
    assert set(read_registry_json(pointer)) == {"alpha", "beta"}
