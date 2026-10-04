# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for inventory completeness and answer separation."""

import base64
import hashlib
import json
from datetime import date
from urllib.parse import parse_qs, urlsplit

import pytest

from experiments.post_training.russell_rsi.corpus import (
    GitHub,
    Inventory,
    add_commit,
    assign_splits,
    freeze_source_splits,
    main,
    search_commits,
    source_snapshot,
    write_inventory,
)


def commit_item(sha: str, repository: str, tree: str) -> dict:
    return {
        "sha": sha,
        "author": {"login": "rjpower"},
        "repository": {"full_name": repository},
        "parents": [{"sha": "parent"}],
        "commit": {
            "tree": {"sha": tree},
            "author": {"name": "Russell Power", "email": "public@example.com", "date": "2025-01-01T00:00:00Z"},
            "committer": {"name": "Russell Power", "date": "2025-01-01T00:00:00Z"},
            "message": "Fix operation",
        },
    }


def test_inventory_fork_dedup_and_shared_history_keep_one_split(tmp_path):
    items = [
        commit_item("one", "upstream/project", "tree-one"),
        commit_item("one", "fork/project", "tree-one"),
        commit_item("two", "fork/project", "tree-two"),
        commit_item("three", "mirror/project", "tree-two"),
    ]
    inventories = []
    for ordered in (items, list(reversed(items))):
        inventory = Inventory()
        for item in ordered:
            add_commit(inventory, item, "author:rjpower")
        assign_splits(inventory, "seed")
        inventories.append(inventory)
    assert len(inventories[0].commits) == 3
    assert inventories[0].commits["one"].repositories == ["fork/project", "upstream/project"]
    assert len({record.split for record in inventories[0].commits.values()}) == 1
    assert {sha: (record.family, record.split) for sha, record in inventories[0].commits.items()} == {
        sha: (record.family, record.split) for sha, record in inventories[1].commits.items()
    }
    api = GitHub(tmp_path / "cache", 1)
    write_inventory(inventories[0], tmp_path / "output", api, {"end": date(2025, 1, 1).isoformat()})
    rows = [json.loads(line) for line in (tmp_path / "output/inventory.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert all(row["parents"] == ["parent"] for row in rows)
    assert json.loads((tmp_path / "output/summary.json").read_text())["answer_artifact"] is True


def test_source_snapshot_added_module_preserves_revision_license_without_answer_in_parent(tmp_path):
    api = GitHub(tmp_path / "cache", 0)
    commit_sha, parent_sha = "a" * 40, "b" * 40
    inventory = Inventory()
    item = commit_item(commit_sha, "project/repository", "tree")
    item["parents"] = [{"sha": parent_sha}]
    add_commit(inventory, item, "author:rjpower")
    assign_splits(inventory, "seed")
    payloads = {
        f"repos/project/repository/commits/{commit_sha}": {
            "parents": [{"sha": parent_sha}],
            "files": [{"filename": "module #?.py", "status": "added"}],
        },
    }
    for revision in (parent_sha, commit_sha):
        names = {"LICENSE": "SPDX-License-Identifier: MIT\n"}
        if revision == commit_sha:
            names["module #?.py"] = "def operation():\n    return 42\n"
        for name, content in names.items():
            encoded = "module%20%23%3F.py" if name == "module #?.py" else name
            payloads[f"repos/project/repository/contents/{encoded}?ref={revision}"] = {
                "type": "file",
                "encoding": "base64",
                "size": len(content.encode()),
                "content": base64.b64encode(content.encode()).decode(),
            }
    for endpoint, payload in payloads.items():
        (api.cache / (hashlib.sha256(endpoint.encode()).hexdigest() + ".json")).write_text(json.dumps(payload))
    snapshot = source_snapshot(
        api,
        {
            "repository": "project/repository",
            "commit_sha": commit_sha,
            "source_paths": ["module #?.py"],
            "license_paths": ["LICENSE"],
            "split": "wrong",
        },
        inventory,
    )
    assert snapshot.parent_files == {"LICENSE": "SPDX-License-Identifier: MIT\n"}
    assert snapshot.reference_files["module #?.py"] == "def operation():\n    return 42\n"
    assert snapshot.reference_files["LICENSE"] == snapshot.parent_files["LICENSE"]
    assert snapshot.parent_sha == parent_sha
    assert snapshot.split == inventory.commits[commit_sha].split


def test_search_divides_at_page_budget_and_collects_the_complete_window(tmp_path):
    class SearchAPI(GitHub):
        def get(self, endpoint):
            query = parse_qs(urlsplit(endpoint).query)
            scope = query["q"][0]
            page = int(query["page"][0])
            if "2025-01-01..2025-01-02" in scope:
                total, indices = 201, range(100)
            elif "2025-01-01..2025-01-01" in scope:
                total, indices = 100, range(100)
            else:
                total = 101
                indices = range(100, 200) if page == 1 else range(200, 201)
            return {
                "total_count": total,
                "incomplete_results": False,
                "items": [commit_item(str(index), "project/repository", str(index)) for index in indices],
            }

    inventory = Inventory()
    search_commits(SearchAPI(tmp_path, 0), inventory, "author:rjpower", date(2025, 1, 1), date(2025, 1, 2), 2)
    assert len(inventory.commits) == 201
    assert [query.status for query in inventory.queries] == ["partitioned", "complete", "complete"]


def test_inventory_ancestry_joins_forks_without_shared_commit_sha():
    inventory = Inventory()
    add_commit(inventory, commit_item("one", "upstream/project", "tree-one"), "scope")
    add_commit(inventory, commit_item("two", "fork/project", "tree-two"), "scope")
    inventory.repositories["fork/project"]["source"] = {"full_name": "upstream/project"}
    assign_splits(inventory, "seed")
    assert inventory.commits["one"].family == inventory.commits["two"].family
    assert inventory.commits["one"].split == inventory.commits["two"].split


def test_eligible_split_freeze_preserves_sealed_test_and_allocates_development():
    inventories = []
    repos = {"project/one", "project/two", "project/three", "project/four"}
    for ordered in (sorted(repos), sorted(repos, reverse=True)):
        inventory = Inventory()
        for repo in ordered:
            add_commit(inventory, commit_item(repo, repo, repo), "scope")
        assign_splits(inventory, "seed")
        freeze_source_splits(inventory, repos, "seed", {"project/one"})
        inventories.append(inventory)
    assignments = {r.family: r.split for r in inventories[0].commits.values()}
    assert assignments["project/one"] == "test"
    assert sorted(assignments.values()) == ["dev", "test", "train", "train"]
    assert assignments == {r.family: r.split for r in inventories[1].commits.values()}


def test_cli_empty_budget_writes_one_partial_scope_error(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["corpus", "--output", str(tmp_path), "--end", "2025-01-01", "--max-requests", "0"])
    main()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "partial"
    assert len(summary["errors"]) == 1
    assert (tmp_path / "inventory.jsonl").read_text() == ""


@pytest.mark.parametrize("path", ["foo_test.py", "test/helper.py", "testing/helper.py", "conftest.py"])
def test_source_snapshot_rejects_reference_test_paths(tmp_path, path):
    class MetadataAPI(GitHub):
        def get(self, endpoint):
            return {"parents": [{"sha": "parent"}], "files": []}

    inventory = Inventory()
    add_commit(inventory, commit_item("one", "project/repository", "tree-one"), "scope")
    assign_splits(inventory, "seed")
    with pytest.raises(ValueError, match="reference source"):
        source_snapshot(
            MetadataAPI(tmp_path, 0),
            {
                "repository": "project/repository",
                "commit_sha": "one",
                "source_paths": [path],
                "license_paths": ["LICENSE"],
            },
            inventory,
        )


@pytest.mark.parametrize(
    "changes",
    [
        [{"filename": "module.py", "status": "modified"}] * 300,
        [{"filename": "module.py", "status": "renamed"}],
        [{"filename": "module.py", "status": "removed"}],
    ],
)
def test_source_snapshot_rejects_incomplete_or_unsupported_commit_changes(tmp_path, changes):
    class MetadataAPI(GitHub):
        def get(self, endpoint):
            return {"parents": [{"sha": "parent"}], "files": changes}

    inventory = Inventory()
    add_commit(inventory, commit_item("one", "project/repository", "tree-one"), "scope")
    assign_splits(inventory, "seed")
    with pytest.raises(ValueError, match=r"truncated|does not support"):
        source_snapshot(
            MetadataAPI(tmp_path, 0),
            {
                "repository": "project/repository",
                "commit_sha": "one",
                "source_paths": ["module.py"],
                "license_paths": ["LICENSE"],
            },
            inventory,
        )
