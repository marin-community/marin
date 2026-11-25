# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from finestore.history import commit_log, row_history
from finestore.layout import CommitToken, OnConflict
from finestore.reader import ReadView
from finestore.store import DataStore


def _open(root: str) -> DataStore:
    store = DataStore(root, writer_id="w", flush_interval=3600)
    store.table("tasks", primary_key=("task_id",), on_conflict=OnConflict.SUPERSEDE)
    return store


def _commit(store: DataStore, rows: list[dict], message: str) -> CommitToken:
    with store.transaction(author="alice", message=message) as transaction:
        transaction.table("tasks").extend(rows)
    assert transaction.token is not None
    return transaction.token


def _revisions(view: ReadView) -> list[tuple[str, int]]:
    return [(row["task_id"], row["rev"]) for row in view.iter_rows("tasks", columns=["task_id", "rev"])]


def test_read_at_a_past_commit_after_later_commits_and_compaction(archive_root):
    with _open(archive_root) as store:
        first = _commit(store, [{"task_id": "a", "rev": 1}, {"task_id": "b", "rev": 1}], "add a and b")
        _commit(store, [{"task_id": "a", "rev": 2}, {"task_id": "c", "rev": 1}], "fix a, add c")
        store.compact("tasks")

    past = ReadView.at(archive_root, first.commit_id)

    assert past.token is not None and past.token.commit_id == first.commit_id
    assert past.commit_info is not None and past.commit_info.message == "add a and b"
    assert _revisions(past) == [("a", 1), ("b", 1)]
    assert _revisions(ReadView(archive_root)) == [("a", 2), ("b", 1), ("c", 1)]


def test_row_history_lists_each_version_with_the_commit_that_wrote_it(archive_root):
    with _open(archive_root) as store:
        _commit(store, [{"task_id": "a", "rev": 1}, {"task_id": "b", "rev": 1}], "add a and b")
        _commit(store, [{"task_id": "b", "rev": 2}], "fix b")
        _commit(store, [{"task_id": "a", "rev": 2}], "fix a")
        store.compact("tasks")
        _commit(store, [{"task_id": "a", "rev": 3}], "fix a again")

    versions = list(row_history(archive_root, "tasks", ("a",)))

    assert [(version.row, version.commit.info and version.commit.info.message) for version in versions] == [
        ({"task_id": "a", "rev": 3}, "fix a again"),
        ({"task_id": "a", "rev": 2}, "fix a"),
        ({"task_id": "a", "rev": 1}, "add a and b"),
    ]
    for version in versions:
        row = ReadView.at(archive_root, version.commit.commit_id).point("tasks", task_id="a")
        assert row is not None and row["rev"] == version.row["rev"]


def test_commit_log_walks_from_head_back_to_the_first_commit(archive_root):
    with _open(archive_root) as store:
        first = _commit(store, [{"task_id": "a", "rev": 1}], "add a")
        second = _commit(store, [{"task_id": "a", "rev": 2}], "fix a")
        log = list(commit_log(archive_root))

    assert [(record.commit_id, record.parent_commit_id) for record in log] == [
        (second.commit_id, first.commit_id),
        (first.commit_id, None),
    ]
    assert [record.info and record.info.message for record in log] == ["fix a", "add a"]
