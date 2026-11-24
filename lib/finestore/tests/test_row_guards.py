# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import threading

import pytest
from finestore.layout import OnConflict
from finestore.reader import ReadView
from finestore.store import DataStore, RowConflict


def _open(root: str, writer: str) -> DataStore:
    store = DataStore(root, writer_id=writer, flush_interval=3600)
    store.table("tasks", primary_key=("task_id",), on_conflict=OnConflict.SUPERSEDE)
    return store


def _seed(root: str, rows: list[dict]) -> None:
    with _open(root, "seed") as store, store.transaction() as transaction:
        transaction.table("tasks").extend(rows)


def test_read_then_concurrent_commit_rejects_stale_write(archive_root):
    _seed(archive_root, [{"task_id": "a", "status": "active", "note": ""}])
    with _open(archive_root, "alice") as alice, _open(archive_root, "bob") as bob:
        stale = alice.transaction(author="alice", message="disable a")
        row = stale.read("tasks", [("a",)])[("a",)]
        bob.update("tasks", [("a",)], lambda r: {**r, "note": "fixed"}, author="bob", message="fix a")
        stale.table("tasks").add({**row, "status": "disabled"})
        with pytest.raises(RowConflict) as raised:
            stale.commit()
    assert raised.value.keys == (("a",),)
    assert ReadView(archive_root).point("tasks", task_id="a")["note"] == "fixed"


def test_guard_ignores_commits_to_other_keys(archive_root):
    _seed(archive_root, [{"task_id": "a", "status": "active"}, {"task_id": "b", "status": "active"}])
    with _open(archive_root, "alice") as alice, _open(archive_root, "bob") as bob:
        transaction = alice.transaction(author="alice", message="disable a")
        row = transaction.read("tasks", [("a",)])[("a",)]
        bob.update("tasks", [("b",)], lambda r: {**r, "status": "disabled"}, author="bob", message="disable b")
        transaction.table("tasks").add({**row, "status": "disabled"})
        transaction.commit()
    view = ReadView(archive_root)
    assert {view.point("tasks", task_id=key)["status"] for key in "ab"} == {"disabled"}


def test_guarded_absent_key_conflicts_with_a_concurrent_insert(archive_root):
    _seed(archive_root, [{"task_id": "a", "status": "active"}])
    with _open(archive_root, "alice") as alice, _open(archive_root, "bob") as bob:
        transaction = alice.transaction(author="alice", message="add n")
        assert transaction.read("tasks", [("n",)]) == {}
        with bob.transaction(author="bob", message="add n") as other:
            other.table("tasks").add({"task_id": "n", "status": "active"})
        transaction.table("tasks").add({"task_id": "n", "status": "disabled"})
        with pytest.raises(RowConflict):
            transaction.commit()


def test_guard_sees_a_concurrent_change_after_compaction(archive_root):
    _seed(archive_root, [{"task_id": key, "n": 0} for key in "abc"])
    with _open(archive_root, "alice") as alice, _open(archive_root, "bob") as bob:
        transaction = alice.transaction(author="alice", message="bump b")
        row = transaction.read("tasks", [("b",)])[("b",)]
        bob.update("tasks", [("b",)], lambda r: {**r, "n": 5}, author="bob", message="set b")
        bob.compact("tasks")
        transaction.table("tasks").add({**row, "n": row["n"] + 1})
        with pytest.raises(RowConflict):
            transaction.commit()
    assert ReadView(archive_root).point("tasks", task_id="b")["n"] == 5


def test_concurrent_updates_lose_no_increment(archive_root):
    _seed(archive_root, [{"task_id": "hot", "w0": 0, "w1": 0, "w2": 0}])
    per_writer = 8
    barrier = threading.Barrier(3)

    def work(worker: int) -> None:
        with _open(archive_root, f"w{worker}") as store:
            barrier.wait()
            for _ in range(per_writer):
                store.update(
                    "tasks",
                    [("hot",)],
                    lambda r, name=f"w{worker}": {**r, name: r[name] + 1},
                    author=f"w{worker}",
                    message="increment",
                    attempts=100,
                )

    threads = [threading.Thread(target=work, args=(worker,)) for worker in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    row = ReadView(archive_root).point("tasks", task_id="hot")
    assert [row[f"w{worker}"] for worker in range(3)] == [per_writer] * 3


def test_update_rejects_missing_keys(archive_root):
    _seed(archive_root, [{"task_id": "a", "status": "active"}])
    with _open(archive_root, "alice") as alice, pytest.raises(KeyError):
        alice.update("tasks", [("a",), ("missing",)], lambda r: r, author="alice", message="noop")


def test_guard_on_an_empty_archive_sees_a_commit_racing_its_head_write(simulated_bucket):
    root = "s3://finestore-tests/empty"
    with _open(root, "alice") as alice, _open(root, "bob") as bob:
        transaction = alice.transaction(author="alice", message="claim x")
        assert transaction.read("tasks", [("x",)]) == {}
        racing = bob.transaction(author="bob", message="create x")
        racing.table("tasks").add({"task_id": "x", "status": "active"})
        write = simulated_bucket.conditional_write
        raced = []

        def bob_commits_first(path, data, *, expected_version):
            if path == f"{root}/HEAD" and not raced:
                raced.append(True)
                racing.commit()
            return write(path, data, expected_version=expected_version)

        simulated_bucket.conditional_write = bob_commits_first
        with pytest.raises(RowConflict):
            transaction.commit()


def test_update_rejects_an_edit_that_changes_the_key(archive_root):
    _seed(archive_root, [{"task_id": "a", "status": "active"}])
    with _open(archive_root, "alice") as alice, pytest.raises(ValueError, match="primary key"):
        alice.update("tasks", [("a",)], lambda r: {**r, "task_id": "b"}, author="alice", message="rename")
    assert ReadView(archive_root).keys("tasks") == {("a",)}


def test_update_rejects_a_table_that_does_not_supersede_rows(archive_root):
    # An ERROR table rejects a second, different row for a key, so an update can never succeed.
    with DataStore(archive_root, writer_id="w", flush_interval=3600) as store:
        store.table("tasks", primary_key=("task_id",))
        with store.transaction() as transaction:
            transaction.table("tasks").add({"task_id": "a", "n": 1})
        with pytest.raises(ValueError, match="SUPERSEDE"):
            store.update("tasks", [("a",)], lambda row: {**row, "n": 2}, author="w", message="bump a")
    assert ReadView(archive_root).point("tasks", task_id="a")["n"] == 1


def test_read_returns_only_the_requested_composite_keys(archive_root):
    with DataStore(archive_root, writer_id="w", flush_interval=3600) as store:
        store.table("runs", primary_key=("run", "i"), on_conflict=OnConflict.SUPERSEDE)
        with store.transaction() as transaction:
            transaction.table("runs").extend(
                {"run": run, "i": i, "v": f"{run}-{i}"} for run in ("r1", "r2") for i in range(3)
            )
        rows = store.transaction().read("runs", [("r1", 2), ("r2", 0), ("r3", 0)])
    assert rows == {("r1", 2): {"run": "r1", "i": 2, "v": "r1-2"}, ("r2", 0): {"run": "r2", "i": 0, "v": "r2-0"}}
