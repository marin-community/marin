# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rollout gate verdicts against a real git history, and the After-check rules."""

import subprocess
from pathlib import Path

import pytest
from finelog.deploy.rollout_checks import (
    BUCKET_MS,
    CATALOG_MIGRATIONS_DIR,
    BuildState,
    FailureRow,
    Status,
    VolumeRow,
    assess_failures,
    assess_volume,
    gate_row,
    order_blockers,
)

CONFIG = "lib/finelog/config/cw-test.yaml"


class Repo:
    def __init__(self, path: Path):
        self.path = path
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "test")

    def git(self, *args: str) -> str:
        return subprocess.run(["git", *args], cwd=self.path, check=True, capture_output=True, text=True).stdout.strip()

    def commit(self, files: dict[str, str]) -> str:
        for name, text in files.items():
            file = self.path / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(text)
            self.git("add", name)
        self.git("commit", "-q", "-m", "change")
        return self.git("rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    repo = Repo(tmp_path)
    repo.commit({f"{CATALOG_MIGRATIONS_DIR}m0.sql": "init", CONFIG: "remote_log_dir: s3://a\nport: 1\n"})
    return repo


def _gate(repo: Repo, deployed: str | None, target: str, *, secret: bool = True):
    return gate_row(
        repo.path,
        server="cw-test",
        deployed=deployed,
        target=target,
        main_ref="main",
        config_path=CONFIG if secret else None,
    )


def test_unrelated_changes_leave_rollout_clear(repo: Repo):
    deployed = repo.commit({"lib/finelog/src/x.py": "1"})
    target = repo.commit({"lib/finelog/src/x.py": "2", CONFIG: "remote_log_dir: s3://a\nport: 2\n"})
    row = _gate(repo, deployed[:10], target)
    assert row.blockers() == []
    assert row.cells()[2:] == ("yes", "yes", "none", "no")


def test_migration_after_deployed_build_blocks(repo: Repo):
    deployed = repo.git("rev-parse", "HEAD")
    target = repo.commit({f"{CATALOG_MIGRATIONS_DIR}m1.sql": "add column"})
    row = _gate(repo, deployed, target)
    assert row.migrations_differ and not row.has_last_migration
    assert row.roll_forward_only and row.blockers()
    assert row.blockers(roll_forward_ok=True) == []


def test_branch_build_with_main_migrations_is_clear(repo: Repo):
    """A build deployed from a branch before it merged has no ancestry path to the target."""
    repo.commit({f"{CATALOG_MIGRATIONS_DIR}m1.sql": "add column"})
    base = repo.git("rev-parse", "HEAD")
    repo.git("checkout", "-q", "-b", "feature")
    deployed = repo.commit({"lib/finelog/src/feature.py": "wip"})
    repo.git("checkout", "-q", "main")
    target = repo.commit({"lib/finelog/src/other.py": "1"})
    row = _gate(repo, deployed, target)
    assert not row.on_main
    assert row.has_last_migration
    assert row.blockers() == []
    assert repo.git("merge-base", deployed, target) == base


def test_branch_build_carrying_its_own_migration_blocks(repo: Repo):
    repo.git("checkout", "-q", "-b", "feature")
    deployed = repo.commit({f"{CATALOG_MIGRATIONS_DIR}m1.sql": "unmerged"})
    repo.git("checkout", "-q", "main")
    target = repo.commit({"lib/finelog/src/other.py": "1"})
    row = _gate(repo, deployed, target)
    assert row.migrations_differ
    assert row.blockers()


def test_secret_field_change_requires_sync_only_for_secret_servers(repo: Repo):
    deployed = repo.git("rev-parse", "HEAD")
    target = repo.commit({CONFIG: "remote_log_dir: s3://b\nport: 1\n"})
    assert _gate(repo, deployed, target).secret_sync_needed
    assert _gate(repo, deployed, target).blockers()
    assert _gate(repo, deployed, target).blockers(secret_synced=True) == []
    gce = _gate(repo, deployed, target, secret=False)
    assert gce.secret_sync_needed is None and gce.blockers() == []


def test_unreadable_or_unfetched_build_blocks(repo: Repo):
    target = repo.git("rev-parse", "HEAD")
    assert _gate(repo, None, target).state is BuildState.UNKNOWN
    missing = _gate(repo, "0123456789", target)
    assert missing.state is BuildState.NOT_FETCHED
    assert missing.blockers() and missing.cells()[2] == "not-fetched"


def test_order_requires_earlier_servers_on_target(repo: Repo):
    old = repo.git("rev-parse", "HEAD")
    target = repo.commit({"lib/finelog/src/x.py": "1"})
    order = ["dev", "hub", "sender"]
    blockers = order_blockers(repo.path, order, "sender", {"dev": target[:10], "hub": old}, target)
    assert len(blockers) == 1 and blockers[0].startswith("hub")
    assert order_blockers(repo.path, order, "dev", {}, target) == []


def test_retries_warn_only_while_they_continue():
    end = 10 * BUCKET_MS
    findings = assess_failures(
        [
            FailureRow("a", "retryable_failure", 4, end - 20 * 60_000),
            FailureRow("b", "retryable_failure", 4, end - 60_000),
            FailureRow("c", "schema_conflict", 1, end - 50 * 60_000),
        ],
        end,
    )
    assert [f.status for f in findings] == [Status.OK, Status.WARN, Status.FAIL]


def test_volume_fails_after_two_empty_complete_buckets():
    end = 10 * BUCKET_MS + 60_000  # one minute into an incomplete bucket
    rows = [
        VolumeRow(8 * BUCKET_MS, "steady", 400_000),
        VolumeRow(9 * BUCKET_MS, "steady", 410_000),
        VolumeRow(9 * BUCKET_MS, "restarted", 90_000),
        VolumeRow(8 * BUCKET_MS, "paused", 300_000),
        VolumeRow(10 * BUCKET_MS, "stopped", 5_000),  # the incomplete bucket doesn't count
    ]
    statuses = {f.subject: f.status for f in assess_volume(rows, ["steady", "restarted", "paused", "stopped"], end)}
    assert statuses == {
        "steady": Status.OK,
        "restarted": Status.OK,
        "paused": Status.WARN,
        "stopped": Status.FAIL,
    }
