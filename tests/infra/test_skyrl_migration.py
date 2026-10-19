# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import os
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from scripts import skyrl_migration as migration


def write_bytes(root: Path, path: str, data: bytes) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def write(root: Path, path: str, data: str) -> None:
    write_bytes(root, path, data.encode())


def save(root: Path, message: str) -> str:
    migration.git(root, "add", ".")
    return migration.commit(root, message)


@pytest.fixture
def snapshot_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "snapshot"
    source = directory / "source"
    source.mkdir(parents=True)
    migration.run("git", "init", "--quiet", "--initial-branch=main", str(source))
    write(source, "cloud/__init__.py", "")
    write(source, "cloud/iris/__init__.py", "")
    write(source, migration.BUNDLE_MANIFEST, "cloud/iris/launch.py\n")
    write(source, "cloud/iris/launch.py", "from skyrl_train.train import train\n")
    write(source, "skyrl-train/skyrl_train/__init__.py", "")
    write(
        source,
        "skyrl-train/skyrl_train/train.py",
        "from .relative import value\nimport importlib\n\ndef train():\n    from .nested import helper\n"
        '    return importlib.import_module("skyrl_train.dynamic").apply(value, helper)\n',
    )
    write(source, "skyrl-train/skyrl_train/relative.py", "value = 1\n")
    write(source, "skyrl-train/skyrl_train/nested.py", "helper = 2\n")
    write(source, "skyrl-train/skyrl_train/dynamic.py", "def apply(a, b):\n    return a + b\n")
    write(source, "skyrl-train/skyrl_train/unused.py", "unused = True\n")
    write(source, "skyrl-train/skyrl_train/assets/old.txt", "data\n")
    write_bytes(source, "skyrl-train/skyrl_train/assets/blob.bin", b"\0old\xff")
    write(source, "skyrl-gym/skyrl_gym/__init__.py", "")
    (source / "skyrl-train/skyrl-gym").symlink_to("../skyrl-gym/")
    write(source, "scripts/run.sh", "#!/bin/sh\nexit 0\n")
    (source / "scripts/run.sh").chmod(0o755)
    write(source, "LICENSE", "Apache-2.0\n")
    write(source, "skyrl-tx/unused.py", "discard = True\n")
    baseline = save(source, "Baseline")
    manifest = """roots = ["cloud.iris.launch"]
discard = ["skyrl-tx/**"]
[[packages]]
directory = "cloud"
import_root = "."
[[packages]]
directory = "skyrl-train/skyrl_train"
import_root = "skyrl-train"
[[keep]]
glob = "LICENSE"
reason = "license"
[[keep]]
glob = "cloud/iris/runtime_bundle_files.txt"
reason = "bundle"
[[keep]]
glob = "skyrl-train/skyrl_train/assets/**"
reason = "runtime data"
[[keep]]
glob = "skyrl-gym/skyrl_gym/__init__.py"
reason = "source symlink target"
[[keep]]
glob = "skyrl-train/skyrl-gym"
reason = "training import path"
[[keep]]
glob = "scripts/run.sh"
reason = "runtime setup"
"""
    write(directory, "manifest.toml", manifest)
    migration.write_json(
        directory / "snapshot.json",
        {
            "schema": 1,
            "source": migration.SOURCE,
            "source_sha": baseline,
            "default_branch": "main",
            "pulls": [],
            "issues": [],
        },
    )
    return directory


def pull(source: Path, number: int, base: str, head: str, base_ref: str = "main") -> dict:
    changed = migration.run("git", "-C", str(source), "diff", "--name-only", "--no-renames", "-z", base, head).stdout
    return {
        "number": number,
        "html_url": f"https://github.com/{migration.SOURCE}/pull/{number}",
        "title": f"Change {number}",
        "body": "A change",
        "user": {"login": "contributor"},
        "merge_base": base,
        "head": {"sha": head, "ref": f"pr-{number}", "repo": {"full_name": migration.SOURCE}},
        "base": {"sha": base, "ref": base_ref, "repo": {"full_name": migration.SOURCE}},
        "paths": [name.decode() for name in changed.split(b"\0") if name],
    }


def prepare(directory: Path, pulls: list[dict]) -> None:
    source = directory / "source"
    record = migration.read_json(directory / "snapshot.json")
    record["source_sha"] = migration.git(source, "rev-parse", "main")
    record["pulls"] = pulls
    migration.write_json(directory / "snapshot.json", record)
    objects = directory / "source.git"
    migration.run("git", "clone", "--quiet", "--bare", str(source), str(objects))
    migration.git(objects, "update-ref", "refs/snapshot/main", record["source_sha"])
    for item in pulls:
        for suffix, sha in (("base", item["merge_base"]), ("head", item["head"]["sha"])):
            migration.git(objects, "update-ref", f"refs/snapshot/pr-{item['number']}-{suffix}", sha)
    migration.plan(directory, directory / "manifest.toml", None)


def test_snapshot_keeps_exact_git_links_and_recently_active_old_issues(
    snapshot_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = snapshot_dir / "source"
    sha = migration.git(source, "rev-parse", "HEAD")
    recent = {
        "number": 1,
        "html_url": f"https://github.com/{migration.SOURCE}/issues/1",
        "created_at": "2026-08-01T00:00:00Z",
        "updated_at": "2026-10-18T00:00:00Z",
    }
    stale = {"number": 2, "updated_at": "2026-08-02T00:00:00Z"}
    pull_item = {"number": 3, "updated_at": "2026-10-18T00:00:00Z", "pull_request": {}}
    actual_run = migration.run

    def process(*args: str, **kwargs) -> subprocess.CompletedProcess[bytes]:
        rewritten = [str(source) if arg == f"https://github.com/{migration.SOURCE}.git" else arg for arg in args]
        return actual_run(*rewritten, **kwargs)

    def api(endpoint: str):
        return {"sha": sha} if "/commits/" in endpoint else {"default_branch": "main"}

    def paginate(endpoint: str):
        return [recent, stale, pull_item] if "/issues?" in endpoint else []

    monkeypatch.setattr(migration, "run", process)
    monkeypatch.setattr(migration, "gh", api)
    monkeypatch.setattr(migration, "github_items", paginate)
    output = tmp_path / "capture"
    migration.snapshot(output, "main", 10, datetime(2026, 10, 19, tzinfo=UTC))
    captured = migration.read_json(output / "snapshot.json")
    assert [issue["number"] for issue in captured["issues"]] == [1]
    assert os.readlink(output / "source/skyrl-train/skyrl-gym") == "../skyrl-gym/"
    migration.plan(output, snapshot_dir / "manifest.toml", None)
    migration.stage(output, tmp_path / "staged")


def test_stage_preserves_nested_imports_assets_modes_and_symlinks(snapshot_dir: Path, tmp_path: Path) -> None:
    prepare(snapshot_dir, [])
    output = tmp_path / "candidate"
    migration.stage(snapshot_dir, output)
    vendor = output / "tree" / migration.VENDOR
    for name in ("relative", "nested", "dynamic"):
        assert (vendor / f"skyrl-train/skyrl_train/{name}.py").is_file()
    assert not (vendor / "skyrl-tx").exists()
    assert not (vendor / "skyrl-train/skyrl_train/unused.py").exists()
    assert (vendor / "skyrl-train/skyrl_train/assets/blob.bin").read_bytes() == b"\0old\xff"
    assert (vendor / "skyrl-train/skyrl-gym").resolve() == vendor / "skyrl-gym"
    assert (vendor / "scripts/run.sh").stat().st_mode & 0o111
    assert (vendor / "LICENSE").read_text() == "Apache-2.0\n"
    provenance = migration.read_json(vendor / migration.PROVENANCE)
    assert provenance["files"]["skyrl-train/skyrl_train/assets/blob.bin"] == migration.file_digest(
        vendor / "skyrl-train/skyrl_train/assets/blob.bin"
    )


def stacked_pulls(directory: Path) -> list[dict]:
    source = directory / "source"
    base = migration.git(source, "rev-parse", "HEAD")
    migration.git(source, "checkout", "--quiet", "-b", "pr-1")
    write(source, "skyrl-train/skyrl_train/relative.py", "value = 10\n")
    write_bytes(source, "skyrl-train/skyrl_train/assets/blob.bin", b"\0new\x80\xff")
    migration.git(source, "mv", "skyrl-train/skyrl_train/assets/old.txt", "skyrl-train/skyrl_train/assets/new.txt")
    first = save(source, "First")
    migration.git(source, "checkout", "--quiet", "-b", "pr-2")
    write(source, "skyrl-train/skyrl_train/relative.py", "value = 20\n")
    second = save(source, "Second")
    migration.git(source, "checkout", "--quiet", "main")
    return [pull(source, 2, first, second, "pr-1"), pull(source, 1, base, first)]


def test_stacked_binary_and_rename_patches_apply_as_separate_deltas(snapshot_dir: Path, tmp_path: Path) -> None:
    prepare(snapshot_dir, stacked_pulls(snapshot_dir))
    output = tmp_path / "candidate"
    migration.stage(snapshot_dir, output)
    report = migration.read_json(output / "report.json")
    assert [(item["number"], item["parent"], item["status"]) for item in report["pulls"]] == [
        (1, None, "ready"),
        (2, 1, "ready"),
    ]
    tree = output / "tree"
    assert (
        migration.git(tree, "show", f"skyrl-migration/pr-1:{migration.VENDOR}/skyrl-train/skyrl_train/relative.py")
        == "value = 10"
    )
    assert (
        migration.git(tree, "show", f"skyrl-migration/pr-2:{migration.VENDOR}/skyrl-train/skyrl_train/relative.py")
        == "value = 20"
    )
    binary = migration.run(
        "git",
        "-C",
        str(tree),
        "show",
        f"skyrl-migration/pr-2:{migration.VENDOR}/skyrl-train/skyrl_train/assets/blob.bin",
    ).stdout
    assert binary == b"\0new\x80\xff"
    assert (
        migration.git(tree, "show", f"skyrl-migration/pr-1:{migration.VENDOR}/skyrl-train/skyrl_train/assets/new.txt")
        == "data"
    )
    missing = migration.run(
        "git",
        "-C",
        str(tree),
        "cat-file",
        "-e",
        f"skyrl-migration/pr-1:{migration.VENDOR}/skyrl-train/skyrl_train/assets/old.txt",
        check=False,
    )
    assert missing.returncode != 0
    assert (snapshot_dir / "source/skyrl-train/skyrl_train/relative.py").read_text() == "value = 1\n"


def test_migration_fix_is_preserved_when_transplanting_open_work(
    snapshot_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = snapshot_dir / "source"
    base = migration.git(source, "rev-parse", "HEAD")
    write(source, "skyrl-train/skyrl_train/relative.py", "value = 7\n")
    patch = snapshot_dir / "runtime.patch"
    patch.write_bytes(migration.run("git", "-C", str(source), "diff", "--binary", "--full-index").stdout)
    migration.git(source, "checkout", "--", ".")
    migration.git(source, "checkout", "--quiet", "-b", "pr-1")
    write(source, "skyrl-train/skyrl_train/nested.py", "helper = 3\n")
    head = save(source, "Open work")
    migration.git(source, "checkout", "--quiet", "main")
    manifest = snapshot_dir / "manifest.toml"
    manifest.write_text('patches = ["runtime.patch"]\n' + manifest.read_text())
    prepare(snapshot_dir, [pull(source, 1, base, head)])
    patch.write_text("The mutable original patch must not be used after capture")
    output = tmp_path / "candidate"
    migration.stage(snapshot_dir, output)
    tree = output / "tree"
    runtime_file = f"{migration.VENDOR}/skyrl-train/skyrl_train/relative.py"
    assert migration.git(tree, "show", f"skyrl-migration/pr-1:{runtime_file}") == "value = 7"
    assert (
        migration.git(tree, "show", f"skyrl-migration/pr-1:{migration.VENDOR}/skyrl-train/skyrl_train/nested.py")
        == "helper = 3"
    )
    provenance = migration.read_json(tree / migration.VENDOR / migration.PROVENANCE)
    assert provenance["files"]["skyrl-train/skyrl_train/relative.py"] == migration.file_digest(tree / runtime_file)
    assert provenance["patches"][0]["sha256"] == migration.file_digest(snapshot_dir / "migration-patches/runtime.patch")
    assert (source / "skyrl-train/skyrl_train/relative.py").read_text() == "value = 1\n"
    github_fake(monkeypatch, migration.read_json(snapshot_dir / "snapshot.json"), "cutover")
    with pytest.raises(ValueError, match="migration fixes"):
        migration.validate_cutover(99, base, provenance["patches"])


def test_partial_pr_is_staged_but_blocks_its_child(snapshot_dir: Path, tmp_path: Path) -> None:
    pulls = stacked_pulls(snapshot_dir)
    source = snapshot_dir / "source"
    migration.git(source, "checkout", "--quiet", "pr-1")
    write(source, "skyrl-tx/unused.py", "discard = False\n")
    first = save(source, "Also change excluded code")
    pulls[1] = pull(source, 1, pulls[1]["merge_base"], first)
    migration.git(source, "checkout", "--quiet", "main")
    prepare(snapshot_dir, pulls)
    output = tmp_path / "candidate"
    migration.stage(snapshot_dir, output)
    report = migration.read_json(output / "report.json")
    assert [(item["number"], item["status"]) for item in report["pulls"]] == [
        (1, "needs-decision"),
        (2, "blocked-parent"),
    ]
    assert report["pulls"][0]["excluded"] == ["skyrl-tx/unused.py"]
    assert (
        migration.git(
            output / "tree", "show", f"skyrl-migration/pr-1:{migration.VENDOR}/skyrl-train/skyrl_train/relative.py"
        )
        == "value = 10"
    )


def test_conflict_is_recorded_without_changing_the_candidate(snapshot_dir: Path, tmp_path: Path) -> None:
    source = snapshot_dir / "source"
    base = migration.git(source, "rev-parse", "HEAD")
    migration.git(source, "checkout", "--quiet", "-b", "pr-1")
    write(source, "skyrl-train/skyrl_train/relative.py", "value = 10\n")
    head = save(source, "PR change")
    item = pull(source, 1, base, head)
    migration.git(source, "checkout", "--quiet", "main")
    write(source, "skyrl-train/skyrl_train/relative.py", "value = 30\n")
    save(source, "Conflicting main change")
    prepare(snapshot_dir, [item])
    output = tmp_path / "candidate"
    migration.stage(snapshot_dir, output)
    assert migration.read_json(output / "report.json")["pulls"][0]["status"] == "conflict"
    assert (output / "tree" / migration.VENDOR / "skyrl-train/skyrl_train/relative.py").read_text() == "value = 30\n"
    assert migration.git(output / "tree", "status", "--porcelain") == ""


def test_changed_snapshot_cannot_claim_original_source_identity(snapshot_dir: Path, tmp_path: Path) -> None:
    prepare(snapshot_dir, [])
    write(snapshot_dir / "source", "skyrl-train/skyrl_train/relative.py", "value = 999\n")
    with pytest.raises(ValueError, match="Snapshot source changed"):
        migration.stage(snapshot_dir, tmp_path / "candidate")
    assert not (tmp_path / "candidate/tree").exists()


def test_vendor_audit_rejects_changed_runtime_assets(snapshot_dir: Path, tmp_path: Path) -> None:
    prepare(snapshot_dir, [])
    candidate = tmp_path / "candidate"
    migration.stage(snapshot_dir, candidate)
    source = candidate / "tree" / migration.VENDOR
    migration.verify_vendor(source)
    write_bytes(source, "skyrl-train/skyrl_train/assets/blob.bin", b"changed runtime data")
    with pytest.raises(ValueError, match="Vendored source changed"):
        migration.verify_vendor(source)


@pytest.mark.parametrize("defect", ["missing", "changed", "untracked"])
def test_wheel_audit_rejects_runtime_contents_outside_provenance(
    snapshot_dir: Path, tmp_path: Path, defect: str
) -> None:
    prepare(snapshot_dir, [])
    candidate = tmp_path / "candidate"
    migration.stage(snapshot_dir, candidate)
    source = candidate / "tree" / migration.VENDOR
    provenance = migration.read_json(source / migration.PROVENANCE)
    wheel = tmp_path / "candidate.whl"
    asset = "skyrl_train/assets/blob.bin"

    def package(omit_asset: bool) -> None:
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.write(source / migration.PROVENANCE, migration.PROVENANCE)
            for name in provenance["files"]:
                member = name.removeprefix("skyrl-train/").removeprefix("skyrl-gym/")
                if omit_asset and member == asset and defect != "untracked":
                    if defect == "changed":
                        archive.writestr(member, b"changed runtime data")
                    continue
                archive.write(source / name, member)
            if omit_asset and defect == "untracked":
                archive.writestr("marinskyrl/untracked.py", "raise RuntimeError('untracked runtime')")

    package(omit_asset=False)
    assert asset in migration.audit_wheel(wheel)
    package(omit_asset=True)
    with pytest.raises(ValueError, match=r"Wheel member missing or changed|outside staged provenance"):
        migration.audit_wheel(wheel)


def github_fake(
    monkeypatch: pytest.MonkeyPatch, record: dict, cutover_sha: str, destination: Path | None = None
) -> dict:
    remote: dict = {"pulls": [], "issues": {}, "transfers": 0, "interrupt": False}
    provenance = {"source": migration.SOURCE, "source_sha": record["source_sha"]}

    def api(endpoint: str, *, method: str = "GET", payload: dict | None = None):
        if endpoint == f"repos/{migration.TARGET}/pulls/99":
            return {"merged": True, "merge_commit_sha": cutover_sha, "base": {"repo": {"full_name": migration.TARGET}}}
        if "/contents/" in endpoint:
            return {"content": base64.b64encode(json.dumps(provenance).encode()).decode()}
        if endpoint.startswith(f"repos/{migration.SOURCE}/commits/"):
            return {"sha": record["source_sha"]}
        if endpoint == f"repos/{migration.TARGET}":
            return {"default_branch": "main"}
        if endpoint == f"repos/{migration.TARGET}/pulls" and method == "POST":
            assert destination is not None and payload is not None
            number = len(remote["pulls"]) + 100
            sha = migration.git(destination, "rev-parse", payload["head"])
            created = payload | {
                "number": number,
                "html_url": f"https://github.com/{migration.TARGET}/pull/{number}",
                "head": {"sha": sha, "ref": payload["head"]},
                "base": {"ref": payload["base"]},
            }
            remote["pulls"].append(created)
            if remote["interrupt"]:
                remote["interrupt"] = False
                raise RuntimeError("Lost response after PR creation")
            return created
        if endpoint.endswith("/labels") and method == "POST":
            return []
        if "/issues/" in endpoint and method == "PATCH":
            if remote["interrupt"]:
                remote["interrupt"] = False
                raise RuntimeError("Lost response after issue transfer")
            assert payload is not None
            remote["issues"]["body"] = payload["body"]
            return remote["issues"]
        raise AssertionError((endpoint, method))

    def paginate(endpoint: str):
        if endpoint.startswith(f"repos/{migration.SOURCE}/pulls?"):
            return record["pulls"]
        if endpoint.startswith(f"repos/{migration.TARGET}/labels?"):
            return [{"name": "agent-generated"}]
        if endpoint.startswith(f"repos/{migration.TARGET}/pulls?"):
            branch = parse_qs(urlsplit(endpoint).query)["head"][0].partition(":")[2]
            return [item for item in remote["pulls"] if item["head"]["ref"] == branch]
        raise AssertionError(endpoint)

    def graph(query: str, **variables: str):
        if "transferIssue" in query:
            remote["issues"] |= {
                "repository": {"nameWithOwner": migration.TARGET},
                "url": f"https://github.com/{migration.TARGET}/issues/100",
                "number": 100,
            }
            remote["transfers"] += 1
            return {"transferIssue": {"issue": dict(remote["issues"])}}
        if "node(id:" in query:
            return {"node": dict(remote["issues"])}
        return {"repository": {"id": "target-id"}}

    monkeypatch.setattr(migration, "gh", api)
    monkeypatch.setattr(migration, "github_items", paginate)
    monkeypatch.setattr(migration, "graphql", graph)
    return remote


def test_issue_transfer_recovers_after_remote_success_and_lost_response(
    snapshot_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = migration.read_json(snapshot_dir / "snapshot.json")
    issue = {
        "number": 3,
        "node_id": "issue-node",
        "html_url": f"https://github.com/{migration.SOURCE}/issues/3",
        "updated_at": "2026-10-19T00:00:00Z",
        "body": "Related to #12 and GH-13.",
    }
    record["issues"] = [issue]
    migration.write_json(snapshot_dir / "snapshot.json", record)
    migration.plan(snapshot_dir, snapshot_dir / "manifest.toml", None)
    remote = github_fake(monkeypatch, record, "cutover")
    remote["issues"] = {
        "body": issue["body"],
        "number": 3,
        "url": issue["html_url"],
        "updatedAt": issue["updated_at"],
        "state": "OPEN",
        "repository": {"nameWithOwner": migration.SOURCE},
    }
    remote["interrupt"] = True
    state = tmp_path / "transferred.json"
    with pytest.raises(RuntimeError, match="Lost response"):
        migration.transfer_issues(snapshot_dir, state, 99)
    migration.transfer_issues(snapshot_dir, state, 99)
    migration.transfer_issues(snapshot_dir, state, 99)
    assert remote["transfers"] == 1
    assert migration.read_json(state)["issue-node"]["destination"].endswith("/issues/100")
    assert f"https://github.com/{migration.SOURCE}/issues/12" in remote["issues"]["body"]
    assert f"https://github.com/{migration.SOURCE}/issues/13" in remote["issues"]["body"]


def test_pr_publication_recovers_lost_response_and_preserves_stack(
    snapshot_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare(snapshot_dir, stacked_pulls(snapshot_dir))
    candidate = tmp_path / "candidate"
    migration.stage(snapshot_dir, candidate)
    destination = tmp_path / "destination.git"
    migration.run("git", "clone", "--quiet", "--bare", str(candidate / "tree"), str(destination))
    cutover = migration.git(destination, "rev-parse", "main")
    record = migration.read_json(snapshot_dir / "snapshot.json")
    remote = github_fake(monkeypatch, record, cutover, destination)
    actual_run = migration.run

    def process(*args: str, **kwargs) -> subprocess.CompletedProcess[bytes]:
        rewritten = [str(destination) if arg == f"https://github.com/{migration.TARGET}.git" else arg for arg in args]
        return actual_run(*rewritten, **kwargs)

    monkeypatch.setattr(migration, "run", process)
    remote["interrupt"] = True
    with pytest.raises(RuntimeError, match="Lost response"):
        migration.publish_prs(candidate, 99)
    migration.publish_prs(candidate, 99)
    migration.publish_prs(candidate, 99)
    assert len(remote["pulls"]) == 2
    assert [item["base"]["ref"] for item in remote["pulls"]] == ["main", "skyrl-migration/pr-1"]
    assert (
        migration.git(
            destination, "show", f"skyrl-migration/pr-2:{migration.VENDOR}/skyrl-train/skyrl_train/relative.py"
        )
        == "value = 20"
    )
