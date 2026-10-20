# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import os
import shutil
import subprocess
import tomllib
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
def snapshot_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for key, value in {
        "GIT_AUTHOR_NAME": "Test User",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test User",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }.items():
        monkeypatch.setenv(key, value)
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
    write(source, "uv.lock", "version = 1\n")
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
glob = "uv.lock"
reason = "independent runtime lock"
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


@pytest.mark.parametrize("reuse_cache", [False, True])
def test_snapshot_keeps_exact_git_links_and_recently_active_old_issues(
    snapshot_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reuse_cache: bool
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
    cache = tmp_path / "cache.git" if reuse_cache else None
    cached_sha = sha
    if cache is not None:
        migration.run("git", "init", "--bare", "--quiet", str(cache))
        migration.git(cache, "fetch", "--quiet", str(source), f"{sha}:refs/snapshot/main")
        write(source, "cloud/iris/launch.py", "from skyrl_train.train import train\n\n")
        sha = save(source, "Upstream advanced after the previous snapshot")
    migration.snapshot(output, "main", 10, datetime(2026, 10, 19, tzinfo=UTC), objects_cache=cache)
    if cache is not None:
        assert migration.git(cache, "rev-parse", "refs/snapshot/main") == cached_sha
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
    assert provenance["patches"][0]["sha256"] == migration.file_digest(snapshot_dir / "migration-patches/runtime.patch")
    assert (source / "skyrl-train/skyrl_train/relative.py").read_text() == "value = 1\n"
    github_fake(monkeypatch, migration.read_json(snapshot_dir / "snapshot.json"), "cutover")
    with pytest.raises(ValueError, match="migration fixes"):
        migration.validate_cutover(99, base, [migration.MigrationPatch(**patch) for patch in provenance["patches"]])


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


def advanced_snapshot(previous: Path, output: Path, edits: dict[str, str], manifest: Path) -> Path:
    output.mkdir()
    migration.run("git", "clone", "--quiet", str(previous / "source"), str(output / "source"))
    for name, contents in edits.items():
        write(output / "source", name, contents)
    save(output / "source", "Advance upstream")
    migration.write_json(output / "snapshot.json", migration.read_json(previous / "snapshot.json"))
    shutil.copyfile(manifest, output / "manifest.toml")
    for name in tomllib.loads(manifest.read_text()).get("patches", []):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(manifest.parent / name, target)
    prepare(output, [])
    return output


def refresh_checkout(snapshot_dir: Path, tmp_path: Path) -> tuple[Path, Path]:
    prepare(snapshot_dir, [])
    migration.stage(snapshot_dir, tmp_path / "original")
    source = tmp_path / "original/tree" / migration.VENDOR
    manifest = tmp_path / "original/tree/config/skyrl-migration.toml"
    manifest.parent.mkdir()
    shutil.copyfile(snapshot_dir / "manifest.toml", manifest)
    return source, manifest


@pytest.mark.parametrize("git_stat_cache", ["default", "minimal"])
def test_refresh_preserves_local_edits_deletions_and_new_files_across_upstream_updates(
    snapshot_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, git_stat_cache: str
) -> None:
    if git_stat_cache == "minimal":
        monkeypatch.setenv("GIT_CONFIG_COUNT", "2")
        monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.checkstat")
        monkeypatch.setenv("GIT_CONFIG_VALUE_0", "minimal")
        monkeypatch.setenv("GIT_CONFIG_KEY_1", "core.trustctime")
        monkeypatch.setenv("GIT_CONFIG_VALUE_1", "false")
        os.utime(snapshot_dir / "source/skyrl-train/skyrl_train/relative.py", (1_700_000_000, 1_700_000_000))
    current, manifest = refresh_checkout(snapshot_dir, tmp_path)
    write(current, "skyrl-train/skyrl_train/relative.py", "value = 7\n")
    if git_stat_cache == "minimal":
        # Archive copies can preserve metadata while changing same-sized contents.
        os.utime(current / "skyrl-train/skyrl_train/relative.py", (1_700_000_000, 1_700_000_000))
    write(current, "cloud/iris/local_extension.py", "local = True\n")
    write(current, "uv.lock", "version = 2\n")
    write_bytes(current, "skyrl-train/skyrl_train/assets/blob.bin", b"\0local\xff")
    (current / "skyrl-train/skyrl_train/assets/old.txt").unlink()
    write(current, ".gitignore", ".venv/\nrelative.py\n")
    write(current, ".venv/keep.txt", "local environment\n")
    updated = advanced_snapshot(
        snapshot_dir,
        tmp_path / "upstream-1",
        {
            "cloud/iris/launch.py": "from .upstream import value\nfrom skyrl_train.train import train\n",
            "cloud/iris/upstream.py": "value = 11\n",
        },
        manifest,
    )
    result = migration.refresh_candidate(snapshot_dir, current, updated, tmp_path / "refresh-1")
    assert result.status is migration.RefreshStatus.READY, result.diagnostic
    migration.apply_refresh(result, current, manifest)
    assert (current / "cloud/iris/upstream.py").read_text() == "value = 11\n"
    assert (current / "skyrl-train/skyrl_train/relative.py").read_text() == "value = 7\n"
    assert (current / "cloud/iris/local_extension.py").read_text() == "local = True\n"
    assert (current / "uv.lock").read_text() == "version = 2\n"
    assert (current / "skyrl-train/skyrl_train/assets/blob.bin").read_bytes() == b"\0local\xff"
    assert not (current / "skyrl-train/skyrl_train/assets/old.txt").exists()
    assert (current / ".venv/keep.txt").read_text() == "local environment\n"
    following = advanced_snapshot(updated, tmp_path / "upstream-2", {"cloud/iris/upstream.py": "value = 12\n"}, manifest)
    migration.git(
        following / "source.git",
        "fetch",
        "--quiet",
        str(updated / "source.git"),
        "refs/snapshot/local-base:refs/snapshot/local-base",
    )
    result = migration.refresh_candidate(updated, current, following, tmp_path / "refresh-2")
    assert result.status is migration.RefreshStatus.READY, result.diagnostic
    migration.apply_refresh(result, current, manifest)
    assert (current / "cloud/iris/upstream.py").read_text() == "value = 12\n"
    assert (current / "skyrl-train/skyrl_train/relative.py").read_text() == "value = 7\n"
    assert (current / "cloud/iris/local_extension.py").read_text() == "local = True\n"
    assert (current / "uv.lock").read_text() == "version = 2\n"
    assert (current / ".venv/keep.txt").read_text() == "local environment\n"
    assert (snapshot_dir / "source/skyrl-train/skyrl_train/relative.py").read_text() == "value = 1\n"


def test_refresh_keeps_a_conflicting_candidate_without_overwriting_local_work(
    snapshot_dir: Path, tmp_path: Path
) -> None:
    current, manifest = refresh_checkout(snapshot_dir, tmp_path)
    write(current, "skyrl-train/skyrl_train/relative.py", "value = 7\n")
    updated = advanced_snapshot(
        snapshot_dir, tmp_path / "upstream", {"skyrl-train/skyrl_train/relative.py": "value = 2\n"}, manifest
    )
    result = migration.refresh_candidate(snapshot_dir, current, updated, tmp_path / "refreshed")
    assert result.status is migration.RefreshStatus.CONFLICT
    assert result.conflicts == [f"{migration.VENDOR}/skyrl-train/skyrl_train/relative.py"]
    assert (current / "skyrl-train/skyrl_train/relative.py").read_text() == "value = 7\n"
    assert migration.git(tmp_path / "refreshed/tree", "diff", "--name-only", "--diff-filter=U")
    resolved = tmp_path / "refreshed/tree" / migration.VENDOR / "skyrl-train/skyrl_train/relative.py"
    resolved.write_text("value = 9\n")
    migration.git(tmp_path / "refreshed/tree", "add", str(resolved))
    migration.finish_refresh(result)
    assert result.status is migration.RefreshStatus.READY
    migration.apply_refresh(result, current, manifest)
    assert (current / "skyrl-train/skyrl_train/relative.py").read_text() == "value = 9\n"
    assert (tmp_path / "refreshed-resolution/tree" / migration.VENDOR / "skyrl-train/skyrl_train/relative.py").exists()


@pytest.mark.parametrize("edited_tree", ["checkout", "candidate"])
def test_refresh_refuses_to_overwrite_an_edit_made_after_preview(
    snapshot_dir: Path, tmp_path: Path, edited_tree: str
) -> None:
    current, manifest = refresh_checkout(snapshot_dir, tmp_path)
    updated = advanced_snapshot(snapshot_dir, tmp_path / "upstream", {"cloud/iris/extra.py": "value = 11\n"}, manifest)
    result = migration.refresh_candidate(snapshot_dir, current, updated, tmp_path / "refreshed")
    assert result.status is migration.RefreshStatus.READY, result.diagnostic
    edited = current if edited_tree == "checkout" else tmp_path / "refreshed/tree" / migration.VENDOR
    write(edited, "skyrl-train/skyrl_train/relative.py", "value = 8\n")
    with pytest.raises(ValueError, match="changed"):
        migration.apply_refresh(result, current, manifest)
    assert (edited / "skyrl-train/skyrl_train/relative.py").read_text() == "value = 8\n"
    assert not (current / "cloud/iris/extra.py").exists()


@pytest.mark.parametrize("defect", ["missing", "changed", "untracked"])
def test_wheel_audit_accepts_source_edits_and_rejects_different_packaged_contents(
    snapshot_dir: Path, tmp_path: Path, defect: str
) -> None:
    prepare(snapshot_dir, [])
    candidate = tmp_path / "candidate"
    migration.stage(snapshot_dir, candidate)
    source = candidate / "tree" / migration.VENDOR
    write_bytes(source, "skyrl-train/skyrl_train/assets/blob.bin", b"edited runtime data")
    write(source, "marinskyrl/normal_edit.py", "value = 7\n")
    wheel = tmp_path / "candidate.whl"
    asset = "skyrl_train/assets/blob.bin"

    def package(omit_asset: bool) -> None:
        with zipfile.ZipFile(wheel, "w") as archive:
            for member, path in migration.wheel_source_files(source).items():
                if omit_asset and member == asset and defect != "untracked":
                    if defect == "changed":
                        archive.writestr(member, b"changed runtime data")
                    continue
                archive.write(path, member)
            if omit_asset and defect == "untracked":
                archive.writestr("marinskyrl/untracked.py", "raise RuntimeError('untracked runtime')")

    package(omit_asset=False)
    assert asset in migration.audit_wheel(wheel, source)
    with zipfile.ZipFile(wheel) as archive:
        assert archive.read(asset) == b"edited runtime data"
        assert archive.read("marinskyrl/normal_edit.py") == b"value = 7\n"
    package(omit_asset=True)
    with pytest.raises(ValueError, match=r"Wheel member missing or changed|outside the checked-out source"):
        migration.audit_wheel(wheel, source)


def github_fake(
    monkeypatch: pytest.MonkeyPatch, record: dict, cutover_sha: str, destination: Path | None = None
) -> dict:
    remote: dict = {"pulls": [], "issues": {}, "transfers": 0, "interrupt": False, "rewrite_references": False}
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
            if remote["rewrite_references"]:
                remote["issues"]["body"] = (
                    remote["issues"]["body"]
                    .replace("#12", f"{migration.SOURCE}#12")
                    .replace("#298", f"{migration.SOURCE}#298")
                )
            remote["transfers"] += 1
            return {"transferIssue": {"issue": dict(remote["issues"])}}
        if "node(id:" in query:
            return {"node": dict(remote["issues"])}
        return {"repository": {"id": "target-id"}}

    monkeypatch.setattr(migration, "gh", api)
    monkeypatch.setattr(migration, "github_items", paginate)
    monkeypatch.setattr(migration, "graphql", graph)
    return remote


@pytest.mark.parametrize("rewrite_references", [False, True])
@pytest.mark.parametrize("human_edit", [False, True])
def test_issue_transfer_recovers_after_remote_success_and_protects_edits(
    snapshot_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rewrite_references: bool, human_edit: bool
) -> None:
    record = migration.read_json(snapshot_dir / "snapshot.json")
    issue = {
        "number": 3,
        "node_id": "issue-node",
        "html_url": f"https://github.com/{migration.SOURCE}/issues/3",
        "updated_at": "2026-10-19T00:00:00Z",
        "body": (
            "Related to #12 and GH-13. Literal `#14`. "
            f"Already qualified {migration.SOURCE}#15. "
            "See [upstream #298](https://github.com/NovaSky-AI/SkyRL/pull/298)."
        ),
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
    remote["rewrite_references"] = rewrite_references
    state = tmp_path / "transferred.json"
    with pytest.raises(RuntimeError, match="Lost response"):
        migration.transfer_issues(snapshot_dir, state, 99)
    if human_edit:
        remote["issues"]["body"] += "\nHuman clarification."
        edited = remote["issues"]["body"]
        with pytest.raises(ValueError, match="body changed"):
            migration.transfer_issues(snapshot_dir, state, 99)
        assert remote["issues"]["body"] == edited
        assert remote["transfers"] == 1
        return
    migration.transfer_issues(snapshot_dir, state, 99)
    migration.transfer_issues(snapshot_dir, state, 99)
    assert remote["transfers"] == 1
    assert migration.read_json(state)["issue-node"]["destination"].endswith("/issues/100")
    assert f"https://github.com/{migration.SOURCE}/issues/12" in remote["issues"]["body"]
    assert f"https://github.com/{migration.SOURCE}/issues/13" in remote["issues"]["body"]
    assert "[upstream #298](https://github.com/NovaSky-AI/SkyRL/pull/298)" in remote["issues"]["body"]
    assert f"Already qualified {migration.SOURCE}#15." in remote["issues"]["body"]


@pytest.mark.parametrize("adapted", [False, True])
def test_pr_publication_recovers_lost_response_and_preserves_stack(
    snapshot_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, adapted: bool
) -> None:
    prepare(snapshot_dir, stacked_pulls(snapshot_dir))
    candidate = tmp_path / "candidate"
    migration.stage(snapshot_dir, candidate)
    destination = tmp_path / "destination.git"
    migration.run("git", "clone", "--quiet", "--bare", str(candidate / "tree"), str(destination))
    cutover = migration.git(destination, "rev-parse", "main")
    for outcome in migration.read_stage_report(candidate / "report.json").pulls:
        migration.git(destination, "update-ref", "-d", f"refs/heads/{outcome.branch}")
    record = migration.read_json(snapshot_dir / "snapshot.json")
    ports_path = None
    if adapted:
        report = migration.read_stage_report(candidate / "report.json")
        original = tmp_path / "adaptations.json"
        ports = []
        for outcome in report.pulls:
            if outcome.status is not migration.PullStatus.READY:
                continue
            tree = candidate / "tree"
            before = migration.git(tree, "rev-parse", outcome.commit + "^")
            migration.git(tree, "checkout", "--quiet", "--detach", outcome.commit)
            write(tree, f"docs/port-{outcome.number}.md", f"Adapted port {outcome.number}\n")
            revised = save(tree, "Add port documentation")
            patch = tmp_path / f"adapted-{outcome.number}.patch"
            patch.write_bytes(migration.run("git", "-C", str(tree), "diff", "--binary", before, revised).stdout)
            ports.append(
                {
                    "number": outcome.number,
                    "source_head": next(p["head"]["sha"] for p in record["pulls"] if p["number"] == outcome.number),
                    "parent": outcome.parent,
                    "patch": str(patch),
                    "title": f"Adapt change {outcome.number}",
                    "body": "Preserve the runtime change and its supporting documentation.",
                    "notes": [],
                }
            )
        migration.write_json(original, ports)
        prepared = tmp_path / "prepared"
        migration.prepare_ports(candidate, original, tree, cutover, prepared)
        ports_path = prepared / "ports.json"
    remote = github_fake(monkeypatch, record, cutover, destination)
    actual_run = migration.run

    def process(*args: str, **kwargs) -> subprocess.CompletedProcess[bytes]:
        rewritten = [str(destination) if arg == f"https://github.com/{migration.TARGET}.git" else arg for arg in args]
        return actual_run(*rewritten, **kwargs)

    monkeypatch.setattr(migration, "run", process)
    remote["interrupt"] = True
    with pytest.raises(RuntimeError, match="Lost response"):
        migration.publish_prs(candidate, 99, ports_path)
    migration.publish_prs(candidate, 99, ports_path)
    migration.publish_prs(candidate, 99, ports_path)
    assert len(remote["pulls"]) == 2
    assert [item["base"]["ref"] for item in remote["pulls"]] == ["main", "skyrl-migration/pr-1"]
    assert (
        migration.git(
            destination, "show", f"skyrl-migration/pr-2:{migration.VENDOR}/skyrl-train/skyrl_train/relative.py"
        )
        == "value = 20"
    )
    if adapted:
        assert migration.git(destination, "show", "skyrl-migration/pr-2:docs/port-2.md") == "Adapted port 2"
        for field in ("title", "body"):
            original_text = remote["pulls"][0][field]
            remote["pulls"][0][field] += "Changed text"
            with pytest.raises(ValueError, match="text changed"):
                migration.publish_prs(candidate, 99, ports_path)
            remote["pulls"][0][field] = original_text
        for field in ("source_head", "sha256"):
            captured = migration.read_json(ports_path)
            captured["ports"][0][field] = "changed"
            migration.write_json(ports_path, captured)
            with pytest.raises(ValueError, match="changed"):
                migration.publish_prs(candidate, 99, ports_path)
            captured["ports"][0][field] = (
                ports[0]["source_head"]
                if field == "source_head"
                else migration.file_digest(prepared / captured["ports"][0]["patch"])
            )
            migration.write_json(ports_path, captured)
