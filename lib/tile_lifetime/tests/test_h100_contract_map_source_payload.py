# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib
import io
import json
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from iris.cluster.bundle import MAX_BUNDLE_SIZE_BYTES
from iris.cluster.client.bundle import create_workspace_zip

payload = importlib.import_module("lib.tile_lifetime.benchmarks.h100_contract_map_source_payload")


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip("\n")


def _source_repository(root: Path, *, symlink_license: bool = False) -> tuple[Path, str, str]:
    source = root / "source"
    source.mkdir()
    _git(source, "init", "--quiet")
    _git(source, "config", "user.name", "Capsule Test")
    _git(source, "config", "user.email", "capsule@example.com")
    for relative in payload.REQUIRED_EXACT_PATHS:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "lib/tile_lifetime/benchmarks/h100_contract_map_source_allowlist.json":
            allowlist = {
                "exact": list(payload.REQUIRED_EXACT_PATHS),
                "recursive": [{"path": "lib/tile_lifetime/src/tile_lifetime", "suffix": ".py"}],
                "schema_version": payload.SCHEMA_VERSION,
            }
            path.write_bytes(payload._canonical_json(allowlist))
        elif relative == "LICENSE" and symlink_license:
            path.symlink_to("pyproject.toml")
        else:
            path.write_text(f"capsule fixture: {relative}\n")
    package = source / "lib/tile_lifetime/src/tile_lifetime"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("\n")
    executable = package / "generated.py"
    executable.write_text("VALUE = 1\n")
    executable.chmod(0o755)
    _git(source, "add", ".")
    _git(source, "commit", "--quiet", "-m", "capsule source")
    return source, _git(source, "rev-parse", "HEAD"), _git(source, "rev-parse", "HEAD^{tree}")


def _prepared_remote(tmp_path: Path, *, symlink_license: bool = False) -> tuple[Path, dict, str, str]:
    source, commit, tree = _source_repository(tmp_path, symlink_license=symlink_license)
    prepared = tmp_path / "prepared"
    result = payload.prepare_source_payload(source, commit, prepared)
    remote = tmp_path / "remote"
    with zipfile.ZipFile(io.BytesIO(create_workspace_zip(prepared))) as bundle:
        bundle.extractall(remote)
    return remote, result, commit, tree


def _restore(remote: Path, result: dict, commit: str, tree: str) -> tuple[Path, dict]:
    return payload.restore_and_verify_source_payload(
        remote,
        expected_manifest_sha256=result["manifest_sha256"],
        expected_source_sha=commit,
        expected_source_tree=tree,
    )


def test_source_capsule_round_trip_restores_exact_files_modes_and_symlinks(tmp_path):
    remote, result, commit, tree = _prepared_remote(tmp_path, symlink_license=True)

    source_root, manifest = _restore(remote, result, commit, tree)

    assert manifest["source"] == {"commit": commit, "tree": tree}
    expected = {record["path"] for record in manifest["members"]}
    actual = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    assert actual == expected
    assert os.readlink(source_root / "LICENSE") == "pyproject.toml"
    assert stat.S_IMODE((source_root / "lib/tile_lifetime/src/tile_lifetime/generated.py").stat().st_mode) == 0o755


def test_source_capsule_creation_rejects_dirty_global_checkout(tmp_path):
    source, commit, _tree = _source_repository(tmp_path)
    (source / "untracked-secret").write_text("not transportable\n")

    with pytest.raises(ValueError, match="globally clean"):
        payload.prepare_source_payload(source, commit, tmp_path / "prepared")


def test_source_capsule_creation_rejects_worktree_mutation_during_preparation(tmp_path, monkeypatch):
    source, commit, _tree = _source_repository(tmp_path)
    load_allowlist = payload._load_allowlist

    def mutate_after_identity(source_root, tree_records):
        paths = load_allowlist(source_root, tree_records)
        (source_root / "LICENSE").write_text("changed during preparation\n")
        return paths

    monkeypatch.setattr(payload, "_load_allowlist", mutate_after_identity)

    with pytest.raises(ValueError, match="globally clean"):
        payload.prepare_source_payload(source, commit, tmp_path / "prepared")


def test_source_capsule_creation_rejects_allowlist_drift(tmp_path):
    source, _commit, _tree = _source_repository(tmp_path)
    allowlist_path = source / payload.ALLOWLIST_RELATIVE_PATH
    allowlist = json.loads(allowlist_path.read_bytes())
    allowlist["exact"].remove("uv.lock")
    allowlist_path.write_bytes(payload._canonical_json(allowlist))
    _git(source, "add", allowlist_path.relative_to(source).as_posix())
    _git(source, "commit", "--quiet", "-m", "drift allowlist")
    commit = _git(source, "rev-parse", "HEAD")

    with pytest.raises(ValueError, match="drifted"):
        payload.prepare_source_payload(source, commit, tmp_path / "prepared")


@pytest.mark.parametrize("mutation", ["changed", "missing", "extra", "mode"])
def test_source_capsule_verifier_rejects_extracted_file_set_or_identity_changes(tmp_path, mutation):
    remote, result, commit, tree = _prepared_remote(tmp_path)
    source_root, manifest = _restore(remote, result, commit, tree)
    target = source_root / "lib/tile_lifetime/src/tile_lifetime/generated.py"
    if mutation == "changed":
        target.write_text("VALUE = 2\n")
    elif mutation == "missing":
        target.unlink()
    elif mutation == "extra":
        (source_root / "unexpected.py").write_text("unexpected\n")
    else:
        target.chmod(0o644)

    with pytest.raises(ValueError, match=r"file set|mode|content"):
        payload._verify_extracted_source(source_root, manifest)


def test_source_capsule_rejects_corruption_before_extraction(tmp_path):
    remote, result, commit, tree = _prepared_remote(tmp_path)
    capsule = remote / payload.CAPSULE_FILENAME
    capsule.write_bytes(capsule.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="capsule SHA-256"):
        _restore(remote, result, commit, tree)

    assert not (remote / "source").exists()


def test_source_capsule_rejects_member_hash_mismatch_with_matching_archive_hash(tmp_path):
    remote, _result, commit, tree = _prepared_remote(tmp_path)
    capsule_path = remote / payload.CAPSULE_FILENAME
    with zipfile.ZipFile(capsule_path) as archive:
        entries = {member.filename: archive.read(member) for member in archive.infolist()}
    entries["LICENSE"] = b"changed\n"
    payload._write_capsule(capsule_path, entries)
    manifest_path = remote / payload.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_bytes())
    manifest["archive"]["sha256"] = hashlib.sha256(capsule_path.read_bytes()).hexdigest()
    raw = payload._canonical_json(manifest)
    manifest_path.write_bytes(raw)

    with pytest.raises(ValueError, match=r"member size|member SHA-256"):
        payload.restore_and_verify_source_payload(
            remote,
            expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            expected_source_sha=commit,
            expected_source_tree=tree,
        )


def test_source_capsule_rejects_traversal_with_matching_transport_hashes(tmp_path):
    remote, _result, commit, tree = _prepared_remote(tmp_path)
    capsule_path = remote / payload.CAPSULE_FILENAME
    with zipfile.ZipFile(capsule_path, "a") as archive:
        info = zipfile.ZipInfo("../credential", date_time=(1980, 1, 1, 0, 0, 0))
        info.external_attr = (stat.S_IFREG | 0o600) << 16
        archive.writestr(info, b"secret")
    manifest_path = remote / payload.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_bytes())
    manifest["members"].insert(
        0,
        {
            "mode": "100644",
            "path": "../credential",
            "sha256": hashlib.sha256(b"secret").hexdigest(),
            "size": 6,
            "type": "file",
        },
    )
    manifest["archive"]["sha256"] = hashlib.sha256(capsule_path.read_bytes()).hexdigest()
    raw = payload._canonical_json(manifest)
    manifest_path.write_bytes(raw)

    with pytest.raises(ValueError, match="normalized relative path"):
        payload.restore_and_verify_source_payload(
            remote,
            expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            expected_source_sha=commit,
            expected_source_tree=tree,
        )
    assert not (remote / "source").exists()
    assert not (remote.parent / "credential").exists()


def test_source_capsule_rejects_escaping_symlink_with_matching_transport_hashes(tmp_path):
    remote, _result, commit, tree = _prepared_remote(tmp_path, symlink_license=True)
    capsule_path = remote / payload.CAPSULE_FILENAME
    with zipfile.ZipFile(capsule_path) as archive:
        entries = {member.filename: archive.read(member) for member in archive.infolist()}
    entries["LICENSE"] = b"../outside"
    payload._write_capsule(capsule_path, entries)
    manifest_path = remote / payload.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_bytes())
    record = next(record for record in manifest["members"] if record["path"] == "LICENSE")
    record["size"] = len(entries["LICENSE"])
    record["sha256"] = hashlib.sha256(entries["LICENSE"]).hexdigest()
    manifest["archive"]["sha256"] = hashlib.sha256(capsule_path.read_bytes()).hexdigest()
    raw = payload._canonical_json(manifest)
    manifest_path.write_bytes(raw)

    with pytest.raises(ValueError, match="escapes"):
        payload.restore_and_verify_source_payload(
            remote,
            expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            expected_source_sha=commit,
            expected_source_tree=tree,
        )


def test_source_capsule_runner_command_fixes_manifest_execute_and_tool_paths(tmp_path):
    arguments = payload.runner_arguments(
        tmp_path,
        "1" * 40,
        "2" * 40,
        tmp_path / payload.MANIFEST_FILENAME,
        "3" * 64,
        tmp_path.parent / "artifacts",
        "0.10.1",
    )

    assert arguments[2:4] == ("--execute", "--source-root")
    assert arguments[arguments.index("--source-tree") + 1] == "2" * 40
    assert arguments[arguments.index("--source-capsule-manifest-sha256") + 1] == "3" * 64
    assert arguments[arguments.index("--nvcc") + 1] == "/usr/local/cuda-13.2/bin/nvcc"
    assert arguments[arguments.index("--ncu") + 1] == "/usr/local/bin/ncu"
    assert arguments[arguments.index("--nsys") + 1] == "/usr/local/bin/nsys"


def test_source_capsule_python_path_prefers_capsule_package_and_benchmark_roots(tmp_path):
    assert payload.capsule_python_path(tmp_path, "/image/site-packages") == os.pathsep.join(
        (str(tmp_path), str(tmp_path / "lib/tile_lifetime/src"), "/image/site-packages")
    )


def test_source_capsule_launcher_identity_rejects_tampering(tmp_path):
    launcher = tmp_path / payload.LAUNCHER_FILENAME
    launcher.write_text("trusted\n")
    identity = hashlib.sha256(launcher.read_bytes()).hexdigest()

    payload.verify_launcher_identity(launcher, identity)
    launcher.write_text("tampered\n")
    with pytest.raises(ValueError, match="trusted launch identity"):
        payload.verify_launcher_identity(launcher, identity)


def test_real_repository_capsule_iris_bundle_is_complete_and_under_limit(tmp_path):
    repository = Path(__file__).resolve().parents[3]
    source = tmp_path / "source-repository"
    subprocess.run(
        ("git", "clone", "--quiet", "--no-local", "--depth=1", repository.as_uri(), str(source)),
        check=True,
    )
    source_sha = _git(source, "rev-parse", "HEAD")
    source_tree = _git(source, "rev-parse", "HEAD^{tree}")
    prepared = tmp_path / "prepared"
    result = payload.prepare_source_payload(source, source_sha, prepared)

    bundle_bytes = create_workspace_zip(prepared)
    assert len(bundle_bytes) < MAX_BUNDLE_SIZE_BYTES
    assert {path.name for path in prepared.iterdir()} == {
        payload.CAPSULE_FILENAME,
        payload.LAUNCHER_FILENAME,
        payload.MANIFEST_FILENAME,
    }
    launcher_record = next(
        record
        for record in result["members"]
        if record["path"] == "lib/tile_lifetime/benchmarks/h100_contract_map_source_payload.py"
    )
    assert result["launcher_sha256"] == launcher_record["sha256"]
    remote = tmp_path / "remote"
    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as bundle:
        bundle.extractall(remote)
    source_root, manifest = _restore(remote, result, source_sha, source_tree)

    payload._verify_extracted_source(source_root, manifest)
    assert len(manifest["members"]) >= 150
    assert (source_root / payload.RUNNER_RELATIVE_PATH).is_file()

    audit_script = """
import importlib
import sys
from pathlib import Path
runner = importlib.import_module("lib.tile_lifetime.benchmarks.h100_contract_map_backend_runner")
importlib.import_module("lib.tile_lifetime.benchmarks.h100_contract_map_backend_training")
tool = Path("/capsule-audit-tool")
config = runner.RunnerConfig(
    source_root=Path(sys.argv[1]),
    source_sha=sys.argv[2],
    source_tree=sys.argv[3],
    source_capsule_manifest=Path(sys.argv[4]),
    source_capsule_manifest_sha256=sys.argv[5],
    artifact_directory=Path(sys.argv[1]).parent / "artifacts",
    tools=runner.ToolPaths(tool, tool, tool, tool, tool, tool, tool),
    require_jax_version="0.10.1",
)
runner.audit_imported_local_modules(config)
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = payload.capsule_python_path(source_root, environment.get("PYTHONPATH"))
    subprocess.run(
        (
            sys.executable,
            "-c",
            audit_script,
            str(source_root),
            source_sha,
            source_tree,
            str(remote / payload.MANIFEST_FILENAME),
            result["manifest_sha256"],
        ),
        cwd=remote,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
