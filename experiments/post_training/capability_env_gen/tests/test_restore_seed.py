import gzip
import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("restore_seed", ROOT / "scripts/restore_seed.py")
assert SPEC and SPEC.loader
restore_seed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore_seed)


def write_archive(path: Path, members: dict[str, bytes], *, executable: set[str] | None = None) -> str:
    executable = executable or set()
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for name, payload in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o755 if name in executable else 0o644
            archive.addfile(info, io.BytesIO(payload))
    path.write_bytes(gzip.compress(raw.getvalue(), mtime=0))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_archive_seed_restores_when_transport_omits_tree(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    seed = source / "restore-seed"
    seed.mkdir()  # Simulates an Iris transport that omits nested seed members.
    members = {"items/c17/status.json": b"{\"state\":\"failed\"}\n", "items/c17/task.txt": b"task\n"}
    archive = source / "restore-seed.tar.gz"
    archive_digest = write_archive(archive, members, executable={"items/c17/task.txt"})
    manifest = source / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "seed_members": {
                    name: hashlib.sha256(payload).hexdigest()
                    for name, payload in members.items()
                },
                "seed_archive": {"name": archive.name, "sha256": archive_digest, "members": len(members)},
            }
        )
    )

    results = tmp_path / "results"
    restore_seed.load_seed(manifest, seed, results)

    assert (results / "items/c17/status.json").read_bytes() == members["items/c17/status.json"]
    restored_task = results / "items/c17/task.txt"
    assert restored_task.read_bytes() == members["items/c17/task.txt"]
    assert restored_task.stat().st_mode & 0o111 == 0o111


def test_direct_seed_mismatch_reports_bounded_path_delta(tmp_path: Path):
    seed = tmp_path / "restore-seed"
    seed.mkdir()
    (seed / "unexpected.txt").write_text("x\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"seed_members": {"expected.txt": "0" * 64}}))

    with pytest.raises(SystemExit, match=r"missing=1 \(expected.txt\); extra=1 \(unexpected.txt\); changed=0"):
        restore_seed.load_seed(manifest, seed, None)


def test_destination_collision_preflight_does_not_partially_restore(tmp_path: Path):
    seed = tmp_path / "restore-seed"
    seed.mkdir()
    members = {"items/a.txt": b"first", "worker.log": b"historical"}
    for name, payload in members.items():
        path = seed / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "seed_members": {
                    name: hashlib.sha256(payload).hexdigest()
                    for name, payload in members.items()
                }
            }
        )
    )
    results = tmp_path / "results"
    results.mkdir()
    (results / "worker.log").write_bytes(b"worker-started")

    with pytest.raises(SystemExit, match="worker.log"):
        restore_seed.load_seed(manifest, seed, results)
    assert not (results / "items/a.txt").exists()
