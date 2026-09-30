import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "proposal_adoption_transport", ROOT / "scripts/proposal_adoption_transport.py"
)
transport = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transport)


def _inputs(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "proposal").mkdir(parents=True)
    (checkpoint / "empty").mkdir()
    (checkpoint / "proposal/proposals.json").write_text("[]\n")
    (checkpoint / "snapshot-capture.json").write_text("capture\n")
    (checkpoint / "pull-manifest.json").write_text("pull\n")
    archive, receipt = tmp_path / "source.tar.gz", tmp_path / "launch.json"
    archive.write_bytes(b"source")
    receipt.write_text("{}\n")
    return checkpoint, archive, receipt


def test_transport_round_trip_preserves_full_checkpoint_and_identity(tmp_path):
    checkpoint, source, launch = _inputs(tmp_path)
    archive, document = tmp_path / "adoption.tar.gz", tmp_path / "transport.json"
    packed = transport.pack(checkpoint, source, launch, archive, document)
    destination = tmp_path / "restored"
    transport._unpack(archive, transport.validate_transport(packed), destination)
    assert (destination / "checkpoint/empty").is_dir()
    assert (destination / "checkpoint/proposal/proposals.json").read_text() == "[]\n"
    assert (destination / "source-archive.tar.gz").read_bytes() == b"source"
    assert json.loads(document.read_text())["archive_sha256"] == packed["archive_sha256"]


def test_transport_rejects_extra_archive_member_and_checkpoint_link(tmp_path):
    checkpoint, source, launch = _inputs(tmp_path)
    (checkpoint / "linked").symlink_to(source)
    with pytest.raises(ValueError, match="unsafe"):
        transport.pack(checkpoint, source, launch, tmp_path / "a.tar.gz", tmp_path / "t.json")

    checkpoint.unlink(missing_ok=True) if checkpoint.is_symlink() else None
    (checkpoint / "linked").unlink()
    archive, document = tmp_path / "a.tar.gz", tmp_path / "t.json"
    packed = transport.pack(checkpoint, source, launch, archive, document)
    packed["members"].append({"path": "unexpected", "kind": "file", "bytes": 1, "sha256": "0" * 64})
    with pytest.raises(ValueError, match="member set differs"):
        transport._unpack(archive, packed, tmp_path / "restored")


def test_transport_streams_large_inputs_without_path_read_bytes(tmp_path, monkeypatch):
    checkpoint, source, launch = _inputs(tmp_path)
    (checkpoint / "proposal/raw.bin").write_bytes(b"x" * (2 * 1024 * 1024))
    monkeypatch.setattr(
        Path, "read_bytes", lambda _path: pytest.fail("transport must stream files")
    )
    transport.pack(
        checkpoint, source, launch, tmp_path / "adoption.tar.gz", tmp_path / "transport.json"
    )
