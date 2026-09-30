import os
from pathlib import Path

import pytest

from capability_pipeline.grading_input import grading_input_fingerprint


def test_fingerprint_tracks_exact_files_response_transcript_and_contract(tmp_path):
    root = tmp_path / "workspace"
    (root / "out").mkdir(parents=True)
    artifact = root / "out" / "cleaned.gpkg"
    artifact.write_bytes(b"gpkg\x00")
    base = grading_input_fingerprint(
        {"id": "one"}, {"submission": "out/cleaned.gpkg"}, "done", root, []
    )
    assert base["workspace_file_count"] == 1
    assert base == grading_input_fingerprint(
        {"id": "one"}, {"submission": "out/cleaned.gpkg"}, "done", root, []
    )
    artifact.write_bytes(b"gpkg\x01")
    changed = grading_input_fingerprint(
        {"id": "one"}, {"submission": "out/cleaned.gpkg"}, "done", root, []
    )
    assert changed["submission_sha256"] != base["submission_sha256"]
    artifact.write_bytes(b"gpkg\x00")
    transcript = grading_input_fingerprint(
        {"id": "one"},
        {"submission": "out/cleaned.gpkg"},
        "done",
        root,
        [{"stdout": "timestamp"}],
    )
    assert transcript["submission_sha256"] == base["submission_sha256"]
    assert transcript["grading_input_sha256"] != base["grading_input_sha256"]
    response = grading_input_fingerprint(
        {"id": "one"}, {"submission": "out/cleaned.gpkg"}, "other", root, []
    )
    assert response["submission_sha256"] != base["submission_sha256"]
    contract = grading_input_fingerprint(
        {"id": "two"}, {"submission": "out/cleaned.gpkg"}, "done", root, []
    )
    assert contract["grading_input_sha256"] != base["grading_input_sha256"]


def test_fingerprint_binds_exact_embedded_spec_payload_and_step_index(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    payload = b'{"specification":{"resource":"embedded-a"},"step_index":1}'
    first = grading_input_fingerprint(
        b'{"resource":"embedded-a"}',
        {"kind": "workspace"},
        None,
        root,
        (),
        step_index=1,
        payload=payload,
    )
    changed_embedded_bytes = grading_input_fingerprint(
        b'{"resource":"embedded-b"}',
        {"kind": "workspace"},
        None,
        root,
        (),
        step_index=1,
        payload=payload,
    )
    changed_step = grading_input_fingerprint(
        b'{"resource":"embedded-a"}',
        {"kind": "workspace"},
        None,
        root,
        (),
        step_index=0,
        payload=payload,
    )
    changed_payload = grading_input_fingerprint(
        b'{"resource":"embedded-a"}',
        {"kind": "workspace"},
        None,
        root,
        (),
        step_index=1,
        payload=payload + b" ",
    )
    assert (
        first["specification_sha256"] != changed_embedded_bytes["specification_sha256"]
    )
    assert (
        first["grading_input_sha256"] != changed_embedded_bytes["grading_input_sha256"]
    )
    assert first["grading_input_sha256"] != changed_step["grading_input_sha256"]
    assert first["payload_sha256"] != changed_payload["payload_sha256"]


def test_fingerprint_includes_empty_directories_delivered_with_workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    before = grading_input_fingerprint({}, {}, None, root, ())
    (root / "__external__" / "mounted").mkdir(parents=True)
    delivered = grading_input_fingerprint({}, {}, None, root, ())
    assert before["workspace_file_count"] == delivered["workspace_file_count"] == 0
    assert before["submission_sha256"] != delivered["submission_sha256"]


def test_fingerprint_rejects_links_and_nonregular_entries(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "link").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        grading_input_fingerprint({}, {}, "", root, [])
    (root / "link").unlink()
    os.mkfifo(root / "pipe")
    with pytest.raises(ValueError, match="non-regular"):
        grading_input_fingerprint({}, {}, "", root, [])


def test_fingerprint_rejects_linked_root(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(root)
    with pytest.raises(ValueError, match="real directory"):
        grading_input_fingerprint({}, {}, "", Path(link), [])
