# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the run_one map function.

These tests don't invoke runsc — they cover the parts that are easy to
exercise in a normal pytest run: argument validation, error fallthrough,
the OCI spec builder, the trace decoder, and the stdout/stderr capper.
"""

from __future__ import annotations

from pathlib import Path


from experiments.swe_rebench_trace.run_one import (
    SANDBOX_ENTRYPOINT_PATH,
    SANDBOX_TRACER_PATH,
    _build_oci_config,
    _cap_text,
    _inject_tracer_and_entrypoint,
    _make_error_result,
    _read_trace_file,
    _sanitize_container_id,
    trace_swe_row,
)

# ---------------------------------------------------------------------------
# trace_swe_row error path
# ---------------------------------------------------------------------------


def test_trace_swe_row_missing_test_cmd_returns_error_row():
    row = {"instance_id": "abc", "image_name": "ghcr.io/x/y:latest", "install_config": {}}
    result = trace_swe_row(row)
    assert result["instance_id"] == "abc"
    assert result["image_name"] == "ghcr.io/x/y:latest"
    assert result["returncode"] == -1
    assert "test_cmd" in (result.get("error") or "").lower()
    # The error path doesn't run a sandbox, so the tracer fields are unknown.
    assert result["tracer"] == "unknown"
    assert result["sandbox_python"] == ""


def test_make_error_result_round_trips_to_dict():
    err = _make_error_result(
        instance_id="abc",
        image_name="ghcr.io/x/y:latest",
        test_cmd="pytest",
        error="boom",
        duration_s=1.5,
    )
    d = err.to_dict()
    assert d["instance_id"] == "abc"
    assert d["error"] == "boom"
    assert d["duration_s"] == 1.5
    assert d["trace_events"] == []
    assert d["returncode"] == -1
    assert d["tracer"] == "unknown"
    assert d["sandbox_python"] == ""


# ---------------------------------------------------------------------------
# _cap_text
# ---------------------------------------------------------------------------


def test_cap_text_short_input_unchanged():
    text, truncated = _cap_text(b"hello world", 100)
    assert text == "hello world"
    assert truncated is False


def test_cap_text_long_input_truncates_with_marker():
    blob = b"a" * 1000 + b"BBBB" + b"c" * 1000
    text, truncated = _cap_text(blob, 200)
    assert truncated is True
    assert "[... truncated ...]" in text
    assert text.startswith("a")
    assert text.endswith("c")


def test_cap_text_handles_invalid_utf8():
    blob = b"\xff\xfe\xfd" * 100
    text, truncated = _cap_text(blob, 50)
    assert truncated is True
    # Replacement characters but no exception.
    assert isinstance(text, str)


# ---------------------------------------------------------------------------
# _inject_tracer_and_entrypoint
# ---------------------------------------------------------------------------


def test_inject_tracer_creates_tracer_and_entrypoint(tmp_path: Path):
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()

    _inject_tracer_and_entrypoint(rootfs=rootfs, test_cmd="pytest -q tests/")

    # tracer.py is copied as sitecustomize.py so the site module loads it
    # automatically at every interpreter startup (PYTHONSTARTUP would be
    # wrong: it only fires for interactive interpreters).
    tracer_dst = rootfs / "_marin" / "sitecustomize.py"
    entry_dst = rootfs / "_marin" / "entrypoint.sh"
    assert tracer_dst.exists()
    assert entry_dst.exists()
    # Entrypoint must invoke the row's test command.
    body = entry_dst.read_text()
    assert "pytest -q tests/" in body
    assert "MARIN_TRACE_FD=9" in body
    # The trace fd must point at the host bind-mount, not a tmpfs.
    assert "/_marin_trace/trace.bin" in body
    # set -e must NOT be present: it would short-circuit the exit-code
    # capture and the trace fd close on test failure.
    assert "set -e" not in body
    # The exit-code capture must be on the same line as the test command so
    # the script keeps running even after a non-zero exit.
    assert "; exit_code=$?" in body
    assert "exec 9>&-" in body
    assert "exit $exit_code" in body
    # Executable bit set.
    assert entry_dst.stat().st_mode & 0o111


# ---------------------------------------------------------------------------
# _sanitize_container_id
# ---------------------------------------------------------------------------


def test_sanitize_container_id_replaces_unsafe_chars():
    cid = _sanitize_container_id("django/django__1.2.3")
    # Strip the random suffix and the swe- prefix to assert the safe slug.
    assert cid.startswith("swe-django-django-1-2-3-")
    parts = cid.rsplit("-", 1)
    assert len(parts[1]) == 8  # uuid suffix


def test_sanitize_container_id_falls_back_for_empty_input():
    cid = _sanitize_container_id("///")
    assert cid.startswith("swe-row-")


def test_sanitize_container_id_uniqueness_across_calls():
    a = _sanitize_container_id("instance-x")
    b = _sanitize_container_id("instance-x")
    assert a != b


# ---------------------------------------------------------------------------
# _read_trace_file metadata extraction
# ---------------------------------------------------------------------------


def _frame(record: dict) -> bytes:
    import json
    import struct

    payload = json.dumps(record).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload


def test_read_trace_file_extracts_meta_record(tmp_path: Path):
    path = tmp_path / "trace.bin"
    path.write_bytes(
        _frame({"e": "meta", "tracer": "sys.monitoring", "py": "3.12.4"})
        + _frame({"e": "call", "f": "/testbed/x.py", "l": 1, "n": "f"})
        + _frame({"e": "return", "f": "/testbed/x.py", "l": 5, "n": "f"})
    )
    events, total, truncated, meta = _read_trace_file(path, max_events=10)
    assert meta == {"e": "meta", "tracer": "sys.monitoring", "py": "3.12.4"}
    assert total == 2  # meta does NOT count toward total
    assert truncated is False
    assert [e["e"] for e in events] == ["call", "return"]


def test_read_trace_file_returns_empty_meta_when_missing(tmp_path: Path):
    path = tmp_path / "trace.bin"
    path.write_bytes(_frame({"e": "call", "f": "/testbed/x.py", "l": 1, "n": "f"}))
    _events, total, truncated, meta = _read_trace_file(path, max_events=10)
    assert meta == {}
    assert total == 1
    assert truncated is False


def test_read_trace_file_handles_missing_path(tmp_path: Path):
    events, total, truncated, meta = _read_trace_file(tmp_path / "nope.bin", max_events=10)
    assert events == []
    assert total == 0
    assert truncated is False
    assert meta == {}


# ---------------------------------------------------------------------------
# _build_oci_config
# ---------------------------------------------------------------------------


def test_build_oci_config_merges_env_and_sets_entrypoint(tmp_path: Path):
    image_config = {
        "config": {
            "Env": ["PATH=/usr/local/bin:/usr/bin", "HOME=/root"],
            "WorkingDir": "/testbed",
        },
    }
    spec = _build_oci_config(
        bundle_dir=tmp_path,
        test_cmd="pytest tests/test_x.py",
        image_config=image_config,
        extra_env={
            "PYTHONSTARTUP": SANDBOX_TRACER_PATH,
            "MARIN_TRACE_ROOTS": "/testbed",
            "PATH": "/marin/bin:/usr/bin",  # override
        },
    )
    assert spec["process"]["args"] == ["/bin/sh", SANDBOX_ENTRYPOINT_PATH]
    assert spec["process"]["cwd"] == "/testbed"

    env = dict(e.split("=", 1) for e in spec["process"]["env"])
    assert env["PYTHONSTARTUP"] == SANDBOX_TRACER_PATH
    assert env["MARIN_TRACE_ROOTS"] == "/testbed"
    assert env["PATH"] == "/marin/bin:/usr/bin"  # override won
    assert env["HOME"] == "/root"  # image-only env preserved
    # Network namespace requested (private netns, no host network).
    namespaces = {ns["type"] for ns in spec["linux"]["namespaces"]}
    assert "network" in namespaces


def test_build_oci_config_rootfs_is_readonly_with_writable_overlays(tmp_path: Path):
    """The rootfs must be read-only; /testbed and friends must be writable tmpfs."""
    spec = _build_oci_config(
        bundle_dir=tmp_path,
        test_cmd="pytest",
        image_config={},
        extra_env={},
    )
    assert spec["root"]["readonly"] is True
    mount_dests = {m["destination"]: m for m in spec["mounts"]}
    # Writable overlays must be present and tmpfs-backed.
    for dest in ("/tmp", "/testbed", "/root", "/var/tmp"):
        assert dest in mount_dests, f"missing writable overlay for {dest}"
        assert mount_dests[dest]["type"] == "tmpfs"


def test_build_oci_config_includes_host_trace_bind_mount(tmp_path: Path):
    """When host_trace_dir is set, the spec must add a bind mount for the trace stream."""
    host_trace_dir = tmp_path / "trace_out"
    host_trace_dir.mkdir()
    spec = _build_oci_config(
        bundle_dir=tmp_path,
        test_cmd="pytest",
        image_config={},
        extra_env={},
        host_trace_dir=host_trace_dir,
    )
    mount_dests = {m["destination"]: m for m in spec["mounts"]}
    assert "/_marin_trace" in mount_dests
    bind = mount_dests["/_marin_trace"]
    assert bind["type"] == "bind"
    assert bind["source"] == str(host_trace_dir.resolve())
    assert "rw" in bind["options"]


def test_build_oci_config_omits_trace_bind_when_not_requested(tmp_path: Path):
    spec = _build_oci_config(
        bundle_dir=tmp_path,
        test_cmd="pytest",
        image_config={},
        extra_env={},
    )
    mount_dests = {m["destination"]: m for m in spec["mounts"]}
    assert "/_marin_trace" not in mount_dests


def test_build_oci_config_handles_image_with_no_config(tmp_path: Path):
    spec = _build_oci_config(
        bundle_dir=tmp_path,
        test_cmd="pytest",
        image_config={},
        extra_env={"X": "1"},
    )
    assert spec["process"]["cwd"] == "/"
    env = dict(e.split("=", 1) for e in spec["process"]["env"])
    assert env == {"X": "1"}
