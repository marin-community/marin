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

    tracer_dst = rootfs / "_marin" / "tracer.py"
    entry_dst = rootfs / "_marin" / "entrypoint.sh"
    assert tracer_dst.exists()
    assert entry_dst.exists()
    # Entrypoint must invoke the row's test command.
    body = entry_dst.read_text()
    assert "pytest -q tests/" in body
    assert "MARIN_TRACE_FD=9" in body
    # Executable bit set.
    assert entry_dst.stat().st_mode & 0o111


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
